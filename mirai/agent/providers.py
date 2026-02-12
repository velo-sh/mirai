"""
LLM Provider implementations for Mirai.

Supports Anthropic direct API and Google Cloud Code Assist (Antigravity) routing.
Uses Pydantic models for validated responses and orjson for fast serialization.
"""

import os
import time
from typing import Any

import anthropic
import httpx
import orjson
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock
from mirai.logging import get_logger

log = get_logger("mirai.providers")


class _RetryableAPIError(Exception):
    """Raised on 429/503 to trigger tenacity retry."""

    def __init__(self, status: int, detail: str):
        self.status = status
        super().__init__(f"API error {status}: {detail}")


class AnthropicProvider:
    """Direct Anthropic API provider."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(4),
        before_sleep=lambda rs: log.warning(
            "anthropic_rate_limited",
            attempt=rs.attempt_number,
            wait=rs.next_action.sleep,  # type: ignore[union-attr]
        ),
    )
    async def generate_response(
        self, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ProviderResponse:
        response = await self.client.messages.create(
            model=model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            max_tokens=4096,
        )
        # Convert Anthropic SDK response to our Pydantic model
        content_blocks: list[TextBlock | ToolUseBlock] = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolUseBlock(id=block.id, name=block.name, input=block.input)  # type: ignore[arg-type]
                )
        return ProviderResponse(content=content_blocks, stop_reason=response.stop_reason or "end_turn")


class AntigravityProvider:
    """
    Routes Claude/Gemini API calls through Google Cloud Code Assist.

    Uses the v1internal:streamGenerateContent endpoint at cloudcode-pa.googleapis.com.
    Messages are sent in Google Generative AI format (contents/parts) and responses
    are parsed back to Pydantic models for AgentLoop compatibility.
    """

    DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
    DAILY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
    ENDPOINTS = [DEFAULT_ENDPOINT, DAILY_ENDPOINT]

    DEFAULT_MODEL = "claude-sonnet-4-5-20250514"
    ANTIGRAVITY_VERSION = "1.15.8"

    # Model name mapping from standard names to Cloud Code Assist internal IDs
    MODEL_MAP = {
        # Claude 4.x series
        "claude-sonnet-4-20250514": "claude-sonnet-4-5",
        "claude-sonnet-4-latest": "claude-sonnet-4-5",
        "claude-opus-4-20250514": "claude-opus-4-5-thinking",
        "claude-opus-4-latest": "claude-opus-4-5-thinking",
        # Claude 3.x series (legacy)
        "claude-3-5-sonnet-20241022": "claude-sonnet-4-5",
        "claude-3-5-sonnet-latest": "claude-sonnet-4-5",
        "claude-3-7-sonnet-20250219": "claude-sonnet-4-5",
        "claude-3-7-sonnet-latest": "claude-sonnet-4-5",
        "claude-3-opus-20240229": "claude-opus-4-5-thinking",
        "claude-3-5-haiku-20241022": "claude-sonnet-4-5",
        "claude-3-haiku-20240307": "claude-sonnet-4-5",
        # Direct Cloud Code Assist IDs (pass-through)
        "claude-sonnet-4-5": "claude-sonnet-4-5",
        "claude-opus-4-5-thinking": "claude-opus-4-5-thinking",
        "gemini-2.0-flash": "gemini-2.0-flash",
    }

    def __init__(self, credentials: dict[str, Any] | None = None, model: str = "claude-sonnet-4-20250514"):
        from mirai.auth.antigravity_auth import load_credentials

        loaded = credentials or load_credentials()
        if not loaded:
            raise FileNotFoundError(
                "No Antigravity credentials found. Run `python -m mirai.auth.auth_cli` to authenticate."
            )
        self.credentials: dict[str, Any] = loaded
        self.model = model
        self._http = httpx.AsyncClient(timeout=120.0, http2=True)

    async def _ensure_fresh_token(self) -> None:
        """Refresh the access token if expired."""
        if time.time() >= self.credentials.get("expires", 0):
            from mirai.auth.antigravity_auth import refresh_access_token, save_credentials

            log.info("token_refresh_started")
            refreshed = await refresh_access_token(self.credentials["refresh"])
            self.credentials["access"] = refreshed["access"]
            self.credentials["expires"] = refreshed["expires"]
            save_credentials(self.credentials)
            log.info("token_refresh_complete")

    def _build_headers(self) -> dict[str, str]:
        """Build request headers for Cloud Code Assist."""
        return {
            "Authorization": f"Bearer {self.credentials['access']}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": f"antigravity/{self.ANTIGRAVITY_VERSION} darwin/arm64",
            "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
            "Client-Metadata": orjson.dumps(
                {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                }
            ).decode(),
        }

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-format messages to Google Generative AI format."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            parts: list[dict[str, Any]] = []
            content = msg.get("content", "")

            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append({"text": block["text"]})
                        elif block.get("type") == "tool_use":
                            parts.append(
                                {  # type: ignore[dict-item]
                                    "functionCall": {
                                        "name": block["name"],
                                        "args": block.get("input", {}),
                                    }
                                }
                            )
                        elif block.get("type") == "tool_result":
                            result_text = block.get("content", "")
                            if isinstance(result_text, list):
                                result_text = " ".join(
                                    b.get("text", "")
                                    for b in result_text
                                    if isinstance(b, dict) and b.get("type") == "text"
                                )
                            parts.append(
                                {  # type: ignore[dict-item]
                                    "functionResponse": {
                                        "name": block.get("tool_use_id", "unknown"),
                                        "response": {"result": str(result_text)},
                                    }
                                }
                            )

            if parts:
                contents.append({"role": role, "parts": parts})
        return contents

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-format tools to Google Generative AI format."""
        if not tools:
            return []
        declarations = []
        for tool in tools:
            decl: dict[str, Any] = {
                "name": tool["name"],
                "description": tool.get("description", ""),
            }
            schema = tool.get("input_schema", {})
            if schema:
                decl["parameters"] = schema
            declarations.append(decl)
        return [{"functionDeclarations": declarations}]

    def _build_request(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Build the Cloud Code Assist request body."""
        import random

        contents = self._convert_messages(messages)
        request: dict[str, Any] = {"contents": contents}

        if system:
            request["systemInstruction"] = {
                "parts": [{"text": system}],
            }

        request["generationConfig"] = {"maxOutputTokens": max_tokens}

        google_tools = self._convert_tools(tools)
        if google_tools:
            request["tools"] = google_tools

        project_id = self.credentials.get("project_id", "")
        return {
            "project": project_id,
            "model": model,
            "request": request,
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": f"agent-{int(time.time())}-{random.randbytes(4).hex()}",
        }

    @staticmethod
    def _parse_sse_response(raw_text: str) -> ProviderResponse:
        """
        Parse SSE stream response and build ProviderResponse.
        Uses orjson for fast JSON parsing of each SSE data line.
        """
        content_blocks: list[TextBlock | ToolUseBlock] = []
        stop_reason = "end_turn"
        tool_call_counter = 0

        # Track last text block for merging consecutive text chunks
        last_text_idx = -1

        for line in raw_text.split("\n"):
            if not line.startswith("data:"):
                continue
            json_bytes = line[5:].strip().encode()
            if not json_bytes:
                continue
            try:
                chunk = orjson.loads(json_bytes)
            except orjson.JSONDecodeError:
                continue

            response_data = chunk.get("response")
            if not response_data:
                continue

            candidate = None
            candidates = response_data.get("candidates", [])
            if candidates:
                candidate = candidates[0]

            if candidate and candidate.get("content", {}).get("parts"):
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        # Merge consecutive text blocks
                        if last_text_idx >= 0 and isinstance(content_blocks[last_text_idx], TextBlock):
                            # TextBlock is frozen, create new merged block
                            old = content_blocks[last_text_idx]
                            assert isinstance(old, TextBlock)
                            content_blocks[last_text_idx] = TextBlock(text=old.text + part["text"])
                        else:
                            content_blocks.append(TextBlock(text=part["text"]))
                            last_text_idx = len(content_blocks) - 1
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        tool_call_counter += 1
                        call_id = fc.get("id", f"{fc['name']}_{int(time.time())}_{tool_call_counter}")
                        content_blocks.append(
                            ToolUseBlock(
                                id=call_id,
                                name=fc["name"],
                                input=fc.get("args", {}),
                            )
                        )
                        stop_reason = "tool_use"
                        last_text_idx = -1  # Reset text merge tracking

            if candidate and candidate.get("finishReason"):
                fr = candidate["finishReason"]
                if fr == "STOP":
                    stop_reason = "end_turn"
                elif fr == "MAX_TOKENS":
                    stop_reason = "max_tokens"

        if not content_blocks:
            content_blocks.append(TextBlock())

        return ProviderResponse(content=content_blocks, stop_reason=stop_reason)

    @retry(
        retry=retry_if_exception_type(_RetryableAPIError),
        wait=wait_exponential(min=2, max=30),
        stop=stop_after_attempt(4),
        before_sleep=lambda rs: log.warning(
            "rate_limited",
            attempt=rs.attempt_number,
            wait=rs.next_action.sleep,  # type: ignore[union-attr]
            status=getattr(rs.outcome.exception(), "status", None),  # type: ignore[union-attr]
        ),
    )
    async def generate_response(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        await self._ensure_fresh_token()

        # Remap model name if needed for Cloud Code Assist
        effective_model = self.MODEL_MAP.get(model, model)
        if effective_model != model:
            log.info("model_remapped", original=model, effective=effective_model)
        elif model not in self.MODEL_MAP:
            log.warning("model_not_in_map", model=model)

        body = self._build_request(effective_model, system, messages, tools)
        headers = self._build_headers()

        # orjson.dumps returns bytes — httpx accepts bytes directly (no decode overhead)
        body_bytes = orjson.dumps(body)

        url = f"{self.DEFAULT_ENDPOINT}/v1internal:streamGenerateContent?alt=sse"
        response = await self._http.post(url, content=body_bytes, headers=headers)

        if response.status_code == 200:
            return self._parse_sse_response(response.text)

        error_text = response.text[:500]
        log.error("api_error", status=response.status_code, detail=error_text[:200])

        if response.status_code == 401:
            self.credentials["expires"] = 0
            raise RuntimeError("Cloud Code Assist: authentication expired")

        if response.status_code in (429, 503):
            raise _RetryableAPIError(response.status_code, error_text[:200])

        raise RuntimeError(f"Cloud Code Assist API error ({response.status_code}): {error_text}")


def create_provider(model: str = "claude-sonnet-4-20250514") -> AnthropicProvider | AntigravityProvider:
    """
    Auto-detect and create the best available provider.

    Priority:
    1. Antigravity credentials (~/.mirai/antigravity_credentials.json)
    2. ANTHROPIC_API_KEY environment variable

    Args:
        model: Default model name to use for generation.
    """
    # Try Antigravity first
    from mirai.auth.antigravity_auth import load_credentials

    creds = load_credentials()
    if creds:
        try:
            provider = AntigravityProvider(credentials=creds, model=model)
            log.info("provider_initialized", provider="antigravity", model=model)
            return provider
        except Exception as e:
            log.warning("antigravity_fallback", error=str(e))

    # Fall back to direct Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        log.info("provider_initialized", provider="anthropic", model=model)
        return AnthropicProvider(api_key=api_key, model=model)

    raise ValueError(
        "No API credentials available. Either:\n"
        "  1. Run `python -m mirai.auth.auth_cli` for Antigravity auth, or\n"
        "  2. Set ANTHROPIC_API_KEY environment variable."
    )


# ---------------------------------------------------------------------------
# Mock providers for testing
# ---------------------------------------------------------------------------


class MockEmbeddingProvider:
    """Provides consistent fake embeddings for testing."""

    def __init__(self, dim: int = 1536):
        self.dim = dim

    async def get_embeddings(self, text: str) -> list[float]:
        """Return a deterministic fake vector based on text hash."""
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        return [(h[i % 32] / 255.0) - 0.5 for i in range(self.dim)]


class MockProvider:
    """Mock provider to test the AgentLoop logic without an API key."""

    def __init__(self) -> None:
        self.call_count = 0
        self.model = "mock-model"

    async def generate_response(
        self, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ProviderResponse:
        self.call_count += 1

        if "Recovered Memories" in system:
            log.debug("mock_memories_present", system_tail=system[system.find("### Recovered Memories") :])

        if "# IDENTITY" in system:
            log.debug("mock_identity_anchors_present")

        last_message = messages[-1]
        last_content = last_message.get("content", "")
        if isinstance(last_content, list):
            last_text = "".join(
                [c.get("text", "") for c in last_content if isinstance(c, dict) and c.get("type") == "text"]
            )
        else:
            last_text = str(last_content)

        # Extract text from all messages to find the user's original intent
        full_history_text = ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                full_history_text += " ".join(
                    [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                )
            else:
                full_history_text += str(content)

        last_text = str(last_text)
        log.debug("mock_request", last_text=last_text[:50])

        # 1. Thinking Turn
        if "analyze the situation" in last_text or "perform a self-reflection" in last_text:
            return ProviderResponse(
                content=[
                    TextBlock(
                        text="<thinking>The system heartbeat has triggered. I should scan the project and summarize progress.</thinking>"
                    )
                ],
                stop_reason="end_turn",
            )

        # 2. Critique Turn
        if "Critique your response" in last_text:
            return ProviderResponse(
                content=[
                    TextBlock(
                        text="The response is aligned with my SOUL.md. Final version: I have successfully completed the proactive scan."
                    )
                ],
                stop_reason="end_turn",
            )

        # 3. Tool Turn / Sequential Logic
        messages[-1].get("role")

        if tools:
            # E2E Proactive Maintenance Workflow
            if "maintenance_check" in full_history_text.lower():
                shell_results = [
                    m
                    for m in messages
                    if m.get("role") == "user"
                    and any("tool_result" in str(c) and "call_shell_maint" in str(c) for c in (m.get("content") or []))
                ]

                if not shell_results:
                    return ProviderResponse(
                        content=[
                            TextBlock(text="Checking for maintenance issues..."),
                            ToolUseBlock(
                                id="call_shell_maint",
                                name="shell_tool",
                                input={"command": "ls maintenance_fixed.txt"},
                            ),
                        ],
                        stop_reason="tool_use",
                    )

                editor_results = [
                    m
                    for m in messages
                    if m.get("role") == "user"
                    and any("tool_result" in str(c) and "call_edit_maint" in str(c) for c in (m.get("content") or []))
                ]

                if not editor_results:
                    return ProviderResponse(
                        content=[
                            TextBlock(text="Fixing the issue..."),
                            ToolUseBlock(
                                id="call_edit_maint",
                                name="editor_tool",
                                input={
                                    "action": "write",
                                    "path": "maintenance_fixed.txt",
                                    "content": "HEALED: Sanity check passed.",
                                },
                            ),
                        ],
                        stop_reason="tool_use",
                    )

            # Default single-tool branch for other tests
            has_any_tool_use = any(
                msg.get("role") == "assistant" and "tool_use" in str(msg.get("content", "")) for msg in messages
            )
            if not has_any_tool_use:
                # Workspace List Request
                if "List the files" in last_text or "SYSTEM_HEARTBEAT" in last_text:
                    # ... (rest of old logic)
                    pass

        # Final Answer if no more tools needed
        return ProviderResponse(
            content=[TextBlock(text="I have finished the complex task.")],
            stop_reason="end_turn",
        )
