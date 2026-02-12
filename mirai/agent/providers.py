import os
import time
from types import SimpleNamespace
from typing import Any

import anthropic


class AnthropicProvider:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = model

    async def generate_response(
        self, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        return await self.client.messages.create(
            model=model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            max_tokens=4096,
        )


class AntigravityProvider:
    """
    Routes Claude/Gemini API calls through Google Cloud Code Assist.

    Uses the v1internal:streamGenerateContent endpoint at cloudcode-pa.googleapis.com.
    Messages are sent in Google Generative AI format (contents/parts) and responses
    are parsed back to Anthropic-compatible objects for AgentLoop compatibility.
    """

    DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
    DAILY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
    ENDPOINTS = [DEFAULT_ENDPOINT, DAILY_ENDPOINT]

    DEFAULT_MODEL = "claude-sonnet-4-5-20250514"
    ANTIGRAVITY_VERSION = "1.15.8"

    # Model name mapping from Anthropic SDK names to Cloud Code Assist names
    MODEL_MAP = {
        "claude-3-5-sonnet-20241022": "claude-sonnet-4-5",
        "claude-3-5-sonnet-latest": "claude-sonnet-4-5",
        "claude-3-7-sonnet-20250219": "claude-sonnet-4-5",
        "claude-3-7-sonnet-latest": "claude-sonnet-4-5",
        "claude-3-opus-20240229": "claude-opus-4-5-thinking",
        "claude-3-5-haiku-20241022": "claude-sonnet-4-5",
        "claude-3-haiku-20240307": "claude-sonnet-4-5",
    }

    def __init__(self, credentials: dict[str, Any] | None = None, model: str = "claude-sonnet-4-20250514"):
        import httpx

        from mirai.auth.antigravity_auth import load_credentials

        loaded = credentials or load_credentials()
        if not loaded:
            raise FileNotFoundError(
                "No Antigravity credentials found. Run `python -m mirai.auth.auth_cli` to authenticate."
            )
        self.credentials: dict[str, Any] = loaded
        self.model = model
        self._http = httpx.AsyncClient(timeout=120.0)

    async def _ensure_fresh_token(self) -> None:
        """Refresh the access token if expired."""
        if time.time() >= self.credentials.get("expires", 0):
            from mirai.auth.antigravity_auth import refresh_access_token, save_credentials

            print("[antigravity] Access token expired, refreshing...")
            refreshed = await refresh_access_token(self.credentials["refresh"])
            self.credentials["access"] = refreshed["access"]
            self.credentials["expires"] = refreshed["expires"]
            save_credentials(self.credentials)
            print("[antigravity] Token refreshed.")

    def _build_headers(self) -> dict[str, str]:
        """Build request headers for Cloud Code Assist."""
        import json as _json

        return {
            "Authorization": f"Bearer {self.credentials['access']}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": f"antigravity/{self.ANTIGRAVITY_VERSION} darwin/arm64",
            "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
            "Client-Metadata": _json.dumps(
                {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                }
            ),
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
            decl = {
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
    def _parse_sse_response(raw_text: str) -> SimpleNamespace:
        """
        Parse SSE stream response and build Anthropic-compatible response object.
        Returns a dict with 'content' list and 'stop_reason'.
        """
        import json as _json
        from types import SimpleNamespace

        content_blocks: list[SimpleNamespace] = []
        stop_reason = "end_turn"
        tool_call_counter = 0

        for line in raw_text.split("\n"):
            if not line.startswith("data:"):
                continue
            json_str = line[5:].strip()
            if not json_str:
                continue
            try:
                chunk = _json.loads(json_str)
            except _json.JSONDecodeError:
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
                        if content_blocks and content_blocks[-1].type == "text":
                            content_blocks[-1].text += part["text"]
                        else:
                            content_blocks.append(SimpleNamespace(type="text", text=part["text"]))
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        tool_call_counter += 1
                        call_id = fc.get("id", f"{fc['name']}_{int(time.time())}_{tool_call_counter}")
                        content_blocks.append(
                            SimpleNamespace(
                                type="tool_use",
                                id=call_id,
                                name=fc["name"],
                                input=fc.get("args", {}),
                            )
                        )
                        stop_reason = "tool_use"

            if candidate and candidate.get("finishReason"):
                fr = candidate["finishReason"]
                if fr == "STOP":
                    stop_reason = "end_turn"
                elif fr == "MAX_TOKENS":
                    stop_reason = "max_tokens"

        if not content_blocks:
            content_blocks.append(SimpleNamespace(type="text", text=""))

        return SimpleNamespace(content=content_blocks, stop_reason=stop_reason)

    async def generate_response(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> SimpleNamespace:
        import asyncio

        await self._ensure_fresh_token()

        # Remap model name if needed for Cloud Code Assist
        effective_model = self.MODEL_MAP.get(model, model)
        if effective_model != model:
            print(f"[antigravity] Model remapped: {model} -> {effective_model}")

        body = self._build_request(effective_model, system, messages, tools)
        headers = self._build_headers()

        import json as _json

        body_json = _json.dumps(body)

        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries + 1):
            url = f"{self.DEFAULT_ENDPOINT}/v1internal:streamGenerateContent?alt=sse"
            try:
                response = await self._http.post(url, content=body_json, headers=headers)
                if response.status_code == 200:
                    return self._parse_sse_response(response.text)

                error_text = response.text[:500]
                print(f"[antigravity] API error ({response.status_code}): {error_text}")

                if response.status_code == 401:
                    self.credentials["expires"] = 0
                    raise Exception("Cloud Code Assist: authentication expired")

                if response.status_code == 429 and attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    print(
                        f"[antigravity] Rate limited. Retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue

                raise Exception(f"Cloud Code Assist API error ({response.status_code}): {error_text}")

            except Exception as e:
                if attempt < max_retries and "rate" in str(e).lower():
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
        raise Exception("Cloud Code Assist: max retries exceeded")


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
            print(f"Using Antigravity (Google Cloud Code Assist) provider. Model: {model}")
            return provider
        except Exception as e:
            print(f"Antigravity provider failed: {e}. Falling back to direct API key.")

    # Fall back to direct Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print(f"Using direct Anthropic API key provider. Model: {model}")
        return AnthropicProvider(api_key=api_key, model=model)

    raise ValueError(
        "No API credentials available. Either:\n"
        "  1. Run `python -m mirai.auth.auth_cli` for Antigravity auth, or\n"
        "  2. Set ANTHROPIC_API_KEY environment variable."
    )


class MockEmbeddingProvider:
    """Provides consistent fake embeddings for testing."""

    def __init__(self, dim: int = 1536):
        self.dim = dim

    async def get_embeddings(self, text: str) -> list[float]:
        # Return a deterministic-looking mock vector based on text hash
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        vector = []
        for i in range(self.dim):
            # Very simple fake vector generation
            val = (h[i % 32] / 255.0) - 0.5
            vector.append(val)
        return vector


class MockProvider:
    """Mock provider to test the AgentLoop logic without an API key."""

    def __init__(self):
        self.call_count = 0

    async def generate_response(
        self, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> SimpleNamespace:
        self.call_count += 1

        if "Recovered Memories" in system:
            print(f"[mock] SYSTEM PROMPT HAS MEMORIES:\n{system[system.find('### Recovered Memories') :]}")

        if "# IDENTITY" in system:
            print("[mock] IDENTITY ANCHORS PRESENT (Sandwich Pattern)")

        last_message = messages[-1]
        last_content = last_message.get("content", "")
        if isinstance(last_content, list):
            # Extract text if list (Anthropic format)
            last_text = "".join(
                [c.get("text", "") for c in last_content if isinstance(c, dict) and c.get("type") == "text"]
            )
        else:
            last_text = str(last_content)

        from types import SimpleNamespace

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
        print(f"[mock] last_text: {last_text[:50]}...")

        # 1. Thinking Turn
        if "analyze the situation" in last_text or "perform a self-reflection" in last_text:
            print("[mock] Handling Thinking Turn")
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text="<thinking>The system heartbeat has triggered. I should scan the project and summarize progress.</thinking>",
                    )
                ],
                stop_reason="end_turn",
            )

        # 2. Critique Turn
        if "Critique your response" in last_text:
            print("[mock] Handling Critique Turn")
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text="The response is aligned with my SOUL.md. Final version: I have successfully completed the proactive scan.",
                    )
                ],
                stop_reason="end_turn",
            )

        # 3. Tool Turn / Normal Chat
        # If the last message is from the assistant and contains text, we might be reaching the end
        # But in our mock, we want to trigger a tool at least once if tools are available.
        # We'll use a local check to see if we already sent a tool in this specific history.

        # 3. Tool Turn / Sequential Logic
        # We need to decide if we should call another tool or give a final answer.

        # Check if the VERY LAST assistant message had a tool_use that hasn't been answered yet
        # Actually, in AgentLoop, if stop_reason is tool_use, it executes and appends result.
        # So when we are called again, the last message is a tool_result from user.

        messages[-1].get("role")

        if tools:
            # E2E Proactive Maintenance Workflow
            if "maintenance_check" in full_history_text.lower():
                print("[mock] Match: E2E Maintenance Workflow")

                shell_results = [
                    m
                    for m in messages
                    if m.get("role") == "user"
                    and any("tool_result" in str(c) and "call_shell_maint" in str(c) for c in (m.get("content") or []))
                ]

                if not shell_results:
                    return SimpleNamespace(
                        content=[
                            SimpleNamespace(type="text", text="Checking for maintenance issues..."),
                            SimpleNamespace(
                                type="tool_use",
                                id="call_shell_maint",
                                name="shell_tool",
                                input={"command": "ls maintenance_fixed.txt"},
                                model_dump=lambda: {
                                    "type": "tool_use",
                                    "id": "call_shell_maint",
                                    "name": "shell_tool",
                                    "input": {"command": "ls maintenance_fixed.txt"},
                                },
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
                    return SimpleNamespace(
                        content=[
                            SimpleNamespace(type="text", text="Fixing the issue..."),
                            SimpleNamespace(
                                type="tool_use",
                                id="call_edit_maint",
                                name="editor_tool",
                                input={
                                    "action": "write",
                                    "path": "maintenance_fixed.txt",
                                    "content": "HEALED: Sanity check passed.",
                                },
                                model_dump=lambda: {
                                    "type": "tool_use",
                                    "id": "call_edit_maint",
                                    "name": "editor_tool",
                                    "input": {
                                        "action": "write",
                                        "path": "maintenance_fixed.txt",
                                        "content": "HEALED: Sanity check passed.",
                                    },
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
        print("[mock] Handling final answer")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="I have finished the complex task.")], stop_reason="end_turn"
        )
