"""
Google Cloud Code Assist (Antigravity) LLM provider.

Routes Claude/Gemini API calls through the v1internal:streamGenerateContent
endpoint at cloudcode-pa.googleapis.com.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

import httpx
import orjson
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mirai.agent.models import ProviderResponse
from mirai.agent.providers.message_converter import convert_messages, convert_tools, parse_sse_response
from mirai.agent.providers.quota import QuotaManager
from mirai.logging import get_logger
from mirai.tracing import get_tracer

if TYPE_CHECKING:
    from mirai.agent.providers.base import ModelInfo, UsageSnapshot

log = get_logger("mirai.providers.antigravity")


class _RetryableAPIError(Exception):
    """Raised on 429/503 to trigger tenacity retry."""

    def __init__(self, status: int, detail: str):
        self.status = status
        super().__init__(f"API error {status}: {detail}")


# Known-good models for failover, in priority order.
FAILOVER_MODELS = [
    "claude-sonnet-4-5",
    "claude-opus-4-5-thinking",
    "claude-opus-4-6-thinking",
    "claude-sonnet-4-5-thinking",
    "gemini-3-pro-high",
    "gemini-3-pro-low",
    "gemini-3-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-thinking",
    "gemini-2.5-flash-lite",
    "gpt-oss-120b-medium",
]

# Model name mapping from standard names to Cloud Code Assist internal IDs
MODEL_MAP = {
    # Claude 4.x series
    "claude-sonnet-4-20250514": "claude-sonnet-4-5",
    "claude-sonnet-4-latest": "claude-sonnet-4-5",
    "claude-opus-4-20250514": "claude-opus-4-5-thinking",
    "claude-opus-4-latest": "claude-opus-4-5-thinking",
    "claude-opus-4-6-thinking": "claude-opus-4-6-thinking",
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
    # Gemini series
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-thinking": "gemini-2.5-flash-thinking",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3-pro-high": "gemini-3-pro-high",
    "gemini-3-pro-low": "gemini-3-pro-low",
    "gemini-3-pro-image": "gemini-3-pro-image",
}


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

    ANTIGRAVITY_VERSION = "1.15.8"

    # Backward-compat aliases — these were static methods, now in message_converter.py
    _parse_sse_response = staticmethod(parse_sse_response)
    _convert_messages = staticmethod(convert_messages)
    _convert_tools = staticmethod(convert_tools)
    MODEL_MAP = MODEL_MAP
    DEFAULT_MODEL = "claude-sonnet-4-5-20250514"

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

    # ------------------------------------------------------------------
    # Provider identity & discovery
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "antigravity"

    async def list_models(self) -> list[ModelInfo]:
        """Return models available through Antigravity (Cloud Code Assist)."""
        from mirai.agent.providers.base import ModelInfo

        return [
            ModelInfo(
                id="claude-sonnet-4-20250514",
                name="Claude Sonnet 4 (Antigravity)",
                description="Claude Sonnet 4 via Google Cloud Code Assist",
                context_window=200_000,
                max_output_tokens=8192,
                supports_tool_use=True,
                supports_vision=True,
                input_modalities=["text", "image"],
                knowledge_cutoff="2025-03",
            ),
        ]

    async def get_usage(self) -> UsageSnapshot:
        """Usage query not supported for Antigravity."""
        from mirai.agent.providers.base import UsageSnapshot

        return UsageSnapshot(provider="antigravity", error="not supported")

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

        # Also initialize/update QuotaManager
        if not hasattr(self, "quota_manager"):
            self.quota_manager = QuotaManager(self.credentials)

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

    def _build_request(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Build the Cloud Code Assist request body."""
        contents = convert_messages(messages)
        request: dict[str, Any] = {"contents": contents}

        if system:
            request["systemInstruction"] = {
                "parts": [{"text": system}],
            }

        request["generationConfig"] = {"maxOutputTokens": max_tokens}

        google_tools = convert_tools(tools)
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

    @retry(
        retry=retry_if_exception_type(_RetryableAPIError),
        wait=wait_exponential(min=2, max=30),
        stop=stop_after_attempt(8),
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
        tracer = get_tracer()
        with tracer.start_as_current_span("provider.antigravity.generate") as span:
            span.set_attribute("llm.model", model)
            await self._ensure_fresh_token()

            # Remap model name if needed for Cloud Code Assist
            effective_model = MODEL_MAP.get(model, model)

            # --- Smart Failover Logic ---
            # If the effective model is exhausted, find the best available alternative
            if not await self.quota_manager.is_available(effective_model):
                log.warning("model_exhausted_failing_over", exhausted_model=effective_model)

                found_fallback = False
                for candidate in FAILOVER_MODELS:
                    if await self.quota_manager.is_available(candidate):
                        log.info("failover_selected_model", selected=candidate)
                        effective_model = candidate
                        found_fallback = True
                        break

                if not found_fallback:
                    log.error("all_failover_models_exhausted")
            # ----------------------------

            if effective_model != model and effective_model not in MODEL_MAP.values():
                log.info("model_remapped", original=model, effective=effective_model)
                span.set_attribute("llm.effective_model", effective_model)
            elif model not in MODEL_MAP:
                # Model not recognized — fail fast with helpful message
                await self.quota_manager._maybe_refresh()
                available = sorted(self.quota_manager._quotas.keys()) or list(MODEL_MAP.values())
                raise ValueError(f"Model '{model}' is not available. Available models: {', '.join(available)}")

            body = self._build_request(effective_model, system, messages, tools)
            headers = self._build_headers()

            # orjson.dumps returns bytes — httpx accepts bytes directly (no decode overhead)
            body_bytes = orjson.dumps(body)
            log.info(
                "api_request_sending",
                requested_model=model,
                effective_model=effective_model,
                body_model=body.get("model"),
            )

            url = f"{self.DEFAULT_ENDPOINT}/v1internal:streamGenerateContent?alt=sse"
            response = await self._http.post(url, content=body_bytes, headers=headers)
            span.set_attribute("http.status_code", response.status_code)

            if response.status_code == 200:
                return parse_sse_response(response.text, model_id=effective_model)

            error_text = response.text[:500]
            log.error("api_error", status=response.status_code, detail=error_text[:200])

            if response.status_code == 401:
                self.credentials["expires"] = 0
                raise RuntimeError("Cloud Code Assist: authentication expired")

            if response.status_code in (429, 503):
                # Mark this model as exhausted so the next retry picks a different one
                self.quota_manager._quotas[effective_model] = 100.0
                log.warning(
                    "model_marked_exhausted_after_429",
                    model=effective_model,
                    remaining_available=[m for m, pct in self.quota_manager._quotas.items() if pct < 100.0],
                )
                raise _RetryableAPIError(response.status_code, error_text[:200])

            raise RuntimeError(f"Cloud Code Assist API error ({response.status_code}): {error_text}")
