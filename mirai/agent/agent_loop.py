"""
Agent Loop — the core agentic execution cycle.

Implements a Think → Act → Critique loop with tool use, memory recall,
and identity-anchored system prompts.

Identity management lives in :mod:`mirai.agent.identity`.
Prompt construction lives in :mod:`mirai.agent.prompt`.
"""

from __future__ import annotations

import asyncio
import enum
import json
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import orjson
from ulid import ULID

from mirai.agent.identity import initialize_collaborator, load_soul
from mirai.agent.identity import update_soul as _update_soul_fn
from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock
from mirai.agent.prompt import build_system_prompt
from mirai.agent.providers import MockEmbeddingProvider
from mirai.agent.providers.base import ProviderProtocol
from mirai.agent.tools.base import BaseTool
from mirai.db.duck import DuckDBStorage
from mirai.db.models import DBTrace
from mirai.errors import ProviderError
from mirai.logging import get_logger
from mirai.memory.vector_db import VectorStore
from mirai.tracing import get_tracer

log = get_logger("mirai.agent")


# ---------------------------------------------------------------------------
# Cycle step events emitted by _execute_cycle
# ---------------------------------------------------------------------------


@dataclass
class ThinkingStep:
    """Phase 1 completed — monologue text available."""

    monologue: str


@dataclass
class ToolCallStep:
    """A tool was invoked during Phase 2."""

    name: str
    input: dict[str, Any]
    result: str


@dataclass
class DraftStep:
    """Phase 2 completed — a draft response is available (pre-critique)."""

    text: str


@dataclass
class RefinedStep:
    """Phase 3 completed — the final refined text is available."""

    text: str
    draft: str = ""  # the pre-critique draft, for reference


CycleEvent = ThinkingStep | ToolCallStep | DraftStep | RefinedStep


class LoopState(enum.Enum):
    """Execution states for the AgentLoop."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    RECOVERING = "recovering"
    DONE = "done"


# _load_soul is now in mirai.agent.identity (re-exported for backward compat)
_load_soul = load_soul


class AgentLoop:
    # State Machine tracking (defaults for compatibility with brittle tests using __new__)
    state: LoopState = LoopState.IDLE
    runtime_overrides: dict[str, Any] = {}

    def __init__(
        self,
        provider: ProviderProtocol,
        tools: Sequence[BaseTool],
        collaborator_id: str,
        base_system_prompt: str = "",
        l3_storage: DuckDBStorage | None = None,
        l2_storage: VectorStore | None = None,
        embedder: Any | None = None,
        fallback_models: list[str] | None = None,
    ):
        self.provider = provider
        self.tools = {tool.definition["name"]: tool for tool in tools}
        self.collaborator_id = collaborator_id
        self.base_system_prompt = base_system_prompt
        self.fallback_models: list[str] = fallback_models or []

        # State Machine tracking
        self.state = LoopState.IDLE
        self.runtime_overrides: dict[str, Any] = {}

        # Dependency Injection with fallbacks
        self.l3_storage = l3_storage or DuckDBStorage()
        self.l2_storage = l2_storage or VectorStore()
        self.embedder = embedder or MockEmbeddingProvider()
        self.name: str = ""
        self.role: str = ""
        self.soul_content: str = ""

    def swap_provider(self, new_provider: ProviderProtocol) -> None:
        """Hot-swap the LLM provider at runtime.

        Takes effect on the next ``generate_response()`` call.
        """
        old_name = self.provider.provider_name
        new_name = new_provider.provider_name
        self.provider = new_provider
        log.info("provider_swapped", old=old_name, new=new_name, model=new_provider.model)

        # Identity attributes (to be loaded)
        self.name = ""
        self.role = ""
        self.base_system_prompt = ""
        self.soul_content = ""

    @classmethod
    async def create(
        cls,
        provider: ProviderProtocol,
        tools: Sequence[BaseTool],
        collaborator_id: str,
        l3_storage: DuckDBStorage | None = None,
        l2_storage: VectorStore | None = None,
        embedder: Any | None = None,
        fallback_models: list[str] | None = None,
    ) -> AgentLoop:
        """Factory method to create and initialize an AgentLoop instance."""
        instance = cls(
            provider,
            tools,
            collaborator_id,
            l3_storage=l3_storage,
            l2_storage=l2_storage,
            embedder=embedder,
            fallback_models=fallback_models,
        )
        await instance._initialize()
        return instance

    async def _initialize(self) -> None:
        """Asynchronously load collaborator metadata and soul."""
        name, role, prompt = await initialize_collaborator(self.collaborator_id)
        self.name = name
        self.role = role
        self.base_system_prompt = prompt

    async def _archive_trace(self, content: str, trace_type: str, metadata: dict[str, Any] | None = None) -> str:
        """Helper to save a trace to the L3 (HDD) storage using DuckDB."""
        trace = DBTrace(
            id=str(ULID()),
            collaborator_id=self.collaborator_id,
            trace_type=trace_type,
            content=content,
            metadata_json=metadata or {},
        )
        await self.l3_storage.append_trace(trace)
        return trace.id

    async def update_soul(self, new_content: str) -> bool:
        """Update the SOUL.md file with new content."""
        success = await _update_soul_fn(self.collaborator_id, new_content)
        if success:
            self.soul_content = new_content
        return success

    # ------------------------------------------------------------------
    # Shared orchestration
    # ------------------------------------------------------------------

    async def _build_system_prompt(self, msg_text: str) -> str:
        """Reload identity, recall memories, and construct the enriched system prompt."""
        self.soul_content = load_soul(self.collaborator_id)
        return await build_system_prompt(
            collaborator_id=self.collaborator_id,
            soul_content=self.soul_content,
            base_system_prompt=self.base_system_prompt,
            provider=self.provider,
            embedder=self.embedder,
            l2_storage=self.l2_storage,
            l3_storage=self.l3_storage,
            msg_text=msg_text,
            tools=self.tools,
        )

    async def _execute_tool(self, tool_call: ToolUseBlock) -> str:
        """Execute a single tool and return the result string."""
        if tool_call.name in self.tools:
            try:
                # Always use .run() to ensure automated tracing spans are captured
                return await self.tools[tool_call.name].run(**tool_call.input)
            except Exception as exc:
                log.error("tool_exec_error", tool=tool_call.name, error=str(exc))
                return (
                    f"⚠️ Tool Error: {exc}\n\n"
                    "💊 Doctor Hint: Consider using `mirai_system(action='status')` "
                    "to check system health, or retry with adjusted parameters."
                )
        return f"Error: Tool {tool_call.name} not found."

    def _transition(self, next_state: LoopState) -> None:
        """Perform a state transition with logging."""
        prev = self.state
        self.state = next_state
        log.info("loop_state_transition", prev=prev.value, next=next_state.value)

    async def _execute_cycle(
        self,
        message: str | list[dict[str, Any]],
        model: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[CycleEvent, None]:
        """State-machine-driven agent loop."""
        tracer = get_tracer()

        with tracer.start_as_current_span("agent.run") as span:
            self._transition(LoopState.THINKING)
            # Normalize message text for embeddings & tracing
            msg_text = ""
            if isinstance(message, str):
                msg_text = message
            elif isinstance(message, list):
                msg_text = " ".join(b["text"] for b in message if b.get("type") == "text")

            span.set_attribute("message.length", len(msg_text))
            if history:
                span.set_attribute("history.turns", len(history))

            model = model or self.provider.model

            # Archive inbound message
            await self._archive_trace(str(message), "message", {"role": "user"})

            # Build enriched system prompt (includes memory recall)
            full_system_prompt = await self._build_system_prompt(msg_text)

            # Build messages with conversation history
            messages: list[dict[str, Any]] = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": message})
            tool_definitions = [tool.definition for tool in self.tools.values()]

            # ---- State-driven tool loop ----
            MAX_TOOL_ROUNDS = 10
            tool_round = 0
            final_text = ""

            while True:
                tool_round += 1
                if tool_round > MAX_TOOL_ROUNDS:
                    log.warning("max_tool_rounds_reached", rounds=MAX_TOOL_ROUNDS)
                    break

                with tracer.start_as_current_span("agent.act"):
                    log.info(
                        "llm_request",
                        tools_count=len(tool_definitions),
                        tool_names=[t["name"] for t in tool_definitions],
                        round=tool_round,
                    )
                    # Attempt primary model, then fallback chain on failure
                    response: ProviderResponse | None = None
                    models_to_try = [model] + [m for m in self.fallback_models if m != model]
                    last_error: Exception | None = None
                    for attempt_idx, attempt_model in enumerate(models_to_try):
                        if attempt_idx > 0:
                            self._transition(LoopState.RECOVERING)
                            backoff = min(0.5 * attempt_idx, 5.0)
                            log.info("fallback_backoff", seconds=backoff, attempt=attempt_idx)
                            await asyncio.sleep(backoff)
                            self._transition(LoopState.THINKING)
                        try:
                            # Prepare config with overrides
                            loop_config: dict[str, Any] = {**self.provider.config_dict(), **self.runtime_overrides}
                            response = await self.provider.generate_response(
                                model=cast(str, attempt_model),
                                system=full_system_prompt,
                                messages=messages,
                                tools=tool_definitions,
                                **loop_config,
                            )
                            if attempt_model != model:
                                log.info("fallback_model_succeeded", model=attempt_model)
                            break
                        except Exception as exc:
                            last_error = exc
                            log.warning(
                                "model_call_failed",
                                model=attempt_model,
                                error=str(exc),
                                error_type=type(exc).__name__,
                                remaining_fallbacks=len(models_to_try) - models_to_try.index(attempt_model) - 1,
                            )
                    if response is None:
                        self._transition(LoopState.DONE)
                        if last_error is not None:
                            raise ProviderError(f"All models in fallback chain failed: {last_error}") from last_error
                        raise ProviderError("All models in fallback chain failed")
                    log.info(
                        "llm_response",
                        stop_reason=response.stop_reason,
                        content_blocks=len(response.content),
                        block_types=[type(b).__name__ for b in response.content],
                    )

                # Parse assistant response
                text_parts: list[str] = []
                tool_calls: list[ToolUseBlock] = []

                for block in response.content:
                    if isinstance(block, TextBlock) and block.text:
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append(block)

                # Build the assistant message in OpenAI format
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                assistant_msg["content"] = "".join(text_parts) if text_parts else None
                if tool_calls:
                    tc_list = []
                    for tc in tool_calls:
                        tc_dict: dict[str, Any] = {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.input),
                            },
                        }
                        if getattr(tc, "thought_signature", None):
                            tc_dict["thought_signature"] = tc.thought_signature
                        tc_list.append(tc_dict)
                    assistant_msg["tool_calls"] = tc_list
                messages.append(assistant_msg)

                # Execute tools if present
                if response.stop_reason == "tool_use" or tool_calls:
                    self._transition(LoopState.ACTING)
                    for tool_call in tool_calls:
                        log.info("tool_call", tool=tool_call.name, input=tool_call.input)
                        await self._archive_trace(
                            orjson.dumps(tool_call.model_dump()).decode(), "tool_use", {"tool": tool_call.name}
                        )

                        result = await self._execute_tool(tool_call)
                        await self._archive_trace(str(result), "tool_result", {"tool": tool_call.name})
                        yield ToolCallStep(name=tool_call.name, input=tool_call.input, result=result)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )
                    # Loop back to THINKING for the next model turn
                    self._transition(LoopState.THINKING)
                else:
                    # Model produced a final text response — done
                    final_text = "".join(text_parts)
                    self._transition(LoopState.DONE)
                    break

            # Fallback if loop ended without text
            if not final_text:
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            final_text = content
                            break
            if not final_text:
                final_text = "I acknowledge your message."

            await self._archive_trace(final_text, "message", {"role": "assistant"})
            yield RefinedStep(text=final_text, draft=final_text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        message: str | list[dict[str, Any]],
        model: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Run the agent loop for a single message with automatic retry."""
        max_attempts = 2
        last_result = ""

        for attempt in range(1, max_attempts + 1):
            result = await self._run_impl(message, model, history)

            if "failed to generate a vocal response" in result:
                log.warning("agent_run_empty_result", attempt=attempt, next_retry=attempt < max_attempts)
                last_result = result
                continue

            return result

        log.error("agent_run_failed_after_retries", attempts=max_attempts)
        return last_result

    async def _run_impl(
        self,
        message: str | list[dict[str, Any]],
        model: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Internal execution — consumes _execute_cycle and returns the final text."""
        result = ""
        try:
            async for step in self._execute_cycle(message, model, history):
                if isinstance(step, RefinedStep):
                    result = step.text
        finally:
            self._transition(LoopState.IDLE)
        return result

    async def stream_run(self, message: str, model: str | None = None) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator that yields SSE event dicts during the agent cycle.

        Events:
            {"event": "thinking", "data": "<monologue text>"}
            {"event": "tool_use", "data": "<tool_name>(<args>)"}
            {"event": "chunk",    "data": "<partial text>"}
            {"event": "done",     "data": "<full final text>"}
        """
        async for step in self._execute_cycle(message, model):
            if isinstance(step, ThinkingStep):
                yield {"event": "thinking", "data": step.monologue}
            elif isinstance(step, ToolCallStep):
                yield {"event": "tool_use", "data": f"{step.name}({step.input})"}
            elif isinstance(step, RefinedStep):
                # Stream refined text in chunks for progressive display
                chunk_size = 50
                for i in range(0, len(step.text), chunk_size):
                    yield {"event": "chunk", "data": step.text[i : i + chunk_size]}
                yield {"event": "done", "data": step.text}
