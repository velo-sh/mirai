"""
Agent Loop — the core agentic execution cycle.

Implements a Think → Act → Critique loop with tool use, memory recall,
and identity-anchored system prompts.
"""

import functools
import os
import re
import shutil
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any

import orjson
from ulid import ULID

from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock
from mirai.agent.providers import MockEmbeddingProvider
from mirai.agent.tools.base import BaseTool
from mirai.db.duck import DuckDBStorage
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


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_soul(collaborator_id: str) -> str:
    """Load SOUL.md from disk (cached after first read)."""
    soul_path = f"mirai/collaborator/{collaborator_id}_SOUL.md"
    if os.path.exists(soul_path):
        with open(soul_path) as f:
            return f.read()
    return ""


class AgentLoop:
    def __init__(
        self,
        provider: Any,
        tools: Sequence[BaseTool],
        collaborator_id: str,
        l3_storage: DuckDBStorage | None = None,
        l2_storage: VectorStore | None = None,
        embedder: Any | None = None,
    ):
        self.provider = provider
        self.tools = {tool.definition["name"]: tool for tool in tools}
        self.collaborator_id = collaborator_id
        
        # Dependency Injection with fallbacks
        self.l3_storage = l3_storage or DuckDBStorage()
        self.l2_storage = l2_storage or VectorStore()
        self.embedder = embedder or MockEmbeddingProvider()

        # Identity attributes (to be loaded)
        self.name = ""
        self.role = ""
        self.base_system_prompt = ""
        self.soul_content = ""

    @classmethod
    async def create(
        cls,
        provider: Any,
        tools: Sequence[BaseTool],
        collaborator_id: str,
        l3_storage: DuckDBStorage | None = None,
        l2_storage: VectorStore | None = None,
        embedder: Any | None = None,
    ):
        """Factory method to create and initialize an AgentLoop instance."""
        instance = cls(
            provider, 
            tools, 
            collaborator_id, 
            l3_storage=l3_storage, 
            l2_storage=l2_storage, 
            embedder=embedder
        )
        await instance._initialize()
        return instance

    async def _initialize(self):
        """Asynchronously load collaborator metadata and soul."""
        from mirai.collaborator.manager import CollaboratorManager
        from mirai.db.session import get_session

        async for session in get_session():
            manager = CollaboratorManager(session)
            collab = await manager.get_collaborator(self.collaborator_id)
            if collab:
                self.name = collab.name
                self.role = collab.role
                self.base_system_prompt = collab.system_prompt
            else:
                self.name = "Unknown Collaborator"
                self.base_system_prompt = "You are a helpful AI collaborator."

    async def _archive_trace(self, content: str, trace_type: str, metadata: dict[str, Any] = None):
        """Helper to save a trace to the L3 (HDD) storage using DuckDB."""
        trace_id = str(ULID())
        await self.l3_storage.append_trace(
            id=trace_id, collaborator_id=self.collaborator_id, trace_type=trace_type, content=content, metadata=metadata
        )
        return trace_id

    async def update_soul(self, new_content: str) -> bool:
        """Update the SOUL.md file with new content.

        This enables autonomous evolution of the collaborator's identity.
        """
        soul_path = f"mirai/collaborator/{self.collaborator_id}_SOUL.md"
        try:
            if os.path.exists(soul_path):
                shutil.copy2(soul_path, f"{soul_path}.bak")

            with open(soul_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            self.soul_content = new_content
            # Clear cache so next load gets new content (or update manually)
            _load_soul.cache_clear()
            # Also update local state immediately
            self.soul_content = new_content
            
            log.info("soul_updated_successfully", collaborator=self.collaborator_id)
            return True
        except Exception as e:
            log.error("soul_update_failed", error=str(e))
            return False

    # ------------------------------------------------------------------
    # Shared orchestration
    # ------------------------------------------------------------------

    async def _build_system_prompt(self, msg_text: str) -> str:
        """Reload identity, recall memories, and construct the enriched system prompt."""
        self.soul_content = _load_soul(self.collaborator_id)

        # Memory recall (L2 → L3)
        query_vector = await self.embedder.get_embeddings(msg_text)
        memories = await self.l2_storage.search(
            vector=query_vector, limit=3, filter=f"collaborator_id = '{self.collaborator_id}'"
        )

        memory_context = ""
        if memories:
            trace_ids = []
            for mem in memories:
                meta = orjson.loads(mem["metadata"])
                if "trace_id" in meta:
                    trace_ids.append(meta["trace_id"])
            raw_traces = await self.l3_storage.get_traces_by_ids(trace_ids)
            if raw_traces:
                memory_context = "\n### Recovered Memories (L3 Raw Context):\n"
                for trace in raw_traces:
                    memory_context += f"- [{trace['trace_type']}] {trace['content']}\n"

        # Sandwich pattern: identity → context → memories → reinforcement → rules
        prompt = f"# IDENTITY\n{self.soul_content}\n\n# CONTEXT\n{self.base_system_prompt}"
        if memory_context:
            prompt += "\n\n" + memory_context + "\nUse the above memories if they are relevant to the current request."
        prompt += (
            "\n\n# IDENTITY REINFORCEMENT\n"
            "Remember, you are operating as defined in the SOUL.md section above. Maintain your persona consistently.\n\n"
            "# TOOL USE RULES\n"
            "You have real tools available. When you need data (status, files, etc.), "
            "you MUST invoke tools through the function call mechanism. "
            "NEVER write tool results in your text response — that is fabrication. "
            "If you want to call a tool, emit a functionCall; do NOT describe the call in prose."
        )
        return prompt

    async def _execute_tool(self, tool_call: ToolUseBlock) -> str:
        """Execute a single tool and return the result string."""
        if tool_call.name in self.tools:
            try:
                return await self.tools[tool_call.name].execute(**tool_call.input)
            except Exception as exc:
                log.error("tool_exec_error", tool=tool_call.name, error=str(exc))
                return (
                    f"⚠️ Tool Error: {exc}\n\n"
                    "💊 Doctor Hint: Consider using `mirai_system(action='status')` "
                    "to check system health, or retry with adjusted parameters."
                )
        return f"Error: Tool {tool_call.name} not found."

    async def _execute_cycle(
        self,
        message: str | list[dict[str, Any]],
        model: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[CycleEvent, None]:
        """Core Think → Act → Critique cycle as an async generator.

        Yields:
            ThinkingStep  — after Phase 1 completes
            ToolCallStep  — for each tool invocation in Phase 2
            DraftStep     — after Phase 2 produces a textual response
            RefinedStep   — after Phase 3 critique produces the final text
        """
        tracer = get_tracer()

        with tracer.start_as_current_span("agent.run") as span:
            # Normalize message text for embeddings & tracing
            msg_text = ""
            if isinstance(message, str):
                msg_text = message
            elif isinstance(message, list):
                msg_text = " ".join(b["text"] for b in message if b.get("type") == "text")

            span.set_attribute("message.length", len(msg_text))
            if history:
                span.set_attribute("history.turns", len(history))

            model = model or getattr(self.provider, "model", "claude-sonnet-4-20250514")

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

            # ---- Phase 1: Thinking ----
            log.info("phase_thinking", collaborator=self.collaborator_id)
            monologue_prompt = (
                "Analyze the user message and plan your response. Use <thinking>...</thinking> tags for your private "
                "internal thoughts. DO NOT output the final response yet. Just strategy and tool choice."
            )
            temp_messages = messages + [{"role": "user", "content": monologue_prompt}]

            with tracer.start_as_current_span("agent.think"):
                think_response: ProviderResponse = await self.provider.generate_response(
                    model=model, system=full_system_prompt, messages=temp_messages, tools=[]
                )

            monologue_text = think_response.text()
            await self._archive_trace(monologue_text, "thinking", {"role": "assistant"})
            try:
                yield ThinkingStep(monologue=monologue_text)
            except GeneratorExit:
                return

            # Inject thinking into conversation so Phase 2 follows through
            monologue_wrapped = monologue_text
            if "<thinking>" not in monologue_text:
                monologue_wrapped = f"<thinking>\n{monologue_text}\n</thinking>"
            messages.append({"role": "assistant", "content": monologue_wrapped})

            # Nudge the model to execute now
            messages.append({
                "role": "user",
                "content": (
                    "Now execute your plan. If you need data, invoke the appropriate tool "
                    "via a function call. Then provide your final response to the user."
                ),
            })

            # ---- Phase 2: Action Loop ----
            MAX_TOOL_ROUNDS = 10
            tool_round = 0
            while True:
                tool_round += 1
                if tool_round > MAX_TOOL_ROUNDS:
                    log.warning("max_tool_rounds_reached", rounds=MAX_TOOL_ROUNDS)
                    break

                with tracer.start_as_current_span("agent.act"):
                    log.info(
                        "phase2_request",
                        tools_count=len(tool_definitions),
                        tool_names=[t["name"] for t in tool_definitions],
                    )
                    response: ProviderResponse = await self.provider.generate_response(
                        model=model, system=full_system_prompt, messages=messages, tools=tool_definitions
                    )
                    log.info(
                        "phase2_response",
                        stop_reason=response.stop_reason,
                        content_blocks=len(response.content),
                        block_types=[type(b).__name__ for b in response.content],
                    )

                # Parse assistant response
                assistant_content: list[dict[str, Any]] = []
                tool_calls: list[ToolUseBlock] = []

                for block in response.content:
                    if isinstance(block, TextBlock):
                        assistant_content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append(block)
                        assistant_content.append(block.model_dump())

                messages.append({"role": "assistant", "content": assistant_content})

                # Execute tools if present
                if response.stop_reason == "tool_use" or tool_calls:
                    for tool_call in tool_calls:
                        log.info("tool_call", tool=tool_call.name, input=tool_call.input)
                        await self._archive_trace(
                            orjson.dumps(tool_call.model_dump()).decode(), "tool_use", {"tool": tool_call.name}
                        )

                        result = await self._execute_tool(tool_call)
                        await self._archive_trace(str(result), "tool_result", {"tool": tool_call.name})
                        try:
                            yield ToolCallStep(name=tool_call.name, input=tool_call.input, result=result)
                        except GeneratorExit:
                            return

                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": tool_call.id, "content": result}
                            ],
                        })
                else:
                    break

            # Extract draft text from the last assistant message
            final_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        final_text = "".join(
                            c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text"
                        )
                    elif isinstance(content, str):
                        final_text = content
                    break

            try:
                yield DraftStep(text=final_text)
            except GeneratorExit:
                return

            # ---- Phase 3: Critique ----
            log.info("phase_critique_start", draft_length=len(final_text))

            critique_prompt = (
                f"Review your draft response below and refine it if needed.\n\n"
                f'Draft: "{final_text}"\n\n'
                f"Rules:\n"
                f"- Check alignment with your SOUL.md identity and behavioral constraints.\n"
                f"- Output ONLY the final polished response — no critique, no analysis, no meta-commentary.\n"
                f"- If the draft is already perfect, output it exactly as-is."
            )

            critique_messages = messages + [{"role": "user", "content": critique_prompt}]
            with tracer.start_as_current_span("agent.critique"):
                refined_response: ProviderResponse = await self.provider.generate_response(
                    model=model, system=full_system_prompt, messages=critique_messages, tools=[]
                )

            refined_text = refined_response.text().strip()
            log.info("phase_critique_done", refined_length=len(refined_text))

            # Fallback cascade: critique → draft → monologue → default
            if not refined_text:
                if final_text:
                    log.warning("critique_returned_empty", fallback="using_final_text")
                    refined_text = final_text
                else:
                    log.error("agent_run_produced_no_text", fallback="salvaging_from_thinking")
                    salvaged = re.sub(
                        r"<thinking>.*?</thinking>", "", monologue_text, flags=re.DOTALL
                    ).strip()
                    if not salvaged and monologue_text:
                        salvaged = monologue_text.replace("<thinking>", "").replace("</thinking>", "").strip()
                    refined_text = salvaged or "I acknowledge your message. (Personality online)"

            await self._archive_trace(refined_text, "message", {"role": "assistant"})
            yield RefinedStep(text=refined_text, draft=final_text)


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
        async for step in self._execute_cycle(message, model, history):
            if isinstance(step, RefinedStep):
                return step.text
        return ""

    async def stream_run(self, message: str, model: str | None = None):
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
