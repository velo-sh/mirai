"""
Agent Loop — the core agentic execution cycle.

Implements a Think → Act → Critique loop with tool use, memory recall,
and identity-anchored system prompts.
"""

import os
import re
import shutil
from collections.abc import Sequence
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


def _load_soul(collaborator_id: str) -> str:
    """Load SOUL.md from disk (cached after first read)."""
    soul_path = f"mirai/collaborator/{collaborator_id}_SOUL.md"
    if os.path.exists(soul_path):
        with open(soul_path) as f:
            return f.read()
    return ""


class AgentLoop:
    def __init__(self, provider: Any, tools: Sequence[BaseTool], collaborator_id: str):
        self.provider = provider
        self.tools = {tool.definition["name"]: tool for tool in tools}
        self.collaborator_id = collaborator_id
        self.l3_storage = DuckDBStorage()
        self.l2_storage = VectorStore()
        self.embedder = MockEmbeddingProvider()

        # Identity attributes (to be loaded)
        self.name = ""
        self.role = ""
        self.base_system_prompt = ""
        self.soul_content = ""

    @classmethod
    async def create(cls, provider: Any, tools: Sequence[BaseTool], collaborator_id: str):
        """Factory method to create and initialize an AgentLoop instance."""
        instance = cls(provider, tools, collaborator_id)
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
                # Default fallback if not in DB
                self.name = "Unknown Collaborator"
                self.base_system_prompt = "You are a helpful AI collaborator."

        # Lazy loaded per-request in _run_impl
        pass

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
            # Create a backup before overwriting
            if os.path.exists(soul_path):
                shutil.copy2(soul_path, f"{soul_path}.bak")

            with open(soul_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Update local state immediately
            self.soul_content = new_content
            log.info("soul_updated_successfully", collaborator=self.collaborator_id)
            return True
        except Exception as e:
            log.error("soul_update_failed", error=str(e))
            return False

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

            # Check if we got the "empty response" error string
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
        """Internal execution of the agent loop for a single message."""
        tracer = get_tracer()
        with tracer.start_as_current_span("agent.run") as span:
            # 0. Reload SOUL per-request for adaptive identity
            self.soul_content = _load_soul(self.collaborator_id)

            msg_text = ""
            if isinstance(message, str):
                msg_text = message
            elif isinstance(message, list):
                msg_text = " ".join(b["text"] for b in message if b.get("type") == "text")

            span.set_attribute("message.length", len(msg_text))
            if history:
                span.set_attribute("history.turns", len(history))

            # Use provider's configured model as default
            model = model or getattr(self.provider, "model", "claude-sonnet-4-20250514")

            # 1. Archive incoming user message
            await self._archive_trace(str(message), "message", {"role": "user"})

            # 2. Total Recall: Semantic search in L2 (RAM)
            query_vector = await self.embedder.get_embeddings(msg_text)
            memories = await self.l2_storage.search(
                vector=query_vector, limit=3, filter=f"collaborator_id = '{self.collaborator_id}'"
            )

            memory_context = ""
            if memories:
                trace_ids = []
                for mem in memories:
                    # Use metadata which is stored as JSON string in LanceDB
                    meta = orjson.loads(mem["metadata"])
                    if "trace_id" in meta:
                        trace_ids.append(meta["trace_id"])

                # 3. Fetch raw context from L3 (HDD)
                raw_traces = await self.l3_storage.get_traces_by_ids(trace_ids)
                if raw_traces:
                    memory_context = "\n### Recovered Memories (L3 Raw Context):\n"
                    for trace in raw_traces:
                        memory_context += f"- [{trace['trace_type']}] {trace['content']}\n"

            # 4. Construct enriched system prompt using Sandwich Pattern
            # Bottom-up Identity Anchor
            full_system_prompt = f"# IDENTITY\n{self.soul_content}\n\n# CONTEXT\n{self.base_system_prompt}"

            if memory_context:
                full_system_prompt += (
                    "\n\n" + memory_context + "\nUse the above memories if they are relevant to the current request."
                )

            # Top-down Identity Anchor (Sandwich)
            full_system_prompt += (
                "\n\n# IDENTITY REINFORCEMENT\n"
                "Remember, you are operating as defined in the SOUL.md section above. Maintain your persona consistently.\n\n"
                "# TOOL USE RULES\n"
                "You have real tools available. When you need data (status, files, etc.), "
                "you MUST invoke tools through the function call mechanism. "
                "NEVER write tool results in your text response — that is fabrication. "
                "If you want to call a tool, emit a functionCall; do NOT describe the call in prose."
            )

            # Build messages with conversation history for multi-turn context
            messages: list[dict[str, Any]] = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": message})
            tool_definitions = [tool.definition for tool in self.tools.values()]

            # Phase 1: Internal Monologue (Thinking)
            log.info("phase_thinking", collaborator=self.collaborator_id)
            monologue_prompt = (
                "Analyze the user message and plan your response. Use <thinking>...</thinking> tags for your private internal thoughts. "
                "DO NOT output the final response yet. Just strategy and tool choice."
            )
            temp_messages = messages + [{"role": "user", "content": monologue_prompt}]

            with tracer.start_as_current_span("agent.think"):
                think_response: ProviderResponse = await self.provider.generate_response(
                    model=model,
                    system=full_system_prompt,
                    messages=temp_messages,
                    tools=[],
                )

            monologue_text = think_response.text()
            await self._archive_trace(monologue_text, "thinking", {"role": "assistant"})

            # Inject thinking into the Action Phase so the model follows through on its plan
            # Enforce thinking tags if the model ignored them to prevent Phase 2 from thinking it already replied
            monologue_wrapped = monologue_text
            if "<thinking>" not in monologue_text:
                monologue_wrapped = f"<thinking>\n{monologue_text}\n</thinking>"
            messages.append({"role": "assistant", "content": monologue_wrapped})

            # Nudge: explicitly tell the model to execute now (prevents Gemini from
            # producing empty output after seeing its own planning turn)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Now execute your plan. If you need data, invoke the appropriate tool "
                        "via a function call. Then provide your final response to the user."
                    ),
                }
            )

            # Phase 2: Action Loop (Tools)
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

                # Process assistant response
                assistant_content: list[dict[str, Any]] = []
                tool_calls: list[ToolUseBlock] = []

                for block in response.content:
                    if isinstance(block, TextBlock):
                        assistant_content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append(block)
                        assistant_content.append(block.model_dump())

                messages.append({"role": "assistant", "content": assistant_content})

                # Execute tools if present (Gemini may return ToolUseBlock with
                # stop_reason='end_turn' instead of 'tool_use', so check both)
                if response.stop_reason == "tool_use" or tool_calls:
                    for tool_call in tool_calls:
                        log.info("tool_call", tool=tool_call.name, input=tool_call.input)

                        await self._archive_trace(
                            orjson.dumps(tool_call.model_dump()).decode(), "tool_use", {"tool": tool_call.name}
                        )

                        if tool_call.name in self.tools:
                            try:
                                result = await self.tools[tool_call.name].execute(**tool_call.input)
                            except Exception as exc:
                                log.error("tool_exec_error", tool=tool_call.name, error=str(exc))
                                result = (
                                    f"⚠️ Tool Error: {exc}\n\n"
                                    "💊 Doctor Hint: Consider using `mirai_system(action='status')` "
                                    "to check system health, or retry with adjusted parameters."
                                )
                        else:
                            result = f"Error: Tool {tool_call.name} not found."

                        await self._archive_trace(str(result), "tool_result", {"tool": tool_call.name})

                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_call.id,
                                        "content": result,
                                    }
                                ],
                            }
                        )
                else:
                    # No tool calls — exit loop, proceed to critique
                    break

            # Phase 3: Critique (runs after loop exits — normal or max-rounds)
            # Collect final text from the last assistant message
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

            if not refined_text:
                if final_text:
                    log.warning("critique_returned_empty", fallback="using_final_text")
                    refined_text = final_text
                else:
                    # If even Phase 2 was empty, salvage from monologue if it looked like an answer
                    log.error("agent_run_produced_no_text", fallback="salvaging_from_thinking")

                    # Remove thinking tags and see if there's anything useful
                    salvaged = re.sub(r"<thinking>.*?</thinking>", "", monologue_text, flags=re.DOTALL).strip()
                    if not salvaged and monologue_text:
                        # If it was ALL thinking but had good content inside, extract it
                        salvaged = monologue_text.replace("<thinking>", "").replace("</thinking>", "").strip()

                    refined_text = salvaged or "I acknowledge your message. (Personality online)"

            await self._archive_trace(refined_text, "message", {"role": "assistant"})
            return refined_text

    async def stream_run(self, message: str, model: str | None = None):
        """Async generator that yields SSE event dicts during the agent cycle.

        Events:
            {"event": "thinking", "data": "<monologue text>"}
            {"event": "tool_use", "data": "<tool_name>(<args>)"}
            {"event": "chunk",    "data": "<partial text>"}
            {"event": "done",     "data": "<full final text>"}
        """

        model = model or getattr(self.provider, "model", "claude-sonnet-4-20250514")
        await self._archive_trace(message, "message", {"role": "user"})

        # Memory recall (same as run())
        query_vector = await self.embedder.get_embeddings(message)
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

        # Build system prompt (same as run())
        full_system_prompt = f"# IDENTITY\n{self.soul_content}\n\n# CONTEXT\n{self.base_system_prompt}"
        if memory_context:
            full_system_prompt += (
                "\n\n" + memory_context + "\nUse the above memories if they are relevant to the current request."
            )
        full_system_prompt += (
            "\n\n# IDENTITY REINFORCEMENT\n"
            "Remember, you are operating as defined in the SOUL.md section above. Maintain your persona consistently.\n\n"
            "# TOOL USE RULES\n"
            "You have real tools available. When you need data (status, files, etc.), "
            "you MUST invoke tools through the function call mechanism. "
            "NEVER write tool results in your text response — that is fabrication. "
            "If you want to call a tool, emit a functionCall; do NOT describe the call in prose."
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
        tool_definitions = [tool.definition for tool in self.tools.values()]

        # Phase 1: Think
        log.info("stream_phase_thinking", collaborator=self.collaborator_id)
        monologue_prompt = "Before responding or using tools, analyze the situation. What is your plan? Use <thinking>...</thinking> tags."
        temp_messages = messages + [{"role": "user", "content": monologue_prompt}]
        think_response: ProviderResponse = await self.provider.generate_response(
            model=model, system=full_system_prompt, messages=temp_messages, tools=[]
        )
        monologue_text = think_response.text()
        await self._archive_trace(monologue_text, "thinking", {"role": "assistant"})
        yield {"event": "thinking", "data": monologue_text}

        # Phase 2: Action Loop
        while True:
            response: ProviderResponse = await self.provider.generate_response(
                model=model, system=full_system_prompt, messages=messages, tools=tool_definitions
            )

            assistant_content: list[dict[str, Any]] = []
            tool_calls: list[ToolUseBlock] = []

            for block in response.content:
                if isinstance(block, TextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(block)
                    assistant_content.append(block.model_dump())

            messages.append({"role": "assistant", "content": assistant_content})

            # Execute tools if present (Gemini may return ToolUseBlock with
            # stop_reason='end_turn' instead of 'tool_use', so check both)
            if response.stop_reason == "tool_use" or tool_calls:
                for tool_call in tool_calls:
                    log.info("stream_tool_call", tool=tool_call.name, input=tool_call.input)
                    yield {"event": "tool_use", "data": f"{tool_call.name}({tool_call.input})"}

                    if tool_call.name in self.tools:
                        try:
                            result = await self.tools[tool_call.name].execute(**tool_call.input)
                        except Exception as exc:
                            log.error("tool_exec_error", tool=tool_call.name, error=str(exc))
                            result = (
                                f"⚠️ Tool Error: {exc}\n\n"
                                "💊 Doctor Hint: Consider using `mirai_system(action='status')` "
                                "to check system health, or retry with adjusted parameters."
                            )
                    else:
                        result = f"Error: Tool {tool_call.name} not found."

                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": result}],
                        }
                    )
            else:
                # Final response — stream text chunks
                final_text = "".join(c["text"] for c in assistant_content if c.get("type") == "text")

                # Phase 3: Critique
                log.info("stream_phase_critique", collaborator=self.collaborator_id)
                critique_prompt = (
                    f"Review your draft response below and refine it if needed.\n\n"
                    f'Draft: "{final_text}"\n\n'
                    f"Rules:\n"
                    f"- Check alignment with your SOUL.md identity and behavioral constraints.\n"
                    f"- Output ONLY the final polished response — no critique, no analysis, no meta-commentary.\n"
                    f"- If the draft is already perfect, output it exactly as-is."
                )
                critique_messages = messages + [{"role": "user", "content": critique_prompt}]
                refined_response: ProviderResponse = await self.provider.generate_response(
                    model=model, system=full_system_prompt, messages=critique_messages, tools=[]
                )
                refined_text = refined_response.text()
                await self._archive_trace(refined_text, "message", {"role": "assistant"})

                # Stream the refined text in chunks for progressive display
                chunk_size = 50
                for i in range(0, len(refined_text), chunk_size):
                    yield {"event": "chunk", "data": refined_text[i : i + chunk_size]}

                yield {"event": "done", "data": refined_text}
                return
