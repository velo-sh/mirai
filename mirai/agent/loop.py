"""
Agent Loop — the core agentic execution cycle.

Implements a Think → Act → Critique loop with tool use, memory recall,
and identity-anchored system prompts.
"""

import os
from collections.abc import Sequence
from functools import lru_cache
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


@lru_cache(maxsize=8)
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

        self.soul_content = _load_soul(self.collaborator_id)

    async def _archive_trace(self, content: str, trace_type: str, metadata: dict[str, Any] = None):
        """Helper to save a trace to the L3 (HDD) storage using DuckDB."""
        trace_id = str(ULID())
        await self.l3_storage.append_trace(
            id=trace_id, collaborator_id=self.collaborator_id, trace_type=trace_type, content=content, metadata=metadata
        )
        return trace_id

    async def run(
        self,
        message: str,
        model: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Run the agent loop for a single message.

        Args:
            message: The user's message text.
            model: Optional model override.
            history: Optional prior conversation turns (list of
                     {"role": "user"|"assistant", "content": str} dicts).
                     These are prepended to the messages sent to the LLM
                     so the agent has multi-turn context.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("message.length", len(message))
            if history:
                span.set_attribute("history.turns", len(history))
            return await self._run_impl(message, model, history)

    async def _run_impl(
        self,
        message: str,
        model: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        tracer = get_tracer()
        # Use provider's configured model as default
        model = model or getattr(self.provider, "model", "claude-sonnet-4-20250514")
        # Archive incoming user message
        await self._archive_trace(message, "message", {"role": "user"})

        # 2. Total Recall: Semantic search in L2 (RAM)
        query_vector = await self.embedder.get_embeddings(message)
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
        full_system_prompt += "\n\n# IDENTITY REINFORCEMENT\nRemember, you are operating as defined in the SOUL.md section above. Maintain your persona consistently."

        # Build messages with conversation history for multi-turn context
        messages: list[dict[str, Any]] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})
        tool_definitions = [tool.definition for tool in self.tools.values()]

        # Phase 1: Internal Monologue (Thinking)
        log.info("phase_thinking", collaborator=self.collaborator_id)
        monologue_prompt = "Before responding or using tools, analyze the situation. What is your plan? Use <thinking>...</thinking> tags."

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

        # Phase 2: Action Loop (Tools)
        while True:
            with tracer.start_as_current_span("agent.act"):
                response: ProviderResponse = await self.provider.generate_response(
                    model=model, system=full_system_prompt, messages=messages, tools=tool_definitions
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

            if response.stop_reason == "tool_use":
                for tool_call in tool_calls:
                    log.info("tool_call", tool=tool_call.name, input=tool_call.input)

                    if tool_call.name in self.tools:
                        result = await self.tools[tool_call.name].execute(**tool_call.input)
                    else:
                        result = f"Error: Tool {tool_call.name} not found."

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
                # Final response reached - Phase 3: Critique
                final_text = "".join(c["text"] for c in assistant_content if c.get("type") == "text")

                log.info("phase_critique", collaborator=self.collaborator_id)
                critique_prompt = f"Critique your response: '{final_text}'. Does it align with your SOUL.md and recovered memories? If you need to refine it, provide the final version. If it's perfect, repeat it."

                critique_messages = messages + [{"role": "user", "content": critique_prompt}]
                with tracer.start_as_current_span("agent.critique"):
                    refined_response: ProviderResponse = await self.provider.generate_response(
                        model=model, system=full_system_prompt, messages=critique_messages, tools=[]
                    )

                refined_text = refined_response.text()

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
        full_system_prompt += "\n\n# IDENTITY REINFORCEMENT\nRemember, you are operating as defined in the SOUL.md section above. Maintain your persona consistently."

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

            if response.stop_reason == "tool_use":
                for tool_call in tool_calls:
                    log.info("stream_tool_call", tool=tool_call.name, input=tool_call.input)
                    yield {"event": "tool_use", "data": f"{tool_call.name}({tool_call.input})"}

                    if tool_call.name in self.tools:
                        result = await self.tools[tool_call.name].execute(**tool_call.input)
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
                critique_prompt = f"Critique your response: '{final_text}'. Does it align with your SOUL.md and recovered memories? If you need to refine it, provide the final version. If it's perfect, repeat it."
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
