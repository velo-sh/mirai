"""System prompt builder — constructs the enriched prompt for the agent.

Extracted from loop.py to separate prompt composition from the core
execution cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson

from mirai.agent.providers.base import ProviderProtocol
from mirai.agent.tools.base import BaseTool

if TYPE_CHECKING:
    from mirai.agent.providers.embeddings import EmbedderProtocol
    from mirai.db.duck import DuckDBStorage
    from mirai.memory.vector_db import VectorStore


def _build_tools_section(tools: dict[str, BaseTool]) -> str:
    """Build the '## Available Tools' section by reading each tool's own definition.

    Each tool carries a ``definition`` property with a ``description`` field.
    We simply echo that description — no hardcoded summaries needed.
    """
    if not tools:
        return ""

    lines = ["## Available Tools", "You have the following tools:"]
    for name, tool in tools.items():
        desc = tool.definition.get("description", "")
        if desc:
            lines.append(f"- **{name}**: {desc}")
        else:
            lines.append(f"- **{name}**")
    return "\n".join(lines)


async def build_system_prompt(
    *,
    collaborator_id: str,
    soul_content: str,
    base_system_prompt: str,
    provider: ProviderProtocol,
    embedder: EmbedderProtocol,
    l2_storage: VectorStore,
    l3_storage: DuckDBStorage,
    msg_text: str,
    tools: dict[str, BaseTool] | None = None,
) -> str:
    """Construct the enriched system prompt with identity, memories, and runtime info.

    Follows the sandwich pattern:
        identity → context → tools → memories → runtime → reinforcement → rules
    """
    # Memory recall (L2 → L3)
    query_vector = await embedder.get_embeddings(msg_text)
    memories = await l2_storage.search(vector=query_vector, limit=3, filter=f"collaborator_id = '{collaborator_id}'")

    memory_context = ""
    if memories:
        trace_ids = []
        for mem in memories:
            meta = orjson.loads(mem["metadata"])
            if "trace_id" in meta:
                trace_ids.append(meta["trace_id"])
        raw_traces = await l3_storage.get_traces_by_ids(trace_ids)
        if raw_traces:
            memory_context = "\n### Recovered Memories (L3 Raw Context):\n"
            for trace in raw_traces:
                memory_context += f"- [{trace.trace_type}] {trace.content}\n"

    # Lightweight runtime info (full model catalog is via list_models tool)
    provider_name = provider.provider_name
    current_model = provider.model
    runtime_info = f"Provider: {provider_name} | Model: {current_model}"

    # Tools section (dynamic — reads descriptions from each tool's definition)
    tools_section = _build_tools_section(tools or {})

    # Sandwich pattern: identity → context → tools → memories → runtime → reinforcement → rules
    prompt = f"# IDENTITY\n{soul_content}\n\n# CONTEXT\n{base_system_prompt}"
    if tools_section:
        prompt += "\n\n" + tools_section
    if memory_context:
        prompt += "\n\n" + memory_context + "\nUse the above memories if they are relevant to the current request."
    prompt += f"\n\n# RUNTIME INFO\n{runtime_info}"
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
