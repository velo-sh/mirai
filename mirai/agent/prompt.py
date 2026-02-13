"""System prompt builder — constructs the enriched prompt for the agent.

Extracted from loop.py to separate prompt composition from the core
execution cycle.
"""

from typing import Any

import orjson

from mirai.logging import get_logger

log = get_logger("mirai.agent.prompt")


async def build_system_prompt(
    *,
    collaborator_id: str,
    soul_content: str,
    base_system_prompt: str,
    provider: Any,
    embedder: Any,
    l2_storage: Any,
    l3_storage: Any,
    msg_text: str,
) -> str:
    """Construct the enriched system prompt with identity, memories, and runtime info.

    Follows the sandwich pattern:
        identity → context → memories → runtime → reinforcement → rules
    """
    # Memory recall (L2 → L3)
    query_vector = await embedder.get_embeddings(msg_text)
    memories = await l2_storage.search(
        vector=query_vector, limit=3, filter=f"collaborator_id = '{collaborator_id}'"
    )

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
                memory_context += f"- [{trace['trace_type']}] {trace['content']}\n"

    # Lightweight runtime info (full model catalog is via list_models tool)
    provider_name = getattr(provider, 'provider_name', 'unknown')
    current_model = getattr(provider, 'model', 'unknown')
    runtime_info = f"Provider: {provider_name} | Model: {current_model}"

    # Sandwich pattern: identity → context → memories → runtime → reinforcement → rules
    prompt = f"# IDENTITY\n{soul_content}\n\n# CONTEXT\n{base_system_prompt}"
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
