from typing import List, Dict, Any, Optional
from mirai.agent.providers import AnthropicProvider, MockEmbeddingProvider
from mirai.agent.tools.base import BaseTool
from mirai.db.duck import DuckDBStorage
from mirai.memory.vector_db import VectorStore
from ulid import ULID
import json

class AgentLoop:
    def __init__(self, provider: AnthropicProvider, tools: List[BaseTool], system_prompt: str, collaborator_id: str):
        self.provider = provider
        self.tools = {tool.definition["name"]: tool for tool in tools}
        self.system_prompt = system_prompt
        self.collaborator_id = collaborator_id
        self.l3_storage = DuckDBStorage()
        self.l2_storage = VectorStore()
        self.embedder = MockEmbeddingProvider()

    async def _archive_trace(self, content: str, trace_type: str, metadata: Dict[str, Any] = None):
        """Helper to save a trace to the L3 (HDD) storage using DuckDB."""
        trace_id = str(ULID())
        await self.l3_storage.append_trace(
            id=trace_id,
            collaborator_id=self.collaborator_id,
            trace_type=trace_type,
            content=content,
            metadata=metadata
        )
        return trace_id

    async def run(self, message: str, model: str = "claude-3-5-sonnet-20241022") -> str:
        # Archive incoming user message
        await self._archive_trace(message, "message", {"role": "user"})
        
        # 2. Total Recall: Semantic search in L2 (RAM)
        query_vector = await self.embedder.get_embeddings(message)
        memories = await self.l2_storage.search(
            vector=query_vector, 
            limit=3, 
            filter=f"collaborator_id = '{self.collaborator_id}'"
        )
        
        memory_context = ""
        if memories:
            trace_ids = []
            for mem in memories:
                # Use metadata which is stored as JSON string in LanceDB
                meta = json.loads(mem["metadata"])
                if "trace_id" in meta:
                    trace_ids.append(meta["trace_id"])
            
            # 3. Fetch raw context from L3 (HDD)
            raw_traces = await self.l3_storage.get_traces_by_ids(trace_ids)
            if raw_traces:
                memory_context = "\n### Recovered Memories (L3 Raw Context):\n"
                for trace in raw_traces:
                    memory_context += f"- [{trace['trace_type']}] {trace['content']}\n"
        
        # 4. Construct enriched system prompt
        full_system_prompt = self.system_prompt
        if memory_context:
            full_system_prompt += "\n\n" + memory_context + "\nUse the above memories if they are relevant to the current request."

        messages = [{"role": "user", "content": message}]
        tool_definitions = [tool.definition for tool in self.tools.values()]

        while True:
            response = await self.provider.generate_response(
                model=model,
                system=full_system_prompt,
                messages=messages,
                tools=tool_definitions
            )

            # Process assistant response
            assistant_content = []
            tool_calls = []
            
            for content in response.content:
                if content.type == "text":
                    assistant_content.append({"type": "text", "text": content.text})
                elif content.type == "tool_use":
                    tool_calls.append(content)
                    assistant_content.append(content.model_dump())

            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "tool_use":
                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_input = tool_call.input
                    
                    print(f"[agent] Using tool: {tool_name} with input: {tool_input}")
                    
                    if tool_name in self.tools:
                        result = await self.tools[tool_name].execute(**tool_input)
                    else:
                        result = f"Error: Tool {tool_name} not found."

                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result,
                            }
                        ],
                    })
            else:
                # Final response reached
                final_text = "".join([c["text"] for c in assistant_content if c["type"] == "text"])
                await self._archive_trace(final_text, "message", {"role": "assistant"})
                return final_text
