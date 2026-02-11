from typing import List, Dict, Any, Optional
from mirai.agent.providers import AnthropicProvider
from mirai.agent.tools.base import BaseTool
from mirai.db.duck import DuckDBStorage
from ulid import ULID
import json

class AgentLoop:
    def __init__(self, provider: AnthropicProvider, tools: List[BaseTool], system_prompt: str, collaborator_id: str):
        self.provider = provider
        self.tools = {tool.definition["name"]: tool for tool in tools}
        self.system_prompt = system_prompt
        self.collaborator_id = collaborator_id
        self.l3_storage = DuckDBStorage()

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
        
        messages = [{"role": "user", "content": message}]
        tool_definitions = [tool.definition for tool in self.tools.values()]

        while True:
            response = await self.provider.generate_response(
                model=model,
                system=self.system_prompt,
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
