import json
from typing import List, Dict, Any, Optional
from mirai.agent.providers import AnthropicProvider
from mirai.agent.tools.base import BaseTool

class AgentLoop:
    def __init__(self, provider: AnthropicProvider, tools: List[BaseTool], system_prompt: str):
        self.provider = provider
        self.tools = {tool.definition["name"]: tool for tool in tools}
        self.system_prompt = system_prompt

    async def run(self, message: str, model: str = "claude-3-5-sonnet-20241022") -> str:
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
                return "".join([c["text"] for c in assistant_content if c["type"] == "text"])
