from typing import List, Dict, Any, Optional
import os
import anthropic

class AnthropicProvider:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def generate_response(
        self, 
        model: str, 
        system: str, 
        messages: List[Dict[str, str]], 
        tools: List[Dict[str, Any]]
    ):
        return await self.client.messages.create(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=4096,
        )

class MockProvider:
    """Mock provider to test the AgentLoop logic without an API key."""
    def __init__(self):
        self.call_count = 0

    async def generate_response(
        self, 
        model: str, 
        system: str, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]]
    ):
        self.call_count += 1
        
        # Simple rule-based mock logic
        last_message = messages[-1]
        
        if self.call_count == 1:
            # First call: trigger a tool use
            from types import SimpleNamespace
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="Let me check that for you."),
                    SimpleNamespace(
                        type="tool_use", 
                        id="call_1", 
                        name="echo", 
                        input={"message": "Hello from Mock!"},
                        model_dump=lambda: {"type": "tool_use", "id": "call_1", "name": "echo", "input": {"message": "Hello from Mock!"}}
                    )
                ],
                stop_reason="tool_use"
            )
        else:
            # Second call: return final text
            from types import SimpleNamespace
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="The tool confirmed: Hello from Mock!")
                ],
                stop_reason="end_turn"
            )
