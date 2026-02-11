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

class MockEmbeddingProvider:
    """Provides consistent fake embeddings for testing."""
    def __init__(self, dim: int = 1536):
        self.dim = dim

    async def get_embeddings(self, text: str) -> List[float]:
        # Return a deterministic-looking mock vector based on text hash
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vector = []
        for i in range(self.dim):
            # Very simple fake vector generation
            val = (h[i % 32] / 255.0) - 0.5
            vector.append(val)
        return vector

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
        
        if "Recovered Memories" in system:
            print(f"[mock] SYSTEM PROMPT HAS MEMORIES:\n{system[system.find('### Recovered Memories'):]}")
        
        # Simple rule-based mock logic
        last_message = messages[-1]
        
        if self.call_count == 1:
            # First call: trigger a 'memorize' tool use
            from types import SimpleNamespace
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="That's an important point about the system architecture. Let me save that to my memory."),
                    SimpleNamespace(
                        type="tool_use", 
                        id="call_mem_1", 
                        name="memorize", 
                        input={"content": "The Mirai system uses a 3-tier memory model inspired by biological brains.", "importance": 0.9},
                        model_dump=lambda: {"type": "tool_use", "id": "call_mem_1", "name": "memorize", "input": {"content": "The Mirai system uses a 3-tier memory model inspired by biological brains.", "importance": 0.9}}
                    )
                ],
                stop_reason="tool_use"
            )
        else:
            # Second call: return final text acknowledging the memory
            from types import SimpleNamespace
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="I've successfully archived that insight. I'll remember the 3-tier memory model for our future discussions.")
                ],
                stop_reason="end_turn"
            )
