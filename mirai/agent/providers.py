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
        
        if "# IDENTITY" in system:
            print(f"[mock] IDENTITY ANCHORS PRESENT (Sandwich Pattern)")

        last_message = messages[-1]
        last_content = last_message.get("content", "")
        if isinstance(last_content, list):
            # Extract text if list (Anthropic format)
            last_text = "".join([c.get("text", "") for c in last_content if isinstance(c, dict) and c.get("type") == "text"])
        else:
            last_text = str(last_content)

        from types import SimpleNamespace

        # 1. Thinking Turn
        if "analyze the situation" in last_text:
            print("[mock] Handling Thinking Turn")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="<thinking>The user wants to store information. I should use the memorize tool.</thinking>")],
                stop_reason="end_turn"
            )
            
        # 2. Critique Turn
        if "Critique your response" in last_text:
            print("[mock] Handling Critique Turn")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="The response is aligned with my SOUL.md. Final version: I have successfully archived that insight.")],
                stop_reason="end_turn"
            )

        # 3. Tool Turn / Normal Chat
        if tools and self.call_count < 10:
            print("[mock] Handling Tool Turn")
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="Let me save that to my memory."),
                    SimpleNamespace(
                        type="tool_use", 
                        id="call_mem_1", 
                        name="memorize", 
                        input={"content": "The Mirai system implements System 2 thinking.", "importance": 0.9},
                        model_dump=lambda: {"type": "tool_use", "id": "call_mem_1", "name": "memorize", "input": {"content": "The Mirai system implements System 2 thinking.", "importance": 0.9}}
                    )
                ],
                stop_reason="tool_use"
            )
        else:
            print("[mock] Handling final answer or normal chat")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="The system now supports reasoning loops.")],
                stop_reason="end_turn"
            )
