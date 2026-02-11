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

        last_text = str(last_text)
        print(f"[mock] last_text: {last_text[:50]}...")

        last_text = str(last_text)
        print(f"[mock] last_text: {last_text[:100]}...")

        # 1. Thinking Turn
        if "analyze the situation" in last_text or "perform a self-reflection" in last_text:
            print("[mock] Handling Thinking Turn")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="<thinking>The system heartbeat has triggered. I should scan the project and summarize progress.</thinking>")],
                stop_reason="end_turn"
            )
            
        # 2. Critique Turn
        if "Critique your response" in last_text:
            print("[mock] Handling Critique Turn")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="The response is aligned with my SOUL.md. Final version: I have successfully completed the proactive scan.")],
                stop_reason="end_turn"
            )

        # 3. Tool Turn / Normal Chat
        # If the last message is from the assistant and contains text, we might be reaching the end
        # But in our mock, we want to trigger a tool at least once if tools are available.
        # We'll use a local check to see if we already sent a tool in this specific history.
        
        has_tool_use = any(msg.get("role") == "assistant" and "tool_use" in str(msg.get("content", "")) for msg in messages)

        if tools and not has_tool_use:
            print(f"[mock] Handling Tool Turn (new interaction)")
            
            # Workspace List Request
            if "List the files" in last_text or "SYSTEM_HEARTBEAT" in last_text:
                 print("[mock] Match: Workspace Scan")
                 return SimpleNamespace(
                    content=[
                        SimpleNamespace(type="text", text="Checking the workspace..."),
                        SimpleNamespace(
                            type="tool_use", 
                            id="call_ws_list_99", 
                            name="workspace_tool", 
                            input={"action": "list", "path": "."},
                            model_dump=lambda: {"type": "tool_use", "id": "call_ws_list_99", "name": "workspace_tool", "input": {"action": "list", "path": "."}}
                        )
                    ],
                    stop_reason="tool_use"
                )

            # Workspace Read Request
            if "Read the content" in last_text:
                 print("[mock] Match: Workspace Read")
                 return SimpleNamespace(
                    content=[
                        SimpleNamespace(type="text", text="Reading the file content..."),
                        SimpleNamespace(
                            type="tool_use", 
                            id="call_ws_read_99", 
                            name="workspace_tool", 
                            input={"action": "read", "path": "main.py"},
                            model_dump=lambda: {"type": "tool_use", "id": "call_ws_read_99", "name": "workspace_tool", "input": {"action": "read", "path": "main.py"}}
                        )
                    ],
                    stop_reason="tool_use"
                )

            # Default
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Acknowledged. No specific tool needed for this mock branch.")],
                stop_reason="end_turn"
            )
        else:
            print(f"[mock] Handling final answer")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="I have finished my workspace tasks.")],
                stop_reason="end_turn"
            )
