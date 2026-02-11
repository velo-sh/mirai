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

        # Extract text from all messages to find the user's original intent
        full_history_text = ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                full_history_text += " ".join([c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"])
            else:
                full_history_text += str(content)
        
        last_text = str(last_text)
        print(f"[mock] last_text: {last_text[:50]}...")

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
        
        # 3. Tool Turn / Sequential Logic
        # We need to decide if we should call another tool or give a final answer.
        
        # Check if the VERY LAST assistant message had a tool_use that hasn't been answered yet
        # Actually, in AgentLoop, if stop_reason is tool_use, it executes and appends result.
        # So when we are called again, the last message is a tool_result from user.
        
        last_role = messages[-1].get("role")
        
        if tools:
            # E2E Proactive Maintenance Workflow
            if "maintenance_check" in full_history_text.lower():
                 print("[mock] Match: E2E Maintenance Workflow")
                 
                 shell_results = [m for m in messages if m.get("role") == "user" and any("tool_result" in str(c) and "call_shell_maint" in str(c) for c in (m.get("content") if isinstance(m.get("content"), list) else []))]
                 
                 if not shell_results:
                    return SimpleNamespace(
                        content=[
                            SimpleNamespace(type="text", text="Checking for maintenance issues..."),
                            SimpleNamespace(
                                type="tool_use", 
                                id="call_shell_maint", 
                                name="shell_tool", 
                                input={"command": "ls maintenance_fixed.txt"},
                                model_dump=lambda: {"type": "tool_use", "id": "call_shell_maint", "name": "shell_tool", "input": {"command": "ls maintenance_fixed.txt"}}
                            )
                        ],
                        stop_reason="tool_use"
                    )
                 
                 editor_results = [m for m in messages if m.get("role") == "user" and any("tool_result" in str(c) and "call_edit_maint" in str(c) for c in (m.get("content") if isinstance(m.get("content"), list) else []))]
                 
                 if not editor_results:
                    return SimpleNamespace(
                        content=[
                            SimpleNamespace(type="text", text="Fixing the issue..."),
                            SimpleNamespace(
                                type="tool_use", 
                                id="call_edit_maint", 
                                name="editor_tool", 
                                input={"action": "write", "path": "maintenance_fixed.txt", "content": "HEALED: Sanity check passed."},
                                model_dump=lambda: {"type": "tool_use", "id": "call_edit_maint", "name": "editor_tool", "input": {"action": "write", "path": "maintenance_fixed.txt", "content": "HEALED: Sanity check passed."}}
                            )
                        ],
                        stop_reason="tool_use"
                    )

            # Default single-tool branch for other tests
            has_any_tool_use = any(msg.get("role") == "assistant" and "tool_use" in str(msg.get("content", "")) for msg in messages)
            if not has_any_tool_use:
                # Workspace List Request
                if "List the files" in last_text or "SYSTEM_HEARTBEAT" in last_text:
                    # ... (rest of old logic)
                    pass

        # Final Answer if no more tools needed
        print(f"[mock] Handling final answer")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="I have finished the complex task.")],
            stop_reason="end_turn"
        )
