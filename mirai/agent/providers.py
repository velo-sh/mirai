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
