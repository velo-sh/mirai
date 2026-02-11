from mirai.agent.tools.base import BaseTool
from typing import Dict, Any

class EchoTool(BaseTool):
    @property
    def definition(self) -> Dict[str, Any]:
        return {
            "name": "echo",
            "description": "Returns the exact message provided. Used for testing connectivity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to repeat."
                    }
                },
                "required": ["message"]
            }
        }

    async def execute(self, message: str) -> str:
        return f"Echoed: {message}"
