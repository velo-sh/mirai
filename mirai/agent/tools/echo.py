from typing import Any

from mirai.agent.tools.base import BaseTool


class EchoTool(BaseTool):
    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "echo",
            "description": "Returns the exact message provided. Used for testing connectivity.",
            "input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "The message to repeat."}},
                "required": ["message"],
            },
        }

    async def execute(self, message: str) -> str:  # type: ignore[override]
        return f"Echoed: {message}"
