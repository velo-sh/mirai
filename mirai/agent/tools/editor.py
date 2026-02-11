import os
import logging
from typing import Dict, Any, Optional
from mirai.agent.tools.base import BaseTool

class EditorTool(BaseTool):
    """Tool to modify or create files in the workspace."""

    @property
    def definition(self) -> Dict[str, Any]:
        return {
            "name": "editor_tool",
            "description": "Create or overwrite files in the workspace. Supports 'write' action.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["write"], "description": "The action to perform."},
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "content": {"type": "string", "description": "Full content to write to the file."}
                },
                "required": ["action", "path", "content"]
            }
        }

    async def execute(self, action: str, path: str, content: str) -> str:
        # Security: Re-use the path validation logic
        from mirai.agent.tools.workspace import WorkspaceTool
        
        # We need a quick way to validate without duplicating logic
        # For now, let's normalize and check traversal
        safe_path = os.path.normpath(path)
        if safe_path.startswith("/") or os.path.isabs(safe_path):
            return "Error: Absolute paths are not allowed."
        if safe_path.startswith(".."):
            return "Error: Path must be within current directory."

        if action == "write":
            try:
                # Ensure directory exists
                directory = os.path.dirname(safe_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                
                with open(safe_path, "w") as f:
                    f.write(content)
                return f"Successfully wrote {len(content)} characters to {safe_path}."
            except Exception as e:
                return f"Error writing file: {str(e)}"
        
        return f"Error: Unknown action '{action}'."
