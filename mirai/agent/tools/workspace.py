import os
import glob
from typing import List, Dict, Any, Optional
from mirai.agent.tools.base import BaseTool

class WorkspaceTool(BaseTool):
    """Tool to allow the agent to read and list project files."""
    
    @property
    def definition(self) -> Dict[str, Any]:
        return {
            "name": "workspace_tool",
            "description": "Read project files or list the current directory. Supports 'list' and 'read' actions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "read"], "description": "The action to perform."},
                    "path": {"type": "string", "description": "Relative path to file or directory."},
                },
                "required": ["action"]
            }
        }

    async def execute(self, action: str, path: str = ".") -> str:
        # Basic security: stay within current workspace
        safe_path = os.path.normpath(path)
        if safe_path.startswith("..") or safe_path.startswith("/"):
            return "Error: Access denied. Paths must be relative and within the workspace."

        if action == "list":
            files = []
            for item in os.listdir(safe_path):
                if item.startswith(".") or "__pycache__" in item:
                    continue
                files.append(item)
            return f"Files in {safe_path}: " + ", ".join(files)

        elif action == "read":
            if not os.path.isfile(safe_path):
                return f"Error: {safe_path} is not a valid file."
            
            try:
                with open(safe_path, "r") as f:
                    content = f.read()
                return f"--- Content of {safe_path} ---\n{content}"
            except Exception as e:
                return f"Error reading file: {e}"

        return "Error: Unknown action."
