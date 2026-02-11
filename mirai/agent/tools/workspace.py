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
        try:
            # Normalize slashes for security (handle backslashes on POSIX)
            path = path.replace("\\", "/")
            cwd = os.getcwd()
            target_path = os.path.abspath(os.path.join(cwd, path))
            
            if not target_path.startswith(cwd):
                raise ValueError(f"Security Error: Path {path} is outside the allowed workspace.")
            
            # Additional check for absolute paths in the input string itself
            if os.path.isabs(path) or path.startswith("/"):
                 raise ValueError("Security Error: Absolute paths are not allowed.")
                 
            safe_path = target_path
        except Exception as e:
            if "Security Error" in str(e): raise
            raise ValueError(f"Security Error: Invalid path provided: {path}")

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
