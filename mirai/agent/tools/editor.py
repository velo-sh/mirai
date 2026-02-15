import os
from typing import Any

from mirai.agent.tools.base import BaseTool


class EditorTool(BaseTool):
    """Tool to modify or create files in the workspace."""

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "editor_tool",
            "description": "Create or overwrite files in the workspace. Supports 'write' action.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["write"], "description": "The action to perform."},
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "content": {"type": "string", "description": "Full content to write to the file."},
                },
                "required": ["action", "path", "content"],
            },
        }

    async def execute(self, action: str, path: str, content: str) -> str:  # type: ignore[override]
        # Security: Robust path validation
        try:
            # Normalize slashes for security (handle backslashes on POSIX)
            path = path.replace("\\", "/")
            cwd = os.getcwd()
            target_path = os.path.abspath(os.path.join(cwd, path))

            if not target_path.startswith(cwd):
                return f"Error: Security Error: Path {path} is outside the allowed workspace."

            if os.path.isabs(path) or path.startswith("/"):
                return "Error: Security Error: Absolute paths are not allowed."

            safe_path = target_path
        except (ValueError, OSError) as e:
            return f"Error: Security Error: Invalid path provided: {str(e)}"

        if action == "write":
            try:
                # Ensure directory exists
                directory = os.path.dirname(safe_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

                with open(safe_path, "w") as f:
                    f.write(content)
                return f"Successfully wrote {len(content)} characters to {safe_path}."
            except OSError as e:
                return f"Error writing file: {str(e)}"

        return f"Error: Unknown action '{action}'."
