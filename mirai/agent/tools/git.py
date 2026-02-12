import asyncio
import os
from typing import Any

from mirai.agent.tools.base import BaseTool


class GitTool(BaseTool):
    """Tool to perform git operations within the workspace."""

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "git_tool",
            "description": "Perform git operations (status, diff, add, commit, log). Essential for version control within the workbench.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "diff", "add", "commit", "log"],
                        "description": "The git command to execute.",
                    },
                    "args": {
                        "type": "string",
                        "description": "Additional arguments (e.g., file paths for 'add', commit message for 'commit').",
                    },
                },
                "required": ["action"],
            },
        }

    async def execute(self, action: str, args: str = "") -> str:  # type: ignore[override]
        """Execute the requested git command."""
        git_cmd = ""
        if action == "status":
            git_cmd = "git status"
        elif action == "diff":
            git_cmd = f"git diff {args}"
        elif action == "add":
            if not args:
                return "Error: 'add' requires target files (e.g., '.' or 'filename')."
            git_cmd = f"git add {args}"
        elif action == "commit":
            if not args:
                return "Error: 'commit' requires a message."
            git_cmd = f'git commit -m "{args}"'
        elif action == "log":
            git_cmd = f"git log -n 5 {args}"
        else:
            return f"Error: Unsupported git action '{action}'."

        try:
            # Basic security: ensure we are in a git repo
            if not os.path.exists(".git"):
                return "Error: Not a git repository."

            process = await asyncio.create_subprocess_shell(
                git_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15.0)

            result = ""
            if stdout:
                result += f"STDOUT:\n{stdout.decode().strip()}\n"
            if stderr:
                result += f"STDERR:\n{stderr.decode().strip()}\n"

            if not result and process.returncode == 0:
                return f"Git {action} completed successfully."

            return result or "No output from git command."

        except TimeoutError:
            return f"Error: git {action} timed out."
        except Exception as e:
            return f"Error executing git command: {str(e)}"
