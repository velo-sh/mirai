import asyncio
from typing import Any

from mirai.agent.tools.base import BaseTool


class ShellTool(BaseTool):
    """Tool to execute shell commands on the system."""

    DEFAULT_TIMEOUT: float = 30.0
    MAX_OUTPUT_CHARS: int = 50_000

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "shell_tool",
            "description": "Execute a Unix/Mac shell command. Use for git, testing, or system operations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The full shell command to execute."},
                },
                "required": ["command"],
            },
        }

    async def execute(self, command: str) -> str:  # type: ignore[override]
        try:
            # Basic safety: No interactive commands or backgrounding like '&'
            if "&" in command or "nohup" in command:
                return "Error: Background processes are not permitted."

            # Execute with configurable timeout
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.DEFAULT_TIMEOUT)
            except TimeoutError:
                process.kill()
                return f"Error: Command timed out after {self.DEFAULT_TIMEOUT} seconds."

            result = ""
            if stdout:
                result += f"STDOUT:\n{stdout.decode().strip()}\n"
            if stderr:
                result += f"STDERR:\n{stderr.decode().strip()}\n"

            if not result:
                result = "Command executed successfully (no output)."

            # Truncate excessively long output
            if len(result) > self.MAX_OUTPUT_CHARS:
                result = result[: self.MAX_OUTPUT_CHARS] + f"\n\n... (truncated at {self.MAX_OUTPUT_CHARS} chars)"

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"
