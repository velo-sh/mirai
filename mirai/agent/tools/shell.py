import asyncio
import logging
from typing import Dict, Any, Optional
from mirai.agent.tools.base import BaseTool

class ShellTool(BaseTool):
    """Tool to execute shell commands on the system."""

    @property
    def definition(self) -> Dict[str, Any]:
        return {
            "name": "shell_tool",
            "description": "Execute a Unix/Mac shell command. Use for git, testing, or system operations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The full shell command to execute."},
                },
                "required": ["command"]
            }
        }

    async def execute(self, command: str) -> str:
        try:
            # Basic safety: No interactive commands or backgrounding like '&'
            if "&" in command or "nohup" in command:
                return "Error: Background processes are not permitted."
            
            # Execute with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                process.kill()
                return "Error: Command timed out after 30 seconds."

            result = ""
            if stdout:
                result += f"STDOUT:\n{stdout.decode().strip()}\n"
            if stderr:
                result += f"STDERR:\n{stderr.decode().strip()}\n"
            
            if not result:
                result = "Command executed successfully (no output)."
                
            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"
