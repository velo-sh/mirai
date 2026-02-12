from unittest.mock import AsyncMock, patch

import pytest

from mirai.agent.tools.editor import EditorTool
from mirai.agent.tools.git import GitTool
from mirai.agent.tools.shell import ShellTool


@pytest.mark.asyncio
async def test_git_tool_status():
    """Verifies that GitTool can execute a basic status command."""
    tool = GitTool()

    # Mock subprocess
    with patch("asyncio.create_subprocess_shell") as mock_sub:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"On branch main\nNothing to commit", b"")
        mock_process.returncode = 0
        mock_sub.return_value = mock_process

        with patch("os.path.exists", return_value=True):  # Simulate .git exists
            result = await tool.execute(action="status")

    assert "On branch main" in result
    print("✅ GitTool status verified.")


@pytest.mark.asyncio
async def test_shell_tool_echo():
    """Verifies that ShellTool can execute an echo command."""
    tool = ShellTool()
    result = await tool.execute(command="echo 'Hello Workbench'")
    assert "Hello Workbench" in result
    print("✅ ShellTool echo verified.")


@pytest.mark.asyncio
async def test_editor_tool_write(tmp_path):
    """Verifies that EditorTool can write a file safely."""
    tool = EditorTool()
    test_file = tmp_path / "test_write.txt"

    # We need to mock os.getcwd to point to the tmp_path for the security check
    with patch("os.getcwd", return_value=str(tmp_path)):
        result = await tool.execute(action="write", path="test_write.txt", content="Workbench Content")

    assert "Successfully wrote" in result
    assert test_file.read_text() == "Workbench Content"
    print("✅ EditorTool write verified.")
