import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.agent.agent_dreamer import AgentDreamer
from mirai.agent.im.feishu_receiver import FeishuEventReceiver
from mirai.agent.tools.editor import EditorTool
from mirai.agent.tools.git import GitTool
from mirai.agent.tools.shell import ShellTool


# --- 1. Security: Path Traversal ---
@pytest.mark.asyncio
async def test_editor_path_traversal():
    """Ensures EditorTool rejects paths outside the workspace."""
    tool = EditorTool()

    # Attempt to write outside workspace
    result = await tool.execute(action="write", path="../secret.txt", content="hack")
    assert "Security Error" in result or "outside the allowed workspace" in result


# --- 2. Robustness: Command Timeout ---
@pytest.mark.asyncio
async def test_shell_command_timeout():
    """Ensures ShellTool terminates long-running commands."""
    tool = ShellTool()
    # We use a command that takes longer than the 30s timeout
    # To run this test quickly, we'll patch the timeout to something small
    with patch("mirai.agent.tools.shell.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await tool.execute(command="sleep 100")

    assert "timed out" in result


# --- 3. Robustness: IM Image Download Error ---
@pytest.mark.asyncio
async def test_feishu_image_download_failure():
    """Verifies FeishuEventReceiver handles download errors gracefully."""
    receiver = FeishuEventReceiver(app_id="id", app_secret="secret", message_handler=AsyncMock())

    # Mock failure response from Lark SDK
    mock_resp = MagicMock()
    mock_resp.success.return_value = False
    mock_resp.code = 404
    mock_resp.msg = "Not Found"

    receiver._reply_client = MagicMock()
    receiver._reply_client.im.v1.message.aget_resource = AsyncMock(return_value=mock_resp)

    data = await receiver._download_image("msg_id", "img_key")
    assert data is None


# --- 4. Logic: Dreamer Safeguards ---
@pytest.mark.asyncio
async def test_dreamer_short_response_safeguard():
    """Verifies Dreamer skips updates if the generated soul is too short."""
    mock_agent = AsyncMock()
    mock_agent.soul_content = "Existing Soul Content"
    mock_agent.provider = AsyncMock()

    mock_storage = AsyncMock()
    mock_storage.get_recent_traces.return_value = [{"trace_type": "thinking", "content": "reflect"}]

    # Mock a response that is too short (< 100 chars)
    mock_resp = MagicMock()
    mock_resp.text.return_value = "Too short soul content."
    mock_agent.provider.generate_response.return_value = mock_resp

    dreamer = AgentDreamer(mock_agent, mock_storage)
    await dreamer.dream()

    # update_soul should NOT have been called
    assert mock_agent.update_soul.call_count == 0


# --- 5. Robustness: Git Outside Repo ---
@pytest.mark.asyncio
async def test_git_outside_repo():
    """Verifies GitTool handles directories without a .git folder."""
    tool = GitTool()
    with patch("os.path.exists", return_value=False):
        result = await tool.execute(action="status")

    assert "Not a git repository" in result


# --- 6. Robustness: Shell Command Metacharacter Injection ---
@pytest.mark.asyncio
async def test_shell_command_injection_attempt():
    """Ensures ShellTool handles potential backgrounding injection."""
    tool = ShellTool()
    result = await tool.execute(command="echo hi & rm -rf /")
    assert "Background processes are not permitted" in result
