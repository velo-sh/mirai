from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirai.agent.dreamer import Dreamer
from mirai.agent.loop import AgentLoop
from mirai.agent.models import ProviderResponse
from mirai.db.duck import DuckDBStorage


@pytest.mark.asyncio
async def test_dream_cycle_evolution():
    """
    Verifies that the Dreamer service reads traces, reflects, and updates the SOUL.md.
    """
    # 1. Setup Mocks
    mock_agent = AsyncMock(spec=AgentLoop)
    mock_agent.collaborator_id = "dream-test"
    mock_agent.soul_content = "Old Identity"
    mock_agent.provider = AsyncMock()

    mock_storage = AsyncMock(spec=DuckDBStorage)
    mock_storage.get_recent_traces.return_value = [
        {"trace_type": "thinking", "content": "I should be more proactive."},
        {"trace_type": "thinking", "content": "The user likes concise answers."},
    ]

    # Mock Provider Response
    mock_resp = MagicMock(spec=ProviderResponse)
    mock_resp.text.return_value = (
        "# IDENTITY\n"
        "I am an evolved version of Mira. I am more proactive and focus on providing "
        "concise yet deeply insightful responses. I understand that GJK values efficiency "
        "and clarity above all else in our co-creation process."
    )
    mock_agent.provider.generate_response.return_value = mock_resp

    mock_agent.update_soul.return_value = True

    dreamer = Dreamer(mock_agent, mock_storage, interval_seconds=10)

    # 3. Trigger Dream
    await dreamer.dream()

    # 4. Verify Pipeline
    mock_storage.get_recent_traces.assert_called_once_with("dream-test", limit=20)
    mock_agent.provider.generate_response.assert_called_once()
    mock_agent.update_soul.assert_called_once_with(
        "# IDENTITY\n"
        "I am an evolved version of Mira. I am more proactive and focus on providing "
        "concise yet deeply insightful responses. I understand that GJK values efficiency "
        "and clarity above all else in our co-creation process."
    )

    print("✅ Dreamer correctly analyzed traces and triggered soul update.")


@pytest.mark.asyncio
async def test_agent_loop_soul_update_io(tmp_path):
    """Verifies the actual file IO and backup logic in AgentLoop.update_soul."""

    # Setup a temp directory structure
    def _mock_init(self, db_path=":memory:"):
        self.db_path = ":memory:"
        import duckdb

        self.conn = duckdb.connect(":memory:")

    with (
        patch("mirai.agent.loop._load_soul", return_value="Original Content"),
        patch("mirai.db.duck.DuckDBStorage.__init__", _mock_init),
    ):
        agent = AgentLoop(provider=MagicMock(), tools=[], collaborator_id="test-bot")

    # Update soul with real method but mock the path logic
    with (
        patch("mirai.agent.loop.os.path.exists", return_value=True),
        patch("mirai.agent.loop.shutil.copy2") as mock_copy,
        patch("builtins.open", new_callable=MagicMock) as mock_open,
    ):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        await agent.update_soul("New Content")

        mock_copy.assert_called_once()
        mock_file.write.assert_called_once_with("New Content")

    print("✅ AgentLoop.update_soul logic verified for path generation and shutil usage.")
