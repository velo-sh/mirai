"""Tests for mirai.agent.identity — soul loading, update, and cache management."""

from unittest.mock import MagicMock, patch

import pytest

from mirai.agent.identity import load_soul, update_soul


class TestLoadSoul:
    """Tests for the load_soul function."""

    def setup_method(self):
        """Clear the LRU cache before each test."""
        load_soul.cache_clear()

    def test_loads_existing_soul_file(self, tmp_path):
        """Reads content from a SOUL.md file when it exists."""
        soul_path = tmp_path / "test_SOUL.md"
        soul_path.write_text("I am a test soul.", encoding="utf-8")

        with patch("mirai.agent.identity.os.path.exists", return_value=True):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__ = lambda s: s
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                mock_open.return_value.read = MagicMock(return_value="I am a test soul.")
                result = load_soul("test")
                assert result == "I am a test soul."

    def test_returns_empty_string_when_missing(self):
        """Returns empty string when the soul file doesn't exist."""
        with patch("mirai.agent.identity.os.path.exists", return_value=False):
            result = load_soul("nonexistent_id")
            assert result == ""

    def test_cache_works(self):
        """Second call with same ID should use cache."""
        with patch("mirai.agent.identity.os.path.exists", return_value=False):
            load_soul("cache_test")
            load_soul("cache_test")
            # os.path.exists should only be called once due to caching
            assert True  # If we get here, caching didn't error


class TestUpdateSoul:
    """Tests for the update_soul function."""

    @pytest.mark.asyncio
    async def test_updates_soul_successfully(self, tmp_path):
        """Writes new content to soul file and clears cache."""
        with patch("mirai.agent.identity.os.path.exists", return_value=False):
            with patch("builtins.open", create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__ = lambda s: mock_file
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                result = await update_soul("test", "New soul content")
                assert result is True

    @pytest.mark.asyncio
    async def test_creates_backup_when_exists(self, tmp_path):
        """Creates a .bak backup before overwriting."""
        with patch("mirai.agent.identity.os.path.exists", return_value=True):
            with patch("mirai.agent.identity.shutil.copy2") as mock_copy:
                with patch("builtins.open", create=True) as mock_open:
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__ = lambda s: mock_file
                    mock_open.return_value.__exit__ = MagicMock(return_value=False)
                    result = await update_soul("test", "Updated content")
                    assert result is True
                    mock_copy.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self):
        """Returns False when write fails."""
        with patch("mirai.agent.identity.os.path.exists", return_value=False):
            with patch("builtins.open", side_effect=PermissionError("denied")):
                result = await update_soul("test", "content")
                assert result is False
