"""Unit tests for mirai.agent.tools — echo, workspace security boundaries."""

import pytest

from mirai.agent.tools.echo import EchoTool
from mirai.agent.tools.workspace import WorkspaceTool

# ---------------------------------------------------------------------------
# EchoTool
# ---------------------------------------------------------------------------


class TestEchoTool:
    def test_definition_has_required_fields(self):
        tool = EchoTool()
        defn = tool.definition
        assert defn["name"] == "echo"
        assert "description" in defn
        assert "input_schema" in defn
        assert "message" in defn["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_echoes_message(self):
        tool = EchoTool()
        result = await tool.execute(message="hello")
        assert result == "Echoed: hello"

    @pytest.mark.asyncio
    async def test_execute_empty_message(self):
        tool = EchoTool()
        result = await tool.execute(message="")
        assert result == "Echoed: "


# ---------------------------------------------------------------------------
# WorkspaceTool — Core functionality
# ---------------------------------------------------------------------------


class TestWorkspaceToolList:
    def test_definition_has_required_fields(self):
        tool = WorkspaceTool()
        defn = tool.definition
        assert defn["name"] == "workspace_tool"
        assert "list" in str(defn)
        assert "read" in str(defn)

    @pytest.mark.asyncio
    async def test_list_current_directory(self):
        tool = WorkspaceTool()
        result = await tool.execute(action="list", path=".")
        # Should list at least some files from the project root
        assert "Files in" in result

    @pytest.mark.asyncio
    async def test_list_excludes_dotfiles(self):
        tool = WorkspaceTool()
        result = await tool.execute(action="list", path=".")
        assert ".git" not in result
        assert "__pycache__" not in result

    @pytest.mark.asyncio
    async def test_read_existing_file(self):
        tool = WorkspaceTool()
        result = await tool.execute(action="read", path="pyproject.toml")
        assert "Content of" in result
        assert "mirai" in result.lower()

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        tool = WorkspaceTool()
        result = await tool.execute(action="read", path="nonexistent_file_qwerty.txt")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = WorkspaceTool()
        result = await tool.execute(action="delete")
        assert "Error" in result


# ---------------------------------------------------------------------------
# WorkspaceTool — Security boundaries (QA critical path)
# ---------------------------------------------------------------------------


class TestWorkspaceToolSecurity:
    """These tests verify that the workspace tool cannot escape the CWD."""

    @pytest.mark.asyncio
    async def test_reject_absolute_path(self):
        tool = WorkspaceTool()
        with pytest.raises(ValueError, match="Security Error"):
            await tool.execute(action="read", path="/etc/passwd")

    @pytest.mark.asyncio
    async def test_reject_parent_traversal(self):
        tool = WorkspaceTool()
        with pytest.raises(ValueError, match="Security Error"):
            await tool.execute(action="read", path="../../etc/passwd")

    @pytest.mark.asyncio
    async def test_reject_backslash_traversal(self):
        """Backslash path segments should be normalized and rejected."""
        tool = WorkspaceTool()
        with pytest.raises(ValueError, match="Security Error"):
            await tool.execute(action="read", path="..\\..\\etc\\passwd")

    @pytest.mark.asyncio
    async def test_reject_home_directory(self):
        tool = WorkspaceTool()
        with pytest.raises(ValueError, match="Security Error"):
            await tool.execute(action="list", path="/Users")

    @pytest.mark.asyncio
    async def test_allow_subdirectory(self):
        """Paths within the workspace should be allowed."""
        tool = WorkspaceTool()
        # 'mirai/' is in the project
        result = await tool.execute(action="list", path="mirai")
        assert "Files in" in result

    @pytest.mark.asyncio
    async def test_allow_nested_file(self):
        tool = WorkspaceTool()
        result = await tool.execute(action="read", path="mirai/config.py")
        assert "Content of" in result
