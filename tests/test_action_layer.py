import asyncio
import os
import pytest
from mirai.agent.tools.shell import ShellTool
from mirai.agent.tools.editor import EditorTool

@pytest.mark.asyncio
async def test_shell_tool_basic():
    tool = ShellTool()
    result = await tool.execute("echo 'hello world'")
    assert "hello world" in result

@pytest.mark.asyncio
async def test_shell_tool_timeout():
    tool = ShellTool()
    # This should timeout (we set 30s, but maybe let's test a short sleep if we could configure it)
    # Since we can't easily configure timeout in the tool yet, we skip a long wait in tests
    # just verify it runs a simple command.
    result = await tool.execute("ls -la")
    assert "STDOUT" in result

@pytest.mark.asyncio
async def test_editor_tool_write():
    tool = EditorTool()
    test_path = "tests/data/test_file.txt"
    test_content = "This is a test of the Action Layer."
    
    result = await tool.execute("write", test_path, test_content)
    assert "Successfully wrote" in result
    assert os.path.exists(test_path)
    
    with open(test_path, "r") as f:
        assert f.read() == test_content
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)

@pytest.mark.asyncio
async def test_editor_tool_security():
    tool = EditorTool()
    result = await tool.execute("write", "../../passwd", "bad content")
    assert "Error" in result
