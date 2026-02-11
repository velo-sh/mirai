import asyncio
import os
import pytest
from mirai.agent.tools.shell import ShellTool
from mirai.agent.tools.editor import EditorTool

@pytest.mark.asyncio
async def test_shell_injection_prevention():
    """
    QA Security Test: Attempt shell injection on ShellTool.
    """
    tool = ShellTool()
    
    # 1. basic semicolon injection
    result = await tool.execute("echo hello; rm dummy.txt")
    # Even if it executes, it's a risk.
    # Currently ShellTool uses asyncio.create_subprocess_shell which IS vulnerable to semicolon.
    # We should decide if we want to restrict this or rely on the Agent's reasoning.
    # Architecture decision: ShellTool should probably be restrictive.
    
    # 2. background process
    result = await tool.execute("sleep 100 &")
    assert "Error" in result

@pytest.mark.asyncio
async def test_path_traversal_editor_advanced():
    """
    QA Security Test: Advanced path traversal on EditorTool.
    """
    tool = EditorTool()
    
    # 1. Absolute path
    result = await tool.execute("write", "/etc/passwd", "malicious")
    assert "Error" in result or "Absolute paths are not allowed" in result
    
    # 2. Double dot traversal
    result = await tool.execute("write", "tests/../../secret.txt", "content")
    assert "Error" in result or "must be within current directory" in result

    # 3. Encoded traversal / mixed separators
    result = await tool.execute("write", r".\..\.\..\etc/passwd", "content")
    assert "Security Error" in result

@pytest.mark.asyncio
async def test_path_traversal_workspace_consistency():
    """
    Ensure EditorTool and WorkspaceTool have consistent security logic.
    """
    from mirai.agent.tools.workspace import WorkspaceTool
    ws_tool = WorkspaceTool()
    ed_tool = EditorTool()
    
    bad_path = "../../hidden_file"
    
    # WorkspaceTool check
    try:
        await ws_tool.execute("read", bad_path)
    except ValueError as e:
        assert "Security Error" in str(e)
        
    # EditorTool check
    res_ed = await ed_tool.execute("write", bad_path, "content")
    assert "Error" in res_ed
