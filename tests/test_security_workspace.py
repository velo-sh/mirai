import asyncio

import pytest

from mirai.agent.tools.workspace import WorkspaceTool


@pytest.mark.asyncio
async def test_workspace_security_path_traversal():
    """
    QA Security Test: Ensure WorkspaceTool prevents path traversal.
    """
    tool = WorkspaceTool()

    # 1. Attempt to escape root via ../
    with pytest.raises(ValueError) as excinfo:
        await tool.execute(action="read", path="../../etc/passwd")
    assert "Path must be within current directory" in str(excinfo.value)

    # 2. Attempt to use absolute path
    with pytest.raises(ValueError) as excinfo:
        await tool.execute(action="read", path="/etc/passwd")
    assert "Absolute paths are not allowed" in str(excinfo.value)

    # 3. Attempt to list outside root
    with pytest.raises(ValueError) as excinfo:
        await tool.execute(action="list", path="..")
    assert "Path must be within current directory" in str(excinfo.value)

    print("\n[QA] Workspace Security Test Passed: Path traversal blocked.")


if __name__ == "__main__":
    asyncio.run(test_workspace_security_path_traversal())
