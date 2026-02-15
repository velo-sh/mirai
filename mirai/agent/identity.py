"""Identity management — soul loading, collaborator init, and soul evolution.

Extracted from loop.py to separate identity concerns from the core
execution cycle.
"""

import functools
import os
import shutil

from mirai.collaborator.manager import CollaboratorManager
from mirai.db.session import get_session
from mirai.logging import get_logger

log = get_logger("mirai.agent.identity")


@functools.lru_cache(maxsize=4)
def load_soul(collaborator_id: str) -> str:
    """Load SOUL.md from disk (cached after first read)."""
    path = f"mirai/collaborator/{collaborator_id}_SOUL.md"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


async def initialize_collaborator(collaborator_id: str) -> tuple[str, str, str]:
    """Load collaborator metadata from the database.

    Returns:
        (name, role, system_prompt) tuple.
    """

    async for session in get_session():
        manager = CollaboratorManager(session)
        collab = await manager.get_collaborator(collaborator_id)
        if collab:
            return collab.name, collab.role, collab.system_prompt
        return "Unknown Collaborator", "", "You are a helpful AI collaborator."

    # Fallback if session iterator is empty
    return "Unknown Collaborator", "", "You are a helpful AI collaborator."


async def update_soul(collaborator_id: str, new_content: str) -> bool:
    """Update the SOUL.md file with new content.

    Creates a backup of the existing file before overwriting.
    Clears the ``load_soul`` cache so subsequent reads pick up
    the new content.

    Returns:
        True on success, False on failure.
    """
    soul_path = f"mirai/collaborator/{collaborator_id}_SOUL.md"
    try:
        if os.path.exists(soul_path):
            shutil.copy2(soul_path, f"{soul_path}.bak")

        with open(soul_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        # Clear cache so next load gets new content
        load_soul.cache_clear()

        log.info("soul_updated_successfully", collaborator=collaborator_id)
        return True
    except Exception as e:
        log.error("soul_update_failed", error=str(e))
        return False
