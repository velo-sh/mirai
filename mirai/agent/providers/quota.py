"""Quota tracking and cache for Google Antigravity model usage."""

import asyncio
import time
from typing import Any

from mirai.auth.antigravity_auth import fetch_usage
from mirai.logging import get_logger

log = get_logger("mirai.providers.quota")


class QuotaManager:
    """
    Tracks and caches model quota usage for Google Antigravity.
    Ensures we don't hit 429s by proactively failing over to available models.
    """

    CACHE_TTL = 60.0  # seconds

    def __init__(self, credentials: dict[str, Any]):
        self.credentials = credentials
        self._quotas: dict[str, float] = {}  # model_id -> used_pct
        self._last_update = 0.0
        self._lock = asyncio.Lock()

    @classmethod
    def for_testing(
        cls,
        *,
        quotas: dict[str, float] | None = None,
        credentials: dict[str, Any] | None = None,
    ) -> "QuotaManager":
        """Create a QuotaManager for testing without real credentials.

        Pre-populates quotas and sets ``_last_update`` to now so the
        manager never tries to refresh from the network.
        """
        qm = cls.__new__(cls)
        qm.credentials = credentials or {}
        qm._quotas = quotas or {}
        qm._last_update = time.time()
        qm._lock = asyncio.Lock()
        return qm

    async def get_used_pct(self, model_id: str) -> float:
        """Get the current usage percentage for a model ID."""
        await self._maybe_refresh()
        return self._quotas.get(model_id, 0.0)

    async def is_available(self, model_id: str) -> bool:
        """Return True if model is not at 100% usage."""
        return await self.get_used_pct(model_id) < 100.0

    async def _maybe_refresh(self):
        """Refresh quotas if cache expired."""
        if time.time() - self._last_update < self.CACHE_TTL:
            return

        async with self._lock:
            # Double check inside lock
            if time.time() - self._last_update < self.CACHE_TTL:
                return

            try:
                log.debug("refreshing_quotas")
                usage = await fetch_usage(self.credentials["access"], self.credentials.get("project_id", ""))
                new_quotas = {}
                for m in usage.get("models", []):
                    new_quotas[m["id"]] = m["used_pct"]
                self._quotas = new_quotas
                self._last_update = time.time()
                log.info("quotas_refreshed", models_count=len(self._quotas))
            except Exception as e:
                log.error("quota_refresh_failed", error=str(e))
