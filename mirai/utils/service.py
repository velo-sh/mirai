"""Base class for background services in Mirai."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from mirai.logging import get_logger

log = get_logger("mirai.service")


class BaseBackgroundService(ABC):
    """Abstract base class for all background services.

    Handles standard task lifecycle: start, stop, and periodic execution.
    """

    def __init__(self, interval_seconds: float):
        self.interval = interval_seconds
        self.is_running = False
        self._task: asyncio.Task[Any] | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the background service task."""
        if self.is_running:
            return
        self.is_running = True

        loop = loop or asyncio.get_running_loop()
        self._task = loop.create_task(self._main_loop())
        log.info(f"{self.__class__.__name__}_started", interval=self.interval)

    async def stop(self) -> None:
        """Gracefully stop the background service."""
        if not self.is_running:
            return
        self.is_running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info(f"{self.__class__.__name__}_stopped")

    async def _main_loop(self) -> None:
        """Internal main loop with error handling and sleep."""
        while self.is_running:
            try:
                await self.tick()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"{self.__class__.__name__}_error", error=str(e), exc_info=True)
                # Brief backoff on error
                await asyncio.sleep(min(self.interval, 60))

    @abstractmethod
    async def tick(self) -> None:
        """Perform a single unit of work. Overridden by subclasses."""
        pass
