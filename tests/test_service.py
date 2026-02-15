"""Tests for mirai.utils.service — BaseBackgroundService lifecycle."""

import asyncio

import pytest

from mirai.utils.service import BaseBackgroundService


class _ConcreteService(BaseBackgroundService):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self, interval: float = 1.0):
        super().__init__(interval)
        self.tick_count = 0

    async def tick(self):
        self.tick_count += 1


class _ErrorService(BaseBackgroundService):
    """Service that raises on first tick, then succeeds."""

    def __init__(self):
        super().__init__(0.01)
        self.calls = 0

    async def tick(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first tick fails")


class TestBaseBackgroundService:
    """Tests for BaseBackgroundService lifecycle."""

    def test_init_state(self):
        svc = _ConcreteService(interval=60.0)
        assert svc.interval == 60.0
        assert svc.is_running is False
        assert svc._task is None

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Service starts, runs ticks, and stops cleanly."""
        svc = _ConcreteService(interval=0.01)
        loop = asyncio.get_running_loop()

        svc.start(loop)
        assert svc.is_running is True
        assert svc._task is not None

        # Let it run a few ticks
        await asyncio.sleep(0.05)
        assert svc.tick_count >= 1

        await svc.stop()
        assert svc.is_running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Calling stop on a non-started service is a no-op."""
        svc = _ConcreteService()
        await svc.stop()
        assert svc.is_running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Calling start twice does not create a second task."""
        svc = _ConcreteService(interval=0.01)
        loop = asyncio.get_running_loop()

        svc.start(loop)
        first_task = svc._task
        svc.start(loop)
        assert svc._task is first_task

        await svc.stop()

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Service continues running after a tick error."""
        svc = _ErrorService()
        loop = asyncio.get_running_loop()

        svc.start(loop)
        # Wait for at least 2 ticks (first fails, second succeeds)
        await asyncio.sleep(0.15)

        assert svc.calls >= 2
        await svc.stop()

    @pytest.mark.asyncio
    async def test_start_uses_running_loop_by_default(self):
        """When no loop is passed, start() uses the running loop."""
        svc = _ConcreteService(interval=0.01)
        svc.start()  # No loop argument
        assert svc.is_running is True
        assert svc._task is not None
        await svc.stop()
