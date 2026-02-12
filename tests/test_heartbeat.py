"""Tests for HeartbeatManager (mirai/agent/heartbeat.py).

Covers: start/stop lifecycle, pulse message format, IM integration.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestHeartbeatLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        from mirai.agent.heartbeat import HeartbeatManager

        agent = MagicMock()
        agent.name = "TestAgent"
        agent.run = AsyncMock(return_value="insight")

        hb = HeartbeatManager(agent, interval_seconds=0.1)

        assert hb.is_running is False
        await hb.start()
        assert hb.is_running is True
        assert hb._task is not None

        await hb.stop()
        assert hb.is_running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        from mirai.agent.heartbeat import HeartbeatManager

        agent = MagicMock()
        agent.name = "TestAgent"
        agent.run = AsyncMock(return_value="insight")

        hb = HeartbeatManager(agent, interval_seconds=3600)
        await hb.start()

        task = hb._task
        assert task is not None
        assert not task.done()

        await hb.stop()
        assert task.done()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        from mirai.agent.heartbeat import HeartbeatManager

        agent = MagicMock()
        agent.name = "TestAgent"
        agent.run = AsyncMock(return_value="insight")

        hb = HeartbeatManager(agent, interval_seconds=3600)
        await hb.start()
        first_task = hb._task

        await hb.start()  # Second call
        assert hb._task is first_task  # Should not create a new task

        await hb.stop()


class TestHeartbeatPulse:
    @pytest.mark.asyncio
    async def test_pulse_calls_agent_run(self):
        from mirai.agent.heartbeat import HeartbeatManager

        agent = MagicMock()
        agent.name = "TestAgent"
        agent.run = AsyncMock(return_value="proactive insight")

        hb = HeartbeatManager(agent, interval_seconds=0.05)
        await hb.start()

        # Wait enough for at least one pulse
        await asyncio.sleep(0.15)
        await hb.stop()

        # agent.run should have been called with the heartbeat pulse message
        assert agent.run.call_count >= 1
        call_args = agent.run.call_args_list[0]
        assert "SYSTEM_HEARTBEAT" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_pulse_with_im_provider(self):
        from mirai.agent.heartbeat import HeartbeatManager

        agent = MagicMock()
        agent.name = "TestAgent"
        agent.run = AsyncMock(return_value="insight for IM")

        im_provider = MagicMock()
        im_provider.send_message = AsyncMock()

        hb = HeartbeatManager(agent, interval_seconds=0.05, im_provider=im_provider)
        await hb.start()

        await asyncio.sleep(0.15)
        await hb.stop()

        # IM provider should have been called with the insight
        assert im_provider.send_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_pulse_error_does_not_crash(self):
        from mirai.agent.heartbeat import HeartbeatManager

        agent = MagicMock()
        agent.name = "TestAgent"
        agent.run = AsyncMock(side_effect=RuntimeError("API down"))

        hb = HeartbeatManager(agent, interval_seconds=0.05)
        await hb.start()

        # Should not crash despite agent error
        await asyncio.sleep(0.15)
        assert hb.is_running is True

        await hb.stop()
