"""Heartbeat Manager — proactive self-reflection pulse."""

import asyncio

from mirai.agent.im.base import BaseIMProvider
from mirai.agent.loop import AgentLoop
from mirai.logging import get_logger

log = get_logger("mirai.heartbeat")


class HeartbeatManager:
    def __init__(
        self,
        agent: AgentLoop,
        interval_seconds: float = 3600,
        im_provider: BaseIMProvider | None = None,
        chat_id: str | None = None,
    ):
        self.agent = agent
        self.interval = interval_seconds
        self.im_provider = im_provider
        self.chat_id = chat_id
        self.is_running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._loop())
        log.info("heartbeat_started", collaborator=self.agent.name, interval=self.interval)

    async def _loop(self):
        while self.is_running:
            try:
                # Wait for interval
                await asyncio.sleep(self.interval)

                log.info("heartbeat_pulse", collaborator=self.agent.name)

                # The Heartbeat Pulse
                pulse_message = (
                    "SYSTEM_HEARTBEAT: Please perform a self-reflection. "
                    "Scan the project workspace and recent L3 traces. "
                    "Do you have any architectural insights or proactive suggestions?"
                )

                # Execute reasoning loop
                response = await self.agent.run(pulse_message)
                log.info("heartbeat_insight", collaborator=self.agent.name, insight=response[:100])

                # Proactive Push to IM
                if self.im_provider:
                    await self.im_provider.send_markdown(
                        content=f"**🤖 Mirai 自省洞察** ({self.agent.name})\n\n{response}",
                        title="Mirai Heartbeat",
                        chat_id=self.chat_id,
                    )

            except Exception as e:
                log.error("heartbeat_error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry

    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("heartbeat_stopped", collaborator=self.agent.name)
