import asyncio

from mirai.agent.im.base import BaseIMProvider
from mirai.agent.loop import AgentLoop


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
        print(f"[heartbeat] Started for {self.agent.name} (Interval: {self.interval}s)")

    async def _loop(self):
        while self.is_running:
            try:
                # Wait for interval
                await asyncio.sleep(self.interval)

                print(f"[heartbeat] Triggering pulse for {self.agent.name}...")

                # The Heartbeat Pulse
                pulse_message = (
                    "SYSTEM_HEARTBEAT: Please perform a self-reflection. "
                    "Scan the project workspace and recent L3 traces. "
                    "Do you have any architectural insights or proactive suggestions?"
                )

                # Execute reasoning loop
                response = await self.agent.run(pulse_message)
                print(f"[heartbeat] Insight generated: {response[:100]}...")

                # Proactive Push to IM
                if self.im_provider:
                    await self.im_provider.send_message(
                        content=f"🤖 **Mirai 自省洞察** ({self.agent.name}):\n\n{response}", chat_id=self.chat_id
                    )

            except Exception as e:
                print(f"[heartbeat] Error during pulse: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print(f"[heartbeat] Stopped for {self.agent.name}")
