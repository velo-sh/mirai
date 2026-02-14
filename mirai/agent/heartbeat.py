"""Heartbeat Manager — proactive self-reflection pulse."""

from mirai.agent.agent_loop import AgentLoop
from mirai.agent.im.base import BaseIMProvider
from mirai.logging import get_logger
from mirai.utils.service import BaseBackgroundService

log = get_logger("mirai.heartbeat")


class HeartbeatManager(BaseBackgroundService):
    def __init__(
        self,
        agent: AgentLoop,
        interval_seconds: float = 3600,
        im_provider: BaseIMProvider | None = None,
        chat_id: str | None = None,
    ):
        super().__init__(interval_seconds)
        self.agent = agent
        self.im_provider = im_provider
        self.chat_id = chat_id

    async def tick(self) -> None:
        """The Heartbeat Pulse."""
        log.info("heartbeat_pulse", collaborator=self.agent.name)

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
