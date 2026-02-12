import asyncio

from mirai.agent.loop import AgentLoop
from mirai.db.duck import DuckDBStorage
from mirai.logging import get_logger

log = get_logger("mirai.dreamer")


class Dreamer:
    """Background service that periodically reflects on past thinking to evolve identity."""

    def __init__(
        self,
        agent: AgentLoop,
        storage: DuckDBStorage,
        interval_seconds: int = 3600,  # Default: 1 hour
    ):
        self.agent = agent
        self.storage = storage
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the dreamer background task."""
        if self._running:
            return
        self._running = True
        self._task = loop.create_task(self._dream_loop())
        log.info("dreamer_started", interval=self.interval_seconds)

    async def stop(self):
        """Stop the dreamer background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("dreamer_stopped")

    async def _dream_loop(self):
        """Infinite loop for periodic dreaming."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)
                await self.dream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("dreamer_loop_error", error=str(e), exc_info=True)

    async def dream(self):
        """Perform a single reflection and evolution cycle."""
        log.info("dream_cycle_starting", collaborator=self.agent.collaborator_id)

        # 1. Fetch recent thinking traces
        traces = await self.storage.get_recent_traces(self.agent.collaborator_id, limit=20)
        thinking_blocks = [t["content"] for t in traces if t["trace_type"] == "thinking"]

        if not thinking_blocks:
            log.info("dream_skipped_no_thinking_traces")
            return

        # 2. Construct the reflection prompt
        monologue_summary = "\n---\n".join(thinking_blocks)
        current_soul = self.agent.soul_content

        reflection_prompt = (
            f"Review the following record of your internal monologue from recent interactions:\n\n"
            f"{monologue_summary}\n\n"
            f"Your current IDENTITY (SOUL.md) is:\n"
            f"```markdown\n{current_soul}\n```\n\n"
            f"Based on your recent thoughts, how should your identity evolve to better collaborate with GJK? "
            f"Construct a NEW version of your SOUL.md. "
            f"Rules:\n"
            f"- Output ONLY the new markdown content.\n"
            f"- Refine your personality, goals, and behavioral constraints based on your realizations.\n"
            f"- Maintain your name and core essence but evolve the nuance.\n"
        )

        # 3. Ask the agent to "re-imagine" itself
        # We use the agent's run method but with a special message to avoid standard loop logic if needed
        # Or just use the provider directly for a clean "pure reflection"
        try:
            from mirai.agent.models import ProviderResponse

            # Use direct provider call for reflection to bypass memory injection/archive logic
            # This is a meta-operation.
            response: ProviderResponse = await self.agent.provider.generate_response(
                model=getattr(self.agent.provider, "model", "gemini-3-flash"),
                system="You are an AI performing a self-reflection and identity evolution cycle. Output only the new SOUL.md content.",
                messages=[{"role": "user", "content": reflection_prompt}],
                tools=[],
            )

            new_soul = response.text().strip()

            # Clean up potential markdown formatting if the model included backticks
            if new_soul.startswith("```"):
                lines = new_soul.splitlines()
                if lines[0].startswith("```markdown"):
                    new_soul = "\n".join(lines[1:-1])
                elif lines[0].startswith("```"):
                    new_soul = "\n".join(lines[1:-1])

            if len(new_soul) < 100:
                log.warning("dream_produced_too_short_soul", length=len(new_soul))
                return

            # 4. Commit the new identity
            success = await self.agent.update_soul(new_soul)
            if success:
                log.info("dream_cycle_completed_successfully")
            else:
                log.error("dream_cycle_failed_to_update_soul")

        except Exception as e:
            log.error("dream_cycle_error", error=str(e), exc_info=True)
