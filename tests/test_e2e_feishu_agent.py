"""End-to-end test: Agent ↔ Feishu full lifecycle.

Reproduces the exact flow verified in live testing (2026-02-12):

  Step 1 — Agent online check-in:
      Agent starts up → FeishuProvider.send_message("Mira is online …")
      → Feishu group receives the check-in message.

  Step 2 — User sends "hello", agent replies "Thinking…":
      FeishuEventReceiver._process_and_reply is called →
      immediately sends a reply with "🤔 Thinking..." (typing indicator).

  Step 3 — LLM processes and sends real reply:
      AgentLoop.run() executes Think → Act → Critique pipeline →
      a NEW reply message is sent with the real LLM response.

All Feishu SDK HTTP calls are mocked; the AgentLoop pipeline is real
(backed by MockProvider).
"""

from unittest.mock import MagicMock

import orjson
import pytest

from mirai.agent.agent_loop import AgentLoop
from mirai.agent.im.base import BaseIMProvider
from mirai.agent.im.feishu_receiver import FeishuEventReceiver
from mirai.agent.providers import MockProvider
from mirai.agent.tools.echo import EchoTool
from mirai.db.duck import DuckDBStorage

# ── Helpers ──────────────────────────────────────────────────────────


def _create_agent() -> AgentLoop:
    """Real AgentLoop with MockProvider (no network, in-memory DB)."""
    storage = DuckDBStorage(db_path=":memory:")
    agent = AgentLoop(
        provider=MockProvider(),
        tools=[EchoTool()],
        collaborator_id="e2e-feishu-test",
        l3_storage=storage,
    )
    agent.name = "Mira"
    agent.role = "collaborator"
    agent.base_system_prompt = "You are Mira, a helpful collaborator."
    agent.soul_content = ""
    return agent


class FakeFeishuReplyResponse:
    """Simulates a successful Feishu reply API response."""

    def __init__(self, message_id: str = "om_placeholder_123"):
        self.data = MagicMock()
        self.data.message_id = message_id

    def success(self):
        return True


# ── The E2E Test ─────────────────────────────────────────────────────


class TestFeishuAgentE2E:
    """Full Agent ↔ Feishu lifecycle, tested as one sequential scenario."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """
        Step 1: Agent comes online → sends check-in to Feishu group
        Step 2: User sends "hello" → agent immediately replies "Thinking…"
        Step 3: LLM processes → agent sends real reply
        """
        agent = _create_agent()

        # ── Step 1: Agent online → check-in message ──────────────────

        class RecordingIMProvider(BaseIMProvider):
            """Records all messages sent through the IM provider."""

            def __init__(self):
                self.sent: list[str] = []

            async def send_message(self, content: str, chat_id=None) -> bool:
                self.sent.append(content)
                return True

            async def send_card(self, card_content: dict, chat_id=None) -> bool:
                return True

        im_provider = RecordingIMProvider()

        # This is what main.py lifespan does after agent init:
        checkin_ok = await im_provider.send_message(f"✅ **{agent.name}** is online and ready to collaborate!")

        assert checkin_ok is True
        assert len(im_provider.sent) == 1
        assert "Mira" in im_provider.sent[0]
        assert "online" in im_provider.sent[0]

        # ── Step 2 + 3: Receive "hello" → Typing → LLM → Reply ──────

        # We'll record every Feishu API call in order
        feishu_calls: list[dict] = []

        mock_reply_client = MagicMock()

        async def mock_areply(request):
            """Captures reply API calls (both typing and real response)."""
            body = request.request_body
            content = orjson.loads(body.content)
            # Handle both plain text and interactive card formats
            if body.msg_type == "interactive":
                # Extract text from card elements (lark_md div)
                elements = content.get("elements", [])
                text = " ".join(
                    el.get("text", {}).get("content", "") for el in elements if el.get("tag") == "div"
                ) or str(content)
            else:
                text = content.get("text", "")
            feishu_calls.append(
                {
                    "action": "reply",
                    "message_id": request.message_id,
                    "text": text,
                }
            )
            return FakeFeishuReplyResponse(f"om_reply_{len(feishu_calls)}")

        async def mock_acreate(request):
            """Captures reaction calls."""
            feishu_calls.append(
                {
                    "action": "reaction",
                    "message_id": request.message_id,
                    "text": "Thinking",  # Proxy for emoji reaction
                }
            )
            return FakeFeishuReplyResponse()

        mock_reply_client.im.v1.message.areply = mock_areply
        mock_reply_client.im.v1.message_reaction.acreate = mock_acreate

        # Build receiver with real AgentLoop as handler
        async def handle_message(sender_id: str, text: str, chat_id: str, history: list) -> str:
            return await agent.run(text, history=history)

        receiver = FeishuEventReceiver(
            app_id="test_app_id",
            app_secret="test_app_secret",
            message_handler=handle_message,
            storage=agent.l3_storage,
        )
        receiver._reply_client = mock_reply_client

        # Simulate: user sends "hello" in Feishu group
        await receiver._process_and_reply(
            message_content="hello",
            sender_id="ou_user_12345",
            message_id="om_original_msg_001",
            chat_id="oc_group_001",
        )

        # ── Verify the complete flow ─────────────────────────────────

        # Exactly 2 Feishu API calls should have been made:
        #   Call 1: typing indicator ("🤔 Thinking...")
        #   Call 2: real LLM response
        assert len(feishu_calls) == 2, (
            f"Expected 2 Feishu API calls (typing + reply), got {len(feishu_calls)}: {feishu_calls}"
        )

        # Call 1: typing indicator (now a reaction)
        typing_call = feishu_calls[0]
        assert typing_call["action"] == "reaction", f"Expected reaction, got {typing_call['action']}"
        assert typing_call["message_id"] == "om_original_msg_001"
        assert "Thinking" in typing_call["text"]

        # Call 2: real LLM response (not empty, not "Thinking...")
        reply_call = feishu_calls[1]
        assert reply_call["action"] == "reply"
        assert reply_call["message_id"] == "om_original_msg_001"
        assert len(reply_call["text"]) > 0, "LLM response should not be empty"
        assert "Thinking" not in reply_call["text"], (
            f"Second reply should be real LLM response, not typing: {reply_call['text']}"
        )

    @pytest.mark.asyncio
    async def test_typing_arrives_before_llm_starts(self):
        """The typing indicator must be sent BEFORE the LLM begins processing."""
        import time

        agent = _create_agent()
        timestamps: list[tuple[str, float]] = []

        async def mock_areply(request):
            body = request.request_body
            content = orjson.loads(body.content)
            text = content.get("text", "")
            label = "typing" if "Thinking" in text else "llm_reply"
            timestamps.append((label, time.monotonic()))
            return FakeFeishuReplyResponse()

        async def mock_acreate(request):
            timestamps.append(("typing", time.monotonic()))
            return FakeFeishuReplyResponse()

        mock_client = MagicMock()
        mock_im = MagicMock()
        mock_v1 = MagicMock()
        mock_msg = MagicMock()
        mock_reaction = MagicMock()

        mock_msg.areply = mock_areply
        mock_reaction.acreate = mock_acreate

        mock_v1.message = mock_msg
        mock_v1.message_reaction = mock_reaction
        mock_im.v1 = mock_v1
        mock_client.im = mock_im

        async def handler(sender_id, text, chat_id, history):
            timestamps.append(("llm_start", time.monotonic()))
            result = await agent.run(text, history=history)
            timestamps.append(("llm_end", time.monotonic()))
            return result

        receiver = FeishuEventReceiver(
            app_id="id",
            app_secret="secret",
            message_handler=handler,
            storage=agent.l3_storage,
        )
        receiver._reply_client = mock_client

        await receiver._process_and_reply("hello", "u1", "m1", "c1")

        labels = [t[0] for t in timestamps]
        assert labels == ["typing", "llm_start", "llm_end", "llm_reply"], (
            f"Expected [typing → llm_start → llm_end → llm_reply], got {labels}"
        )
        # typing must happen strictly before LLM starts
        assert timestamps[0][1] < timestamps[1][1]

    @pytest.mark.asyncio
    async def test_llm_error_does_not_crash_receiver(self):
        """If the LLM explodes, the receiver logs the error but does not crash."""
        mock_client = MagicMock()
        replies: list[str] = []

        async def mock_areply(request):
            body = request.request_body
            content = orjson.loads(body.content)
            replies.append(content.get("text", ""))
            return FakeFeishuReplyResponse()

        async def mock_acreate(request):
            replies.append("Thinking")
            return FakeFeishuReplyResponse()

        mock_client = MagicMock()
        mock_im = MagicMock()
        mock_v1 = MagicMock()
        mock_msg = MagicMock()
        mock_reaction = MagicMock()

        mock_msg.areply = mock_areply
        mock_reaction.acreate = mock_acreate

        mock_v1.message = mock_msg
        mock_v1.message_reaction = mock_reaction
        mock_im.v1 = mock_v1
        mock_client.im = mock_im

        async def exploding_handler(sender_id, text, chat_id, history):
            raise RuntimeError("LLM service unavailable")

        receiver = FeishuEventReceiver(
            app_id="id",
            app_secret="secret",
            message_handler=exploding_handler,
            storage=None,
        )
        receiver._reply_client = mock_client

        # Should NOT raise — the receiver catches exceptions internally
        await receiver._process_and_reply("hello", "u1", "m1", "c1")

        # Typing indicator should still have been sent
        assert len(replies) >= 1
        assert "Thinking" in replies[0]
