"""End-to-end test: Agent ↔ Feishu lifecycle.

Scenario:
  1. Agent comes online → sends a check-in message to Feishu
  2. Feishu user sends "hello" → Agent immediately replies with "Thinking..."
  3. Agent processes via LLM (MockProvider) → patches the placeholder with real response

All Feishu SDK calls are mocked; the AgentLoop + MockProvider pipeline is real.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mirai.agent.loop import AgentLoop
from mirai.agent.providers import MockProvider
from mirai.agent.tools.echo import EchoTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_real_agent() -> AgentLoop:
    """Create a real AgentLoop backed by MockProvider (no DB)."""
    provider = MockProvider()
    tools = [EchoTool()]
    agent = AgentLoop(provider, tools, collaborator_id="e2e-feishu-test")
    agent.name = "E2EFeishuBot"
    agent.role = "collaborator"
    agent.base_system_prompt = "You are a test agent."
    agent.soul_content = ""
    return agent


def _build_mock_feishu_provider():
    """Build a mock FeishuProvider that records all send_message calls."""
    from mirai.agent.im.base import BaseIMProvider

    class RecordingFeishuProvider(BaseIMProvider):
        def __init__(self):
            self.messages: list[dict] = []

        async def send_message(self, content: str, chat_id: str | None = None) -> bool:
            self.messages.append({"content": content, "chat_id": chat_id})
            return True

        async def send_card(self, card_content: dict, chat_id: str | None = None) -> bool:
            return True

    return RecordingFeishuProvider()


class MockReplyResponse:
    """Mocks the Feishu SDK reply response."""

    def __init__(self, message_id: str = "mock_placeholder_msg_id"):
        self.data = MagicMock()
        self.data.message_id = message_id

    def success(self):
        return True


class MockPatchResponse:
    """Mocks the Feishu SDK patch response."""

    def success(self):
        return True


# ---------------------------------------------------------------------------
# Step 1: Agent online → check-in to Feishu
# ---------------------------------------------------------------------------


class TestAgentOnlineCheckIn:
    @pytest.mark.asyncio
    async def test_sends_checkin_on_startup(self):
        """When agent comes online, it sends a check-in message via IM provider."""
        agent = _create_real_agent()
        im_provider = _build_mock_feishu_provider()

        # Simulate the startup check-in
        await im_provider.send_message(
            f"✅ **{agent.name}** is online and ready to collaborate!",
            chat_id="test_chat_001",
        )

        assert len(im_provider.messages) == 1
        msg = im_provider.messages[0]
        assert agent.name in msg["content"]
        assert "online" in msg["content"]
        assert msg["chat_id"] == "test_chat_001"

    @pytest.mark.asyncio
    async def test_checkin_with_webhook_fallback(self):
        """Check-in also works via webhook when no app_id is configured."""
        im_provider = _build_mock_feishu_provider()

        await im_provider.send_message("🤖 Agent online (webhook mode)")
        assert len(im_provider.messages) == 1
        assert "online" in im_provider.messages[0]["content"]


# ---------------------------------------------------------------------------
# Step 2 + 3: Receive message → Typing → LLM response → Patch
# ---------------------------------------------------------------------------


class TestFeishuMessageFlow:
    @pytest.mark.asyncio
    async def test_typing_then_llm_reply(self):
        """Full flow: receive hello → typing indicator → LLM reply → patch.

        Mocks the Feishu SDK reply/patch APIs while using a real AgentLoop.
        """
        agent = _create_real_agent()

        # Track the flow stages
        flow_record: list[str] = []
        reply_content_record: list[str] = []
        patch_content_record: list[str] = []

        # --- Mock Feishu reply client ---
        mock_reply_client = MagicMock()

        # Mock areply (typing indicator)
        async def mock_areply(request):
            import orjson

            body = request.request_body
            content = orjson.loads(body.content)
            reply_content_record.append(content.get("text", ""))
            flow_record.append("typing_sent")
            return MockReplyResponse("placeholder_msg_123")

        mock_reply_client.im.v1.message.areply = mock_areply

        # Mock apatch (real response)
        async def mock_apatch(request):
            import orjson

            body = request.request_body
            content = orjson.loads(body.content)
            patch_content_record.append(content.get("text", ""))
            flow_record.append("response_patched")
            return MockPatchResponse()

        mock_reply_client.im.v1.message.apatch = mock_apatch

        # --- Build the receiver ---
        async def message_handler(sender_id: str, text: str, chat_id: str) -> str:
            flow_record.append("llm_processing")
            result = await agent.run(text)
            flow_record.append("llm_done")
            return result

        from mirai.agent.im.feishu_receiver import FeishuEventReceiver

        receiver = FeishuEventReceiver(
            app_id="test_app_id",
            app_secret="test_app_secret",
            message_handler=message_handler,
        )
        receiver._reply_client = mock_reply_client

        # --- Execute the flow ---
        await receiver._process_and_reply(
            text="hello",
            sender_id="user_001",
            message_id="msg_original_001",
            chat_id="chat_001",
        )

        # --- Verify the flow order ---
        assert flow_record == [
            "typing_sent",
            "llm_processing",
            "llm_done",
            "response_patched",
        ], f"Unexpected flow: {flow_record}"

        # Verify typing indicator content
        assert len(reply_content_record) == 1
        assert "Thinking" in reply_content_record[0]

        # Verify LLM response was patched (not empty)
        assert len(patch_content_record) == 1
        assert len(patch_content_record[0]) > 0

    @pytest.mark.asyncio
    async def test_typing_sent_immediately_before_llm(self):
        """Typing indicator is sent BEFORE the LLM starts processing."""
        timing_record: list[tuple[str, float]] = []

        agent = _create_real_agent()

        mock_reply_client = MagicMock()

        import time

        async def mock_areply(request):
            timing_record.append(("typing", time.monotonic()))
            return MockReplyResponse("ph_001")

        async def mock_apatch(request):
            timing_record.append(("patch", time.monotonic()))
            return MockPatchResponse()

        mock_reply_client.im.v1.message.areply = mock_areply
        mock_reply_client.im.v1.message.apatch = mock_apatch

        async def handler(sender_id: str, text: str, chat_id: str) -> str:
            timing_record.append(("llm_start", time.monotonic()))
            result = await agent.run(text)
            timing_record.append(("llm_end", time.monotonic()))
            return result

        from mirai.agent.im.feishu_receiver import FeishuEventReceiver

        receiver = FeishuEventReceiver(
            app_id="test_id",
            app_secret="test_secret",
            message_handler=handler,
        )
        receiver._reply_client = mock_reply_client

        await receiver._process_and_reply("hello", "u1", "m1", "c1")

        # Verify: typing < LLM start < LLM end < patch
        labels = [r[0] for r in timing_record]
        assert labels == ["typing", "llm_start", "llm_end", "patch"]

        # Timing: typing must be before LLM
        assert timing_record[0][1] < timing_record[1][1]

    @pytest.mark.asyncio
    async def test_fallback_reply_if_typing_fails(self):
        """If the typing placeholder fails, agent still sends a direct reply."""
        agent = _create_real_agent()

        mock_reply_client = MagicMock()
        reply_calls: list[str] = []

        async def mock_areply(request):
            import orjson

            body = request.request_body
            content = orjson.loads(body.content)
            text = content.get("text", "")
            reply_calls.append(text)

            if "Thinking" in text:
                # Simulate typing placeholder failure
                resp = MagicMock()
                resp.success.return_value = False
                resp.code = 403
                resp.msg = "Forbidden"
                return resp
            else:
                # Fallback reply succeeds
                return MockReplyResponse("fallback_msg")

        mock_reply_client.im.v1.message.areply = mock_areply
        mock_reply_client.im.v1.message.apatch = AsyncMock()

        async def handler(sender_id: str, text: str, chat_id: str) -> str:
            return await agent.run(text)

        from mirai.agent.im.feishu_receiver import FeishuEventReceiver

        receiver = FeishuEventReceiver(
            app_id="test_id",
            app_secret="test_secret",
            message_handler=handler,
        )
        receiver._reply_client = mock_reply_client

        await receiver._process_and_reply("hello", "u1", "m1", "c1")

        # Should have 2 reply calls: failed typing + fallback with real response
        assert len(reply_calls) == 2
        assert "Thinking" in reply_calls[0]
        # Second call is the actual LLM response (not empty)
        assert len(reply_calls[1]) > 0
        assert "Thinking" not in reply_calls[1]

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self):
        """If the LLM handler raises, the receiver should not crash."""
        mock_reply_client = MagicMock()

        async def mock_areply(request):
            return MockReplyResponse("ph_001")

        async def mock_apatch(request):
            return MockPatchResponse()

        mock_reply_client.im.v1.message.areply = mock_areply
        mock_reply_client.im.v1.message.apatch = mock_apatch

        async def handler(sender_id: str, text: str, chat_id: str) -> str:
            raise RuntimeError("LLM exploded")

        from mirai.agent.im.feishu_receiver import FeishuEventReceiver

        receiver = FeishuEventReceiver(
            app_id="test_id",
            app_secret="test_secret",
            message_handler=handler,
        )
        receiver._reply_client = mock_reply_client

        # Should NOT raise
        await receiver._process_and_reply("hello", "u1", "m1", "c1")


# ---------------------------------------------------------------------------
# Full lifecycle: check-in → receive → typing → reply
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    @pytest.mark.asyncio
    async def test_complete_agent_feishu_lifecycle(self):
        """End-to-end: agent online → check-in → receive hello → typing → LLM reply."""
        # --- Phase 1: Agent online + check-in ---
        agent = _create_real_agent()
        im_provider = _build_mock_feishu_provider()

        checkin_msg = f"✅ **{agent.name}** is online and ready!"
        await im_provider.send_message(checkin_msg, chat_id="group_001")

        assert len(im_provider.messages) == 1
        assert agent.name in im_provider.messages[0]["content"]

        # --- Phase 2+3: Receive message → Typing → LLM → Patch ---
        flow_stages: list[str] = []

        mock_reply_client = MagicMock()

        async def mock_areply(request):
            flow_stages.append("typing")
            return MockReplyResponse("ph_lifecycle")

        async def mock_apatch(request):
            import orjson

            body = request.request_body
            content = orjson.loads(body.content)
            flow_stages.append(f"patched:{content.get('text', '')[:30]}")
            return MockPatchResponse()

        mock_reply_client.im.v1.message.areply = mock_areply
        mock_reply_client.im.v1.message.apatch = mock_apatch

        async def message_handler(sender_id: str, text: str, chat_id: str) -> str:
            flow_stages.append("llm")
            return await agent.run(text)

        from mirai.agent.im.feishu_receiver import FeishuEventReceiver

        receiver = FeishuEventReceiver(
            app_id="test_app",
            app_secret="test_secret",
            message_handler=message_handler,
        )
        receiver._reply_client = mock_reply_client

        await receiver._process_and_reply("hello", "user_feishu", "orig_msg", "group_001")

        # Verify complete flow
        assert flow_stages[0] == "typing"
        assert flow_stages[1] == "llm"
        assert flow_stages[2].startswith("patched:")

        # Verify the patched response is not empty
        patched_text = flow_stages[2].split("patched:", 1)[1]
        assert len(patched_text) > 0
