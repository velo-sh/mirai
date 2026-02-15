"""Tests for mirai.integrations — Feishu, dreamer, and check-in helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# create_im_provider
# ---------------------------------------------------------------------------


class TestCreateImProvider:
    """Tests for IM provider factory."""

    def test_returns_none_when_disabled(self):
        """Returns None when Feishu is disabled."""
        from mirai.integrations import create_im_provider

        config = MagicMock()
        config.feishu.enabled = False
        assert create_im_provider(config) is None

    def test_returns_none_when_no_credentials(self):
        """Returns None when enabled but no credentials are set."""
        from mirai.integrations import create_im_provider

        config = MagicMock()
        config.feishu.enabled = True
        config.feishu.app_id = None
        config.feishu.app_secret = None
        config.feishu.webhook_url = None
        assert create_im_provider(config) is None

    def test_creates_app_api_provider(self):
        """Creates FeishuProvider with app_id + app_secret."""
        from mirai.integrations import create_im_provider

        config = MagicMock()
        config.feishu.enabled = True
        config.feishu.app_id = "test_app_id"
        config.feishu.app_secret = "test_secret"

        with patch("mirai.agent.im.feishu.FeishuProvider") as mock_cls:
            result = create_im_provider(config)
            mock_cls.assert_called_once_with(app_id="test_app_id", app_secret="test_secret")
            assert result is not None

    def test_creates_webhook_provider(self):
        """Creates FeishuProvider with webhook_url when no app credentials."""
        from mirai.integrations import create_im_provider

        config = MagicMock()
        config.feishu.enabled = True
        config.feishu.app_id = None
        config.feishu.app_secret = None
        config.feishu.webhook_url = "https://example.com/webhook"

        with patch("mirai.agent.im.feishu.FeishuProvider") as mock_cls:
            result = create_im_provider(config)
            mock_cls.assert_called_once_with(webhook_url="https://example.com/webhook")
            assert result is not None


# ---------------------------------------------------------------------------
# start_feishu_receiver
# ---------------------------------------------------------------------------


class TestStartFeishuReceiver:
    """Tests for Feishu WebSocket receiver startup."""

    def test_noop_when_disabled(self):
        """Does nothing when Feishu is not enabled."""
        from mirai.integrations import start_feishu_receiver

        config = MagicMock()
        config.feishu.enabled = False
        agent = MagicMock()

        # Should not raise and not instantiate any receiver
        start_feishu_receiver(agent, config)

    def test_noop_when_no_app_credentials(self):
        """Does nothing when enabled but missing app_id/app_secret."""
        from mirai.integrations import start_feishu_receiver

        config = MagicMock()
        config.feishu.enabled = True
        config.feishu.app_id = None
        config.feishu.app_secret = "secret"

        agent = MagicMock()
        start_feishu_receiver(agent, config)

    def test_starts_receiver_when_configured(self):
        """Creates and starts FeishuEventReceiver when properly configured."""
        from mirai.integrations import start_feishu_receiver

        config = MagicMock()
        config.feishu.enabled = True
        config.feishu.app_id = "app_id"
        config.feishu.app_secret = "secret"

        agent = MagicMock()
        agent.l3_storage = MagicMock()

        with patch("mirai.agent.im.feishu_receiver.FeishuEventReceiver") as mock_receiver_cls:
            mock_receiver = MagicMock()
            mock_receiver_cls.return_value = mock_receiver
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value = MagicMock()
                start_feishu_receiver(agent, config)

            mock_receiver_cls.assert_called_once()
            mock_receiver.start.assert_called_once()


# ---------------------------------------------------------------------------
# send_checkin
# ---------------------------------------------------------------------------


class TestSendCheckin:
    """Tests for startup check-in card."""

    @pytest.mark.asyncio
    async def test_noop_when_no_provider(self):
        """Does nothing when im_provider is None."""
        from mirai.integrations import send_checkin

        agent = MagicMock()
        config = MagicMock()
        await send_checkin(agent, None, config)  # Should not raise

    @pytest.mark.asyncio
    async def test_sends_card_on_success(self):
        """Sends interactive card and logs success."""
        from mirai.integrations import send_checkin

        agent = MagicMock()
        agent.name = "Mira"
        config = MagicMock()
        config.llm.default_model = "test-model"
        config.feishu.curator_chat_id = "chat_123"

        im_provider = AsyncMock()
        im_provider.send_card = AsyncMock(return_value=True)

        await send_checkin(agent, im_provider, config)
        im_provider.send_card.assert_awaited_once()
        call_kwargs = im_provider.send_card.call_args
        assert call_kwargs.kwargs["chat_id"] == "chat_123"

    @pytest.mark.asyncio
    async def test_logs_warning_on_failure(self):
        """Doesn't raise when send_card fails."""
        from mirai.integrations import send_checkin

        agent = MagicMock()
        agent.name = "Mira"
        config = MagicMock()
        config.llm.default_model = "test-model"
        config.feishu.curator_chat_id = None

        im_provider = AsyncMock()
        im_provider.send_card = AsyncMock(return_value=False)

        await send_checkin(agent, im_provider, config)  # Should not raise


# ---------------------------------------------------------------------------
# start_dreamer
# ---------------------------------------------------------------------------


class TestStartDreamer:
    """Tests for dreamer service startup."""

    def test_creates_and_starts_dreamer(self):
        """Creates AgentDreamer with correct interval and starts it."""
        from mirai.integrations import start_dreamer

        agent = MagicMock()
        agent.l3_storage = MagicMock()
        config = MagicMock()
        config.dreamer.interval = 7200

        with patch("mirai.agent.agent_dreamer.AgentDreamer") as mock_dreamer_cls:
            mock_dreamer = MagicMock()
            mock_dreamer_cls.return_value = mock_dreamer
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value = MagicMock()
                result = start_dreamer(agent, config)

            mock_dreamer_cls.assert_called_once_with(
                agent,
                agent.l3_storage,
                interval_seconds=7200,
            )
            mock_dreamer.start.assert_called_once()
            assert result is mock_dreamer
