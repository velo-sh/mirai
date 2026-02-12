import pytest

from mirai.agent.im.feishu import FeishuProvider


@pytest.mark.asyncio
async def test_feishu_provider_webhook_mock(monkeypatch):
    """
    Test FeishuProvider with a mocked webhook response.
    """

    class MockResponse:
        status_code = 200

    async def mock_post(*args, **kwargs):
        return MockResponse()

    # Mock httpx.AsyncClient.post
    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = FeishuProvider(webhook_url="https://fake-webhook.com/abc")
    success = await provider.send_message("Test message")
    assert success is True


@pytest.mark.asyncio
async def test_feishu_provider_card_mock(monkeypatch):
    """
    Test FeishuProvider card sending with mocked webhook.
    """

    class MockResponse:
        status_code = 200

    async def mock_post(*args, **kwargs):
        # Verify content includes "interactive"
        assert kwargs["json"]["msg_type"] == "interactive"
        return MockResponse()

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = FeishuProvider(webhook_url="https://fake-webhook.com/abc")
    success = await provider.send_card({"header": {"title": {"content": "Test"}}})
    assert success is True
