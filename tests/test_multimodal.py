import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from mirai.agent.im.feishu_receiver import FeishuEventReceiver
from mirai.agent.providers import AntigravityProvider


@pytest.mark.asyncio
async def test_image_perception_pipeline():
    """
    Simulates a Feishu image message and verifies it flows through to the provider.
    """
    # 1. Setup Mocks
    mock_handler = AsyncMock(return_value="I see an image!")
    receiver = FeishuEventReceiver(app_id="test_app", app_secret="test_secret", message_handler=mock_handler)
    receiver._loop = asyncio.get_running_loop()

    # Mock Feishu Resource Download
    fake_image_bytes = b"fake-image-content"
    mock_resource_resp = MagicMock()
    mock_resource_resp.success.return_value = True
    
    # Mock response.file as BytesIO
    from io import BytesIO
    mock_resource_resp.file = BytesIO(fake_image_bytes)
    
    receiver._reply_client = MagicMock()
    # Correct path: im.v1.message_resource.aget
    receiver._reply_client.im.v1.message_resource.aget = AsyncMock(return_value=mock_resource_resp)
    receiver._reply_client.im.v1.message.areply = AsyncMock(return_value=MagicMock(success=lambda: True))
    receiver._reply_client.im.v1.message_reaction.acreate = AsyncMock(return_value=MagicMock(success=lambda: True))

    # Mock Storage
    receiver._storage = AsyncMock()
    receiver._storage.get_feishu_history.return_value = []

    # 2. Simulate Image Event
    event_data = MagicMock()
    event_data.event = MagicMock()
    event_data.event.message = MagicMock()
    event_data.event.message.message_id = "om_123"
    event_data.event.message.chat_id = "oc_456"
    event_data.event.message.message_type = "image"
    event_data.event.message.content = orjson.dumps({"image_key": "img_789"}).decode()
    event_data.event.sender = MagicMock()
    event_data.event.sender.sender_id = MagicMock(open_id="user_abc")

    # 3. Trigger Receiver
    receiver._on_message_received(event_data)

    # Wait for the async task to complete
    await asyncio.sleep(0.5)

    # 4. Verify Handler Call
    # The handler should receive a list of blocks
    assert mock_handler.called
    sender_id, content, chat_id, history = mock_handler.call_args[0]

    assert sender_id == "user_abc"
    assert chat_id == "oc_456"
    assert isinstance(content, list)
    assert content[1]["type"] == "image"
    assert content[1]["source"]["data"] == base64.b64encode(fake_image_bytes).decode()
    assert content[1]["source"]["media_type"] == "image/png"

    print("✅ Feishu receiver correctly processed image and sent to handler.")


@pytest.mark.asyncio
async def test_provider_image_conversion():
    """Verifies that AntigravityProvider converts internal image blocks to Google format."""
    provider = AntigravityProvider(credentials={"access": "test-token", "project": "test-proj"})

    fake_b64 = "YmFzZTY0LWRhdGE="
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": fake_b64}},
            ],
        }
    ]

    google_contents = provider._convert_messages(messages)

    assert len(google_contents) == 1
    parts = google_contents[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "What is this?"
    assert "inlineData" in parts[1]
    assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert parts[1]["inlineData"]["data"] == fake_b64

    print("✅ AntigravityProvider correctly converted image blocks.")
