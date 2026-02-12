import io
from unittest.mock import AsyncMock, MagicMock

import pytest

from mirai.agent.im.feishu_receiver import FeishuEventReceiver


@pytest.mark.asyncio
async def test_feishu_receiver_download_image_success_via_file():
    """Test successful image download using the response.file (BytesIO) path."""

    receiver = FeishuEventReceiver(app_id="test_id", app_secret="test_secret", message_handler=AsyncMock())

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.success.return_value = True

    # Simulate binary data in a BytesIO (realistic for lark-oapi)
    image_data = b"fake_image_binary_data"
    mock_response.file = io.BytesIO(image_data)

    mock_aget = AsyncMock(return_value=mock_response)
    mock_client.im.v1.message_resource.aget = mock_aget
    receiver._reply_client = mock_client

    result = await receiver._download_image("msg_123", "img_456")

    assert result == image_data
    mock_aget.assert_called_once()


@pytest.mark.asyncio
async def test_feishu_receiver_download_image_success_via_raw_content():
    """Test successful image download using the response.raw.content fallback."""

    receiver = FeishuEventReceiver(app_id="test_id", app_secret="test_secret", message_handler=AsyncMock())

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.success.return_value = True
    mock_response.file = None  # No file provided

    # Simulate content in raw response
    image_data = b"fallback_image_data"
    mock_raw = MagicMock()
    mock_raw.content = image_data
    mock_response.raw = mock_raw

    mock_aget = AsyncMock(return_value=mock_response)
    mock_client.im.v1.message_resource.aget = mock_aget
    receiver._reply_client = mock_client

    result = await receiver._download_image("msg_123", "img_456")

    assert result == image_data


@pytest.mark.asyncio
async def test_feishu_receiver_download_image_failure_api_error():
    """Test image download failure handling when API returns non-success."""

    receiver = FeishuEventReceiver(app_id="test_id", app_secret="test_secret", message_handler=AsyncMock())

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.success.return_value = False
    mock_response.code = 400
    mock_response.msg = "Permission denied"

    mock_aget = AsyncMock(return_value=mock_response)
    mock_client.im.v1.message_resource.aget = mock_aget
    receiver._reply_client = mock_client

    result = await receiver._download_image("msg_123", "img_456")

    assert result is None
