"""DM-topic reply opt-out across Telegram's outbound transport paths."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


METADATA = {
    "thread_id": "42",
    "telegram_dm_topic_reply_fallback": True,
    "telegram_reply_to_message_id": "12345",
}


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", reply_to_mode="off"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=701))
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    adapter._bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=700))
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["draft", "final", "split", "reasoning", "suffix", "overflow"])
async def test_text_egress_keeps_topic_without_reply_anchor(path):
    adapter = _adapter()
    if path == "draft":
        result = await adapter.send_draft("12345", 1, "preview", metadata=METADATA)
        calls = adapter._bot.send_message_draft.call_args_list
    elif path == "overflow":
        adapter.truncate_message = lambda *args, **kwargs: ["head", "tail"]
        result = await adapter._edit_overflow_split("12345", "700", "head tail", finalize=True, metadata=METADATA)
        calls = adapter._bot.send_message.call_args_list
    else:
        if path == "split":
            adapter._split_replies_enabled = True
            adapter._split_replies_delay_seconds = 0
            text = "part one\n---\npart two"
        else:
            text = path
        if path == "reasoning":
            result = await adapter.send_reasoning("12345", text, metadata=METADATA)
        else:
            result = await adapter.send("12345", text, reply_to="999", metadata=METADATA)
        calls = adapter._bot.send_message.call_args_list
    assert result.success
    assert calls
    for call in calls:
        assert call.kwargs.get("reply_to_message_id") is None
        assert call.kwargs.get("message_thread_id") == 42


@pytest.mark.asyncio
async def test_rich_final_and_media_routing_respect_off():
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(return_value={"message_id": 701})
    result = await adapter._try_send_rich("12345", "rich final", "999", METADATA)
    assert result.success
    payload = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert "reply_parameters" not in payload
    assert payload["message_thread_id"] == 42
    reply_id, media = adapter._media_send_kwargs("12345", "999", METADATA)
    assert reply_id is None
    assert media["reply_to_message_id"] is None
    assert media["message_thread_id"] == 42
