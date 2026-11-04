"""Conversation-scoped plugin deletion uses the delivery bot and never falls back to sends."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.plugin_messages import PostTarget
from gateway.session_context import scoped_session_key
from tests.gateway.test_plugin_post_message import _Adapter, _in_session, _setup


class DeletingAdapter(_Adapter):
    async def delete_message(self, chat_id, message_id):
        self.calls.append(("delete", chat_id, message_id))
        return True


@pytest.mark.asyncio
async def test_deletion_uses_current_conversation_and_delivery_adapter(tmp_path):
    receiving = DeletingAdapter()
    runner, ctx, entry = _setup(tmp_path, receiving)
    # Provenance is resolved by the gateway, not by the plugin's configured default token.
    receiving.delete_message = AsyncMock(return_value=True)
    with patch("gateway.run._gateway_runner_ref", lambda: runner), \
            patch.object(type(runner), "_plugin_messaging_session_owned", return_value=True):
        result = await _in_session(entry.session_key, lambda: ctx.delete_message("321"))
    assert result == {"ok": True, "message_id": "321"}
    receiving.delete_message.assert_awaited_once_with("42", "321")
    assert receiving.calls == []


@pytest.mark.asyncio
async def test_deletion_refuses_foreign_profile_without_calling_transport(tmp_path):
    receiving = DeletingAdapter()
    runner, ctx, entry = _setup(tmp_path, receiving)
    with patch("gateway.run._gateway_runner_ref", lambda: runner), \
            patch.object(type(runner), "_plugin_messaging_session_owned", return_value=False):
        result = await _in_session(entry.session_key, lambda: ctx.delete_message("321"))
    assert not result["ok"] and result["error"] == "no_conversation"
    assert receiving.calls == []


@pytest.mark.asyncio
async def test_deletion_requires_bound_session(tmp_path):
    _runner, ctx, _entry = _setup(tmp_path, DeletingAdapter())
    with scoped_session_key(""):
        result = await asyncio.to_thread(ctx.delete_message, "321")
    assert result["error"] == "no_conversation"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [False, RuntimeError("transport failure")])
async def test_failed_delete_does_not_send_replacement(outcome):
    from gateway.plugin_messages import delete
    adapter = SimpleNamespace(delete_message=AsyncMock(), send=AsyncMock())
    if isinstance(outcome, Exception):
        adapter.delete_message.side_effect = outcome
    else:
        adapter.delete_message.return_value = outcome
    result = await delete(PostTarget(platform="telegram", chat_id="42", adapter=adapter), "321", plugin_id="test")
    assert not result["ok"] and result["error"] == "delete_failed"
    adapter.send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh_final", [False, True])
async def test_recalled_stream_bubble_is_not_republished(fresh_final, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake", extra={"rich_messages": False}))
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=1, text="Already said")),
        edit_message_text=AsyncMock(return_value=True), delete_message=AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_chat_outbound_slot_remaining", lambda *_: 0)
    monkeypatch.setattr(adapter, "_hold_chat_outbound_slot", lambda *_: None)
    monkeypatch.setattr(adapter, "_retrigger_typing", AsyncMock())
    visible = asyncio.Event()
    consumer = GatewayStreamConsumer(adapter, "42", StreamConsumerConfig(buffer_threshold=1, cursor=""),
                                     on_new_message=visible.set)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Already said")
    await asyncio.wait_for(visible.wait(), 5)
    if fresh_final:
        monkeypatch.setattr(adapter, "prefers_fresh_final_streaming", lambda *a, **kw: True)
    assert await adapter.delete_message("42", "1")
    consumer.finish("Already said, completed")
    await asyncio.wait_for(task, 5)
    assert adapter._bot.send_message.await_count == 1
    adapter._bot.edit_message_text.assert_not_called()
    assert consumer.delivered_final_matches("Already said, completed") is True
