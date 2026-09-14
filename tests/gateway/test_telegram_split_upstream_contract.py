"""Delimiter splitting composes with upstream preview, cap and partial-send contracts."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def adapter():
    obj = TelegramAdapter(PlatformConfig(enabled=True, token="fake", extra={
        "split_replies": {"enabled": True, "delay_between_ms": 0, "max_parts": 20},
    }))
    obj._bot = MagicMock()
    obj._bot.send_message = AsyncMock(side_effect=lambda **kw: SimpleNamespace(message_id=1))
    obj._chat_outbound_slot_remaining = lambda *_: 0
    obj._hold_chat_outbound_slot = lambda *_: None
    return obj


@pytest.mark.asyncio
async def test_editable_delimiter_preview_is_one_bubble_until_fresh_final(adapter):
    await adapter.send("123", "Alpha\n---\nBeta", metadata={"expect_edits": True})
    assert adapter._bot.send_message.await_count == 1
    assert adapter.prefers_fresh_final_streaming("Alpha\n---\nBeta more", {})


def test_delimiter_plan_is_included_in_whole_reply_cap(adapter):
    content = "\n---\n".join(f"Bubble {i}" for i in range(9))
    assert adapter.reply_message_count(content, {}) == 9


@pytest.mark.asyncio
async def test_uncertain_multipart_send_keeps_upstream_never_attempted_tail(adapter):
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="Timed out", raw_response={
        "partial_overflow": True, "delivery_uncertain": True,
        "delivered_prefix": "confirmed", "unattempted_chunks": ["never attempted"],
    }))
    adapter._resume_partial_send = AsyncMock(return_value=SendResult(success=True, message_id="9"))
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="123", config=StreamConsumerConfig())
    assert await consumer._first_send("confirmed timed-out never attempted", finalize=True)
    adapter._resume_partial_send.assert_awaited_once()
    assert consumer._delivery_uncertain


@pytest.mark.asyncio
async def test_delimiter_timeout_resumes_never_attempted_later_bubbles(adapter):
    from telegram.error import TimedOut
    adapter._bot.send_message = AsyncMock(side_effect=[TimedOut(), SimpleNamespace(message_id=2)])
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="123", config=StreamConsumerConfig())
    assert await consumer._first_send("Maybe delivered\n---\nNever attempted", finalize=True)
    texts = [call.kwargs["text"] for call in adapter._bot.send_message.await_args_list]
    assert len(texts) == 2
    assert "Never attempted" in texts[1]
    assert "Maybe delivered" not in texts[1]
    assert consumer._delivery_uncertain


@pytest.mark.asyncio
async def test_delimiter_resume_keeps_inner_multipart_receipt(adapter):
    adapter._send_single_message = AsyncMock(side_effect=[
        SendResult(success=True, message_id="1"),
        SendResult(success=False, message_id="2", retryable=True, raw_response={
            "partial_overflow": True, "delivered_prefix": "Inner head", "message_ids": ["2"],
            "delivered_message_ids": ["2"], "undelivered_chunks": ["inner tail"],
        }),
        SendResult(success=True, message_id="4"),
    ])
    adapter._send_chunks = AsyncMock(return_value=SendResult(success=True, message_id="2", raw_response={"message_ids": ["2", "3"]}))
    result = await adapter.send("123", "First\n---\nInner head and tail\n---\nLast")
    resumed = await adapter._resume_partial_send("123", result, reply_to=None, metadata={})
    assert resumed is not None and resumed.success
    assert resumed.raw_response["message_ids"] == ["1", "2", "3", "4"]
    assert [call.args[1] for call in adapter._send_single_message.await_args_list] == ["First", "Inner head and tail", "Last"]
    assert adapter._send_chunks.await_args.args[1] == ["inner tail"]


@pytest.mark.asyncio
async def test_consumer_resumes_inner_chunks_not_the_whole_delimiter_part(adapter):
    adapter._send_single_message = AsyncMock(side_effect=[
        SendResult(success=True, message_id="1"),
        SendResult(success=False, message_id="2", retryable=True, raw_response={
            "partial_overflow": True, "message_ids": ["2"],
            "delivered_message_ids": ["2"], "undelivered_chunks": ["inner tail"],
        }),
        SendResult(success=True, message_id="4"),
    ])
    adapter._send_chunks = AsyncMock(return_value=SendResult(success=True, message_id="2", raw_response={"message_ids": ["2", "3"]}))
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="123", config=StreamConsumerConfig())
    content = "First\n---\nInner head and tail\n---\nLast"
    assert not await consumer._first_send(content, finalize=True)
    await consumer._send_fallback_final(content)
    assert adapter._send_chunks.await_count == 1
    assert adapter._send_single_message.await_count == 3
    assert consumer.final_response_sent


@pytest.mark.asyncio
async def test_repeated_partial_failure_never_degrades_to_a_guessed_raw_suffix(adapter):
    failure = SendResult(success=False, message_id="2", retryable=True, raw_response={
        "partial_overflow": True, "message_ids": ["2"],
        "delivered_message_ids": ["2"], "undelivered_chunks": ["inner tail"],
    })
    adapter._send_single_message = AsyncMock(side_effect=[SendResult(success=True, message_id="1"), failure])
    adapter._send_chunks = AsyncMock(return_value=failure)
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="123", config=StreamConsumerConfig())
    content = "First\n---\nInner head and tail\n---\nLast"
    assert not await consumer._first_send(content, finalize=True)
    await consumer._send_fallback_final(content)
    assert adapter._send_single_message.await_count == 2
    assert not consumer.final_response_sent  # incomplete is not a confirmed delivery
    assert consumer.has_pending_final_delivery(content)
    assert consumer.delivery_uncertain  # normal completion must ledger, never blindly resend
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType, SessionSource
    from tests.gateway.test_run_progress_topics import _make_runner
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    response = {"final_response": content}
    ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=source, session_key="agent:main:telegram:dm:123")
    await runner._run_agent_mark_streamed_delivery(response, ctx)
    assert response["already_sent"] and response["delivery_uncertain"]
    adapter.ledger_attempted_final = AsyncMock(return_value=True)
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")
    await runner._hmwa_deliver_turn_response(event, source, SimpleNamespace(session_id="s"), ctx.session_key,
                                            None, response, [], content, None, False)
    adapter.ledger_attempted_final.assert_awaited_once()


@pytest.mark.asyncio
async def test_uncertain_resume_retains_progress_when_its_later_tail_fails(adapter):
    adapter._send_single_message = AsyncMock(side_effect=[
        SendResult(success=True, message_id="1"),
        SendResult(success=False, message_id="2", raw_response={
            "partial_overflow": True, "delivery_uncertain": True, "message_ids": ["2"],
            "unattempted_chunks": ["tail A", "tail B"],
        }),
        SendResult(success=True, message_id="5"),
    ])
    adapter._send_chunks = AsyncMock(side_effect=[
        SendResult(success=False, message_id="3", retryable=True, raw_response={
            "partial_overflow": True, "delivered_message_ids": ["2", "3"], "undelivered_chunks": ["tail B"],
        }),
        SendResult(success=True, message_id="4", raw_response={"message_ids": ["2", "3", "4"]}),
    ])
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="123", config=StreamConsumerConfig())
    content = "First\n---\nInner head and tail\n---\nLast"
    assert not await consumer._first_send(content, finalize=True)
    await consumer._send_fallback_final(content)
    assert [call.args[1] for call in adapter._send_chunks.await_args_list] == [["tail A", "tail B"], ["tail B"]]
    assert consumer.delivery_uncertain
