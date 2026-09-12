"""Regression: partial Telegram delimiter fanout preserves delivery receipts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _message(message_id: int | str) -> SimpleNamespace:
    return SimpleNamespace(message_id=message_id)


@pytest.fixture
def telegram_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._split_replies_enabled = True
    adapter._split_replies_max_parts = 8
    adapter._split_replies_delay_seconds = 0.0
    return adapter


@pytest.mark.asyncio
async def test_delimiter_fanout_failure_on_last_part_preserves_prefix_receipts(telegram_adapter):
    """When part N fails after parts 1..N-1 succeed, adapter returns partial_overflow with receipts."""
    content = "First bubble\n\n---\n\nSecond bubble\n\n---\n\nThird bubble (fails)"

    # Mock: parts 1 and 2 succeed, part 3 fails.
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[
            _message(101),
            _message(102),
            RuntimeError("telegram send failed"),
        ]
    )

    result = await telegram_adapter.send("12345", content, reply_to=None, metadata={})

    assert result.success is False
    assert result.message_id == "102"  # last successful
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["message_ids"] == ["101", "102"]
    assert result.raw_response["delivered_parts"] == 2
    assert result.raw_response["total_parts"] == 3
    # delivered_prefix is everything up to where the failed part begins.
    delivered_prefix = result.raw_response["delivered_prefix"]
    assert "First bubble" in delivered_prefix
    assert "Second bubble" in delivered_prefix
    assert "Third bubble" not in delivered_prefix


@pytest.mark.asyncio
async def test_delimiter_fanout_failure_on_first_part_returns_plain_failure(telegram_adapter):
    """When the first part fails, there's no partial delivery to preserve."""
    content = "First bubble (fails)\n\n---\n\nSecond bubble"

    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=RuntimeError("telegram send failed")
    )

    result = await telegram_adapter.send("12345", content, reply_to=None, metadata={})

    assert result.success is False
    # No partial_overflow when nothing was delivered.
    assert result.raw_response is None or not result.raw_response.get("partial_overflow")


@pytest.mark.asyncio
async def test_consumer_enters_fallback_on_partial_fanout_first_send(telegram_adapter):
    """Consumer's _first_send detects partial_overflow and enters fallback mode."""
    consumer = GatewayStreamConsumer(
        adapter=telegram_adapter,
        chat_id="12345",
        config=StreamConsumerConfig(),
    )

    content = "Part A\n\n---\n\nPart B\n\n---\n\nPart C (fails)"

    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[
            _message(201),
            _message(202),
            RuntimeError("telegram send failed"),
        ]
    )

    # Simulate the finalize send path through _first_send.
    success = await consumer._first_send(content, finalize=True)

    # Partial delivery: _first_send returns False, but fallback mode is armed.
    assert success is False
    assert consumer._fallback_final_send is True
    assert consumer._message_id == "202"
    assert "Part A" in consumer._last_sent_text
    assert "Part B" in consumer._last_sent_text
    assert "Part C" not in consumer._last_sent_text


@pytest.mark.asyncio
async def test_consumer_fallback_recovers_unsent_tail_after_partial_fanout(telegram_adapter):
    """After partial fanout, _send_fallback_final sends only the missing tail."""
    consumer = GatewayStreamConsumer(
        adapter=telegram_adapter,
        chat_id="12345",
        config=StreamConsumerConfig(),
    )

    content = "Delivered A\n\n---\n\nDelivered B\n\n---\n\nMissing C"

    # First attempt: parts 1-2 succeed, part 3 fails → partial_overflow.
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[
            _message(301),
            _message(302),
            RuntimeError("telegram send failed"),
        ]
    )

    success = await consumer._first_send(content, finalize=True)
    assert success is False
    assert consumer._fallback_final_send is True

    # Now simulate got_done calling _send_fallback_final to recover the tail.
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[_message(303)]
    )

    await consumer._send_fallback_final(content)

    # Fallback should have sent only the continuation (the missing tail).
    sent_calls = telegram_adapter._bot.send_message.call_args_list
    assert len(sent_calls) == 1
    sent_text = sent_calls[0].kwargs.get("text", "")
    # The continuation should NOT include the delivered prefix.
    assert "Delivered A" not in sent_text
    assert "Delivered B" not in sent_text
    assert "Missing C" in sent_text
    # After successfully sending the tail, the consumer marks the turn delivered.
    assert consumer._final_response_sent is True
    assert consumer._already_sent is True


@pytest.mark.asyncio
async def test_split_with_offsets_returns_accurate_part_boundaries():
    """split_reply_delimited(with_offsets=True) returns correct start offsets."""
    from plugins.platforms.telegram.telegram_split_replies import split_reply_delimited

    text = "First\n\n---\n\nSecond\n\n-----\n\nThird"
    parts = split_reply_delimited(text, with_offsets=True)

    assert len(parts) == 3
    assert parts[0] == ("First", 0, 0)
    assert parts[1] == ("Second", 3, text.find("Second"))
    assert parts[2] == ("Third", 5, text.find("Third"))

    # Verify offsets produce valid prefixes.
    for i in range(1, len(parts)):
        _, _, offset = parts[i]
        prefix = text[:offset]
        suffix = text[offset:]
        # The suffix should start with the current part's content.
        assert suffix.startswith(parts[i][0])
