"""A cancel that lands while a send is in flight must not strand a delivered final.

Telegram delimiter fanout paces each bubble, so one ``send()`` can outlive the
gateway's 5s flush wait; cancelling mid-fanout used to drop the delivery receipt
and the gateway re-sent the whole reply from bubble one."""

import asyncio
from contextlib import suppress
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _slow_adapter(delay: float):
    adapter = MagicMock()
    sends: list[str] = []

    async def send(*args, **kwargs):
        sends.append(kwargs.get("content", args[1] if len(args) > 1 else ""))
        await asyncio.sleep(delay)
        return SendResult(success=True, message_id=str(len(sends)))

    adapter.send = send
    adapter.MAX_MESSAGE_LENGTH = 4096
    return adapter, sends


@pytest.mark.asyncio
async def test_cancel_during_final_send_keeps_delivery_receipt():
    adapter, sends = _slow_adapter(0.3)
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="1", config=StreamConsumerConfig())
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("hello there")
    consumer.finish("hello there")

    # Same shape as run_turn._await_stream_task: bounded wait, then cancel + await.
    try:
        await asyncio.wait_for(task, timeout=0.1)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert len(sends) == 1
    assert consumer.final_response_sent or consumer.final_content_delivered
