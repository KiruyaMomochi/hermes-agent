"""Bubble receipts follow upstream's physical media/delete operations after refactors."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_delete_receipt_comes_from_actual_delete(monkeypatch, confirmed):
    observed = []
    monkeypatch.setattr("plugins.platforms.telegram.telegram_message_events.observe_message",
                        lambda *args, **kwargs: observed.append((args, kwargs)))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._bot = MagicMock(delete_message=AsyncMock())
    operation = adapter.delete_message_confirmed if confirmed else adapter.delete_message
    assert await operation("123", "456")
    assert len(observed) == 1
    assert observed[0][1]["operation"] == "deleted"
    assert adapter.message_is_deleted("123", "456")
