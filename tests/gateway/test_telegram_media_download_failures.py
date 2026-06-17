"""Regression tests for Telegram media download failure handling."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _message(*, photo=None, document=None, caption=None):
    chat = SimpleNamespace(
        id=123,
        type="private",
        title=None,
        full_name="Test Chat",
        is_forum=False,
    )
    user = SimpleNamespace(id=456, full_name="Test User")
    return SimpleNamespace(
        chat=chat,
        from_user=user,
        message_id=789,
        message_thread_id=None,
        is_topic_message=False,
        date=None,
        text="",
        caption=caption,
        photo=photo,
        video=None,
        audio=None,
        voice=None,
        document=document,
        sticker=None,
        reply_to_message=None,
        media_group_id=None,
    )


@pytest.mark.asyncio
async def test_media_download_retries_transient_failure(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    monkeypatch.setattr("gateway.platforms.telegram.asyncio.sleep", AsyncMock())

    file_obj = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"ok")))
    source = SimpleNamespace(get_file=AsyncMock(side_effect=[RuntimeError("proxy reset"), file_obj]))

    returned_file, data = await adapter._download_telegram_file_bytes(source, label="photo")

    assert returned_file is file_obj
    assert data == b"ok"
    assert source.get_file.await_count == 2
    file_obj.download_as_bytearray.assert_awaited_once()


@pytest.mark.asyncio
async def test_photo_download_success_keeps_existing_cache_and_batch_flow(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    adapter._enqueue_photo_event = MagicMock()
    monkeypatch.setattr(
        "gateway.platforms.telegram.cache_image_from_bytes",
        MagicMock(return_value="/tmp/hermes-photo.jpg"),
    )

    file_obj = SimpleNamespace(
        file_path="photos/file.jpg",
        download_as_bytearray=AsyncMock(return_value=bytearray(b"image")),
    )
    photo = SimpleNamespace(get_file=AsyncMock(return_value=file_obj))
    update = SimpleNamespace(update_id=1, message=_message(photo=[photo], caption="look"))

    await adapter._handle_media_message(update, SimpleNamespace())

    adapter._enqueue_photo_event.assert_called_once()
    event = adapter._enqueue_photo_event.call_args.args[1]
    assert event.text == "look"
    assert event.media_urls == ["/tmp/hermes-photo.jpg"]
    assert event.media_types == ["image/jpg"]
    photo.get_file.assert_awaited_once()
    file_obj.download_as_bytearray.assert_awaited_once()


@pytest.mark.asyncio
async def test_photo_download_failure_dispatches_failure_note_not_blank(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr("gateway.platforms.telegram.asyncio.sleep", AsyncMock())

    photo = SimpleNamespace(get_file=AsyncMock(side_effect=RuntimeError("proxy down")))
    update = SimpleNamespace(update_id=1, message=_message(photo=[photo]))

    await adapter._handle_media_message(update, SimpleNamespace())

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "[Image received but download failed]"
    assert event.media_urls == []
    assert photo.get_file.await_count == 3


@pytest.mark.asyncio
async def test_photo_download_failure_preserves_caption(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr("gateway.platforms.telegram.asyncio.sleep", AsyncMock())

    photo = SimpleNamespace(get_file=AsyncMock(side_effect=RuntimeError("proxy down")))
    update = SimpleNamespace(update_id=1, message=_message(photo=[photo], caption="please inspect"))

    await adapter._handle_media_message(update, SimpleNamespace())

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "please inspect"
    assert event.media_urls == []
    assert photo.get_file.await_count == 3


@pytest.mark.asyncio
async def test_document_download_failure_dispatches_failure_note_not_blank(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr("gateway.platforms.telegram.asyncio.sleep", AsyncMock())

    doc = SimpleNamespace(
        file_name="report.pdf",
        mime_type="application/pdf",
        file_size=128,
        get_file=AsyncMock(side_effect=RuntimeError("proxy down")),
    )
    update = SimpleNamespace(update_id=1, message=_message(document=doc))

    await adapter._handle_media_message(update, SimpleNamespace())

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "[Document received but download failed]"
    assert event.media_urls == []
    assert doc.get_file.await_count == 3
