"""Bubble receipts for observers, without exposing the bot or its credentials."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def observe_message(chat_id, message_id, text, *, thread_id=None, operation="sent"):
    from hermes_cli.lifecycle import has_hook, invoke_hook
    if not has_hook("gateway_outbound_message"):
        return
    try:
        invoke_hook("gateway_outbound_message", platform="telegram", chat_id=str(chat_id),
                    message_id=str(message_id), thread_id=str(thread_id) if thread_id else None,
                    text=text, operation=operation)
    except Exception:
        # A delivered bubble must never be retried because an observer failed.
        logger.warning("Telegram outbound observer failed", exc_info=True)
