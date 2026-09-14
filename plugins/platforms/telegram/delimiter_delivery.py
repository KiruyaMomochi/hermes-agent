"""Compose delimiter bubbles with Telegram's rich/legacy partial-send receipts.

The outer plan never guesses where an inner formatted chunk ends. Resumption first
uses the inner transport's receipt, then continues the never-attempted bubbles.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from gateway.platforms.base import SendResult
from plugins.platforms.telegram.telegram_split_replies import split_reply_delimited, split_reply_delay_seconds


def _raw(result):
    return result.raw_response if isinstance(result.raw_response, dict) else {}


def _ids(result):
    raw = _raw(result)
    return [str(i) for i in (raw.get("message_ids") or raw.get("delivered_message_ids") or
                            ([result.message_id] if result.message_id else []))]


def _uncertain(result):
    return bool(_raw(result).get("delivery_uncertain"))


def _receipt(result, content, parts, index, ids, uncertain):
    raw = dict(_raw(result))
    landed = list(dict.fromkeys(ids + _ids(result)))
    prefix = content[:parts[index][2]]
    inner_prefix = raw.get("delivered_prefix")
    if isinstance(inner_prefix, str) and parts[index][0].startswith(inner_prefix):
        prefix += inner_prefix
    raw.update({
        "message_ids": landed, "last_message_id": landed[-1] if landed else None,
        "delivered_prefix": prefix.rstrip(), "delivered_parts": index, "total_parts": len(parts),
        "delimited_resume": {"content": content, "index": index, "ids": ids,
                             "inner": asdict(result), "uncertain": uncertain},
    })
    if landed:
        raw["partial_overflow"] = True
    if index + 1 < len(parts):
        raw["unattempted_delimited_parts"] = True
    if uncertain:
        raw["delivery_uncertain"] = True
    return SendResult(success=False, message_id=landed[-1] if landed else result.message_id,
                      error=result.error, retryable=result.retryable, retry_after=result.retry_after,
                      error_kind=result.error_kind, raw_response=raw)


async def send_delimited(adapter, chat_id, content, parts, reply_to, metadata, *, index=0,
                         ids=None, inner=None, uncertain=False):
    ids = list(ids or [])
    while index < len(parts):
        part, dash_count, _ = parts[index]
        anchor = reply_to if index == 0 else None
        if inner is not None:
            resumed = await adapter._resume_partial_send(chat_id, inner, reply_to=anchor, metadata=metadata)
            if resumed is None:
                # The timed-out request itself must never be repeated. A definite refusal
                # with no landed chunks can retry this one bubble, not the preceding ones.
                resumed = inner if _uncertain(inner) else await adapter._send_single_message(chat_id, part, anchor, metadata)
            inner = None
            result = resumed
        else:
            if index:
                delay = split_reply_delay_seconds(getattr(adapter, "_split_replies_delay_seconds", 0), dash_count)
                if delay:
                    await asyncio.sleep(delay)
            result = await adapter._send_single_message(chat_id, part, anchor, metadata)
        adapter._trace_send("split_part", index=index, total=len(parts), success=result.success,
                            retryable=result.retryable, error_kind=result.error_kind,
                            retry_after=result.retry_after, delivered=len(ids))
        raw = _raw(result)
        uncertain = uncertain or _uncertain(result)
        if not result.success:
            if not (_uncertain(result) and not raw.get("unattempted_chunks")):
                return _receipt(result, content, parts, index, ids, uncertain)
            # This bubble is complete as far as the wire can tell. Continue only the
            # untouched bubbles; the gateway's delivery ledger owns the ambiguous one.
        ids = list(dict.fromkeys(ids + _ids(result)))
        index += 1
    if uncertain:
        return SendResult(success=False, message_id=ids[-1] if ids else None, error="Timed out: delimiter part may have landed",
                          raw_response={"message_ids": ids, "delivery_uncertain": True})
    return SendResult(success=True, message_id=ids[-1] if ids else None, raw_response={"message_ids": ids})


async def resume_delimited(adapter, chat_id, result, reply_to, metadata):
    state = _raw(result)["delimited_resume"]
    content = state["content"]
    parts = split_reply_delimited(content, max_parts=getattr(adapter, "_split_replies_max_parts", 8), with_offsets=True)
    async with adapter._chat_send_lock(chat_id):
        return await send_delimited(adapter, chat_id, content, parts, reply_to, metadata,
                                    index=state["index"], ids=state["ids"],
                                    inner=SendResult(**state["inner"]), uncertain=state["uncertain"])
