"""Helpers for rendering gateway message timestamps exactly once."""
from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Optional, Tuple

_TIMESTAMP_PREFIX_RE = re.compile(
    r"^\[(?:"
    r"(?P<dow>[A-Z][a-z]{2}) (?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<time>\d{2}:\d{2}:\d{2})(?: (?P<tz>[A-Za-z0-9_+\-/:]+))?"
    r"|(?P<iso>\d{4}-\d{2}-\d{2}T[^\]]+)"
    r")\]\s*"
)
_SHORT_TIMESTAMP_RE = re.compile(
    r"^\[(?:(?P<full_date>\d{4}-\d{2}-\d{2}) "
    r"(?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun) |(?P<mmdd>\d{2}-\d{2} )?)"
    r"(?P<hhmm>\d{2}:\d{2})\]\s*"
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _localize(dt: datetime, tz) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz) if tz is not None else dt.astimezone()
    return float(dt.timestamp())


def _parse_iso(text: str, tz=None) -> Optional[float]:
    for parse in (datetime.fromisoformat, lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z")):
        try:
            return _localize(parse(text), tz)
        except (TypeError, ValueError):
            continue
    return None


def _parse_timestamp_match(match: re.Match, tz=None) -> Optional[float]:
    groups = match.groupdict()
    if groups.get("iso"):
        return _parse_iso(groups["iso"], tz)
    if groups.get("full_date") and groups.get("hhmm"):
        try:
            return _localize(datetime.strptime(f"{groups['full_date']} {groups['hhmm']}", "%Y-%m-%d %H:%M"), tz)
        except ValueError:
            return None
    if not groups.get("date"):
        return None
    try:
        dt = datetime.strptime(f"{groups['date']} {groups['time']}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return _localize(dt, tz)


def coerce_message_timestamp(ts_value: Any, tz=None) -> Optional[float]:
    if isinstance(ts_value, (int, float)):
        return float(ts_value)
    if hasattr(ts_value, "timestamp"):
        try:
            return float(ts_value.timestamp())
        except Exception:
            return None
    text = ts_value.strip() if isinstance(ts_value, str) else ""
    if not text:
        return None
    match = _TIMESTAMP_PREFIX_RE.match(text)
    parsed = _parse_timestamp_match(match, tz=tz) if match is not None else None
    if parsed is not None:
        return parsed
    try:
        return float(text)
    except (TypeError, ValueError):
        return _parse_iso(text, tz)


def format_message_timestamp(ts_value: Any, tz=None) -> str:
    epoch = coerce_message_timestamp(ts_value, tz=tz)
    if epoch is None:
        return ""
    dt = datetime.fromtimestamp(epoch, tz=tz) if tz is not None else datetime.fromtimestamp(epoch).astimezone()
    return f"[{dt:%H:%M}]"


def _format_legacy_timestamp(ts_value: Any, tz=None) -> str:
    epoch = coerce_message_timestamp(ts_value, tz=tz)
    if epoch is None:
        return ""
    dt = datetime.fromtimestamp(epoch, tz=tz) if tz is not None else datetime.fromtimestamp(epoch).astimezone()
    return f"[{dt:%a %Y-%m-%d %H:%M:%S %Z}]"


def inbound_timestamp_prefix(current: Any, previous: Optional[datetime] = None, tz=None) -> str:
    epoch = coerce_message_timestamp(current, tz=tz)
    if epoch is None:
        return ""
    cur = datetime.fromtimestamp(epoch, tz=tz) if tz is not None else datetime.fromtimestamp(epoch).astimezone()
    if previous is None or cur.date() != previous.date():
        return f"[{cur:%Y-%m-%d} {_WEEKDAYS[cur.weekday()]} {cur:%H:%M}]"
    if (cur - previous).total_seconds() < 60:
        return ""
    return f"[{cur:%H:%M}]"


def strip_leading_message_timestamps(content: str, tz=None) -> Tuple[str, Optional[float]]:
    if not isinstance(content, str) or not content:
        return content, None
    text, embedded_epoch = content, None
    while True:
        match = _TIMESTAMP_PREFIX_RE.match(text) or _SHORT_TIMESTAMP_RE.match(text)
        if match is None:
            break
        if not (match.groupdict().get("hhmm") and not match.groupdict().get("full_date")):
            parsed = _parse_timestamp_match(match, tz=tz)
            if parsed is not None:
                embedded_epoch = parsed
        text = text[match.end():]
    return text, embedded_epoch


def render_user_content_with_timestamp(content: str, ts_value: Any = None, tz=None) -> str:
    clean_content, embedded_epoch = strip_leading_message_timestamps(content, tz=tz)
    prefix = _format_legacy_timestamp(ts_value if embedded_epoch is None else embedded_epoch, tz=tz)
    return f"{prefix} {clean_content}" if prefix and clean_content else (prefix or clean_content)


def _parse_timestamp_prefix(text: str, tz=None) -> Optional[float]:
    match = _TIMESTAMP_PREFIX_RE.match(text) or _SHORT_TIMESTAMP_RE.match(text)
    return _parse_timestamp_match(match, tz=tz) if match else None
