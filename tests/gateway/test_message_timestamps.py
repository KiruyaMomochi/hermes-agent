from datetime import datetime
from zoneinfo import ZoneInfo

from gateway.message_timestamps import (
    coerce_message_timestamp,
    strip_leading_message_timestamps,
)


BERLIN = ZoneInfo("Europe/Berlin")


def _epoch(year, month, day, hour, minute, second):
    return datetime(year, month, day, hour, minute, second, tzinfo=BERLIN).timestamp()


# ---------------------------------------------------------------------------
# Opt-in gate: gateway.message_timestamps.enabled (default OFF)
# ---------------------------------------------------------------------------


def test_message_timestamps_enabled_defaults_off():
    from gateway.run import _message_timestamps_enabled

    assert _message_timestamps_enabled(None) is False
    assert _message_timestamps_enabled({}) is False
    assert _message_timestamps_enabled({"gateway": {}}) is False
    assert (
        _message_timestamps_enabled({"gateway": {"message_timestamps": {}}}) is False
    )


def test_build_history_injects_only_when_enabled():
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "hello", "timestamp": _epoch(2026, 4, 28, 13, 40, 53)},
        {"role": "assistant", "content": "hi"},
    ]

    # Default (off): user content stays clean, no timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history)
    assert agent_history[0]["content"] == "hello"

    # Enabled: user content gets exactly one timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history, inject_timestamps=True)
    assert agent_history[0]["content"].startswith("[")
    assert agent_history[0]["content"].endswith("hello")
    # Assistant message is never timestamped.
    assert agent_history[1]["content"] == "hi"


def test_build_history_uses_compact_prefixes_and_sixty_second_burst_rule():
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "first", "timestamp": _epoch(2026, 4, 28, 13, 40, 0)},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "burst", "timestamp": _epoch(2026, 4, 28, 13, 40, 30)},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "spaced", "timestamp": _epoch(2026, 4, 28, 13, 42, 0)},
        {"role": "assistant", "content": "three"},
        {"role": "user", "content": "next day", "timestamp": _epoch(2026, 4, 29, 9, 0, 0)},
    ]

    agent_history, _ = _build_gateway_agent_history(history, inject_timestamps=True)
    users = [entry["content"] for entry in agent_history if entry["role"] == "user"]
    assert users == [
        "[2026-04-28 Tue 11:40] first",
        "burst",
        "[11:42] spaced",
        "[2026-04-29 Wed 07:00] next day",
    ]
