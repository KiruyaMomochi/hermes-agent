"""Telegram --- split replies behavior."""
import pytest

from plugins.platforms.telegram.telegram_split_replies import (
    split_reply_delimited,
    split_reply_delay_seconds,
)


def test_split_reply_recognizes_standalone_dash_lines():
    text = "First bubble\n\n---\n\nSecond bubble"
    parts = split_reply_delimited(text)
    assert len(parts) == 2
    assert parts[0][0] == "First bubble"
    assert parts[1][0] == "Second bubble"


def test_split_reply_ignores_dashes_inside_code_fence():
    text = "Before\n\n```\n---\n```\n\nAfter"
    parts = split_reply_delimited(text)
    assert len(parts) == 1


def test_split_reply_tracks_dash_count_for_delay():
    text = "A\n\n---\n\nB\n\n-----\n\nC"
    parts = split_reply_delimited(text)
    assert len(parts) == 3
    assert parts[0] == ("A", 0)
    assert parts[1] == ("B", 3)
    assert parts[2] == ("C", 5)


def test_split_delay_scales_with_dash_count():
    base = 0.35
    assert split_reply_delay_seconds(base, 0) == base
    assert split_reply_delay_seconds(base, 3) == base
    assert split_reply_delay_seconds(base, 4) > base
    assert split_reply_delay_seconds(base, 10) > split_reply_delay_seconds(base, 4)


def test_split_caps_at_max_parts():
    text = "\n\n---\n\n".join(f"Part {i}" for i in range(20))
    parts = split_reply_delimited(text, max_parts=5)
    assert len(parts) == 5
    # Overflow merged into last part
    assert "Part 19" in parts[-1][0]


def test_yaml_config_bridge():
    """Config propagation from YAML to adapter extras."""
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    split_cfg = {"enabled": True, "delay_between_ms": 250, "max_parts": 4}
    extras = _apply_yaml_config({}, {"split_replies": split_cfg})

    assert extras is not None
    assert extras["split_replies"] == split_cfg
