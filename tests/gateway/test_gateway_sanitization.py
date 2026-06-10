"""Regression tests for gateway user-visible response sanitization."""

from gateway.run import _allows_gateway_reasoning_display
from gateway.stream_consumer import GatewayStreamConsumer


def test_telegram_never_allows_gateway_reasoning_display():
    assert _allows_gateway_reasoning_display("telegram") is False


def test_non_telegram_gateway_reasoning_display_policy_unchanged():
    assert _allows_gateway_reasoning_display("discord") is True


def test_stream_display_strips_formatted_reasoning_preamble():
    text = (
        "💭 **Reasoning:**\n"
        "```\n"
        "internal scratchpad must stay hidden\n"
        "```\n\n"
        "Visible answer"
    )

    cleaned = GatewayStreamConsumer._clean_for_display(text)

    assert cleaned == "Visible answer"
    assert "Reasoning" not in cleaned
    assert "internal scratchpad" not in cleaned
