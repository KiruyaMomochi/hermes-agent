"""Anthropic replay keeps intact thinking and drops locally invalidated blocks."""

from agent.anthropic_message_convert import convert_messages_to_anthropic


def test_intact_historical_thinking_replays_but_repaired_turn_drops_it():
    intact_thinking = {"type": "thinking", "thinking": "intact plan", "signature": "sig-intact"}
    intact_redacted = {"type": "redacted_thinking", "data": "opaque-intact"}
    invalidated_thinking = {"type": "thinking", "thinking": "stale plan", "signature": "sig-stale"}
    messages = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "first answer",
            "reasoning_details": [intact_thinking, intact_redacted],
        },
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "checking",
            "reasoning_details": [invalidated_thinking],
            "tool_calls": [{
                "id": "orphan",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "user", "content": "continue"},
    ]

    _system, converted = convert_messages_to_anthropic(messages)

    assistants = [message for message in converted if message.get("role") == "assistant"]
    assert intact_thinking in assistants[0]["content"]
    assert intact_redacted in assistants[0]["content"]
    assert not any(
        block.get("type") in {"thinking", "redacted_thinking"}
        for block in assistants[1]["content"]
        if isinstance(block, dict)
    )
    assert "_thinking_signature_invalidated" not in assistants[1]
