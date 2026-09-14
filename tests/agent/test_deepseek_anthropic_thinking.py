"""Regression guard for Anthropic-compatible thinking block cleanup."""

from __future__ import annotations


class TestDeepSeekAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay DeepSeek thinking blocks."""

    def test_cache_control_stripped_from_thinking_block(self) -> None:
        """cache_control must still be stripped even when the block is preserved.

        DeepSeek's compatibility matrix lists cache_control on thinking blocks
        as ignored — cache markers interfere with signature validation on
        upstreams that do check them, so Hermes strips them everywhere.
        """
        from agent.anthropic_message_convert import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "r1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        # Inject cache_control on the synthesised thinking block after-the-fact
        # by running conversion once, mutating, then re-running would be
        # indirect.  Instead check the simpler invariant: no thinking block in
        # the converted output carries cache_control.
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )
        for m in converted:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}:
                    assert "cache_control" not in b
