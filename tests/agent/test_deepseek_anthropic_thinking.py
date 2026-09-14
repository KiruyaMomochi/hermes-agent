"""Regression guards for uniform Anthropic-format thinking replay."""

from __future__ import annotations

import pytest


class TestDeepSeekAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay DeepSeek thinking blocks."""



    def test_signed_anthropic_thinking_block_is_preserved(self) -> None:
        """Endpoint identity does not alter intact signed thinking blocks."""
        from agent.anthropic_message_convert import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "anthropic-signed payload",
                        "signature": "anthropic-sig-xyz",
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == [messages[1]["content"][0]]

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


@pytest.mark.parametrize("url", [None, "https://api.anthropic.com", "https://inference-api.nousresearch.com/anthropic"])
def test_deepseek_model_name_does_not_override_native_signature_contract(url):
    from agent.anthropic_message_convert import _manage_thinking_signatures
    block = {"type": "thinking", "thinking": "signed native reasoning", "signature": "sig"}
    messages = [{"role": "assistant", "content": [dict(block), {"type": "text", "text": "answer"}]}]
    _manage_thinking_signatures(messages, url, "deepseek-v4")
    assert messages[0]["content"][0] == block


def test_proxy_replays_all_intact_thinking_in_older_tool_turns():
    import copy
    from agent.anthropic_message_convert import convert_messages_to_anthropic
    history = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "checking", "reasoning_details": [
            {"type": "thinking", "thinking": "unsigned", "cache_control": {"type": "ephemeral"}},
            {"type": "thinking", "thinking": "foreign signed", "signature": "sig"},
            {"type": "redacted_thinking", "data": "redacted-signature"},
        ], "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    snapshot = copy.deepcopy(history)
    _, result = convert_messages_to_anthropic(history, base_url="https://proxy.example/anthropic", model="vendor/other-model")
    assistant = next(m for m in result if m["role"] == "assistant")
    assert [b for b in assistant["content"] if b.get("type") in {"thinking", "redacted_thinking"}] == [
        {"type": "thinking", "thinking": "unsigned"},
        {"type": "thinking", "thinking": "foreign signed", "signature": "sig"},
        {"type": "redacted_thinking", "data": "redacted-signature"},
    ]
    assert history == snapshot
