"""Native audio bytes are media, not text tokens in preflight estimates."""

from copy import deepcopy

import pytest

from agent.model_metadata import estimate_messages_tokens_rough, estimate_request_tokens_rough


@pytest.mark.parametrize("field", ["content", "output"])
def test_encoded_audio_size_does_not_trigger_text_compaction(field):
    def message(data):
        return {"role": "user", field: [
            {"type": "text", "text": "Listen to this clip"},
            {"type": "input_audio", "input_audio": {"data": data, "format": "mp3"}},
        ]}

    short = message("YWJj")
    large = message("YWJj" * 250_000)
    before = deepcopy(large)
    short_cost = estimate_request_tokens_rough([short], system_prompt="system")
    large_cost = estimate_request_tokens_rough([large], system_prompt="system")
    text_cost = estimate_messages_tokens_rough([
        {"role": "user", "content": [{"type": "text", "text": "Listen to this clip"}]}
    ])
    assert large_cost == short_cost
    assert text_cost < large_cost < 10_000
    assert large == before
    assert estimate_request_tokens_rough([large], system_prompt="system") == large_cost
    large[field].append(deepcopy(large[field][-1]))
    assert estimate_request_tokens_rough([large], system_prompt="system") > large_cost + 4_000
