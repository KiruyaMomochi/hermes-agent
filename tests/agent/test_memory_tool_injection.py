"""Request-only recalled memory uses the same conversion path as real tools."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.turn_context import build_api_messages
from tests.agent.test_api_content_sidecar import (
    _chat_requests,
    _tc_resp,
    _text_resp,
    wire_env,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "hy_search",
            "description": "Search memory",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


@pytest.fixture
def tool_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("memory:\n  recall_injection: tool_result\n")


def make_agent(mode="chat_completions", tools=TOOLS):
    return SimpleNamespace(
        api_mode=mode,
        tools=tools,
        _current_turn_timestamp=1.0,
        _current_turn_id="test-turn",
        ephemeral_system_prompt="",
        _copy_reasoning_content_for_api=lambda source, target: None,
        _should_sanitize_tool_calls=lambda: False,
    )


def build(agent, messages, recall="I remember tea.", index=0):
    from agent.memory_context import build_memory_tool_messages

    agent._memory_tool_messages = build_memory_tool_messages(
        agent, recall, messages[index]["content"]
    )
    return build_api_messages(
        agent,
        messages,
        current_turn_user_idx=index,
        ext_prefetch_cache=recall,
        plugin_user_context="",
        moa_config=None,
        active_system_prompt="SYSTEM",
    )[0]


def test_request_only_pair_after_user_is_stable_within_turn(tool_mode):
    history = [{"role": "user", "content": "What do you remember?"}]
    original = copy.deepcopy(history)
    payload = build(make_agent(), history)
    assert [m["role"] for m in payload] == ["system", "user", "assistant", "tool"]
    call = payload[2]["tool_calls"][0]
    assert call["function"]["name"] == "hy_search"
    assert json.loads(call["function"]["arguments"]) == {}
    assert payload[3]["tool_call_id"] == call["id"]
    assert payload[3]["content"] == "I remember tea."
    assert payload == build(make_agent(), history)
    assert history == original
    later = history + [
        {"role": "assistant", "content": "Tea."},
        {"role": "user", "content": "Next"},
    ]
    assert "I remember tea." not in json.dumps(
        build(make_agent(), later, recall="", index=2)
    )


@pytest.mark.parametrize("recall", ["", " ", "\n\t"])
def test_empty_recall_injects_nothing(tool_mode, recall):
    assert build(make_agent(), [{"role": "user", "content": "Hello"}], recall) == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "Hello"},
    ]


@pytest.mark.parametrize(
    "mode,tools",
    [("chat_completions", []), ("codex_app_server", TOOLS), ("unsupported", TOOLS)],
)
def test_unavailable_tool_or_transport_uses_text(tool_mode, mode, tools):
    payload = build(make_agent(mode, tools), [{"role": "user", "content": "Hello"}])
    assert len(payload) == 2
    assert payload[1]["content"].startswith("<memory-context>\n")


def test_default_mode_keeps_existing_text_composition():
    from agent.turn_context import compose_user_api_content

    payload = build(make_agent(), [{"role": "user", "content": "Hello"}])
    assert payload[1]["content"] == compose_user_api_content(
        "Hello", "I remember tea.", ""
    )


def test_sanitized_empty_result_falls_back_to_text(tool_mode):
    payload = build(
        make_agent(),
        [{"role": "user", "content": "Hello"}],
        "<memory-context>\nwrapped\n</memory-context>",
    )
    assert [m["role"] for m in payload] == ["system", "user"]


def test_client_without_hermes_tools_uses_text(tool_mode):
    agent = make_agent()
    agent.client = SimpleNamespace(SUPPORTS_HERMES_TOOL_CALLS=False)
    payload = build(agent, [{"role": "user", "content": "Hello"}])
    assert [m["role"] for m in payload] == ["system", "user"]


def test_request_without_recall_does_not_reuse_parent_pair(tool_mode):
    agent = make_agent()
    messages = [{"role": "user", "content": "Hello"}]
    build(agent, messages)
    payload, _ = build_api_messages(
        agent,
        messages,
        current_turn_user_idx=0,
        ext_prefetch_cache="",
        plugin_user_context="",
        moa_config=None,
        active_system_prompt="SYSTEM",
    )
    assert [m["role"] for m in payload] == ["system", "user"]


@pytest.mark.parametrize(
    "mode",
    [
        "chat_completions",
        "anthropic_messages",
        "codex_responses",
        "bedrock_converse",
        "gemini",
    ],
)
def test_existing_converters_preserve_call_result_pair(tool_mode, mode):
    from agent.transports import get_transport

    agent = make_agent("chat_completions" if mode == "gemini" else mode)
    payload = build(agent, [{"role": "user", "content": "Hello"}])
    call_id = payload[2]["tool_calls"][0]["id"]
    if mode == "gemini":
        from agent.gemini_native_adapter import build_gemini_request

        wire = build_gemini_request(
            messages=payload, tools=TOOLS, model="gemini-3-pro-preview"
        )
        call = wire["contents"][-2]["parts"][0]["functionCall"]
        result = wire["contents"][-1]["parts"][0]["functionResponse"]
        assert call["name"] == result["name"] == "hy_search"
        assert call["id"] == result["id"] == call_id
    else:
        wire = get_transport(mode).build_kwargs("test-model", payload, TOOLS)
        if mode == "chat_completions":
            assert wire["messages"][-2] == payload[-2]
            assert wire["messages"][-1]["tool_call_id"] == call_id
        elif mode == "anthropic_messages":
            call = wire["messages"][-2]["content"][0]
            result = wire["messages"][-1]["content"][0]
            assert call["type"] == "tool_use" and call["name"] == "hy_search"
            assert call["id"] == result["tool_use_id"] == call_id
            assert result["type"] == "tool_result"
        elif mode == "codex_responses":
            call, result = wire["input"][-2:]
            assert call["type"] == "function_call" and call["name"] == "hy_search"
            assert result["type"] == "function_call_output"
            assert call["call_id"] == result["call_id"] == call_id
        else:
            call = wire["messages"][-2]["content"][0]["toolUse"]
            result = wire["messages"][-1]["content"][0]["toolResult"]
            assert call["name"] == "hy_search"
            assert call["toolUseId"] == result["toolUseId"] == call_id
    assert "I remember tea." in json.dumps(wire)


@pytest.mark.parametrize("multimodal", [False, True])
def test_real_turn_does_not_persist_synthetic_recall(wire_env, multimodal):
    make, handler, db, sid = wire_env
    from hermes_cli.config import load_config, atomic_config_write
    from hermes_constants import get_hermes_home

    config = load_config()
    config["memory"]["recall_injection"] = "tool_result"
    atomic_config_write(get_hermes_home() / "config.yaml", config)
    agent = make()
    agent.tools = TOOLS
    handler.response_queue.extend(
        [_tc_resp("read_file", '{"path":"/nonexistent"}'), _text_resp("Done")]
    )
    content = (
        [{"type": "text", "text": "Remember our tea?"}]
        if multimodal
        else "Remember our tea?"
    )
    deltas = []
    with patch(
        "agent.turn_context._memory_turn_start_and_prefetch",
        return_value="I remember tea.",
    ):
        result = agent.run_conversation(
            content, conversation_history=[], task_id="t", stream_callback=deltas.append
        )
    requests = _chat_requests(handler)
    assert len(requests) == 2
    pairs = [
        [
            m
            for m in r["messages"]
            if m.get("role") == "tool"
            and m.get("tool_call_id", "").startswith("call_memory_")
        ]
        for r in requests
    ]
    assert len(pairs[0]) == len(pairs[1]) == 1
    assert pairs[0] == pairs[1]
    assert "I remember tea." not in "".join(deltas)
    assert "I remember tea." not in json.dumps(result)
    assert "I remember tea." not in json.dumps(db.get_messages(sid))
    assert "hy_search" not in json.dumps(db.get_messages(sid))
    handler.captured_requests = []
    make().run_conversation(
        "Next question",
        conversation_history=db.get_messages_as_conversation(sid),
        task_id="next",
    )
    assert "I remember tea." not in json.dumps(_chat_requests(handler))


def test_configuration_change_waits_until_next_turn(wire_env):
    from hermes_cli.config import load_config, atomic_config_write
    from hermes_constants import get_hermes_home

    make, handler, db, sid = wire_env
    config = load_config()
    config["memory"]["recall_injection"] = "tool_result"
    atomic_config_write(get_hermes_home() / "config.yaml", config)
    agent = make()
    agent.tools = TOOLS
    handler.response_queue.extend([_tc_resp("read_file"), _text_resp("Done")])

    def change_config(*args, **kwargs):
        config["memory"]["recall_injection"] = "text"
        atomic_config_write(get_hermes_home() / "config.yaml", config)
        return "Tool done"

    with (
        patch(
            "agent.turn_context._memory_turn_start_and_prefetch",
            return_value="I remember tea.",
        ),
        patch("model_tools.handle_function_call", side_effect=change_config),
    ):
        agent.run_conversation("Remember our tea?", conversation_history=[])
    requests = _chat_requests(handler)
    assert len(requests) == 2
    assert (
        requests[0]["messages"]
        == requests[1]["messages"][: len(requests[0]["messages"])]
    )
    handler.captured_requests = []
    with patch(
        "agent.turn_context._memory_turn_start_and_prefetch",
        return_value="I remember tea.",
    ):
        agent.run_conversation(
            "Remember again?", conversation_history=db.get_messages_as_conversation(sid)
        )
    messages = _chat_requests(handler)[0]["messages"]
    assert not any(
        m.get("tool_call_id", "").startswith("call_memory_") for m in messages
    )
    assert "I remember tea." in messages[-1]["content"]
