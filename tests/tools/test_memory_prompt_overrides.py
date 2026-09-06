"""Profile-local memory presentation, independent of tool result wording."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.system_prompt import _memory_parts
from hermes_yaml import safe_dump
from tools.memory_tool_store import (
    MEMORY_BLOCK_HEADERS, MemoryStore, get_memory_block_headers, get_memory_block_order,
)


def _overrides(monkeypatch, home, data):
    monkeypatch.setenv("HERMES_HOME", str(home))
    home.mkdir(exist_ok=True)
    (home / "prompt_overrides.yaml").write_text(safe_dump(data), encoding="utf-8")


def test_defaults_without_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert get_memory_block_headers() == MEMORY_BLOCK_HEADERS
    assert get_memory_block_order() == ["memory", "user"]


@pytest.mark.parametrize("value, expected", [
    ({"memory": "我的筆記", "user": "關於33"}, {"memory": "我的筆記", "user": "關於33"}),
    ({"user": "關於33", "unknown": "ignored"}, {**MEMORY_BLOCK_HEADERS, "user": "關於33"}),
    ({"memory": None, "user": ""}, {"memory": "", "user": ""}),
    ({"memory": 42, "user": []}, MEMORY_BLOCK_HEADERS),
    ([], MEMORY_BLOCK_HEADERS), (None, MEMORY_BLOCK_HEADERS),
])
def test_header_overrides(monkeypatch, tmp_path, value, expected):
    _overrides(monkeypatch, tmp_path, {"MEMORY_BLOCK_HEADERS": value})
    assert get_memory_block_headers() == expected
    store = MemoryStore()
    for target in ("memory", "user"):
        block = store._render_block(target, ["A fact"])
        assert block.splitlines()[1].startswith(expected[target] + " [")
        assert block.endswith("A fact")
        assert store._render_block(target, []) == ""


def test_file_edits_and_profile_switch_do_not_stale_or_mutate_defaults(monkeypatch, tmp_path):
    original = dict(MEMORY_BLOCK_HEADERS)
    for home, title, order in [
        (tmp_path / "a", "First", ["user", "memory"]),
        (tmp_path / "a", "A longer replacement", ["memory", "user"]),
        (tmp_path / "b", "Other profile", ["user", "memory"]),
        (tmp_path / "a", "Return to first profile", ["memory", "user"]),
    ]:
        _overrides(monkeypatch, home, {"MEMORY_BLOCK_HEADERS": {"memory": title}, "MEMORY_BLOCK_ORDER": order})
        assert get_memory_block_headers()["memory"] == title
        assert get_memory_block_order() == order
    (home / "prompt_overrides.yaml").unlink()
    assert get_memory_block_headers() == original
    assert MEMORY_BLOCK_HEADERS == original


@pytest.mark.parametrize("text", ["[invalid", "- not a mapping\n"])
def test_bad_yaml_falls_back(monkeypatch, tmp_path, text):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "prompt_overrides.yaml").write_text(text, encoding="utf-8")
    assert get_memory_block_headers() == MEMORY_BLOCK_HEADERS
    assert get_memory_block_order() == ["memory", "user"]


@pytest.mark.parametrize("order", [
    ["memory", "user"], ["user", "memory"], [], ["memory"],
    ["memory", "memory"], ["user", "memory", "other"],
    "user,memory", None, {"user": "memory"}, [[], {}],
])
@pytest.mark.parametrize("enabled", [(True, True), (False, True), (True, False), (False, False)])
def test_block_order_keeps_gates_and_external_provider_last(monkeypatch, tmp_path, order, enabled):
    _overrides(monkeypatch, tmp_path, {"MEMORY_BLOCK_ORDER": order})
    store = Mock()
    store.format_for_system_prompt.side_effect = lambda kind: kind
    manager = Mock()
    manager.build_system_prompt.return_value = "external"
    agent = SimpleNamespace(_memory_store=store, _memory_enabled=enabled[0],
                            _user_profile_enabled=enabled[1], _memory_manager=manager)
    monkeypatch.setattr("agent.memory_manager_tools.memory_provider_tools_exposed", lambda _: True)
    expected_order = ["user", "memory"] if order == ["user", "memory"] else ["memory", "user"]
    gates = dict(zip(("memory", "user"), enabled))
    assert _memory_parts(agent) == [kind for kind in expected_order if gates[kind]] + ["external"]
    assert [call.args[0] for call in store.format_for_system_prompt.call_args_list] == [
        kind for kind in expected_order if gates[kind]
    ]
