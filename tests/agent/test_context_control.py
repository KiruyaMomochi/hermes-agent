from __future__ import annotations

from agent.memory_manager import build_memory_context_block
from agent.turn_context import _system_memory_context, compose_user_api_content


def test_context_control_positions_are_reloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    control = tmp_path / "context_control.yaml"
    control.write_text("openviking:\n  position: before\n  min_score: 0.8\n  top_k: 2\n  max_items: 1\n", encoding="utf-8")
    expected = build_memory_context_block("remembered")
    assert compose_user_api_content("question", "remembered", "") == expected + "\n\nquestion"

    control.write_text("openviking:\n  position: system\n", encoding="utf-8")
    assert compose_user_api_content("question", "remembered", "") is None
    assert _system_memory_context("remembered") == expected


def test_context_control_disabled_keeps_plugin_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "context_control.yaml").write_text("enabled: false\n", encoding="utf-8")
    assert compose_user_api_content("question", "remembered", "plugin") == "question\n\nplugin"
