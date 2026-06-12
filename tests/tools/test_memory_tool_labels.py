"""Regression tests for configurable memory prompt labels."""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.config as hermes_config
from tools import memory_tool


def _store_with_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> memory_tool.MemoryStore:
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: tmp_path)
    memory_tool._reset_memory_labels_cache()
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "USER.md").write_text("prefers concise responses", encoding="utf-8")
    (tmp_path / "MEMORY.md").write_text("project uses pytest", encoding="utf-8")
    store = memory_tool.MemoryStore(memory_char_limit=200, user_char_limit=200)
    store.load_from_disk()
    return store


def test_memory_prompt_labels_default_when_config_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = _store_with_entries(tmp_path / "memories", monkeypatch)

    user_block = store.format_for_system_prompt("user")
    memory_block = store.format_for_system_prompt("memory")

    assert user_block is not None
    assert memory_block is not None
    assert "USER PROFILE (who the user is) [" in user_block
    assert "MEMORY (your personal notes) [" in memory_block


def test_memory_prompt_labels_read_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  labels:\n"
        "    user: Custom User Facts\n"
        "    agent: Custom Agent Notes\n",
        encoding="utf-8",
    )
    store = _store_with_entries(tmp_path / "memories", monkeypatch)

    user_block = store.format_for_system_prompt("user")
    memory_block = store.format_for_system_prompt("memory")

    assert user_block is not None
    assert memory_block is not None
    assert "Custom User Facts [" in user_block
    assert "Custom Agent Notes [" in memory_block
    assert "USER PROFILE (who the user is) [" not in user_block
    assert "MEMORY (your personal notes) [" not in memory_block


def test_memory_prompt_labels_invalid_config_values_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  labels:\n"
        "    user: ''\n"
        "    agent:\n"
        "      nested: invalid\n",
        encoding="utf-8",
    )
    store = _store_with_entries(tmp_path / "memories", monkeypatch)

    user_block = store.format_for_system_prompt("user")
    memory_block = store.format_for_system_prompt("memory")

    assert user_block is not None
    assert memory_block is not None
    assert "USER PROFILE (who the user is) [" in user_block
    assert "MEMORY (your personal notes) [" in memory_block


def test_memory_prompt_labels_config_load_failure_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated config load failure")),
    )
    store = _store_with_entries(tmp_path / "memories", monkeypatch)

    user_block = store.format_for_system_prompt("user")
    memory_block = store.format_for_system_prompt("memory")

    assert user_block is not None
    assert memory_block is not None
    assert "USER PROFILE (who the user is) [" in user_block
    assert "MEMORY (your personal notes) [" in memory_block
