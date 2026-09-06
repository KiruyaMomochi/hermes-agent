"""Profile-local overrides for the background-review prompts and toolsets.

The sweep in ``agent/prompt_builder.py`` only covers public ALL_CAPS strings defined in that
module, so these underscore-prefixed names in ``agent/background_review.py`` need their own
application step. These tests pin that a written override actually reaches the value the review
spawn reads, not just that the YAML parses.
"""

import importlib
import sys

import pytest


def _reload_with_overrides(tmp_path, monkeypatch, yaml_text: str):
    """Import ``background_review`` fresh against a HERMES_HOME holding *yaml_text*."""
    (tmp_path / "prompt_overrides.yaml").write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("agent.prompt_builder", "agent.background_review"):
        sys.modules.pop(name, None)
    return importlib.import_module("agent.background_review")


def test_review_prompts_are_overridable(tmp_path, monkeypatch):
    """A string override replaces the prompt the spawn path hands to the fork."""
    module = _reload_with_overrides(
        tmp_path, monkeypatch,
        "_COMBINED_REVIEW_PROMPT: 'combined override'\n"
        "_MEMORY_REVIEW_PROMPT: 'memory override'\n"
        "_SKILL_REVIEW_PROMPT: 'skill override'\n",
    )

    assert module._COMBINED_REVIEW_PROMPT == "combined override"
    assert module._MEMORY_REVIEW_PROMPT == "memory override"
    assert module._SKILL_REVIEW_PROMPT == "skill override"


@pytest.mark.parametrize(
    "review_memory,review_skills,expected",
    [(True, True, "combined override"), (True, False, "memory override"), (False, True, "skill override")],
)
def test_spawn_uses_overridden_prompt(tmp_path, monkeypatch, review_memory, review_skills, expected):
    """The scope table resolves to the OVERRIDDEN text, so the fork receives it."""
    module = _reload_with_overrides(
        tmp_path, monkeypatch,
        "_COMBINED_REVIEW_PROMPT: 'combined override'\n"
        "_MEMORY_REVIEW_PROMPT: 'memory override'\n"
        "_SKILL_REVIEW_PROMPT: 'skill override'\n",
    )

    class _Agent:
        pass

    _target, prompt = module.spawn_background_review_thread(
        _Agent(), [], review_memory=review_memory, review_skills=review_skills, task_cfg={},
    )
    assert prompt == expected


def test_toolsets_override_is_applied(tmp_path, monkeypatch):
    """A list override widens the toolsets the review whitelist is built from."""
    module = _reload_with_overrides(
        tmp_path, monkeypatch,
        "_BACKGROUND_REVIEW_TOOLSETS:\n  - memory\n  - skills\n  - file\n",
    )

    assert module._BACKGROUND_REVIEW_TOOLSETS == ["memory", "skills", "file"]


def test_malformed_toolsets_override_keeps_the_default(tmp_path, monkeypatch):
    """A non-list (or empty) override must not strip the review fork of every tool."""
    module = _reload_with_overrides(
        tmp_path, monkeypatch, "_BACKGROUND_REVIEW_TOOLSETS: 'skills'\n",
    )

    assert module._BACKGROUND_REVIEW_TOOLSETS == ["memory", "skills"]


def test_memory_gate_still_wins_over_the_toolsets_override(tmp_path, monkeypatch):
    """An override naming ``memory`` cannot hand the memory tool to a memory-disabled profile."""
    module = _reload_with_overrides(
        tmp_path, monkeypatch,
        "_BACKGROUND_REVIEW_TOOLSETS:\n  - memory\n  - skills\n",
    )

    class _MemoryDisabledAgent:
        _memory_enabled = False
        _user_profile_enabled = False

    seen: dict = {}
    model_tools = importlib.import_module("model_tools")

    def _fake_get_tool_definitions(*, enabled_toolsets, quiet_mode):  # noqa: ARG001
        seen["toolsets"] = list(enabled_toolsets)
        return []

    # Patch where production reads: _review_tool_whitelist imports it inside the function.
    monkeypatch.setattr(model_tools, "get_tool_definitions", _fake_get_tool_definitions)

    module._review_tool_whitelist(_MemoryDisabledAgent(), {})
    assert "memory" not in seen["toolsets"]
    assert "skills" in seen["toolsets"]
