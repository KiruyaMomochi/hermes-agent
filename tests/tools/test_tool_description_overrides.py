"""Behavior contracts for profile-scoped tool schema descriptions."""

from pathlib import Path

from hermes_yaml import safe_dump

from tools.registry import registry

_TEST_TOOL = "_description_override_test_tool"
registry.register(
    name=_TEST_TOOL,
    toolset="test",
    schema={"name": _TEST_TOOL, "description": "original description", "parameters": {}},
    handler=lambda args: "ok",
)


def _description() -> str:
    from tools.prompt_text import apply_tool_text
    definitions = apply_tool_text(registry.get_definitions({_TEST_TOOL}, quiet=True))
    return next(item["function"]["description"] for item in definitions)


def test_tool_description_override_is_read_from_profile_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = _description()
    (Path(tmp_path) / "prompt_overrides.yaml").write_text(
        safe_dump({"tool_descriptions": {_TEST_TOOL: "profile description"}}),
        encoding="utf-8",
    )

    assert _description() == "profile description"
    assert original != "profile description"


def test_tool_description_without_override_keeps_schema_value(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert _description() == "original description"


def test_vault_note_on_input_tools_follows_profile_override(tmp_path, monkeypatch):
    from model_tools import _rewrite_input_tool_for_vault

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    td = {"type": "function", "function": {"name": "browser_type", "description": "Type text.", "parameters": {}}}
    default = _rewrite_input_tool_for_vault(td, {"browser_vault"})["function"]["description"]
    (Path(tmp_path) / "prompt_overrides.yaml").write_text(
        safe_dump({"VAULT_NO_PASSWORD_NOTE": " profile note"}), encoding="utf-8"
    )
    from tools.prompt_text import apply_tool_text
    overridden = apply_tool_text([_rewrite_input_tool_for_vault(td, {"browser_vault"})])[0]["function"]["description"]

    assert overridden == "Type text. profile note"
    assert default != overridden
