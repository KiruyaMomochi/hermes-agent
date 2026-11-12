"""Explicit instruction-file exemptions, through real config loading and file writes."""

import json
from pathlib import Path

import pytest
import yaml

from tools.file_tools import write_file_tool


@pytest.fixture
def configured_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    state = tmp_path / "hermes-state"
    state.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(state))
    monkeypatch.setattr(Path, "home", lambda: home)

    def configure(paths):
        (state / "config.yaml").write_text(yaml.safe_dump({
            "security": {"protected_instruction_exempt_paths": paths},
        }))

    return home, configure


def test_explicit_tilde_path_writes_without_approval(configured_home):
    home, configure = configured_home
    configure(["~/AGENTS.md"])
    result = json.loads(write_file_tool(str(home / "AGENTS.md"), "my rules"))
    assert not result.get("error"), result
    assert (home / "AGENTS.md").read_text() == "my rules"


@pytest.mark.parametrize("relative", ["AGENTS.md", "Projects/repo/SOUL.md", "repo/.hermes/config.yaml"])
def test_directory_exempts_descendants(configured_home, relative):
    home, configure = configured_home
    configure(["~"])
    target = home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(write_file_tool(str(target), "my rules"))
    assert not result.get("error"), result
    assert target.read_text() == "my rules"


@pytest.mark.parametrize("sibling", ["other", "home2"])
def test_directory_does_not_exempt_outside_or_prefix_sibling(configured_home, sibling):
    home, configure = configured_home
    configure(["~"])
    target = home.parent / sibling / "AGENTS.md"
    target.parent.mkdir()
    result = json.loads(write_file_tool(str(target), "injected"))
    assert "protected agent-instruction" in result.get("error", ""), result
    assert not target.exists()


def test_file_exemption_does_not_exempt_neighbor(configured_home):
    home, configure = configured_home
    configure(["~/SOUL.md"])
    result = json.loads(write_file_tool(str(home / "AGENTS.md"), "injected"))
    assert "protected agent-instruction" in result.get("error", ""), result
    assert not (home / "AGENTS.md").exists()


@pytest.mark.require_symlinks
@pytest.mark.parametrize("directory_link", [False, True])
def test_symlink_escaping_exempt_directory_still_needs_approval(configured_home, directory_link):
    home, configure = configured_home
    configure(["~"])
    outside = home.parent / "other-user"
    outside.mkdir()
    target = outside / "AGENTS.md"
    target.write_text("original")
    link = home / ("linked-repo" if directory_link else "AGENTS.md")
    link.symlink_to(outside if directory_link else target, target_is_directory=directory_link)
    write_path = link / "AGENTS.md" if directory_link else link
    result = json.loads(write_file_tool(str(write_path), "injected"))
    assert "protected agent-instruction" in result.get("error", ""), result
    assert target.read_text() == "original"


@pytest.mark.require_symlinks
def test_alias_to_exempt_target_uses_realpath(configured_home):
    home, configure = configured_home
    configure(["~/AGENTS.md"])
    alias = home.parent / "alias.md"
    alias.symlink_to(home / "AGENTS.md")
    result = json.loads(write_file_tool(str(alias), "my rules"))
    assert not result.get("error"), result
    assert (home / "AGENTS.md").read_text() == "my rules"


@pytest.mark.parametrize("paths", ["~", None, {}, [None, 42, "", "AGENTS.md"]])
def test_invalid_exemptions_fail_closed(configured_home, paths):
    home, configure = configured_home
    configure(paths)
    result = json.loads(write_file_tool(str(home / "AGENTS.md"), "injected"))
    assert "protected agent-instruction" in result.get("error", ""), result
    assert not (home / "AGENTS.md").exists()


def test_config_load_failure_keeps_gate_on(configured_home, monkeypatch):
    home, configure = configured_home
    configure(["~"])

    def unavailable():
        raise OSError("unreadable config")

    monkeypatch.setattr("hermes_cli.config.load_config", unavailable)
    result = json.loads(write_file_tool(str(home / "AGENTS.md"), "injected"))
    assert "protected agent-instruction" in result.get("error", ""), result
    assert not (home / "AGENTS.md").exists()


def test_exemptions_do_not_bypass_config_hard_block(configured_home):
    home, configure = configured_home
    configure([str(home.parent)])
    config = home.parent / "hermes-state" / "config.yaml"
    original = config.read_text()
    result = json.loads(write_file_tool(str(config), "injected"))
    assert result.get("error"), result
    assert config.read_text() == original
