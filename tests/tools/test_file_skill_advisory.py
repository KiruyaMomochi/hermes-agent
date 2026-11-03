"""Coverage hints through current discovery and real file operations."""
import json
from pathlib import Path

import pytest

from agent.skill_utils import TIER_LOCAL, TIER_PROJECT, parse_frontmatter
from tools import file_tools, skills_tool
from tools.file_skill_advisory import matching_skills


@pytest.fixture
def coverage(tmp_path, monkeypatch):
    skills = tmp_path / "skills"
    skills.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(skills_tool, "skill_search_dirs", lambda: ([(TIER_LOCAL, skills)], skills))
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda: set())
    skills_tool._SKILLS_CACHE.clear()
    monkeypatch.chdir(tmp_path)

    def add(name="nix-infra", extra="", root=skills):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(
            f'---\nname: {name}\ndescription: Nix conventions\ncovers: ["**/*.nix", "**/flake.lock"]\n{extra}---\nBody\n')
        skills_tool._SKILLS_CACHE.clear()
    add()
    yield tmp_path, skills, add
    skills_tool._SKILLS_CACHE.clear()


def test_covers_normalization():
    fm, _ = parse_frontmatter("---\ncovers: ['../secret', '/etc/*', '~/*', 3, '**/*.nix']\n---\n")
    assert fm["covers"] == ["**/*.nix"]
    fm, _ = parse_frontmatter("---\ncovers: '**/*.nix'\n---\n")
    assert fm["covers"] == ["**/*.nix"]
    fm, _ = parse_frontmatter("---\nname: usable\ncovers: [broken\n---\n")
    assert fm["name"] == "usable"
    assert "covers" not in fm


def test_matching_and_home_boundaries(coverage, monkeypatch):
    workspace, _, _ = coverage
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/home/yozakura")))
    assert matching_skills(["flake.nix", "nix/module.nix", "flake.lock"], workspace) == ["nix-infra"]
    for path in ("/home/kyaru/Projects/nix/flake.nix", "/home/yozakura/notes/a.nix"):
        assert matching_skills([path], workspace) == ["nix-infra"]
    assert matching_skills(["../a.nix", "/tmp/a.nix", "/etc/nixos/a.nix", "a.py"], workspace) == []


def test_dotfile_relative_to_home(coverage, monkeypatch):
    workspace, skills, _ = coverage
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/home/yozakura")))
    (skills / "nix-infra" / "SKILL.md").write_text("---\nname: ssh\ncovers: '.ssh/config'\n---\n")
    assert matching_skills(["/home/yozakura/.ssh/config"], workspace) == ["ssh"]


def test_visibility_and_tool_conditions(coverage, monkeypatch):
    workspace, _, add = coverage
    add(extra="metadata:\n  hermes:\n    requires_toolsets: [terminal]\n")
    assert matching_skills(["flake.nix"], workspace, enabled_toolsets={"file"}) == []
    assert matching_skills(["flake.nix"], workspace, enabled_toolsets={"terminal"}) == ["nix-infra"]
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda: {"nix-infra"})
    assert matching_skills(["flake.nix"], workspace) == []
    monkeypatch.setattr(skills_tool, "_get_disabled_skill_names", lambda: set())
    add(extra="platforms: [nonexistent-platform]\n")
    assert matching_skills(["flake.nix"], workspace) == []


def test_current_catalog_precedence(coverage, monkeypatch):
    workspace, local, _ = coverage
    project = workspace / "project-skills"
    directory = project / "nix-infra"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: nix-infra\ncovers: '**/*.py'\n---\n")
    monkeypatch.setattr(skills_tool, "skill_search_dirs", lambda: ([(TIER_PROJECT, project), (TIER_LOCAL, local)], local))
    monkeypatch.setattr("agent.skill_utils.iter_project_skill_files", lambda root: root.rglob("SKILL.md"))
    assert matching_skills(["flake.nix"], workspace) == []
    assert matching_skills(["a.py"], workspace) == ["nix-infra"]


def test_file_operations_and_session_deduplication(coverage):
    workspace, _, _ = coverage
    path = workspace / "flake.nix"
    path.write_text("{}\n")
    result = json.loads(file_tools._handle_read_file({"path": str(path)}, session_id="coverage-read"))
    assert "nix-infra" in result["_skill_advisory"]
    repeated = json.loads(file_tools._handle_read_file({"file_path": str(path)}, session_id="coverage-read"))
    assert "_skill_advisory" not in repeated
    other = json.loads(file_tools._handle_read_file({"path": str(path)}, session_id="coverage-other", task_id="coverage-other"))
    assert "nix-infra" in other["_skill_advisory"]
    written = json.loads(file_tools._handle_write_file({"path": str(workspace / "new.nix"), "content": "{}\n"}, session_id="coverage-write"))
    assert "nix-infra" in written["_skill_advisory"]
    patched = json.loads(file_tools._handle_patch({"path": str(path), "old_string": "{}", "new_string": "{ x = 1; }"}, session_id="coverage-patch"))
    assert "nix-infra" in patched["_skill_advisory"]


@pytest.mark.parametrize("target,extra", [("files", {}), ("content", {}), ("content", {"output_mode": "count"})])
def test_search_result_paths(coverage, target, extra):
    workspace, _, _ = coverage
    path = workspace / "flake.nix"
    path.write_text("\n".join("needle" for _ in range(8)))
    result = json.loads(file_tools._handle_search_files({"path": str(workspace), "pattern": "*.nix" if target == "files" else "needle", "target": target, **extra}, session_id=f"coverage-search-{target}-{extra}"))
    assert "nix-infra" in result["_skill_advisory"]


def test_failure_and_nonmatching_paths_are_unchanged(coverage):
    workspace, _, _ = coverage
    result = json.loads(file_tools._handle_read_file({"path": str(workspace / "missing.nix")}))
    assert result.get("error")
    assert "_skill_advisory" not in result
    path = workspace / "a.txt"
    path.write_text("hello")
    result = json.loads(file_tools._handle_read_file({"path": str(path)}))
    assert "_skill_advisory" not in result


def test_skill_edit_refreshes_coverage(coverage, monkeypatch):
    from tools import skill_manager_tool
    workspace, skills, _ = coverage
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills)
    assert matching_skills(["flake.nix"], workspace) == ["nix-infra"]
    result = json.loads(skill_manager_tool.skill_manage(
        action="patch", name="nix-infra", old_string='"**/*.nix", "**/flake.lock"',
        new_string='"**/*.toml"'))
    assert result.get("success"), result
    assert matching_skills(["flake.nix"], workspace) == []
    assert matching_skills(["config.toml"], workspace) == ["nix-infra"]
