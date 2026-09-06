"""Profile-local prompt overrides."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_prompt_overrides_load_and_apply(monkeypatch):
    """Overrides from prompt_overrides.yaml replace constants."""
    with tempfile.TemporaryDirectory() as tmpdir:
        override_file = Path(tmpdir) / "prompt_overrides.yaml"
        override_file.write_text(
            "DEFAULT_AGENT_IDENTITY: 'Custom identity'\n"
            "PLATFORM_HINTS:\n"
            "  telegram: 'Custom Telegram hint'\n"
        )

        monkeypatch.setenv("HERMES_HOME", tmpdir)

        # Force reload by clearing cached module
        import sys
        if "agent.prompt_builder" in sys.modules:
            del sys.modules["agent.prompt_builder"]

        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS

        assert DEFAULT_AGENT_IDENTITY == "Custom identity"
        assert PLATFORM_HINTS["telegram"] == "Custom Telegram hint"


def test_prompt_overrides_null_removes_section(monkeypatch):
    """Setting a constant to null removes it (empty string)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        override_file = Path(tmpdir) / "prompt_overrides.yaml"
        override_file.write_text("MEMORY_GUIDANCE: null\n")

        monkeypatch.setenv("HERMES_HOME", tmpdir)

        import sys
        if "agent.prompt_builder" in sys.modules:
            del sys.modules["agent.prompt_builder"]

        from agent.prompt_builder import MEMORY_GUIDANCE

        assert MEMORY_GUIDANCE == ""


def test_skills_index_header_footer_overridable(monkeypatch):
    """SKILLS_INDEX_HEADER/FOOTER overrides wrap the skills index; {basic_tools} is substituted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        override_file = Path(tmpdir) / "prompt_overrides.yaml"
        override_file.write_text(
            "SKILLS_INDEX_HEADER: 'Custom header ({basic_tools})'\n"
            "SKILLS_INDEX_FOOTER: null\n"
        )

        monkeypatch.setenv("HERMES_HOME", tmpdir)

        import sys
        if "agent.prompt_builder" in sys.modules:
            del sys.modules["agent.prompt_builder"]

        import agent.oneshot_footprint as oneshot
        monkeypatch.setattr(oneshot, "is_single_query_session", lambda: False)
        from agent.prompt_builder import _render_skills_index

        out = _render_skills_index({"cat": [("demo", "a demo skill")]}, {}, None, {"terminal"})

        assert out.startswith("Custom header (terminal)\n\n<available_skills>\n")
        assert "    - demo: a demo skill" in out
        assert out.rstrip().endswith("</available_skills>")
        assert "MUST load" not in out


@pytest.mark.parametrize("component", [
    "MEMORY_GUIDANCE_FRAME", "USER_PROFILE_GUIDANCE_FRAME",
    "MEMORY_GUIDANCE_SKILL_ROUTING", "MEMORY_GUIDANCE_NO_SKILL_ROUTING",
    "MEMORY_GUIDANCE_BODY", "all",
])
@pytest.mark.parametrize("value", ["custom", "empty", "null", "missing"])
def test_memory_yaml_overrides_reach_assembled_prompt(tmp_path, component, value):
    """Override one component without erasing other components or runtime gates."""
    components = [
        "MEMORY_GUIDANCE_FRAME", "USER_PROFILE_GUIDANCE_FRAME",
        "MEMORY_GUIDANCE_SKILL_ROUTING", "MEMORY_GUIDANCE_NO_SKILL_ROUTING",
        "MEMORY_GUIDANCE_BODY",
    ] if component == "all" else [component]
    values = {"empty": "''", "null": "null"}
    (tmp_path / "prompt_overrides.yaml").write_text(
        "".join(f"{name}: 'CUSTOM_{name} '\n" if value == "custom" else f"{name}: {values[value]}\n"
                for name in components) if value != "missing" else "{}\n",
        encoding="utf-8",
    )
    baseline_home = tmp_path / "baseline"
    baseline_home.mkdir()
    code = """
import importlib
import os
import sys
from agent import prompt_builder as pb
from agent.system_prompt import build_system_prompt
from tests.agent.test_system_prompt import _make_agent

components = ("MEMORY_GUIDANCE_FRAME", "USER_PROFILE_GUIDANCE_FRAME",
              "MEMORY_GUIDANCE_SKILL_ROUTING", "MEMORY_GUIDANCE_NO_SKILL_ROUTING",
              "MEMORY_GUIDANCE_BODY")
defaults = {name: getattr(pb, name) for name in components}
baseline = os.environ["HERMES_HOME"]
# Exercise fresh imports for A -> B -> A, keeping the existing import-time
# override lifecycle (no per-turn reloads in production).
for home in (sys.argv[1], baseline, sys.argv[1]):
    os.environ["HERMES_HOME"] = home
    importlib.reload(pb)
    expected = dict(defaults)
    if home != baseline and sys.argv[3] != "missing":
        for name in components if sys.argv[2] == "all" else [sys.argv[2]]:
            expected[name] = f"CUSTOM_{name} " if sys.argv[3] == "custom" else ""
    for name in components:
        assert getattr(pb, name) == expected[name]
    for memory_enabled, profile_enabled in ((True, True), (True, False), (False, True), (False, False)):
        for skill_manage in (True, False):
            active = ["MEMORY_GUIDANCE_FRAME" if memory_enabled else "USER_PROFILE_GUIDANCE_FRAME",
                      "MEMORY_GUIDANCE_SKILL_ROUTING" if skill_manage else "MEMORY_GUIDANCE_NO_SKILL_ROUTING",
                      "MEMORY_GUIDANCE_BODY"]
            guidance = "".join(expected[name] for name in active) if memory_enabled or profile_enabled else ""
            assert pb.build_memory_guidance(memory_enabled, profile_enabled,
                                            skill_manage_available=skill_manage) == guidance
            for memory_tool in (True, False):
                names = {"memory"} if memory_tool else set()
                if skill_manage:
                    names.add("skill_manage")
                agent = _make_agent(valid_tool_names=names, skip_context_files=True,
                                    _memory_enabled=memory_enabled, _user_profile_enabled=profile_enabled)
                prompt = build_system_prompt(agent)
                if guidance and memory_tool:
                    assert guidance.rstrip() in prompt
                for name in components:
                    if expected[name]:
                        assert (expected[name].rstrip() in prompt) == (memory_tool and bool(guidance) and name in active)
print("memory override integration OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), component, value],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(baseline_home)},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "memory override integration OK" in result.stdout
