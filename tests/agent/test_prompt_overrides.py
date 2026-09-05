"""Profile-local prompt overrides."""
import tempfile
from pathlib import Path


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
