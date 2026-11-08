"""Notice recognition follows the same live profile cache and child policy as emission."""

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.delegation_context import delegated_child_context
from hermes_yaml import safe_dump
from tools.notice_text import notice_prefixes


def test_notice_prefixes_refresh_without_reparsing_per_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "prompt_overrides.yaml"
    path.write_text(safe_dump({"notifications": {"process": {"completion": "custom {session_id}"}}}))
    with patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened:
        for _ in range(10):
            prefixes = notice_prefixes()
            assert "custom " in prefixes
            assert "[IMPORTANT: Background process " in prefixes
        assert opened.call_count == 1
    path.write_text(safe_dump({"notifications": {"process": {"completion": "changed {session_id}"}}}))
    assert "changed " in notice_prefixes()
    assert "custom " not in notice_prefixes()
    with delegated_child_context():
        assert "changed " not in notice_prefixes()
        assert "[IMPORTANT: Background process " in notice_prefixes()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))
    assert "changed " not in notice_prefixes()


@pytest.mark.parametrize("template,expected", [
    ("", None), ("{session_id}: result", None), ("   {session_id}", None),
    ("literal notice", "literal notice"), ("\n  custom {session_id}", "custom "),
])
def test_notice_prefixes_use_only_nonempty_literal_text(tmp_path, monkeypatch, template, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({"notifications": {
        "process": {"completion": template}}}))
    prefixes = notice_prefixes("process")
    assert all(prefix.strip() for prefix in prefixes)
    assert not "ordinary human ask".startswith(prefixes)
    if expected:
        assert expected in prefixes
    assert not any("ASYNC" in prefix for prefix in prefixes)
