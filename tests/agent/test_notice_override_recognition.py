"""Configured notice headers remain machinery, including persisted English notices."""

import pytest

from hermes_yaml import safe_dump
from tools.process_registry_notifications import format_process_notification


@pytest.mark.parametrize("key,event", [
    ("notifications.process.completion", {"session_id": "proc", "exit_code": 0}),
    ("notifications.async_delegation.header", {"type": "async_delegation", "delegation_id": "deleg"}),
    ("notifications.async_delegation.batch_header", {"type": "async_delegation", "delegation_id": "deleg", "n_tasks": 2, "results": [{}, {}]}),
    ("notifications.async_delegation.task_failed_header", {"type": "async_delegation", "delegation_id": "deleg", "task_failure_notice": True, "results": [{}], "n_tasks": 2}),
    ("notifications.process.watch_header", {"type": "watch_match", "session_id": "proc", "pattern": "ready", "output": "ready"}),
])
@pytest.mark.parametrize("prefix", ["【測試通知 ", " \n【測試通知\n  "])
def test_overridden_and_original_notices_are_not_human_turns(tmp_path, monkeypatch, key, event, prefix):
    from agent.context_compressor import ContextCompressor, _build_verbatim_user_section, _synthetic_user_row
    from agent.conversation_compression_scaffolding import is_user_role_scaffolding
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = format_process_notification(event)
    group, field = key.removeprefix("notifications.").split(".")
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({
        "notifications": {group: {field: prefix + ("{session_id}]" if group == "process" else "{deleg_id}]")}},
    }), encoding="utf-8")
    replacement = format_process_notification(event)
    assert replacement.startswith(prefix)
    notices = [{"role": "user", "content": text} for text in (original, replacement)]
    for notice in notices:
        assert _synthetic_user_row(notice["content"])
        # Delegation results remain actionable tail anchors, unlike process noise.
        assert is_user_role_scaffolding(notice) == (group == "process")
        assert ContextCompressor._is_synthetic_compression_user_turn(notice) == (group == "process")
    assert _build_verbatim_user_section(notices) == ""
    assert not is_user_role_scaffolding({"role": "user", "content": "human ask"})
    from hermes_state_timeline import get_session_timeline

    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session(session_id="notices", source="cli")
        db.append_messages_batch("notices", [*notices, {"role": "user", "content": "human ask"}])
        assert [row["preview"] for row in get_session_timeline(db, "notices")["entries"]] == ["human ask"]


def test_watch_header_override_is_shared_by_gateway(tmp_path, monkeypatch):
    from gateway.run import _format_gateway_process_notification

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = {"type": "watch_match", "session_id": "proc", "pattern": "ready", "output": "{output}",
             "command": "echo ready", "suppressed": 2}
    original = format_process_notification(event)
    assert _format_gateway_process_notification(event) == original
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({"notifications": {
        "process": {"watch_header": "【測試 {session_id}: {pattern}】\n"}}}), encoding="utf-8")
    text = format_process_notification(event)
    assert text.startswith("【測試 proc: ready】\n")
    assert text.endswith('Command: echo ready\nMatched output:\n{output}\n(2 earlier matches were suppressed by rate limit)]')
    assert _format_gateway_process_notification(event) == text
