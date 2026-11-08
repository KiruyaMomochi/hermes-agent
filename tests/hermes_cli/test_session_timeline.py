"""Real read-only timeline requests against temporary profile stores."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_state import SessionDB


@pytest.fixture
def timeline_store(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="timeline-root", source="desktop")
    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        yield db, client, home
    db.close()


def test_timeline_pages_project_human_prompts_without_tool_payloads(timeline_store):
    db, client, _ = timeline_store
    db.append_messages_batch("timeline-root", [
        {"role": "user", "content": " First\n  prompt ", "timestamp": 200},
        {"role": "assistant", "content": "answer", "tool_calls": [
            {"id": "tool-1", "function": {"name": "terminal", "arguments": "x" * 10000}}]},
        {"role": "tool", "content": "secret tool payload" * 1000, "tool_call_id": "tool-1"},
        {"role": "user", "content": "é" * 150, "timestamp": 100},
    ])
    expected_ids = [row["id"] for row in db.get_messages("timeline-root") if row["role"] == "user"]
    response = client.get("/api/sessions/timeline-root/timeline?limit=1")
    assert response.status_code == 200
    first = response.json()
    assert first["session_id"] == "timeline-root"
    assert first["profile"] == "default"
    assert first["entries"] == [{"row_id": expected_ids[0], "preview": "First prompt", "timestamp": 200}]
    assert first["pagination"]["total"] == 2
    assert first["pagination"]["returned"] == 1
    assert first["pagination"]["has_more"] is True
    second = client.get("/api/sessions/timeline-root/timeline", params={
        "after_row_id": first["pagination"]["next_cursor"], "limit": 1}).json()
    assert second["entries"][0]["row_id"] == expected_ids[1]
    assert len(second["entries"][0]["preview"]) == 120
    assert second["pagination"]["has_more"] is False
    assert second["pagination"]["next_cursor"] is None
    assert b"secret tool payload" not in response.content


@pytest.mark.parametrize("legacy", [False, True])
def test_compacted_timeline_and_jump_share_display_order_and_visibility(timeline_store, legacy):
    from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT, SUMMARY_PREFIX, _SUMMARY_END_MARKER

    db, client, _ = timeline_store
    sid = "timeline-root"
    db.append_messages_batch(sid, [
        {"role": "user", "content": "old ask", "timestamp": 10},
        {"role": "assistant", "content": "old answer", "timestamp": 11},
        {"role": "user", "content": "retained ask", "timestamp": 12},
        {"role": "assistant", "content": "retained answer", "timestamp": 13},
    ])
    retained = db.get_messages(sid)[2:]
    db.archive_and_compact(sid, [
        {"role": "user", "content": SUMMARY_PREFIX + "summary", "_compressed_summary": True},
        *retained,
    ])
    db.append_messages_batch(sid, [
        {"role": "user", "content": "hidden ask", "display_kind": "hidden"},
        {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT},
        {"role": "user", "content": "[IMPORTANT: Background process 12 completed]"},
        {"role": "user", "content": "[ASYNC DELEGATION BATCH COMPLETE] completed"},
        {"role": "user", "content": "A background subagent you dispatched earlier has finished."},
        {"role": "user", "content": SUMMARY_PREFIX + "handoff\n" + _SUMMARY_END_MARKER + "\nlive ask",
         "_compressed_summary": True, "display_kind": "hidden"},
        {"role": "assistant", "content": "live answer"},
        {"role": "user", "content": "rewound ask"},
    ])
    rewound_id = db.get_messages(sid)[-1]["id"]
    db._write_sql("UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (rewound_id,))
    if legacy:
        db._write_sql("UPDATE messages SET display_identity = NULL, display_order = NULL WHERE session_id = ?", (sid,))
    timeline = client.get(f"/api/sessions/{sid}/timeline").json()
    assert [entry["preview"] for entry in timeline["entries"]] == ["old ask", "retained ask", "live ask"]
    anchor = timeline["entries"][1]["row_id"]
    response = client.get(f"/api/sessions/{sid}/messages/around?row_id={anchor}&limit=2")
    assert response.status_code == 200
    jump = response.json()
    assert [row["content"] for row in jump["messages"]] == ["retained ask", "retained answer"]
    assert jump["pagination"]["has_older"] is True
    assert jump["pagination"]["has_newer"] is True
    live_id = timeline["entries"][-1]["row_id"]
    last = client.get(f"/api/sessions/{sid}/messages/around?row_id={live_id}").json()
    assert last["messages"][0]["display_content"] == "live ask"
    assert last["messages"][1]["content"] == "live answer"
    assert last["pagination"]["has_newer"] is False
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={rewound_id}").status_code == 404


def test_cursor_survives_compaction_between_pages(timeline_store):
    db, client, _ = timeline_store
    sid = "timeline-root"
    db.append_messages_batch(sid, [
        {"role": "user", "content": f"ask {i}", "timestamp": 100 - i} for i in range(4)
    ])
    first = client.get(f"/api/sessions/{sid}/timeline?limit=2").json()
    old_row_id = first["entries"][-1]["row_id"]
    retained = db.get_messages(sid)[1:]
    db.archive_and_compact(sid, retained)
    second = client.get(f"/api/sessions/{sid}/timeline", params={
        "limit": 2, "after_row_id": first["pagination"]["next_cursor"]}).json()
    assert [r["preview"] for r in first["entries"] + second["entries"]] == [f"ask {i}" for i in range(4)]
    assert second["pagination"]["total"] == 4
    assert second["pagination"]["has_more"] is False
    current = client.get(f"/api/sessions/{sid}/timeline").json()
    assert current["entries"][1]["row_id"] > old_row_id
    empty = client.get(f"/api/sessions/{sid}/timeline?after_row_id=99999").json()
    assert empty["entries"] == []
    assert empty["pagination"]["total"] == 4
    assert empty["pagination"]["has_more"] is False


def test_exact_owner_lineage_validation_and_bounded_jump(timeline_store):
    db, client, home = timeline_store
    sid = "timeline-root"
    db.end_session(sid, end_reason="compression")
    db.create_session(session_id="timeline-tip", source="desktop", parent_session_id=sid)
    db.append_messages_batch("timeline-tip", [
        {"role": "user", "content": "tip ask"},
        *[{"role": "assistant", "content": f"step {i}"} for i in range(130)],
        {"role": "user", "content": "last ask"},
        {"role": "assistant", "content": "last answer"},
    ])
    db.create_session(session_id="delegate", source="tool", parent_session_id="timeline-tip")
    db.append_message("delegate", role="user", content="delegate ask")
    db.create_session(session_id="foreign", source="desktop")
    foreign = db.append_message("foreign", role="user", content="foreign ask")
    expected_sid = client.get(f"/api/sessions/{sid}/messages").json()["session_id"]
    first = client.get(f"/api/sessions/{sid}/timeline?limit=1").json()
    assert first["session_id"] == expected_sid == "timeline-tip"
    row_id = first["entries"][0]["row_id"]
    jump = client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id}").json()
    assert jump["messages"][0]["id"] == row_id
    assert len(jump["messages"]) == jump["pagination"]["returned"] == 120
    assert jump["pagination"]["has_older"] is False
    assert jump["pagination"]["has_newer"] is True
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={foreign}").status_code == 404
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id + 1}").status_code == 404
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id}&limit=121").status_code == 422
    assert client.get(f"/api/sessions/{sid}/timeline?limit=501").status_code == 422
    assert client.get("/api/sessions/timeline-ro/timeline").status_code == 404
    assert client.get("/api/sessions/absent/timeline").status_code == 404
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    with SessionDB(db_path=work / "state.db") as other:
        other.create_session(session_id=sid, source="desktop")
        other.append_message(sid, role="user", content="work ask")
    for profile, expected in (("default", "tip ask"), ("work", "work ask"), ("default", "tip ask")):
        page = client.get(f"/api/sessions/{sid}/timeline?profile={profile}").json()
        assert page["profile"] == profile
        assert page["entries"][0]["preview"] == expected
    db._write_sql("UPDATE sessions SET profile_name = 'wrong-owner' WHERE id = 'timeline-tip'")
    assert client.get(f"/api/sessions/{sid}/timeline").status_code == 404
    assert client.get(f"/api/sessions/{sid}/messages/around?row_id={row_id}").status_code == 404


@pytest.mark.parametrize("launch_profile", ["default", "work"])
def test_timeline_notice_headers_use_requested_profile(timeline_store, monkeypatch, launch_profile):
    db, client, home = timeline_store
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    headers = {"default": "【本機完成】", "work": "【工作完成】"}
    for profile, directory in (("default", home), ("work", work)):
        (directory / "prompt_overrides.yaml").write_text(
            f'notifications:\n  process:\n    completion: "{headers[profile]} {{session_id}}"\n',
            encoding="utf-8",
        )
    with SessionDB(db_path=work / "state.db") as other:
        other.create_session(session_id="timeline-root", source="desktop")
        for profile, store in (("default", db), ("work", other)):
            foreign = headers["work" if profile == "default" else "default"] + " human wording"
            store.append_messages_batch("timeline-root", [
                {"role": "user", "content": headers[profile] + " proc-1"},
                {"role": "user", "content": "[IMPORTANT: Background process old completed]"},
                {"role": "user", "content": foreign},
                {"role": "user", "content": "real human ask"},
            ])
    launch_home = home if launch_profile == "default" else work
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", launch_home / "state.db")
    # Interleave owners, then the implicit launch profile: no context/cache leakage.
    for profile in ("work", "default", "work", launch_profile):
        params = {} if profile == launch_profile else {"profile": profile}
        response = client.get("/api/sessions/timeline-root/timeline", params=params)
        assert response.status_code == 200
        page = response.json()
        foreign = headers["work" if profile == "default" else "default"] + " human wording"
        assert page["profile"] == profile
        assert [row["preview"] for row in page["entries"]] == [foreign, "real human ask"]
        assert page["pagination"]["total"] == 2
        # Jump validation must apply the same owner-specific notice classification.
        for row_id in (1, 2):
            jump = client.get("/api/sessions/timeline-root/messages/around", params={
                **params, "row_id": row_id})
            assert jump.status_code == 404
        jump = client.get("/api/sessions/timeline-root/messages/around", params={
            **params, "row_id": page["entries"][0]["row_id"]})
        assert jump.status_code == 200
        assert jump.json()["messages"][0]["content"] == foreign


def test_timeline_sql_never_reads_tool_columns_or_writes(timeline_store, monkeypatch):
    import sqlite3
    from contextlib import contextmanager
    from hermes_state_timeline import get_session_timeline

    db, _, home = timeline_store
    db.append_messages_batch("timeline-root", [
        {"role": "user", "content": [{"type": "text", "text": "multimodal ask"},
                                      {"type": "image_url", "image_url": {"url": "data:unused"}}]},
        {"role": "tool", "content": "unread tool result", "tool_calls": [{"unused": "payload"}]},
    ])
    reader = SessionDB(db_path=home / "state.db", read_only=True)
    original = reader._read_ctx
    accesses = []

    def authorize(action, table, column, *_):
        if action == sqlite3.SQLITE_READ:
            accesses.append((table, column))
            if table == "messages" and column in {"tool_calls", "reasoning", "api_content", "codex_reasoning_items"}:
                return sqlite3.SQLITE_DENY
        if action in {sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @contextmanager
    def guarded():
        with original() as conn:
            conn.set_authorizer(authorize)
            try:
                yield conn
            finally:
                conn.set_authorizer(None)

    monkeypatch.setattr(reader, "_read_ctx", guarded)
    try:
        page = get_session_timeline(reader, "timeline-root")
        assert page["entries"][0]["preview"] == "multimodal ask"
        assert page["pagination"]["total"] == 1
        assert accesses
    finally:
        reader.close()
