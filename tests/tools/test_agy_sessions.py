"""Read-only Antigravity recall through the public session_search entry point."""
import json
from pathlib import Path
import sqlite3

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session

import pytest

from tools.session_search_tool import session_search


@pytest.fixture
def agy(tmp_path, monkeypatch):
    app = tmp_path / "antigravity"
    app.mkdir()
    monkeypatch.setenv("HERMES_AGY_DIRS", str(app))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return app


def write_steps(app, cid, steps):
    path = app / "brain" / cid / ".system_generated/logs/transcript_full.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in steps))
    return path


def step(idx, text, role="user", **extra):
    return {"step_index": idx, "content": text,
            "source": "USER_EXPLICIT" if role == "user" else "MODEL",
            "type": "USER_INPUT" if role == "user" else "PLANNER_RESPONSE",
            "created_at": "2026-06-02T12:00:00Z", **extra}


def test_read_extracts_request_and_rewinds_even_nonvisible_steps(agy):
    write_steps(agy, "abc", [step(0, "context <USER_REQUEST>\n你好\n</USER_REQUEST> context",
                                  media=[{"mime_type": "audio/ogg"}]),
                            step(2, "abandoned", "assistant"), step(4, "stale"),
                            step(2, "hidden", type="TOOL_CALL"),
                            step(3, "Visible\nMEDIA:/tmp/audio.ogg\nanswer", "assistant")])
    result = json.loads(session_search(session_id="agy:abc"))
    assert result["success"], result
    assert result["source"] == "agy"
    assert result["message_count"] == 2
    assert [(m["id"], m["message_id"], m["role"], m["content"]) for m in result["messages"]] == [
        (0, 0, "user", "你好\n[語音訊息]"), (3, 3, "assistant", "Visible\n\nanswer")]


@pytest.fixture
def store(tmp_path):
    with open_persistence(profile_scope(home=tmp_path / ".hermes"), eager=True) as handle:
        seed_session(handle, "native", "cli")
        seed_messages(handle, "native", {"role": "user", "content": "Hitohira native history"})
        yield handle


def test_browse_and_search_merge_native_and_agy(agy, store):
    write_steps(agy, "abc", [step(0, "Hitohira 一片花瓣"), step(7, "Hitohira reply", "assistant")])
    for kwargs, mode in [({}, "browse"), ({"query": "Hitohira"}, "discover")]:
        result = json.loads(session_search(persistence=store, **kwargs))
        assert result["mode"] == mode
        assert {r["session_id"] for r in result["results"]} == {"native", "agy:abc"}
        assert result["count"] == 2
        assert json.loads(session_search(persistence=store, limit=1, **kwargs))["count"] == 1
    result = json.loads(session_search(query="花瓣", persistence=store))
    assert result["results"][0]["match_message_id"] == 0
    assert result["results"][0]["detail"] == "full"


def test_scroll_links_and_closest_anchor(agy):
    write_steps(agy, "abc", [step(i, str(i)) for i in range(0, 20, 2)])
    result = json.loads(session_search(session_id="@session:default/agy:abc", around_message_id=7, window=1))
    assert result["success"], result
    assert [m["id"] for m in result["messages"]] == [4, 6, 8]
    assert result["messages"][1]["anchor"]
    assert result["messages_before"] == 2 and result["messages_after"] == 5


def test_metadata_hides_subagents_in_all_shapes(agy, store):
    for cid in ("parent", "child", "nested", "linked"):
        write_steps(agy, cid, [step(0, "Hitohira")])
    with sqlite3.connect(agy / "conversation_summaries.db") as con:
        con.execute("CREATE TABLE conversation_summaries (conversation_id TEXT, title TEXT, parent_conversation_id TEXT, nesting_depth INTEGER)")
        con.executemany("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?)", [
            ("parent", "Custom title", None, 0), ("child", "Child", "parent", 0), ("nested", "Nested", None, 1)])
        con.execute("CREATE TABLE conversation_parents (conversation_id TEXT, parent_id TEXT)")
        con.execute("INSERT INTO conversation_parents VALUES ('linked', 'parent')")
    for query in ("", "Hitohira"):
        result = json.loads(session_search(query=query, persistence=store))
        external = [r for r in result["results"] if r["source"] == "agy"]
        assert [r["session_id"] for r in external] == ["agy:parent"]
        assert external[0]["title"] == "Custom title"
    assert not json.loads(session_search(session_id="agy:child"))["success"]


@pytest.mark.parametrize("query,expected", [
    ('"alpha beta"', {"a"}), ("alpha beta", {"a", "b"}),
    ("alpha OR gamma", {"a", "b", "c"}), ("alpha NOT beta", set()),
    ("alph*", {"a", "b", "d"}), ("花瓣", {"a"}), ("alpha", {"a", "b"}),
])
def test_keyword_boolean_phrase_prefix_and_cjk(agy, store, query, expected):
    for cid, text in {"a": "alpha beta 一片花瓣", "b": "beta alpha", "c": "gamma", "d": "alphabet"}.items():
        write_steps(agy, cid, [step(0, text)])
    result = json.loads(session_search(query=query, persistence=store, limit=10))
    assert {r["session_id"][4:] for r in result["results"] if r["source"] == "agy"} == expected


def test_bounds_roles_exclusions_sort_and_adaptive_detail(agy, store):
    for cid, date in (("early", "2026-06-01T00:00:00Z"), ("late", "2026-06-03T00:00:00Z")):
        write_steps(agy, cid, [step(0, "greeting", created_at=date),
                               step(5, "needle", "assistant", created_at=date), step(8, "tail", created_at=date)])
    def search(**kwargs):
        return json.loads(session_search(query="needle", persistence=store, **kwargs))["results"]
    assert [r["session_id"] for r in search(after="2026-06-01", before="2026-06-03")] == ["agy:early"]
    assert [r["session_id"] for r in search(after="2026-06-03")] == ["agy:late"]
    assert search(role_filter="user") == []
    assert all(r["matched_role"] == "assistant" for r in search(role_filter="assistant"))
    assert [r["session_id"] for r in search(exclude_session_ids=["@session:default/agy:early"])] == ["agy:late"]
    newest = search(sort="newest")
    assert [r["session_id"] for r in newest] == ["agy:late", "agy:early"]
    assert [r["detail"] for r in newest] == ["full", "compact"]
    assert len(newest[0]["messages"]) == 3 and len(newest[1]["messages"]) == 1
    assert [r["session_id"] for r in search(sort="oldest")] == ["agy:early", "agy:late"]
    assert all(r["detail"] == "full" for r in search(detail="full"))


def test_secondary_profile_never_receives_agy(agy, store, tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    write_steps(agy, "private", [step(0, "Hitohira private")])
    root = tmp_path / ".hermes"
    secondary = root / "profiles/eva"
    secondary.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n")
    (secondary / "config.yaml").write_text("{}\n")
    with open_persistence(profile_scope(home=secondary), eager=True) as other:
        seed_session(other, "own", "cli")
        seed_messages(other, "own", {"role": "user", "content": "Hitohira own"})
        for kwargs in ({}, {"query": "Hitohira"}, {"session_id": "agy:private"}):
            result = json.loads(session_search(profile="eva", **kwargs))
            assert "Hitohira private" not in json.dumps(result)
            assert all(r.get("source") != "agy" for r in result.get("results", []))
            if kwargs.get("session_id"):
                assert result["success"] is False
        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        token = set_hermes_home_override(secondary)
        try:
            for kwargs in ({}, {"query": "Hitohira"}, {"session_id": "agy:private"},
                           {"session_id": "@session:default/agy:private"}, {"profile": "default", "query": "Hitohira"}):
                result = json.loads(session_search(persistence=other, **kwargs))
                assert "Hitohira private" not in json.dumps(result)
                assert all(r.get("source") != "agy" for r in result.get("results", []))
        finally:
            reset_hermes_home_override(token)


def test_cache_refresh_truncation_and_malformed_live_tail(agy):
    path = write_steps(agy, "abc", [step(i, "x" * 3000) for i in range(40)])
    first = json.loads(session_search(session_id="agy:abc"))
    assert first["truncated"] and len(first["messages"]) == 30
    assert first["messages"][0]["content_truncated"]
    with path.open("a") as stream:
        stream.write(json.dumps(step(1, "rewound")) + "\n{partial")
    second = json.loads(session_search(session_id="agy:abc"))
    assert second["message_count"] == 2 and second["messages"][-1]["content"] == "rewound"
    assert not json.loads(session_search(session_id="agy:.."))["success"]


def test_live_wal_metadata_hides_subagent_without_touching_source(agy, store):
    write_steps(agy, "child", [step(0, "Hitohira child")])
    db_path = agy / "conversation_summaries.db"
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE conversation_parents (conversation_id TEXT, parent_id TEXT)")
        con.execute("INSERT INTO conversation_parents VALUES ('child', 'parent')")
        con.commit()
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in agy.glob("conversation_summaries.db*")}
        result = json.loads(session_search(query="Hitohira", persistence=store))
        assert all(r["source"] != "agy" for r in result["results"])
        assert not json.loads(session_search(session_id="agy:child"))["success"]
        after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in agy.glob("conversation_summaries.db*")}
        assert before == after


def test_temporal_merge_fully_hydrates_promoted_native_hit(agy, store):
    from tests.fixtures.storage_holder import raw_holder
    seed_session(store, "older", "cli")
    seed_messages(store, "older", {"role": "user", "content": "Hitohira oldest"})
    seed_messages(store, "older", {"role": "assistant", "content": "extra context"})
    raw_holder(store)._execute_write(lambda con: con.executemany(
        "UPDATE sessions SET title=?, started_at=? WHERE id=?", [
            ("Hitohira", 1800000000, "native"), ("Older history", 1700000000, "older")]))
    write_steps(agy, "abc", [step(0, "Hitohira")])
    result = json.loads(session_search(query="Hitohira", sort="oldest", persistence=store))
    assert result["results"][0]["session_id"] == "older"
    assert result["results"][0]["detail"] == "full"
    assert len(result["results"][0]["messages"]) == 2


def test_registry_dispatch_and_transcript_cache_are_read_only(agy, store, monkeypatch):
    from tools.registry import registry
    path = write_steps(agy, "abc", [step(0, "Hitohira"), step(9, "reply", "assistant")])
    entry = registry.get_entry("session_search")
    assert entry is not None
    result = json.loads(entry.handler({"query": "Hitohira"}, db=store))
    hit = next(r for r in result["results"] if r["source"] == "agy")
    original_open = Path.open
    def no_reparse(self, *args, **kwargs):
        assert self != path, "unchanged transcript should come from memory"
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_reparse)
    result = json.loads(entry.handler({"session_id": hit["session_id"], "around_message_id": hit["match_message_id"]}))
    assert [m["message_id"] for m in result["messages"]] == [0, 9]
    monkeypatch.setenv("HERMES_AGY_DIRS", "")
    result = json.loads(entry.handler({"query": "Hitohira"}, db=store))
    assert all(r["source"] != "agy" for r in result["results"])
