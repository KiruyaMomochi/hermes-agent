"""Unavailable CJK indexes are preserved on open and isolated on disposable repair-probe copies."""

import argparse
import sqlite3

import pytest

import hermes_state
from hermes_cli import sessions_cmd
from hermes_state import SessionDB

_CJK_TRIGGERS = ("messages_fts_cjk_insert", "messages_fts_cjk_delete", "messages_fts_cjk_update")


def _outside_insert(db_path) -> None:
    with sqlite3.connect(db_path) as raw:
        raw.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', 'user', '北京', 0)")


@pytest.fixture
def cjk_db(tmp_path, monkeypatch):
    """The on-disk shape of a built index, as seen by a process that cannot load the tokenizer."""
    monkeypatch.setenv("HERMES_CJK_FTS", "0")
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="cli", model="m")
    db.append_message("s1", "user", "东京都")
    db.close()
    raw = sqlite3.connect(db_path)
    raw.isolation_level = None
    raw.executescript(
        "CREATE VIEW messages_fts_cjk_src AS SELECT id, role, content, tool_name, tool_calls FROM messages"
        " WHERE role <> 'tool';"
        "CREATE VIRTUAL TABLE messages_fts_cjk USING fts5(content, tool_name, tool_calls,"
        " content='messages_fts_cjk_src', content_rowid='id', tokenize='unicode61');"
        "INSERT INTO state_meta (key, value) VALUES ('fts_cjk_rebuild_high_water', '1'),"
        " ('fts_cjk_rebuild_progress', '0'), ('fts_cjk_stale', '1');"
    )
    for name, event, row in zip(_CJK_TRIGGERS, ("INSERT", "DELETE", "UPDATE"), ("new", "old", "new")):
        raw.execute(f"CREATE TRIGGER {name} AFTER {event} ON messages BEGIN"
                    f" INSERT INTO messages_fts_cjk(rowid, content) VALUES ({row}.id, {row}.content); END")
    raw.execute("PRAGMA writable_schema=ON")
    raw.execute("UPDATE sqlite_master SET sql = replace(sql, \"'unicode61'\", \"'cjk_unicode61'\")"
                " WHERE name = 'messages_fts_cjk'")
    version = raw.execute("PRAGMA schema_version").fetchone()[0]
    raw.execute(f"PRAGMA schema_version={version + 1}")
    raw.execute("PRAGMA writable_schema=OFF")
    raw.close()
    with pytest.raises(sqlite3.OperationalError, match="no such tokenizer: cjk_unicode61"):
        _outside_insert(db_path)
    return db_path


def _cjk_residue(db_path) -> list:
    with sqlite3.connect(db_path) as raw:
        objects = raw.execute("SELECT name FROM sqlite_master WHERE name LIKE 'messages\\_fts\\_cjk%' ESCAPE '\\'")
        markers = raw.execute("SELECT key FROM state_meta WHERE key LIKE 'fts\\_cjk\\_%' ESCAPE '\\'")
        return [row[0] for row in (*objects.fetchall(), *markers.fetchall())]


def test_open_preserves_unavailable_index_and_restores_outside_writers(cjk_db):
    before = set(_cjk_residue(cjk_db))
    SessionDB(db_path=cjk_db).close()
    assert set(_cjk_residue(cjk_db)) == before - set(_CJK_TRIGGERS)
    _outside_insert(cjk_db)


def test_second_open_preserves_unavailable_index(cjk_db):
    SessionDB(db_path=cjk_db).close()
    before = _store_snapshot(cjk_db)
    db = SessionDB(db_path=cjk_db)
    try:
        db.append_message("s1", "user", "서울")
        assert db.search_messages("东京都")
    finally:
        db.close()
    assert _store_snapshot(cjk_db) == before


def _store_snapshot(db_path) -> tuple:
    with sqlite3.connect(db_path) as raw:
        schema = raw.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
        return schema, raw.execute("SELECT key, value FROM state_meta ORDER BY key").fetchall()


def test_repair_check_only_leaves_the_store_unchanged(cjk_db, monkeypatch):
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", cjk_db)
    before = _store_snapshot(cjk_db)

    assert not sessions_cmd._cmd_repair(argparse.Namespace(check_only=True, no_backup=False))

    assert _store_snapshot(cjk_db) == before
    assert "messages_fts_cjk" in _cjk_residue(cjk_db)
