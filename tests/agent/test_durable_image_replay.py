"""Retained image parts survive SQLite reload and gateway replay."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from agent.image_eviction_policy import OUTBOUND_IMAGE_LIMIT
from agent.session_persistence import _db_flush_row
from gateway.run import _build_gateway_agent_history
from hermes_state import SessionDB


def _image(data=b"image"):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def _open(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="images", source="telegram")
    return db


def _persist(db, message):
    agent = SimpleNamespace(_session_db=db)
    db.append_messages_batch("images", [_db_flush_row(agent, message, False)])


def _replay(db):
    stored = db.get_messages_as_conversation("images")
    history, _ = _build_gateway_agent_history(stored)
    return stored, history


def test_first_turn_user_image_and_resumed_turn(tmp_path):
    db = _open(tmp_path)
    try:
        image = _image()
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "What is this?"}, image]})
        raw = db._conn.execute("SELECT content, image_refs FROM messages WHERE session_id='images'").fetchone()
        assert raw[0] == "What is this?\n[screenshot]"
        assert "base64" not in raw[1]
        stored, history = _replay(db)
        assert history[0]["content"] == [{"type": "text", "text": "What is this?"}, image]
        assert stored[0]["content"] == history[0]["content"]
        from agent.message_metadata import without_persistence_fields
        assert "image_refs" not in without_persistence_fields(history[0])
        db.close()
        db = SessionDB(db_path=tmp_path / "state.db")
        _persist(db, {"role": "assistant", "content": "A picture."})
        _, resumed = _replay(db)
        assert resumed[0]["content"] == history[0]["content"]
        assert resumed[1]["content"] == "A picture."
    finally:
        db.close()


def test_tool_image_missing_snapshot_rewrites_claim(tmp_path):
    db = _open(tmp_path)
    try:
        _persist(db, {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call", "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}]})
        _persist(db, {"role": "tool", "tool_name": "vision_analyze", "tool_call_id": "call",
            "content": {"_multimodal": True, "text_summary": "Image loaded — you can see it natively now.",
                "content": [{"type": "text", "text": "Image loaded — you can see it natively now."}, _image(b"vision")]}})
        stored, history = _replay(db)
        assert history[1]["content"][1] == _image(b"vision")
        refs = json.loads(db._conn.execute("SELECT image_refs FROM messages WHERE role='tool'").fetchone()[0])
        snapshot = Path(db.db_path).parent / "cache" / "session_images" / refs[1]["sha256"]
        snapshot.unlink()
        stored, history = _replay(db)
        assert "you can see it natively now" not in str(history[1]["content"])
        assert "image unavailable" in str(history[1]["content"])
        assert not any(p["type"] == "image_url" for p in history[1]["content"])
        assert stored[1]["content"] == history[1]["content"]
    finally:
        db.close()


def test_source_file_snapshot_and_missing_user_image(tmp_path):
    db = _open(tmp_path)
    source = tmp_path / "incoming.png"
    source.write_bytes(b"original")
    try:
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "Look"},
            {"type": "image_url", "image_url": {"url": str(source)}}]})
        source.write_bytes(b"changed")
        _, history = _replay(db)
        assert history[0]["content"][1] == _image(b"original")
        refs = json.loads(db._conn.execute("SELECT image_refs FROM messages").fetchone()[0])
        (tmp_path / "cache" / "session_images" / refs[1]["sha256"]).unlink()
        _, history = _replay(db)
        assert history[0]["content"][1] == {"type": "text", "text": "[image unavailable]"}
    finally:
        db.close()


def test_remote_user_image_url_survives_replay(tmp_path):
    db = _open(tmp_path)
    remote = "https://example.com/photo.png"
    try:
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "Look"},
            {"type": "image_url", "image_url": {"url": remote}}]})
        _, history = _replay(db)
        assert history[0]["content"][1] == {"type": "image_url", "image_url": {"url": remote}}
    finally:
        db.close()


def test_replace_message_content_rebuilds_image_sidecar(tmp_path):
    db = _open(tmp_path)
    try:
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "original caption"}, _image()]})
        message = db.get_messages_as_conversation("images")[0]
        message["content"][0]["text"] = "updated caption"
        db.replace_messages("images", [message])
        _, history = _replay(db)
        assert history[0]["content"][0] == {"type": "text", "text": "updated caption"}
        assert history[0]["content"][1] == _image()
        message = db.get_messages_as_conversation("images")[0]
        message["content"].pop()
        db.replace_messages("images", [message])
        _, history = _replay(db)
        assert history[0]["content"] == [{"type": "text", "text": "updated caption"}]
        assert "image_refs" not in history[0]
    finally:
        db.close()


def test_20_image_boundary_retains_tool_images_for_send_path_eviction(tmp_path):
    from agent.context_compressor import evict_stale_outbound_tool_images
    db = _open(tmp_path)
    try:
        for i in range(OUTBOUND_IMAGE_LIMIT + 1):
            _persist(db, {"role": "tool", "tool_call_id": str(i), "content": [
                {"type": "text", "text": f"frame {i}"}, _image(str(i).encode())]})
        _, history = _replay(db)
        assert evict_stale_outbound_tool_images(history[:OUTBOUND_IMAGE_LIMIT]) == 0
        assert sum(any(p.get("type") == "image_url" for p in m["content"])
                   for m in history) == OUTBOUND_IMAGE_LIMIT + 1
        assert evict_stale_outbound_tool_images(history) > 0
        _, still_retained = _replay(db)
        assert still_retained[0]["content"][1] == _image(b"0")
    finally:
        db.close()


def test_legacy_vision_claim_is_not_replayed_as_visible(tmp_path):
    db = _open(tmp_path)
    try:
        db.append_message("images", "tool", "Image loaded — you can see it natively now.",
                          tool_name="vision_analyze", tool_call_id="old")
        _, history = _replay(db)
        assert "image unavailable" in history[0]["content"]
        assert "you can see it natively now" not in history[0]["content"]
    finally:
        db.close()


def test_gateway_transcript_writer_keeps_user_image(tmp_path):
    from gateway.session_transcript import SessionTranscriptMixin

    db = _open(tmp_path)

    class Transcript(SessionTranscriptMixin):
        def _db_for_session_id(self, session_id):
            return db

    try:
        Transcript()._append_transcript_message("images", {"role": "user", "content": [
            {"type": "text", "text": "caption"}, _image(b"gateway")]})
        _, history = _replay(db)
        assert history[0]["content"] == [{"type": "text", "text": "caption"}, _image(b"gateway")]
    finally:
        db.close()
