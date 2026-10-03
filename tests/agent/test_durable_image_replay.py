"""Retained image parts survive SQLite reload and gateway replay."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from agent.image_eviction_policy import OUTBOUND_IMAGE_LIMIT
from agent.session_persistence import _db_flush_row
from gateway.run import _build_gateway_agent_history
from persistence_assembly import open_persistence, profile_scope
from persistence.conversation.messages import (
    ConversationRequest, HeldMessage, NewMessage, PrunedRowChange, ReplaceMode, TranscriptRead, TranscriptView, Watermark,
)
from tests.fixtures.storage_seed import seed_session
from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_holder import raw_holder


def _image(data=b"image"):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def _open(tmp_path):
    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, "images", "telegram")
    return db


def _persist(db, message):
    agent = SimpleNamespace(_session_db=db)
    db.messages.append_batch("images", [NewMessage.from_live(_db_flush_row(agent, message, False))])


def _replay(db):
    stored = conversation(db, "images")
    history, _ = _build_gateway_agent_history(stored)
    return stored, history


def test_first_turn_user_image_and_resumed_turn(tmp_path):
    db = _open(tmp_path)
    try:
        image = _image()
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "What is this?"}, image]})
        raw = raw_holder(db)._read_one("SELECT content, image_refs FROM messages WHERE session_id='images'")
        assert raw[0] == "What is this?\n[screenshot]"
        assert "base64" not in raw[1]
        stored, history = _replay(db)
        assert history[0]["content"] == [{"type": "text", "text": "What is this?"}, image]
        assert stored[0]["content"] == history[0]["content"]
        from agent.message_metadata import without_persistence_fields
        assert "image_refs" not in without_persistence_fields(history[0])
        db.close()
        db = open_persistence(profile_scope(home=tmp_path))
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
        refs = json.loads(raw_holder(db)._read_one("SELECT image_refs FROM messages WHERE role='tool'")[0])
        snapshot = Path(raw_holder(db).db_path).parent / "cache" / "session_images" / refs[1]["sha256"]
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
        refs = json.loads(raw_holder(db)._read_one("SELECT image_refs FROM messages")[0])
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
        message = conversation(db, "images")[0]
        message["content"][0]["text"] = "updated caption"
        db.messages.replace_transcript("images", [NewMessage.from_live(message)], ReplaceMode.DESTRUCTIVE)
        _, history = _replay(db)
        assert history[0]["content"][0] == {"type": "text", "text": "updated caption"}
        assert history[0]["content"][1] == _image()
        message = conversation(db, "images")[0]
        message["content"].pop()
        db.messages.replace_transcript("images", [NewMessage.from_live(message)], ReplaceMode.DESTRUCTIVE)
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
        db.messages.append("images", NewMessage(role="tool", content="Image loaded — you can see it natively now.",
                          tool_name="vision_analyze", tool_call_id="old"))
        _, history = _replay(db)
        assert "image unavailable" in history[0]["content"]
        assert "you can see it natively now" not in history[0]["content"]
    finally:
        db.close()


def test_gateway_transcript_writer_keeps_user_image(tmp_path):
    from gateway.session_transcript import SessionTranscriptMixin

    db = _open(tmp_path)

    class Transcript(SessionTranscriptMixin):
        def _persistence_for_session_id(self, session_id):
            return db

    try:
        Transcript()._append_transcript_message("images", {"role": "user", "content": [
            {"type": "text", "text": "caption"}, _image(b"gateway")]})
        _, history = _replay(db)
        assert history[0]["content"] == [{"type": "text", "text": "caption"}, _image(b"gateway")]
    finally:
        db.close()


def test_rewind_image_turn_pins_restored_content(tmp_path):
    # /undo pins the rehydrated list; the in-txn check must compare that, not the raw "[screenshot]" text.
    db = _open(tmp_path)
    try:
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "look"}, _image()]})
        _persist(db, {"role": "assistant", "content": "ok"})
        outcome = db.messages.rewind_user_turn("images", -1)
        assert outcome.rewound_count == 2
        assert conversation(db, "images") == []
    finally:
        db.close()


def test_image_replay_is_adopted_without_reinserting(tmp_path):
    db = _open(tmp_path)
    try:
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "caption"}, _image()]})
        replay = conversation(db, "images")[0]
        replay = {k: v for k, v in replay.items() if not k.startswith("_")}
        replay["_restored_from_store"] = True
        result = db.messages.append_batch("images", [NewMessage.from_live(replay)])
        assert result.inserted == 0
        assert len(db.messages.read_transcript("images")) == 1
        assert conversation(db, "images")[0]["content"][1] == _image()
    finally:
        db.close()


def test_image_prune_clears_sidecar_and_cas_detects_rewrite(tmp_path):
    db = _open(tmp_path)
    try:
        _persist(db, {"role": "tool", "tool_call_id": "call", "tool_name": "vision_analyze",
                      "content": [{"type": "text", "text": "caption"}, _image()]})
        before = conversation(db, "images", ConversationRequest(include_row_ids=True))[0]
        held = HeldMessage(role="tool", content=before["content"], tool_call_id="call",
                           row_id=before["_row_id"], persisted=True)
        assert db.messages.rewritten_held_row_ids("images", [held]) == ()
        replacement = {**before, "content": [before["content"][0], {"type": "text", "text": "[image pruned]"}]}
        result = db.messages.rewrite_pruned_rows("images", [PrunedRowChange(
            original=held, before=NewMessage.from_live(before), replacement=NewMessage.from_live(replacement))])
        assert result.rewritten == 1
        assert db.messages.rewritten_held_row_ids("images", [held]) == (held.row_id,)
        active = db.messages.read_transcript("images")[0]
        assert active.image_refs is None
        assert conversation(db, "images")[0]["content"] == replacement["content"]
    finally:
        db.close()


def test_compaction_claims_restored_image_tail_once(tmp_path):
    db = _open(tmp_path)
    try:
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "caption"}, _image()]})
        replay = conversation(db, "images", ConversationRequest(include_row_ids=True))[0]
        db.messages.compact("images", [NewMessage.from_live(replay)], watermark=Watermark(None))
        assert len(db.messages.read_transcript("images")) == 1
        assert conversation(db, "images")[0]["content"] == replay["content"]
    finally:
        db.close()


def test_discovery_redacts_sidecars_without_reading_images(tmp_path, monkeypatch):
    from tools.session_search_tool import _shape_message
    from persistence.sqlite.conversation._store import image_refs

    db = _open(tmp_path)
    try:
        _persist(db, {"role": "user", "content": [{"type": "text", "text": "caption"}, _image(b"secret pixels")]})
        def no_image_reads(*args):
            raise AssertionError("discovery must not load image bytes")
        monkeypatch.setattr(image_refs, "load_image", no_image_reads)
        row = db.messages.read_transcript("images")[0]
        shaped = _shape_message(row, derived=True)
        assert shaped["content"][0] == {"type": "text", "text": "caption"}
        assert shaped["content"][1]["type"] == "image_url"
        assert "c2VjcmV0IHBpeGVscw==" not in json.dumps(shaped)
        assert "sha256" not in json.dumps(shaped)
    finally:
        db.close()


def test_image_only_rewrite_invalidates_cas_and_text_edit_removes_image(tmp_path):
    db = _open(tmp_path)
    try:
        result = db.messages.append_batch("images", [NewMessage("user", [_image(b"before")])])
        row_id = result.outcomes[0].id
        before = conversation(db, "images")[0]
        held = HeldMessage(role="user", content=before["content"], row_id=row_id, persisted=True,
                           db_row_snapshot=result.outcomes[0].db_row_snapshot)
        assert db.messages.set_user_message_content("images", row_id, [_image(b"after")])
        assert db.messages.rewritten_held_row_ids("images", [held]) == (row_id,)
        assert conversation(db, "images")[0]["content"] == [_image(b"after")]
        assert db.messages.set_user_message_content("images", row_id, "image removed")
        assert db.messages.read_transcript("images")[0].image_refs is None
        assert conversation(db, "images")[0]["content"] == "image removed"
    finally:
        db.close()


def test_distinct_images_with_equal_text_and_timestamp_stay_distinct(tmp_path):
    db = _open(tmp_path)
    try:
        db.messages.append("images", NewMessage("user", [_image(b"first")], timestamp=100.0))
        db.messages.append("images", NewMessage("user", [_image(b"second")], timestamp=100.0))
        shown = db.messages.read_transcript("images", TranscriptRead(view=TranscriptView.DISPLAY))
        assert len(shown) == 2
        assert shown[0].image_refs != shown[1].image_refs
    finally:
        db.close()


def test_low_detail_survives_replay_and_corruption_fails_closed(tmp_path):
    db = _open(tmp_path)
    try:
        image = _image()
        image["image_url"]["detail"] = "low"
        _persist(db, {"role": "user", "content": [image]})
        assert conversation(db, "images")[0]["content"] == [image]
        row = db.messages.read_transcript("images")[0]
        digest = json.loads(row.image_refs)[0]["sha256"]
        (tmp_path / "cache" / "session_images" / digest).write_bytes(b"corrupt")
        assert conversation(db, "images")[0]["content"] == [{"type": "text", "text": "[image unavailable]"}]
    finally:
        db.close()
