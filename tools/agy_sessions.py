"""On-demand, read-only Antigravity transcripts. No store, sync, or tool registration.

Only visible user/planner messages survive replay; step indices remain scroll anchors.
The cache holds immutable snapshots keyed by the transcript's mtime_ns and size.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from itertools import zip_longest
from datetime import datetime, UTC
from pathlib import Path
import sqlite3
from typing import Callable


@dataclass(frozen=True)
class Message:
    id: int
    role: str
    content: str
    timestamp: float | None


@dataclass(frozen=True)
class Conversation:
    session_id: str
    title: str
    started_at: float
    last_activity: float
    messages: tuple[Message, ...]


_CACHE: dict[Path, tuple[tuple[int, int], Conversation]] = {}
_USER_REQUEST = re.compile(r"<USER_REQUEST>\n?(.*?)\n?</USER_REQUEST>", re.DOTALL)
_MEDIA_LINE = re.compile(r"^MEDIA:\S+\s*$", re.MULTILINE)
_LOCAL_TIME = re.compile(r"The current local time is:\s*([0-9T:+-]+)")


def app_dirs() -> list[Path]:
    """An explicit override accepts app directories or their .gemini parent."""
    override = os.environ.get("HERMES_AGY_DIRS")  # health: allow HX002 -- explicit legacy path override required by the local agy integration
    roots = [Path(p).expanduser() for p in override.split(os.pathsep) if p] if override is not None else [
        Path.home() / ".gemini", *(Path("/home").glob("*/.gemini") if Path("/home").is_dir() else ())]
    found = set()
    for root in roots:
        candidates = [root] if root.name.startswith("antigravity") or (root / "brain").is_dir() else sorted(root.glob("antigravity*"))
        found.update(p.resolve() for p in candidates if p.is_dir() and os.access(p, os.R_OK | os.X_OK))
    return sorted(found)


def _metadata_stamp(db: Path) -> tuple:
    stamps = []
    for path in (db, db.with_name(db.name + "-wal")):
        try:
            stat = path.stat()
            stamps.append((stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            stamps.append(None)
    return tuple(stamps)


@contextmanager
def _metadata_reader(db: Path):
    """Never let SQLite create/modify WAL shared memory in the application's dir.

    Immutable reads ignore WAL. Only for a pending WAL, read a disposable copy of
    existing metadata + WAL instead (not a transcript index), then delete it.
    """
    wal = db.with_name(db.name + "-wal")
    if not wal.exists() or not wal.stat().st_size:
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True)) as con:
            yield con
        return
    with tempfile.TemporaryDirectory(prefix="agy-metadata-") as directory:
        snapshot = Path(directory) / db.name
        stamp = _metadata_stamp(db)
        shutil.copyfile(db, snapshot)
        shutil.copyfile(wal, snapshot.with_name(snapshot.name + "-wal"))
        if stamp != _metadata_stamp(db):
            raise OSError("Antigravity metadata changed during snapshot; retry on next call")
        with closing(sqlite3.connect(snapshot)) as con:
            yield con


_METADATA_CACHE: dict[Path, tuple[tuple, tuple[dict[str, str], set[str]]]] = {}


def _metadata(app: Path) -> tuple[dict[str, str], set[str]] | None:
    titles, hidden = {}, set()
    db = app / "conversation_summaries.db"
    if not db.is_file():
        return titles, hidden
    try:
        stamp = _metadata_stamp(db)
        cached = _METADATA_CACHE.get(db)
        if cached and cached[0] == stamp:
            return cached[1]
        with _metadata_reader(db) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "conversation_summaries" in tables:
                con.row_factory = sqlite3.Row
                for row in con.execute("SELECT * FROM conversation_summaries"):
                    row = dict(row)
                    cid = row.get("conversation_id")
                    if row.get("title"):
                        titles[cid] = row["title"]
                    if row.get("parent_conversation_id") or (row.get("nesting_depth") or 0) > 0:
                        hidden.add(cid)
            if "conversation_parents" in tables:
                hidden.update(r[0] for r in con.execute(
                    "SELECT conversation_id FROM conversation_parents WHERE coalesce(parent_id, '') != ''"))
        if stamp != _metadata_stamp(db):
            raise OSError("Antigravity metadata changed while reading; retry on next call")
        _METADATA_CACHE[db] = (stamp, (titles, hidden))
    except (OSError, sqlite3.Error) as exc:
        logging.debug("Cannot read Antigravity metadata %s: %s", db, exc)
        return None  # Existing but unreadable metadata must not expose subagents.
    return titles, hidden


def _timestamp(step: dict) -> float | None:
    match = _LOCAL_TIME.search(str(step.get("content") or ""))
    raw = match.group(1) if match else step.get("created_at")
    try:
        if isinstance(raw, (int, float)):
            return float(raw)
        if not isinstance(raw, str):
            return None
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=UTC).timestamp() if dt.tzinfo is None else dt.timestamp()
    except (ValueError, AttributeError, OverflowError):
        return None


def _message(step: dict) -> Message | None:
    text = step.get("content") or ""
    if not isinstance(text, str):
        return None
    if step.get("source") == "USER_EXPLICIT" and step.get("type") == "USER_INPUT":
        match = _USER_REQUEST.search(text)
        text = (match.group(1) if match else text).strip()
        if any(isinstance(m, dict) and str(m.get("mime_type", "")).startswith("audio/")
               for m in (step.get("media") or [])):
            text = (text + "\n[語音訊息]").lstrip()
        role = "user"
    elif step.get("source") == "MODEL" and step.get("type") == "PLANNER_RESPONSE":
        text, role = _MEDIA_LINE.sub("", text).strip(), "assistant"
    else:
        return None
    return Message(step["step_index"], role, text, _timestamp(step)) if text else None


def _load(path: Path) -> Conversation | None:
    try:
        stat = path.stat()
        fingerprint = (stat.st_mtime_ns, stat.st_size)
        cached = _CACHE.get(path)
        if cached and cached[0] == fingerprint:
            return cached[1]
        timeline = []
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    step = json.loads(line)
                except ValueError:
                    continue  # A live writer may have left an incomplete trailing row.
                if not isinstance(step, dict) or type(step.get("step_index")) is not int:
                    continue
                idx = step["step_index"]
                while timeline and timeline[-1]["step_index"] >= idx:
                    timeline.pop()
                timeline.append(step)
        messages = tuple(m for s in timeline if (m := _message(s)) is not None)
        times = [m.timestamp for m in messages if m.timestamp is not None]
        conversation = Conversation("agy:" + path.parts[-4], "", min(times, default=stat.st_mtime),
                                    max([stat.st_mtime, *times]), messages)
        _CACHE[path] = (fingerprint, conversation)
        return conversation
    except (OSError, UnicodeError) as exc:
        logging.debug("Cannot read Antigravity transcript %s: %s", path, exc)
        return None


def conversations(session_id: str | None = None) -> list[Conversation]:
    # Never interpolate an untrusted session id into a filesystem path.
    cid = session_id[4:] if session_id else None
    if cid is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", cid):
        return []
    found = {}
    for app in app_dirs():
        metadata = _metadata(app)
        if metadata is None:
            continue
        titles, hidden = metadata
        paths = ([app / "brain" / cid / ".system_generated/logs/transcript_full.jsonl"] if cid is not None
                 else app.glob("brain/*/.system_generated/logs/transcript_full.jsonl"))
        for path in paths:
            if path.parts[-4] in hidden:
                continue
            conv = _load(path)
            if conv is None or not conv.messages:
                continue
            conv = replace(conv, title=titles.get(path.parts[-4]) or conv.messages[0].content[:100])
            previous = found.get(conv.session_id)
            if previous is None or conv.last_activity > previous.last_activity:
                found[conv.session_id] = conv
    return list(found.values())


def _shape(message: Message, cap: int | None = None, anchor=None, redact=lambda x: x) -> dict:
    content = redact(message.content)
    result = {"id": message.id, "message_id": message.id, "role": message.role,
              "content": content, "timestamp": message.timestamp}
    if message.id == anchor:
        result["anchor"] = True
    if cap and len(content) > cap:
        result.update(content=content[:cap] + "…", content_truncated=True, original_content_chars=len(content))
    return result


def read(session_id: str, around_message_id=None, window=5) -> str:
    matches = conversations(session_id)
    if not matches:
        return json.dumps({"success": False, "error": f"session_id not found: {session_id}"})
    conv = matches[0]
    messages, total = conv.messages, len(conv.messages)
    result = {"success": True, "session_id": session_id, "source": "agy", "title": conv.title,
              "started_at": conv.started_at, "last_activity": conv.last_activity, "message_count": total,
              "link": f"@session:default/{session_id}",
              "session_meta": {"when": conv.started_at, "source": "agy", "title": conv.title, "model": "unknown"}}
    if around_message_id is None:
        truncated = total > 30
        result.update(mode="read", truncated=truncated,
                      messages=[_shape(m, 2000) for m in (messages[:20] + messages[-10:] if truncated else messages)])
        if truncated:
            result["message"] = f"Session has {total} messages; showing first 20 + last 10. Pass around_message_id to scroll."
    else:
        try:
            anchor = int(around_message_id)
        except (ValueError, TypeError):
            return json.dumps({"success": False, "error": "scroll requires integer around_message_id"})
        try:
            window = max(1, min(int(window), 20))
        except (ValueError, TypeError):
            window = 5
        center = min(range(total), key=lambda i: abs(messages[i].id - anchor))
        start, end = max(0, center - window), min(total, center + window + 1)
        result.update(mode="scroll", around_message_id=messages[center].id, window=window,
                      messages=[_shape(m, anchor=messages[center].id) for m in messages[start:end]],
                      messages_before=start, messages_after=total - end,
                      hint="Scroll forward with the last message's id; backward with the first message's id.")
    return json.dumps(result, ensure_ascii=False)


_TOKEN = re.compile(r'"(?:[^"]|"")*"\*?|\(|\)|[^\s()]+')


def _query_tree(query: str):
    """Small FTS-like boolean grammar: NOT binds tighter than AND, then OR."""
    tokens = _TOKEN.findall(query)
    position = 0

    def atom():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("missing search term")
        token = tokens[position]
        position += 1
        if token == "(":
            node = expression()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("unclosed search group")
            position += 1
            return node
        if token in ("AND", "OR", "NOT", ")"):
            raise ValueError("unexpected search operator")
        prefix = token.endswith("*")
        text = (token[:-1] if prefix else token).strip('"').replace('""', '"')
        if not text:
            raise ValueError("empty search term")
        # CJK words use substring semantics; Latin words retain token boundaries.
        cjk = any("\u2e80" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" for ch in text)
        pattern = re.escape(text)
        if not cjk:
            pattern = r"(?<!\w)" + pattern + ("" if prefix else r"(?!\w)")
        return re.compile(pattern, re.IGNORECASE)

    def conjunction():
        nonlocal position
        nodes = [atom()]
        while position < len(tokens) and tokens[position] not in ("OR", ")"):
            op = tokens[position]
            if op in ("AND", "NOT"):
                position += 1
            node = atom()
            nodes.append(("not", node) if op == "NOT" else node)
        return ("and", nodes)

    def expression():
        nonlocal position
        nodes = [conjunction()]
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            nodes.append(conjunction())
        return ("or", nodes)

    tree = expression()
    if position != len(tokens):
        raise ValueError("unexpected search group")
    return tree


def _matches(tree, text: str) -> bool:
    if isinstance(tree, re.Pattern):
        return tree.search(text) is not None
    op, nodes = tree
    if op == "not":
        return not _matches(nodes, text)
    results = (_matches(node, text) for node in nodes)
    return all(results) if op == "and" else any(results)


def _browse(conv: Conversation, redact: Callable) -> dict:
    return {"session_id": conv.session_id, "source": "agy", "title": redact(conv.title),
            "started_at": conv.started_at, "last_active": conv.last_activity, "last_activity": conv.last_activity,
            "message_count": len(conv.messages), "preview": redact(conv.messages[0].content)[:200],
            "link": f"@session:default/{conv.session_id}"}


def _discover(conv: Conversation, center: int, redact: Callable) -> dict:
    messages, anchor = conv.messages, conv.messages[center]
    start, end = max(0, center - 5), min(len(messages), center + 6)
    return {**_browse(conv, redact), "when": conv.started_at, "model": "unknown",
            "matched_role": anchor.role, "match_message_id": anchor.id,
            "snippet": redact(anchor.content)[:300],
            "bookend_start": [_shape(m, 1200, redact=redact) for m in messages[:3]],
            "messages": [_shape(m, 4000, anchor.id, redact) for m in messages[start:end]],
            "bookend_end": [_shape(m, 1200, redact=redact) for m in messages[-3:]],
            "messages_before": start, "messages_after": len(messages) - end, "detail": "full"}


def merge(payload: str, *, limit: int, redact: Callable, started: Callable, hydrate: Callable, role_filter=None,
          sort=None, after_ts=None, before_ts=None, exclude_session_ids=None, current_session_id=None) -> str:
    """Merge bounded native results, retaining native ranking within its source.

    BM25 scores cannot be compared to unindexed transcripts: relevance uses a stable
    round-robin merge, with automation still demoted. Temporal sort is global.
    """
    data = json.loads(payload)
    if not data.get("success"):
        return payload
    browse = data["mode"] == "browse"
    available = conversations()
    if not available:
        return payload
    tree = None
    if not browse:
        try:
            tree = _query_tree(data["query"])
        except ValueError:
            return payload  # Native FTS owns syntax errors/fallback behavior.
    roles = {r.strip() for r in role_filter.split(",") if r.strip()} if isinstance(role_filter, str) else set()
    roles = roles or {"user", "assistant"}
    excluded = {s.rsplit("/", 1)[-1].removeprefix("@session:") for s in (exclude_session_ids or [])}
    external = []
    for conv in available:
        if conv.session_id in excluded or conv.session_id == current_session_id:
            continue
        if (after_ts is not None and conv.started_at < after_ts) or (before_ts is not None and conv.started_at >= before_ts):
            continue
        if browse:
            external.append(_browse(conv, redact))
            continue
        center = next((i for i, m in enumerate(conv.messages) if m.role in roles and _matches(tree, m.content)), None)
        if center is not None:
            external.append(_discover(conv, center, redact))
    native = data["results"]
    if not external:
        return payload
    external.sort(key=lambda r: r["last_activity"], reverse=True)
    # Interleave without starving either source at the usual limit=3.
    combined = [r for pair in zip_longest(native, external) for r in pair if r is not None]
    if browse:
        combined.sort(key=lambda r: r.get("last_active") or r.get("started_at") or 0, reverse=True)
    elif sort in ("newest", "oldest"):
        combined.sort(key=lambda r: (r["started_at"] if r["source"] == "agy" else started(r)) or 0,
                      reverse=sort == "newest")
    if not browse:
        combined.sort(key=lambda r: r.get("source") == "cron")
    results = combined[:limit]
    if not browse and data.get("detail") != "full":
        for index, entry in enumerate(results):
            if not index and entry.get("detail") != "full":
                full = hydrate(entry)
                if full is not None:
                    entry.update(full)
            if index:
                entry.update(detail="compact", bookend_start=[], bookend_end=[],
                             messages=[m for m in entry["messages"] if m["id"] == entry["match_message_id"]])
    data.update(results=results, count=len(results))
    if browse:
        data["message"] = f"Showing {len(results)} most recent sessions. Pass query= to search, or session_id to read."
    else:
        data.pop("message", None)
        data["sessions_searched"] = data.get("sessions_searched", 0) + len(external)
    return json.dumps(data, ensure_ascii=False)
