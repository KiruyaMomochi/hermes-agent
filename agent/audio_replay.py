"""Wire-only native audio replay from profile-owned, digest-validated retained media.

Batch frontiers match image replay so old cached prefixes change only once per batch.
The image hook runs first; its list text runs can still contain durable audio markers.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
from typing import Any

from agent.audio_routing import _audio_part, _read_audio_bytes, decide_audio_input_mode, supports_audio_format, supports_native_audio
from agent.image_replay import replay_frontier

logger = logging.getLogger(__name__)
DEFAULT_REPLAY_RECENT_AUDIO = 3
_AUDIO_TYPES = frozenset({'audio', 'input_audio'})
_HINT_RE = re.compile(r'^\[Audio attached at: (.+)\]$', re.MULTILINE)
_MARKER_RE = re.compile(r'^\[audio\]$', re.MULTILINE)


def resolve_replay_recent_audio(cfg: dict) -> int:
    """Non-negative audio.replay_recent_audio (default 3, malformed values disable)."""
    audio = cfg.get('audio') if isinstance(cfg, dict) else None
    raw = audio.get('replay_recent_audio', DEFAULT_REPLAY_RECENT_AUDIO) if isinstance(audio, dict) else None
    if raw is None:
        return DEFAULT_REPLAY_RECENT_AUDIO
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    logger.warning('audio.replay_recent_audio=%r is not a non-negative integer; replay disabled', raw)
    return 0


def _retained_audio_part(ref: str) -> dict | None:
    """Never replay arbitrary paths, cross-profile paths, symlinks, or changed bytes."""
    from gateway.session_ingress_media import native_inputs_root
    path = Path(ref)
    try:
        root = native_inputs_root()
        if (not path.is_absolute() or path.parent.parent != root
                or path.parent.is_symlink() or path.is_symlink()
                or path.resolve().parent.parent != root
                or re.fullmatch(r'[0-9a-f]{64}', path.parent.name) is None):
            return None
        raw = _read_audio_bytes(path)
        if hashlib.sha256(raw).hexdigest() != path.parent.name:
            return None
    except (ValueError, OSError):
        return None
    return _audio_part(raw, path)


def _row_parts(content: Any) -> tuple[list[dict], list[str]]:
    """Content parts and validated 1:1 marker hints, including mixed image/audio rows."""
    if isinstance(content, dict) and content.get('_multimodal') is True:
        content = content.get('content')
    parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
    if not isinstance(parts, list):
        return [], []
    texts = [p['text'] for p in parts if isinstance(p, dict) and p.get('type') == 'text'
             and isinstance(p.get('text'), str)]
    hints = _HINT_RE.findall('\n'.join(texts))
    markers = sum(len(_MARKER_RE.findall(text)) for text in texts)
    return parts, hints if markers and markers == len(hints) else []


def _rebuild_row(parts: list[dict], hints: list[str], slot: int, frontier: int, model: str = '') -> tuple[list[dict], int]:
    rebuilt, attached = [], 0
    refs = iter(hints)
    for part in parts:
        if not isinstance(part, dict) or part.get('type') != 'text' or not isinstance(part.get('text'), str):
            rebuilt.append(part)
            if isinstance(part, dict) and part.get('type') in _AUDIO_TYPES:
                slot += 1
            continue
        text = part['text']
        start = 0
        for marker in _MARKER_RE.finditer(text):
            ref = next(refs)
            eligible = not model or supports_audio_format(model, Path(ref).suffix)
            audio = _retained_audio_part(ref) if slot >= frontier and eligible else None
            slot += 1
            if audio is None:
                continue
            # Durable projection joins parts with a newline; consume just those separators.
            before = text[start:marker.start()].removesuffix('\n')
            if before:
                rebuilt.append({**part, 'text': before})
            rebuilt.append(audio)
            attached += 1
            start = marker.end() + (text[marker.end():marker.end() + 1] == '\n')
        if start == 0:
            rebuilt.append(part)
        elif start < len(text):
            rebuilt.append({**part, 'text': text[start:]})
    return rebuilt, attached


def replay_recent_audio(api_messages: list[dict[str, Any]], keep: int, *, replay_users: bool = True, model: str = '') -> int:
    """Reattach audio in place on wire copies only; live parts also hold frontier slots."""
    if keep <= 0:
        return 0
    slots, rows = 0, []
    for msg in api_messages:
        if msg.get('role') not in {'user', 'tool'}:
            continue
        parts, hints = _row_parts(msg.get('content'))
        rows.append((msg, parts, hints, slots))
        slots += len(hints) + sum(isinstance(p, dict) and p.get('type') in _AUDIO_TYPES for p in parts)
    frontier = replay_frontier(slots, keep)
    attached = 0
    for msg, parts, hints, slot in rows:
        if not hints or (msg.get('role') == 'user' and not replay_users):
            continue
        rebuilt, count = _rebuild_row(parts, hints, slot, frontier, model)
        if count:
            msg['content'] = rebuilt
            attached += count
    return attached


def _turn_replay_window(agent: Any) -> tuple[int, bool]:
    """Freeze config for one admitted turn, but re-evaluate on a fallback/model switch."""
    provider, model = getattr(agent, 'provider', '') or '', getattr(agent, 'model', '') or ''
    key = (getattr(agent, '_current_turn_timestamp', None), provider, model)
    cached = getattr(agent, '_audio_replay_window', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly()
    # Inbound STT preference does not override an explicit audio_listen request.
    window = (resolve_replay_recent_audio(cfg), decide_audio_input_mode(provider, model, cfg) == 'native')
    agent._audio_replay_window = (key, window)
    return window


def replay_for_agent(agent: Any, api_messages: list[dict[str, Any]]) -> None:
    """build_api_messages hook; strip live audio after a non-capable model switch."""
    model = getattr(agent, 'model', '')
    native = supports_native_audio(getattr(agent, 'provider', ''), model)
    for msg in api_messages:
        content = msg.get('content')
        if msg.get('role') == 'user' and isinstance(content, list):
            msg['content'] = [
                {'type': 'text', 'text': '[audio]'}
                if (isinstance(p, dict) and p.get('type') in _AUDIO_TYPES
                    and (not native or not supports_audio_format(model, p.get('input_audio', {}).get('format', ''))))
                else p for p in content]
    if not native:
        return
    if not any(_row_parts(msg.get('content'))[1] for msg in api_messages if msg.get('role') in {'user', 'tool'}):
        return
    keep, replay_users = _turn_replay_window(agent)
    if keep > 0:
        replay_recent_audio(api_messages, keep, replay_users=replay_users, model=model)
