"""Conservative native audio routing and bounded, fail-closed attachment reads."""
from __future__ import annotations

import base64
import os
from pathlib import Path
import re
import stat
from typing import Any

from agent import file_safety

MAX_AUDIO_BYTES = 20 * 1024 * 1024
_AUDIO_FORMATS = frozenset({'ogg', 'opus', 'mp3', 'wav', 'm4a', 'flac', 'aac', 'webm'})


def supports_native_audio(provider: str, model: str) -> bool:
    """Only known audio families on native or OpenAI-compatible routes; never Claude."""
    route = (provider or '').strip().lower()
    name = (model or '').strip().lower()
    if not route or 'claude' in name or route in {'anthropic', 'bedrock', 'openai-codex', 'moa'}:
        return False
    # Publisher-qualified Vertex and router identifiers share the terminal model slug.
    slug = name.rsplit('/', 1)[-1]
    if {'tts', 'image', 'embedding'} & set(slug.split('-')):
        return False
    gemini = slug.startswith('gemini-') and not slug.startswith(('gemini-1.0', 'gemini-pro-vision'))
    openai = re.fullmatch(r'(?:gpt-4o(?:-mini)?-audio-preview|gpt-audio(?:-mini)?)(?:-\d{4}-\d{2}-\d{2})?', slug)
    return bool(gemini or openai)


def decide_audio_input_mode(provider: str, model: str, cfg: dict | None) -> str:
    """``agent.audio_input_mode`` cannot force bytes onto a non-audio model."""
    agent = cfg.get('agent') if isinstance(cfg, dict) else None
    raw = agent.get('audio_input_mode') if isinstance(agent, dict) else None
    mode = raw.strip().lower() if isinstance(raw, str) else 'auto'
    return 'native' if mode != 'stt' and supports_native_audio(provider, model) else 'stt'


def supports_audio_format(model: str, suffix: str) -> bool:
    """Gemini accepts the supported containers; OpenAI audio input accepts WAV/MP3."""
    format_ = suffix.lower().lstrip('.')
    slug = (model or '').lower().rsplit('/', 1)[-1]
    return format_ in (_AUDIO_FORMATS if slug.startswith('gemini-') else {'wav', 'mp3'})


def native_audio_path_eligible(path: Path, model: str) -> bool:
    """Preflight before bypassing STT; actual attachment reads remain bounded/guarded."""
    if not supports_audio_format(model, path.suffix):
        return False
    try:
        file_safety.raise_if_read_blocked(str(path))
        return path.is_file() and 0 < path.stat().st_size <= MAX_AUDIO_BYTES
    except (ValueError, OSError):
        return False


def _read_audio_bytes(path: Path) -> bytes:
    """Read a regular audio file with the safety guard and a hard allocation ceiling."""
    if path.suffix.lower().lstrip('.') not in _AUDIO_FORMATS:
        raise ValueError('Unsupported audio format')
    try:
        file_safety.raise_if_read_blocked(str(path))
    except Exception as exc:
        raise ValueError('Audio attachment read denied') from exc
    if not path.is_file():
        raise ValueError('Audio attachment is not a regular file')
    with path.open('rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_AUDIO_BYTES:
            raise ValueError('Audio attachment must be a regular file at most 20 MiB')
        raw = stream.read(MAX_AUDIO_BYTES + 1)
    if not raw or len(raw) > MAX_AUDIO_BYTES:
        raise ValueError('Audio attachment is empty or exceeds 20 MiB')
    return raw


def _audio_part(raw: bytes, path: Path) -> dict[str, Any]:
    """Encode already-validated bytes (replay hashes these exact bytes)."""
    return {'type': 'input_audio', 'input_audio': {
        'data': base64.b64encode(raw).decode('ascii'), 'format': path.suffix.lower().lstrip('.')}}


def audio_file_part(path: Path) -> dict[str, Any]:
    """OpenAI input_audio part; invalid/blocked files raise ValueError or OSError."""
    return _audio_part(_read_audio_bytes(path), path)


def build_native_audio_content_parts(user_text: str, audio_paths: list[str]) -> tuple[list[dict], list[str]]:
    """Caption and local path hints followed by one part per readable file."""
    attached, skipped, hints = [], [], []
    for ref in audio_paths:
        try:
            part = audio_file_part(Path(ref))
        except (ValueError, OSError):
            skipped.append(str(ref))
            continue
        attached.append(part)
        hints.append(f'[Audio attached at: {ref}]')
    text = (user_text or '').strip()
    if attached:
        from agent.prompt_builder import AUDIO_ONLY_CAPTION
        text = f'{text or AUDIO_ONLY_CAPTION}\n\n' + '\n'.join(hints)
    return ([{'type': 'text', 'text': text}] if text else []) + attached, skipped
