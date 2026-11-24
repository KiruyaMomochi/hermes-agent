"""Session-local inbound audio routing; retention is owned by the generic admission capture."""
from __future__ import annotations

import logging
from pathlib import Path

from agent.audio_routing import build_native_audio_content_parts, decide_audio_input_mode, native_audio_path_eligible
from hermes_cli.config import load_config

logger = logging.getLogger(__name__)


def event_audio_paths(event) -> list[str]:
    """Voice notes with missing/octet-stream MIME still carry native audio."""
    from gateway.run import _event_media_is_audio, _event_media_is_stt_input
    return [path for i, path in enumerate(event.media_urls or [])
            if _event_media_is_audio(event, i) or _event_media_is_stt_input(event, i)]


def native_audio_event(runner, event, session_key: str) -> bool:
    """Resolve the upcoming turn's route, never the process's last-used main model."""
    paths = event_audio_paths(event)
    if not paths:
        return False
    try:
        cfg = load_config()
        model, runtime = runner._resolve_session_agent_runtime(
            source=event.source, session_key=session_key, user_config=cfg, record_notice=False,
        )
        return (decide_audio_input_mode(runtime.get('provider') or '', model, cfg) == 'native'
                and all(native_audio_path_eligible(Path(path), model) for path in paths))
    except Exception as exc:
        logger.debug('Audio route resolution failed; using STT: %s', exc)
        return False


def consume_audio_paths(runner, session_key: str) -> list[str]:
    state = runner._peek_session_state(session_key)
    if state is None:
        return []
    paths = state.persistent.native_audio_paths
    state.persistent.native_audio_paths = []
    return paths


def append_audio_parts(parts: list[dict], text: str, audio_paths: list[str]) -> list[dict]:
    """Compose with native images without creating a second text part or losing path hints."""
    caption = parts[0]['text'] if parts and parts[0].get('type') == 'text' else text
    audio, skipped = build_native_audio_content_parts(caption, audio_paths)
    if skipped:
        logger.warning('Native audio attachment: skipped %d unreadable path(s)', len(skipped))
        # Files may disappear after preflight. Never turn a failed clip into a silent empty turn.
        notice = '\n'.join(f'[Audio could not be loaded: {path}]' for path in skipped)
        audio.insert(0, {'type': 'text', 'text': notice})
    if not any(part.get('type') == 'input_audio' for part in audio):
        return [*parts, *audio] if skipped else parts
    return [*audio[:1], *(parts[1:] if parts and parts[0].get('type') == 'text' else parts), *audio[1:]]
