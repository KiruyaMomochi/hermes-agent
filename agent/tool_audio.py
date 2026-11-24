"""Wire-only audio tool delivery; canonical tool results remain provider-neutral.

OpenAI Chat Completions accepts input_audio on user messages, not tool results.
Keep the tool pairing intact and append its audio after the complete result batch.
Native Gemini instead accepts audio beside functionResponse on its user content.
"""
from __future__ import annotations

from typing import Any

from agent.audio_routing import supports_audio_format, supports_native_audio
from agent.gemini_native_adapter import is_native_gemini_route
from session_rules.turns import _is_multimodal_tool_result, _multimodal_text_summary

_AUDIO_TYPES = frozenset({"audio", "input_audio"})


def _audio_parts(content: Any) -> tuple[list, list]:
    parts = content.get("content") if _is_multimodal_tool_result(content) else content
    if not isinstance(parts, list):
        return [], []
    audio = [p for p in parts if isinstance(p, dict) and p.get("type") in _AUDIO_TYPES]
    return parts, audio


def prepare_tool_audio_for_api(agent: Any, api_messages: list[dict[str, Any]]) -> None:
    """Transform a wire copy AFTER audio replay; never mutate canonical messages.

    One attachment user row follows each contiguous tool-result batch, so parallel
    results all satisfy their calls before user audio starts. Subsequent user input
    is merged on the wire to avoid adjacent user roles. Gemini keeps audio in the
    tool list for its adapter; non-audio models get text, never raw media bytes.
    """
    native = supports_native_audio(getattr(agent, "provider", ""), getattr(agent, "model", ""))
    gemini = native and is_native_gemini_route(
        getattr(agent, "base_url", ""), getattr(agent, "provider", ""),
    )
    rebuilt: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for msg in api_messages:
        if msg.get("role") != "tool" and pending:
            if msg.get("role") == "user":
                content = msg.get("content")
                parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
                msg = {**msg, "content": [*pending, *parts]}
            else:
                rebuilt.append({"role": "user", "content": pending})
            pending = []
        content = msg.get("content")
        parts, audio = _audio_parts(content) if msg.get("role") == "tool" else ([], [])
        if audio:
            can_attach = native and all(supports_audio_format(
                getattr(agent, 'model', ''), part.get('input_audio', {}).get('format', ''),
            ) for part in audio)
            if gemini:
                msg = {**msg, "content": parts}
            else:
                text = _multimodal_text_summary({"_multimodal": True, "content": [
                    p for p in parts if not (isinstance(p, dict) and p.get("type") in _AUDIO_TYPES)
                ]})
                if not can_attach:
                    text = (text + "\n\n" if text else "") + (
                        "[Audio omitted: native audio is unavailable for this model or format. "
                        "Use audio_listen again for speech transcription only.]"
                    )
                msg = {**msg, "content": text or "[audio]"}
                if can_attach:
                    pending.extend([{"type": "text", "text": f"Audio from tool call {msg.get('tool_call_id', '')}:"}, *audio])
        rebuilt.append(msg)
    if pending:
        rebuilt.append({"role": "user", "content": pending})
    api_messages[:] = rebuilt
