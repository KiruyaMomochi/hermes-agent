"""Load audio into the active main model, or return a speech transcript.

Native audio is a tool-result attachment, not an auxiliary model request. Inbound
voice routing preferences do not disable explicit requests to listen to a file.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tools.prompt_text import tool_text
from tools.registry import registry, tool_error


_AUDIO_SUFFIXES = frozenset({".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".webm"})
_MAX_AUDIO_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class _AudioRequest:
    path: Path
    question: str

    @classmethod
    def parse(cls, args: dict[str, Any]) -> "_AudioRequest":
        source = args.get("audio_path", "")
        question = args.get("question", "")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("audio_path is required and must be a local file path.")
        if not isinstance(question, str):
            raise ValueError("question must be a string.")
        return cls(Path(source).expanduser(), question.strip())


def _validate_audio_path(path: Path) -> None:
    # Guard failures must propagate: never send an unchecked file to any backend.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(str(path))
    if path.suffix.lower() not in _AUDIO_SUFFIXES:
        raise ValueError("Unsupported audio format. Supported: ogg, opus, mp3, wav, m4a, flac, aac, webm.")
    if not path.is_file():
        raise ValueError("Audio path must name an existing regular file.")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("Audio file is empty.")
    if size > _MAX_AUDIO_BYTES:
        raise ValueError("Audio file exceeds the 20 MiB size limit. Trim or compress it and retry.")


def _native_audio_result(request: _AudioRequest, retained: Path, audio_part: dict) -> dict[str, Any]:
    text = tool_text(
        "tool_results.audio_listen.native",
        "Audio loaded into your context. Listen with your built-in audio understanding and answer the user.",
    )
    if request.question:
        text += "\n\n" + tool_text("tool_results.audio_listen.question", "Question: {question}").format(
            question=request.question,
        )
    text += f"\n\n[Audio attached at: {retained}]"
    return {
        "_multimodal": True,
        "content": [{"type": "text", "text": text}, audio_part],
        # Persistence uses this instead of content: preserve the exact text and
        # marker projection so a resumed request reconstructs the same prefix.
        "text_summary": text + "\n[audio]",
    }


def _transcript_result(path: Path) -> str:
    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio(str(path), source="audio_listen")
    if not result.get("success"):
        raise ValueError(result.get("error") or "Audio transcription failed.")
    transcript = result.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Audio transcription returned no speech text.")
    return json.dumps({
        "success": True,
        "transcript": transcript,
        "note": tool_text(
            "tool_results.audio_listen.transcript",
            "Native audio is unavailable for this model or file format. Speech transcription only; non-speech audio was not analyzed.",
        ),
    }, ensure_ascii=False)


def _handle_audio_listen(args: dict[str, Any], **kwargs: Any) -> str | dict[str, Any]:
    from agent.audio_routing import audio_file_part, supports_audio_format, supports_native_audio
    from agent.auxiliary_client import _runtime_main_value

    try:
        request = _AudioRequest.parse(args)
        _validate_audio_path(request.path)
        model = _runtime_main_value("model")
        if (supports_native_audio(_runtime_main_value("provider"), model)
                and supports_audio_format(model, request.path.suffix)):
            from gateway.session_ingress_media import capture_native_media

            retained = Path(capture_native_media([request.path.absolute()])[0]["path"])
            return _native_audio_result(request, retained, audio_file_part(retained))
        return _transcript_result(request.path)
    except Exception as exc:
        return tool_error(tool_text("tool_results.audio_listen.error", "Could not listen to audio: {error}").format(
            error=str(exc),
        ), success=False)


AUDIO_LISTEN_SCHEMA = {
    "name": "audio_listen",
    "description": (
        "Listen to local audio and answer questions about speech, music, sounds, or tone. "
        "When native audio is unavailable, returns speech transcription only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Local audio file path (ogg, opus, mp3, wav, m4a, flac, aac, webm); maximum 20 MiB.",
            },
            "question": {
                "type": "string",
                "description": "Optional question or listening instructions about the audio.",
                "default": "",
            },
        },
        "required": ["audio_path"],
    },
}


registry.register(
    name="audio_listen",
    toolset="vision",
    schema=AUDIO_LISTEN_SCHEMA,
    handler=_handle_audio_listen,
    emoji="🎧",
)
