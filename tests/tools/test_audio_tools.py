"""Audio listening contracts through the real registry and profile-local wording."""

import base64
import json
from pathlib import Path
import wave
from unittest.mock import Mock

import pytest

from agent.auxiliary_client import scoped_runtime_main
from hermes_yaml import safe_dump
from tools import audio_tools
from tools.prompt_text import apply_tool_text
from tools.registry import registry


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 160)
    return path


@pytest.fixture
def transcriber(monkeypatch):
    from tools import transcription_tools

    transcribe = Mock(return_value={"success": True, "transcript": "Hello, 世界."})
    monkeypatch.setattr(transcription_tools, "transcribe_audio", transcribe)
    return transcribe


def _dispatch(path, question=""):
    return registry.dispatch("audio_listen", {"audio_path": str(path), "question": question})


def test_native_audio_attaches_bytes_to_main_runtime(audio_path, transcriber, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Explicit inbound STT preference does not override an explicit listen request.
    (tmp_path / "config.yaml").write_text(safe_dump({"agent": {"audio_input_mode": "stt"}}))
    with scoped_runtime_main({"provider": "google", "model": "gemini-2.5-flash"}):
        result = _dispatch(audio_path, "What can you hear?")
    assert result["_multimodal"] is True
    assert [part["type"] for part in result["content"]] == ["text", "input_audio"]
    assert "What can you hear?" in result["content"][0]["text"]
    payload = result["content"][1]["input_audio"]
    assert payload["format"] == "wav"
    assert base64.b64decode(payload["data"]) == audio_path.read_bytes()
    from agent.audio_replay import replay_recent_audio
    from session_rules.turns import durable_content

    text = result["content"][0]["text"]
    retained = Path(text.split("[Audio attached at: ", 1)[1].split("]", 1)[0])
    assert retained != audio_path
    assert retained.read_bytes() == audio_path.read_bytes()
    stored = durable_content(result)
    assert stored == text + "\n[audio]"
    audio_path.unlink()
    rows = [{"role": "tool", "tool_call_id": "listen", "content": stored}]
    assert replay_recent_audio(rows, 3) == 1
    assert rows[0]["content"] == result["content"]
    transcriber.assert_not_called()


def test_main_runtime_is_context_local_and_claude_uses_stt(audio_path, transcriber):
    with scoped_runtime_main({"provider": "google", "model": "gemini-2.5-flash"}):
        with scoped_runtime_main({"provider": "anthropic", "model": "claude-sonnet-4-5"}):
            result = json.loads(_dispatch(audio_path))
        native = _dispatch(audio_path)
    assert result["success"] is True
    assert result["transcript"] == "Hello, 世界."
    assert "non-speech" in result["note"]
    assert native["_multimodal"] is True
    transcriber.assert_called_once_with(str(audio_path), source="audio_listen")


def test_unbound_runtime_does_not_use_launch_model(audio_path, transcriber, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    with scoped_runtime_main(None):
        result = json.loads(_dispatch(audio_path))
    assert result["transcript"] == "Hello, 世界."
    transcriber.assert_called_once()


def test_openai_unsupported_container_uses_stt(audio_path, transcriber):
    renamed = audio_path.with_suffix('.ogg')
    audio_path.rename(renamed)
    with scoped_runtime_main({'provider': 'openai', 'model': 'gpt-audio'}):
        result = json.loads(_dispatch(renamed))
    assert result['success'] is True
    assert result['transcript'] == 'Hello, 世界.'
    transcriber.assert_called_once_with(str(renamed), source='audio_listen')


@pytest.mark.parametrize("suffix", ["ogg", "opus", "mp3", "wav", "m4a", "flac", "aac", "webm"])
def test_supported_formats_reach_stt(tmp_path, transcriber, suffix):
    path = tmp_path / f"audio.{suffix.upper()}"
    path.write_bytes(b"audio")
    with scoped_runtime_main({"provider": "anthropic", "model": "claude-sonnet-4-5"}):
        assert json.loads(_dispatch(path))["success"] is True
    transcriber.assert_called_once_with(str(path), source="audio_listen")


@pytest.mark.parametrize("provider,model", [("anthropic", "claude-sonnet-4-5"), ("google", "gemini-2.5-flash")])
@pytest.mark.parametrize("kind", ["missing", "directory", "unsupported", "empty", "oversized", "secret"])
def test_invalid_files_fail_before_native_or_stt(tmp_path, transcriber, monkeypatch, provider, model, kind):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import audio_routing

    encode = Mock(side_effect=AssertionError("Unsafe input reached encoder"))
    monkeypatch.setattr(audio_routing, "audio_file_part", encode)
    path = tmp_path / "input.wav"
    if kind == "directory":
        path.mkdir()
    elif kind == "unsupported":
        path = tmp_path / "input.txt"
        path.write_bytes(b"not audio")
    elif kind == "empty":
        path.touch()
    elif kind == "oversized":
        with path.open("wb") as handle:
            handle.truncate(20 * 1024 * 1024 + 1)
    elif kind == "secret":
        secret = tmp_path / "auth.json"
        secret.write_text('{"api_key":"never disclose"}')
        path.symlink_to(secret)
    with scoped_runtime_main({"provider": provider, "model": model}):
        result = json.loads(_dispatch(path))
    assert result["success"] is False
    assert result["error"]
    transcriber.assert_not_called()
    encode.assert_not_called()


def test_file_safety_guard_failure_is_closed(audio_path, transcriber, monkeypatch):
    from agent import file_safety

    guard = Mock(side_effect=RuntimeError("guard unavailable"))
    monkeypatch.setattr(file_safety, "raise_if_read_blocked", guard)
    result = json.loads(_dispatch(audio_path))
    assert result["success"] is False
    assert "guard unavailable" in result["error"]
    transcriber.assert_not_called()


@pytest.mark.parametrize("failure", [
    {"success": False, "error": "No STT provider available"},
    {"success": True, "transcript": ""},
    RuntimeError("STT backend unavailable"),
])
def test_stt_failure_is_clean_tool_error(audio_path, transcriber, failure):
    if isinstance(failure, Exception):
        transcriber.side_effect = failure
    else:
        transcriber.return_value = failure
    with scoped_runtime_main({"provider": "anthropic", "model": "claude-sonnet-4-5"}):
        result = json.loads(_dispatch(audio_path))
    assert result["success"] is False
    assert result["error"].startswith("Could not listen to audio:")


def test_question_is_optional_in_real_registry(audio_path, transcriber):
    with scoped_runtime_main({"provider": "google", "model": "gemini-2.5-flash"}):
        result = registry.dispatch("audio_listen", {"audio_path": str(audio_path)})
    assert result["_multimodal"] is True
    assert "Question:" not in result["content"][0]["text"]
    transcriber.assert_not_called()


def test_schema_and_result_wording_are_profile_overridable(audio_path, transcriber, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({
        "tool_descriptions": {"audio_listen": {
            "description": "Profile listening instructions",
            "params": {"audio_path": "Profile file instructions", "question": "Profile question instructions"},
        }},
        "tool_results": {"audio_listen": {
            "native": "Profile native result", "question": "Focus: {question}",
            "text_summary": "Profile summary", "transcript": "Profile STT note", "error": "Profile error: {error}",
        }},
    }))
    definitions = registry.get_definitions({"audio_listen"}, quiet=True)
    schema = apply_tool_text(definitions)[0]["function"]
    assert schema["description"] == "Profile listening instructions"
    assert schema["parameters"]["properties"]["audio_path"]["description"] == "Profile file instructions"
    assert schema["parameters"]["properties"]["question"]["description"] == "Profile question instructions"
    assert registry.get_schema("audio_listen")["description"] == audio_tools.AUDIO_LISTEN_SCHEMA["description"]
    with scoped_runtime_main({"provider": "google", "model": "gemini-2.5-flash"}):
        native = _dispatch(audio_path, "music?")
    assert native["content"][0]["text"].startswith("Profile native result\n\nFocus: music?\n\n[Audio attached at: ")
    assert native["text_summary"] == native["content"][0]["text"] + "\n[audio]"
    with scoped_runtime_main({"provider": "anthropic", "model": "claude-sonnet-4-5"}):
        assert json.loads(_dispatch(audio_path))["note"] == "Profile STT note"
    assert json.loads(_dispatch(audio_path.parent / "missing.wav"))["error"].startswith("Profile error:")


def test_toolset_resolution_exposes_audio_alongside_vision():
    from toolsets import TOOLSETS, resolve_toolset

    assert registry.get_toolset_for_tool("audio_listen") == "vision"
    for bundle in ("vision", "hermes-cli", "hermes-gateway", "hermes-telegram", "safe"):
        names = resolve_toolset(bundle)
        assert {"vision_analyze", "audio_listen"} <= set(names)
    for name in TOOLSETS:
        if name.startswith("hermes-") and name != "hermes-webhook":
            assert "audio_listen" in resolve_toolset(name)
    assert "audio_listen" not in resolve_toolset("hermes-webhook")
