"""Native audio admission contracts."""

import base64

import pytest


def test_audio_attachment_is_bounded_and_guarded(tmp_path, monkeypatch):
    from agent.audio_routing import audio_file_part, build_native_audio_content_parts
    path = tmp_path / 'voice.ogg'
    path.write_bytes(b'OggSvoice')
    part = audio_file_part(path)
    assert part == {'type': 'input_audio', 'input_audio': {
        'data': base64.b64encode(b'OggSvoice').decode(), 'format': 'ogg'}}
    parts, skipped = build_native_audio_content_parts('', [str(path)])
    assert not skipped
    assert parts == [{'type': 'text', 'text': f'Listen to this audio.\n\n[Audio attached at: {path}]'}, part]
    def blocked(_path):
        raise RuntimeError('guard unavailable')
    monkeypatch.setattr('agent.file_safety.raise_if_read_blocked', blocked)
    with pytest.raises(ValueError):
        audio_file_part(path)
    assert build_native_audio_content_parts('caption', [str(path)]) == (
        [{'type': 'text', 'text': 'caption'}], [str(path)])


@pytest.mark.parametrize('provider,model,native', [
    ('gemini', 'gemini-2.5-flash', True),
    ('gemini', 'gemini/gemini-4-carbon', True),
    ('custom', 'gemini/gemini-4-argon', True),
    ('custom', 'gemini-4-neon', True),
    ('gemini', 'gemini-4-hydrogen', True),
    ('gemini', 'gemini-pro-latest', True),
    ('vertex', 'gemini-3.8-flash', True),
    ('vertex', 'publishers/google/models/gemini-3-pro-preview', True),
    ('openrouter', 'google/gemini-2.5-pro', True),
    ('custom:relay', 'gemini-2.0-flash', True),
    ('openai', 'gpt-4o-audio-preview', True),
    ('openai', 'gpt-audio-mini', True),
    ('anthropic', 'gemini-2.5-pro', False),
    ('openrouter', 'anthropic/claude-sonnet-4', False),
    ('openai', 'gpt-4o', False),
    ('gemini', 'gemini-embedding-001', False),
    ('gemini', 'gemini-2.5-flash-preview-tts', False),
    ('gemini', 'gemini-2.5-flash-image', False),
    ('custom', 'not-gemini-2.5-pro', False),
])
def test_capability_guard_also_applies_to_forced_native(provider, model, native):
    from agent.audio_routing import supports_native_audio, decide_audio_input_mode
    assert supports_native_audio(provider, model) is native
    for mode in ('native', 'auto', 'invalid'):
        assert decide_audio_input_mode(provider, model, {'agent': {'audio_input_mode': mode}}) == (
            'native' if native else 'stt')
    assert decide_audio_input_mode(provider, model, {'agent': {'audio_input_mode': 'stt'}}) == 'stt'


@pytest.mark.parametrize('suffix', ['ogg', 'opus', 'mp3', 'wav', 'm4a', 'flac', 'aac', 'webm'])
def test_supported_file_formats(tmp_path, suffix):
    from agent.audio_routing import audio_file_part
    path = tmp_path / f'voice.{suffix}'
    path.write_bytes(b'audio')
    assert audio_file_part(path)['input_audio']['format'] == suffix


def test_oversized_empty_and_unknown_files_are_rejected(tmp_path):
    from agent.audio_routing import MAX_AUDIO_BYTES, audio_file_part
    path = tmp_path / 'large.wav'
    with path.open('wb') as stream:
        stream.truncate(MAX_AUDIO_BYTES + 1)
    with pytest.raises(ValueError):
        audio_file_part(path)
    path.write_bytes(b'')
    with pytest.raises(ValueError):
        audio_file_part(path)
    unknown = tmp_path / 'voice.txt'
    unknown.write_bytes(b'audio')
    with pytest.raises(ValueError):
        audio_file_part(unknown)


def test_audio_caption_uses_prompt_override_and_keeps_successful_files(tmp_path, monkeypatch):
    from agent.audio_routing import build_native_audio_content_parts
    monkeypatch.setattr('agent.prompt_builder.AUDIO_ONLY_CAPTION', 'Custom audio caption')
    path = tmp_path / 'voice.wav'
    path.write_bytes(b'audio')
    missing = str(tmp_path / 'missing.wav')
    parts, skipped = build_native_audio_content_parts('', [missing, str(path)])
    assert skipped == [missing]
    assert parts[0]['text'] == f'Custom audio caption\n\n[Audio attached at: {path}]'
    assert parts[1]['type'] == 'input_audio'
