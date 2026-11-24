"""Native inbound audio keeps the original media and never pays STT on the native route."""
import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import wave

import pytest
from PIL import Image

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _wrap_current_message_with_observed_context
from gateway.session import SessionSource
from gateway.run_turn_runner import TurnRunner, _observed_turn_persist_message
from gateway.session_envelope import snapshot_native, restore_native
from gateway.session_ingress_media import release_admission_media
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import storage_handle


@pytest.fixture
def audio_rig(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '42')
    monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', tmp_path / 'voice_modes.json')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='audio-chat', chat_type='dm', user_id='42')
    config = {'agent': {'audio_input_mode': 'auto'}, 'model': {'provider': 'anthropic', 'default': 'claude-test'}}
    monkeypatch.setattr('gateway.run_inbound_audio.load_config', lambda: config)
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return 'gemini/gemini-4-carbon', {'provider': 'gemini'}

    monkeypatch.setattr(runner, '_resolve_session_agent_runtime', resolve)
    stt = AsyncMock(return_value='lossy transcript')
    monkeypatch.setattr(runner, '_enrich_inbound_voice', stt)
    clip = tmp_path / 'note.wav'
    with wave.open(str(clip), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b'\x00\x00' * 80)
    return runner, source, clip, stt, calls, config


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', [MessageType.VOICE, MessageType.AUDIO, MessageType.DOCUMENT])
async def test_native_audio_skips_stt_and_preserves_user_caption(audio_rig, kind):
    runner, source, clip, stt, calls, _ = audio_rig
    event = MessageEvent(text='Describe the mood, not just the words.', source=source,
                         message_type=kind, media_urls=[str(clip)], media_types=['audio/wav'])
    text = await runner._prepare_inbound_message_text(event=event, source=source, history=[], session_key='actual-route')
    assert text == event.text
    stt.assert_not_awaited()
    assert calls[-1]['session_key'] == 'actual-route'
    assert calls[-1]['record_notice'] is False
    turn = object.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = SimpleNamespace(session_key='actual-route', message=text)
    parts = turn._native_image_run_message()
    assert parts[0]['text'].startswith(event.text)
    assert any(part['type'] == 'input_audio' for part in parts)
    assert str(clip) in '\n'.join(part.get('text', '') for part in parts)
    assert turn._native_image_run_message() == text


@pytest.mark.asyncio
async def test_busy_native_audio_queues_before_steer_stt(audio_rig, monkeypatch):
    runner, source, clip, stt, _, _ = audio_rig
    event = MessageEvent(text='Listen carefully', source=source, message_type=MessageType.VOICE,
                         media_urls=[str(clip)], media_types=['audio/wav'])
    prepare = AsyncMock(return_value='lossy busy transcript')
    monkeypatch.setattr(runner, '_prepare_busy_steer_text', prepare)
    agent = SimpleNamespace(steer=lambda text: pytest.fail('native audio must not steer as text'))
    outcome = await runner._resolve_busy_steer_or_redirect(event, 'actual-route', 'steer', agent)
    assert outcome.effective_mode == 'queue'
    assert outcome.steered is False
    prepare.assert_not_awaited()
    stt.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_native_audio_drain_skips_transcription(audio_rig, monkeypatch):
    runner, source, clip, _, calls, _ = audio_rig
    event = MessageEvent(text='Listen carefully', source=source, message_type=MessageType.VOICE,
                         media_urls=[str(clip)], media_types=['audio/wav'])
    adapter = SimpleNamespace(get_pending_message=lambda key: event)
    transcribe = AsyncMock(return_value=('lossy transcript', ['lossy transcript']))
    monkeypatch.setattr(runner, '_transcribe_and_echo_pending_voice', transcribe)
    monkeypatch.setattr(runner, '_merge_leftover_steer', lambda *args: args[-2:])
    restored, text = await runner._run_agent_drain_pending({'final_response': 'done'}, adapter, source, 'actual-route')
    assert restored is event
    assert text == 'Listen carefully'
    assert calls[-1]['session_key'] == runner._session_key_for_source(source)
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['auto', 'native'])
async def test_claude_falls_back_to_voice_transcription(audio_rig, monkeypatch, mode):
    runner, source, clip, stt, _, config = audio_rig
    config['agent']['audio_input_mode'] = mode
    monkeypatch.setattr(runner, '_resolve_session_agent_runtime',
                        lambda **_: ('anthropic/claude-opus-4-6', {'provider': 'anthropic'}))
    event = MessageEvent(text='Caption', source=source, message_type=MessageType.VOICE,
                         media_urls=[str(clip)], media_types=['audio/wav'])
    text = await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    assert text == 'lossy transcript'
    stt.assert_awaited_once()


@pytest.mark.asyncio
async def test_captionless_audio_uses_prompt_builder_override(audio_rig, monkeypatch):
    runner, source, clip, _, _, _ = audio_rig
    monkeypatch.setattr('agent.prompt_builder.AUDIO_ONLY_CAPTION', 'Hear the emotion in this clip.')
    event = MessageEvent(text='', source=source, message_type=MessageType.VOICE,
                         media_urls=[str(clip)], media_types=['audio/wav'])
    key = runner._session_key_for_source(source)
    text = await runner._prepare_inbound_message_text(event=event, source=source, history=[], session_key=key)
    turn = object.__new__(TurnRunner)
    turn._runner, turn._ctx = runner, SimpleNamespace(session_key=key, message=text)
    assert turn._native_image_run_message()[0]['text'].startswith('Hear the emotion in this clip.')


@pytest.mark.asyncio
async def test_native_images_and_audio_share_one_user_content_list(audio_rig, monkeypatch, tmp_path):
    runner, source, clip, stt, _, _ = audio_rig
    image = tmp_path / 'scene.png'
    Image.new('RGB', (2, 2)).save(image)
    monkeypatch.setattr(runner, '_decide_image_input_mode', lambda **_: 'native')
    key = runner._session_key_for_source(source)
    event = MessageEvent(text='Compare the scene and the sound.', source=source, message_type=MessageType.PHOTO,
                         media_urls=[str(image), str(clip)], media_types=['image/png', 'audio/wav'])
    text = await runner._prepare_inbound_message_text(event=event, source=source, history=[], session_key=key)
    turn = object.__new__(TurnRunner)
    turn._runner, turn._ctx = runner, SimpleNamespace(session_key=key, message=text)
    parts = turn._native_image_run_message()
    assert [part['type'] for part in parts] == ['text', 'image_url', 'input_audio']
    assert str(image) in parts[0]['text'] and str(clip) in parts[0]['text']
    wrapped = _wrap_current_message_with_observed_context(parts, 'Observed room discussion')
    assert 'Observed room discussion' in wrapped[0]['text']
    persisted = _observed_turn_persist_message(parts, 'Clean caption', text)
    assert persisted[0]['text'].startswith('Clean caption')
    assert str(clip) in persisted[0]['text'] and str(image) in persisted[0]['text']
    assert persisted[1:] == parts[1:]
    stt.assert_not_awaited()


@pytest.mark.asyncio
async def test_clarification_voice_reply_still_uses_spoken_text(audio_rig, monkeypatch):
    runner, source, clip, _, _, _ = audio_rig
    event = MessageEvent(text='', source=source, message_type=MessageType.VOICE,
                         media_urls=[str(clip)], media_types=['audio/wav'])
    transcribe = AsyncMock(return_value=('speech note', ['Option two']))
    monkeypatch.setattr(runner, '_transcribe_pending_audio_event_once', transcribe)
    assert await runner._prepare_clarify_reply_text(event) == 'Option two'
    transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_retained_admission_audio_survives_staging_deletion(audio_rig, tmp_path, monkeypatch):
    runner, source, clip, stt, _, _ = audio_rig
    adapter = SimpleNamespace()
    monkeypatch.setattr(runner, '_adapter_for_source', lambda _: adapter)
    original = clip.read_bytes()
    event = MessageEvent(text='Listen to the whole recording.', source=source, message_id='audio-1',
                         message_type=MessageType.VOICE, media_urls=[str(clip)], media_types=['audio/wav'])
    payload = snapshot_native(runner, event)
    db = storage_handle(tmp_path / 'home')
    try:
        seed_session(db, 'native-audio', 'telegram')
        epoch = db.runtime.begin_epoch(instance_id='audio-test')
        row, _ = db.admissions.admit(epoch=epoch, principal_id='user:42', session_id='native-audio',
                                    request_id='audio-1', payload=payload)
        clip.unlink()
        durable = db.admissions.get(admission_id=row.admission_id)
        restored = restore_native(durable.payload, runner)
        retained = Path(restored.media_urls[0])
        assert 'native-inputs' in retained.parts
        assert retained.read_bytes() == original
        key = runner._session_key_for_source(restored.source)
        text = await runner._prepare_inbound_message_text(event=restored, source=restored.source,
                                                         history=[], session_key=key)
        turn = object.__new__(TurnRunner)
        turn._runner, turn._ctx = runner, SimpleNamespace(session_key=key, message=text)
        parts = turn._native_image_run_message()
        audio = next(part['input_audio'] for part in parts if part['type'] == 'input_audio')
        assert base64.b64decode(audio['data']) == original
        assert str(retained) in parts[0]['text']
        from session_rules.turns import durable_content
        from tests.fixtures.storage_read import conversation
        from agent.audio_replay import replay_recent_audio
        seed_messages(db, 'native-audio', {'role': 'user', 'content': durable_content(parts)})
        rows = conversation(db, 'native-audio')
        assert '[audio]' in rows[0]['content']
        assert audio['data'] not in rows[0]['content']
        assert replay_recent_audio(rows, 3) == 1
        assert rows[0]['content'] == parts
        claim = db.admissions.claim(epoch=epoch, session_id='native-audio')
        db.admissions.settle(epoch=epoch, admission_id=row.admission_id, generation=claim.generation,
                             outcome='completed')
        release_admission_media(db, row.admission_id)
        assert retained.exists(), 'the transcript still names the original audio for follow-up listening'
        stt.assert_not_awaited()
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('mime', ['', 'application/octet-stream'])
async def test_native_voice_with_unknown_mime_skips_stt_and_echo(audio_rig, monkeypatch, mime):
    runner, source, clip, _, _, config = audio_rig
    config['agent']['audio_input_mode'] = 'native'
    # Exercise the real enrichment entry point, not a mocked STT decision.
    monkeypatch.delattr(runner, '_enrich_inbound_voice')
    transcribe = AsyncMock(side_effect=AssertionError('native voice must not transcribe'))
    echo = AsyncMock(side_effect=AssertionError('native voice must not echo'))
    monkeypatch.setattr(runner, '_transcribe_one_clip', transcribe)
    monkeypatch.setattr(runner, '_echo_stt_transcripts', echo)
    event = MessageEvent(text='', source=source, message_type=MessageType.VOICE,
                         media_urls=[str(clip)], media_types=[mime])
    key = runner._session_key_for_source(source)
    text = await runner._prepare_inbound_message_text(event=event, source=source, history=[], session_key=key)
    turn = object.__new__(TurnRunner)
    turn._runner, turn._ctx = runner, SimpleNamespace(session_key=key, message=text)
    assert turn._native_image_run_message()[-1]['type'] == 'input_audio'
    transcribe.assert_not_awaited()
    echo.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_audio_config_and_caption_override_from_profile_yaml(audio_rig, monkeypatch, tmp_path):
    import importlib
    import agent.prompt_builder as prompts
    from hermes_cli.config import load_config
    runner, source, clip, stt, _, _ = audio_rig
    home = tmp_path / 'yaml-home'
    home.mkdir()
    (home / 'config.yaml').write_text('agent:\n  audio_input_mode: native\naudio:\n  replay_recent_audio: 0\n')
    (home / 'prompt_overrides.yaml').write_text('AUDIO_ONLY_CAPTION: Hear the emotion in this clip.\n')
    with monkeypatch.context() as profile:
        profile.setenv('HERMES_HOME', str(home))
        profile.setattr('gateway.run_inbound_audio.load_config', load_config)
        importlib.reload(prompts)
        try:
            event = MessageEvent(text='', source=source, message_type=MessageType.VOICE,
                                 media_urls=[str(clip)], media_types=['audio/wav'])
            key = runner._session_key_for_source(source)
            text = await runner._prepare_inbound_message_text(event=event, source=source, history=[], session_key=key)
            turn = object.__new__(TurnRunner)
            turn._runner, turn._ctx = runner, SimpleNamespace(session_key=key, message=text)
            parts = turn._native_image_run_message()
            assert parts[0]['text'].startswith('Hear the emotion in this clip.')
            assert parts[-1]['type'] == 'input_audio'
            stt.assert_not_awaited()
            from agent.audio_replay import resolve_replay_recent_audio
            assert resolve_replay_recent_audio(load_config()) == 0
        finally:
            profile.undo()
            importlib.reload(prompts)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['openai-ogg', 'oversized', 'empty', 'unsupported'])
async def test_ineligible_native_voice_keeps_stt_fallback(audio_rig, monkeypatch, kind):
    runner, source, clip, stt, _, _ = audio_rig
    if kind == 'openai-ogg':
        monkeypatch.setattr(runner, '_resolve_session_agent_runtime', lambda **_: ('gpt-audio', {'provider': 'openai'}))
        renamed = clip.with_suffix('.ogg')
        clip.rename(renamed)
        clip = renamed
    elif kind == 'oversized':
        with clip.open('wb') as output:
            output.truncate(20 * 1024 * 1024 + 1)
    elif kind == 'empty':
        clip.write_bytes(b'')
    else:
        renamed = clip.with_suffix('.oga')
        clip.rename(renamed)
        clip = renamed
    event = MessageEvent(text='', source=source, message_type=MessageType.VOICE,
                         media_urls=[str(clip)], media_types=['audio/ogg'])
    assert await runner._prepare_inbound_message_text(event=event, source=source, history=[]) == 'lossy transcript'
    stt.assert_awaited_once()


def test_file_disappearing_after_native_preflight_is_reported(audio_rig):
    runner, source, clip, _, _, _ = audio_rig
    key = runner._session_key_for_source(source)
    runner._session_state(key).persistent.native_audio_paths = [str(clip)]
    clip.unlink()
    turn = object.__new__(TurnRunner)
    turn._runner, turn._ctx = runner, SimpleNamespace(session_key=key, message='Listen')
    parts = turn._native_image_run_message()
    assert 'Audio could not be loaded' in str(parts)
    assert 'Listen' in str(parts)
