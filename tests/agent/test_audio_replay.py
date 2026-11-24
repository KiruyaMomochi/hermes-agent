"""Durable audio and wire-only replay behavior, using real retained media."""
import copy
from pathlib import Path

import pytest

from agent.audio_routing import build_native_audio_content_parts
from gateway.session_ingress_media import capture_native_media
from session_rules.turns import durable_content


def _turn(tmp_path, name='voice.ogg', data=b'OggSvoice'):
    path = tmp_path / name
    path.write_bytes(data)
    retained = capture_native_media([path])[0]['path']
    parts, skipped = build_native_audio_content_parts('caption', [retained])
    assert not skipped
    return {'role': 'user', 'content': parts}, Path(retained)


def test_durable_marker_and_replay_are_byte_identical(tmp_path):
    from agent.audio_replay import replay_recent_audio
    live, _ = _turn(tmp_path)
    stored = durable_content(live['content'])
    assert stored.endswith('\n[audio]')
    assert 'T2dn' not in stored
    rows = [{'role': 'user', 'content': stored}]
    assert replay_recent_audio(rows, 3) == 1
    assert rows == [live]


def test_batch_frontier_counts_live_and_missing_slots(tmp_path):
    from agent.audio_replay import replay_recent_audio
    turns = [_turn(tmp_path, f'voice{i}.ogg', f'audio{i}'.encode()) for i in range(8)]
    rows = [{'role': 'user', 'content': durable_content(turn['content'])} for turn, _ in turns[:7]]
    turns[5][1].unlink()
    rows.append(copy.deepcopy(turns[7][0]))
    assert replay_recent_audio(rows, 3) == 3
    # Eight slots -> frontier 3: slots 3,4,6 restored, missing 5 still counts; 7 already live.
    assert all(isinstance(row['content'], str) for row in rows[:3])
    for index in (3, 4, 6, 7):
        assert rows[index] == turns[index][0]
    assert isinstance(rows[5]['content'], str)


@pytest.mark.platforms('posix')
def test_replay_rejects_changed_unretained_cross_profile_and_symlink_bytes(tmp_path):
    from agent.audio_replay import replay_recent_audio
    from gateway.session_ingress_media import native_inputs_root
    turn, retained = _turn(tmp_path)
    stored = durable_content(turn['content'])
    retained.write_bytes(b'tampered')
    assert replay_recent_audio([{'role': 'user', 'content': stored}], 3) == 0
    outside = tmp_path / 'outside.ogg'
    outside.write_bytes(b'OggSvoice')
    for path in (outside, native_inputs_root().parent / retained.parent.name / retained.name):
        row = {'role': 'user', 'content': stored.replace(str(retained), str(path))}
        assert replay_recent_audio([row], 3) == 0
    retained.unlink()
    retained.symlink_to(outside)
    assert replay_recent_audio([{'role': 'user', 'content': stored}], 3) == 0
    assert replay_recent_audio([{'role': 'user', 'content': 'literal\n[audio]'}], 3) == 0


def test_mixed_image_audio_replay_preserves_order(tmp_path):
    from agent.audio_replay import replay_recent_audio
    from agent.image_replay import replay_recent_images
    from agent.image_routing import build_native_content_parts
    from tests.agent.test_image_replay import _PNG
    turn, _ = _turn(tmp_path)
    staged = tmp_path / 'image.png'
    staged.write_bytes(_PNG)
    path = capture_native_media([staged])[0]['path']
    image, _ = build_native_content_parts('', [path])
    # User text carries both path hints; the durable order is audio, image, hook.
    parts = [dict(turn['content'][0]), turn['content'][1], image[1], {'type': 'text', 'text': 'hook'}]
    parts[0]['text'] += f'\n[Image attached at: {path}]'
    rows = [{'role': 'user', 'content': durable_content(parts)}]
    assert replay_recent_images(rows, 3) == 1
    assert replay_recent_audio(rows, 3) == 1
    assert rows[0]['content'] == parts


class _Agent:
    provider, model = 'gemini', 'gemini-2.5-flash'
    _current_turn_timestamp = 1000.0
    ephemeral_system_prompt = None
    base_url = ''

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _wire(agent, history):
    from agent.turn_context import build_api_messages
    return build_api_messages(agent, history, current_turn_user_idx=len(history) - 1,
                              ext_prefetch_cache='', plugin_user_context='', moa_config=None,
                              active_system_prompt='stable')[0]


def test_wire_hook_preserves_history_and_fences_model_switch(tmp_path):
    turn, _ = _turn(tmp_path)
    history = [copy.deepcopy(turn), {'role': 'assistant', 'content': 'heard it'},
               {'role': 'user', 'content': 'again'}]
    original = copy.deepcopy(history)
    agent = _Agent()
    assert _wire(agent, history)[1] == turn
    agent.provider, agent.model = 'anthropic', 'claude-sonnet-4'
    wire = _wire(agent, history)
    assert wire[1]['content'][1] == {'type': 'text', 'text': '[audio]'}
    assert history == original
    assert wire[0] == {'role': 'system', 'content': 'stable'}


def test_audio_model_switch_still_fences_unsupported_containers(tmp_path):
    turn, _ = _turn(tmp_path)
    agent = _Agent()
    agent.provider, agent.model = 'openai', 'gpt-audio'
    history = [copy.deepcopy(turn), {'role': 'assistant', 'content': 'heard it'},
               {'role': 'user', 'content': 'again'}]
    original = copy.deepcopy(history)
    assert _wire(agent, history)[1]['content'][1] == {'type': 'text', 'text': '[audio]'}
    history[0]['content'] = durable_content(turn['content'])
    assert isinstance(_wire(agent, history)[1]['content'], str)
    assert original[0] == turn


def test_replay_config_frozen_within_turn_and_zero_disables(tmp_path, monkeypatch):
    from agent.audio_replay import resolve_replay_recent_audio
    turn, _ = _turn(tmp_path)
    history = [{'role': 'user', 'content': durable_content(turn['content'])},
               {'role': 'assistant', 'content': 'reply'}, {'role': 'user', 'content': 'again'}]
    cfg = {'audio': {'replay_recent_audio': 3}}
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: cfg)
    agent = _Agent()
    assert _wire(agent, history)[1] == turn
    cfg['audio']['replay_recent_audio'] = 0
    assert _wire(agent, history)[1] == turn
    agent._current_turn_timestamp += 1
    assert isinstance(_wire(agent, history)[1]['content'], str)
    for raw in (-1, True, '3', [], 1.5):
        assert resolve_replay_recent_audio({'audio': {'replay_recent_audio': raw}}) == 0
    assert resolve_replay_recent_audio({}) == 3


def test_db_reload_replays_audio_without_persisting_bytes(tmp_path, monkeypatch):
    from tests.fixtures.storage_read import conversation
    from tests.fixtures.storage_seed import seed_messages, seed_session
    from tests.gateway.fixtures.storage_handles import storage_handle
    turn, _ = _turn(tmp_path)
    db = storage_handle(tmp_path)
    seed_session(db, 'audio-session', 'api_server')
    seed_messages(db, 'audio-session', {'role': 'user', 'content': durable_content(turn['content']), 'timestamp': 1000.0})
    seed_messages(db, 'audio-session', {'role': 'assistant', 'content': 'reply', 'timestamp': 1001.0})
    history = conversation(db, 'audio-session') + [{'role': 'user', 'content': 'again'}]
    assert isinstance(history[0]['content'], str)
    assert turn['content'][1]['input_audio']['data'] not in history[0]['content']
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', dict)
    assert _wire(_Agent(), history)[1]['content'] == turn['content']
    assert isinstance(history[0]['content'], str)
