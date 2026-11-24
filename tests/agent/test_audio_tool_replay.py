"""Native listening tool history survives durable projection without changing wire prefixes."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.audio_replay import replay_recent_audio
from agent.auxiliary_client import scoped_runtime_main

from session_rules.turns import durable_content
from tools import audio_tools  # register the real listening tool
from tools.registry import registry


def _listen(tmp_path, name='clip.wav'):
    path = tmp_path / name
    path.write_bytes(b'RIFF-audio-' + name.encode())
    with scoped_runtime_main({'provider': 'google', 'model': 'gemini-2.5-flash'}):
        result = registry.dispatch('audio_listen', {'audio_path': str(path), 'question': 'Identify this sound.'})
    assert isinstance(result, dict)
    return {'role': 'tool', 'tool_call_id': name, 'content': result}, path


def _agent(provider='google', model='gemini-2.5-flash'):
    return SimpleNamespace(provider=provider, model=model, base_url='', _current_turn_timestamp=1000,
                           ephemeral_system_prompt=None, _should_sanitize_tool_calls=lambda: False,
                           _copy_reasoning_content_for_api=lambda source, target: None)


def _wire(agent, rows):
    from agent.turn_context import build_api_messages
    return build_api_messages(agent, rows, current_turn_user_idx=0, ext_prefetch_cache='',
                              plugin_user_context='', moa_config=None, active_system_prompt='')[0]


@pytest.mark.parametrize('provider,model', [('google', 'gemini-2.5-flash'), ('openai', 'gpt-audio')])
def test_db_tool_reload_keeps_exact_wire_prefix(tmp_path, monkeypatch, provider, model):
    from tests.fixtures.storage_read import conversation
    from tests.fixtures.storage_seed import seed_messages, seed_session
    from tests.gateway.fixtures.storage_handles import storage_handle

    monkeypatch.setattr('hermes_cli.config.load_config_readonly', dict)
    tool, source = _listen(tmp_path)
    calls = [{'id': source.name, 'type': 'function', 'function': {'name': 'audio_listen', 'arguments': '{}'}}]
    live = [{'role': 'user', 'content': 'Listen'}, {'role': 'assistant', 'content': '', 'tool_calls': calls}, tool]
    db = storage_handle(tmp_path)
    seed_session(db, 'audio-tool', 'api_server')
    for index, row in enumerate(live):
        seed_messages(db, 'audio-tool', {**row, 'content': durable_content(row['content']), 'timestamp': 1000 + index})
    source.unlink()
    restored = conversation(db, 'audio-tool')
    assert isinstance(restored[-1]['content'], str)
    assert restored[-1]['content'].endswith('\n[audio]')
    assert tool['content']['content'][1]['input_audio']['data'] not in restored[-1]['content']
    before = copy.deepcopy(restored)
    agent = _agent(provider, model)
    wire = _wire(agent, restored)
    fields = {'role', 'content', 'tool_calls', 'tool_call_id'}
    assert [{k: v for k, v in row.items() if k in fields} for row in wire] == _wire(agent, live)
    assert [row['tool_call_id'] for row in wire if row['role'] == 'tool'] == [source.name]
    assert restored == before


def test_shared_frontier_counts_live_tool_and_user_audio(tmp_path):
    rows = []
    for index in range(9):
        tool, _ = _listen(tmp_path, f'{index}.wav')
        if index % 2:
            tool['role'] = 'user'
            tool.pop('tool_call_id')
        if index != 8:
            tool['content'] = durable_content(tool['content'])
        rows.append(tool)
    assert replay_recent_audio(rows, 3) == 2
    assert all(isinstance(row['content'], str) for row in rows[:6])
    assert all(isinstance(row['content'], list) for row in rows[6:8])


@pytest.mark.parametrize('live', [False, True])
def test_noncapable_model_never_receives_tool_audio(tmp_path, live):
    tool, _ = _listen(tmp_path)
    data = tool['content']['content'][1]['input_audio']['data']
    if not live:
        tool['content'] = durable_content(tool['content'])
    before = copy.deepcopy(tool)
    wire = _wire(_agent('anthropic', 'claude-sonnet-4'), [tool])
    assert data not in str(wire)
    assert wire[0]['tool_call_id'] == tool['tool_call_id']
    assert tool == before


def test_tool_replay_zero_disables_and_stt_preference_does_not(tmp_path, monkeypatch):
    tool, _ = _listen(tmp_path)
    tool['content'] = durable_content(tool['content'])
    cfg = {'agent': {'audio_input_mode': 'stt'}, 'audio': {'replay_recent_audio': 3}}
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: cfg)
    agent = _agent()
    assert isinstance(_wire(agent, [tool])[0]['content'], list)
    cfg['audio']['replay_recent_audio'] = 0
    agent._current_turn_timestamp += 1
    assert _wire(agent, [tool]) == [tool]


@pytest.mark.parametrize('invalid', ['changed', 'outside', 'foreign'])
def test_tool_replay_fences_retained_bytes(tmp_path, monkeypatch, invalid):
    tool, source = _listen(tmp_path)
    tool['content'] = durable_content(tool['content'])
    ref = tool['content'].split('[Audio attached at: ', 1)[1].split(']', 1)[0]
    if invalid == 'changed':
        Path(ref).write_bytes(b'changed')
    elif invalid == 'outside':
        tool['content'] = tool['content'].replace(ref, str(source))
    else:
        monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'other-profile'))
    before = copy.deepcopy(tool)
    assert replay_recent_audio([tool], 3) == 0
    assert tool == before
