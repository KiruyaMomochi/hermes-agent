"""Tests for gateway /fast support and Priority Processing routing."""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None
    last_run = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(
        self,
        user_message,
        conversation_history=None,
        task_id=None,
        persist_user_message=None,
        persist_user_timestamp=None,
    ):
        type(self).last_run = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
            "persist_user_message": persist_user_message,
            "persist_user_timestamp": persist_user_timestamp,
        }
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def test_turn_route_injects_priority_processing_without_changing_runtime():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)

    assert route["runtime"]["provider"] == "openrouter"
    assert route["runtime"]["api_mode"] == "chat_completions"
    assert route["request_overrides"] == {"service_tier": "priority"}


def test_turn_route_skips_priority_processing_for_unsupported_models():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.3-codex", runtime_kwargs)

    assert route["request_overrides"] == {}


@pytest.mark.asyncio
async def test_handle_fast_command_persists_config(monkeypatch, tmp_path):
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    response = await runner._handle_fast_command(_make_event("/fast fast"))

    assert "FAST" in response
    assert runner._service_tier == "priority"

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["agent"]["service_tier"] == "fast"


@pytest.mark.asyncio
async def test_run_agent_passes_priority_processing_to_gateway_agent(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    (tmp_path / "config.yaml").write_text("agent:\n  service_tier: fast\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    # ``_load_service_tier`` was refactored to call ``_load_gateway_runtime_config``
    # (which wraps ``_load_gateway_config`` plus env-expansion).  Since the test
    # stubs ``_load_gateway_config`` to ``{}``, also stub the runtime wrapper
    # directly so the priority routing assertions still exercise the live tier.
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"service_tier": "fast"}},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init["service_tier"] == "priority"
    assert _CapturingAgent.last_init["request_overrides"] == {"service_tier": "priority"}


@pytest.mark.asyncio
async def test_run_agent_persists_plain_text_for_native_image_turn(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake image bytes")
    session_key = "agent:main:telegram:dm:12345"
    runner._pending_native_image_paths_by_session = {session_key: [str(image_path)]}

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    result = await runner._run_agent(
        message="[14:02]",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key=session_key,
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_run is not None
    assert isinstance(_CapturingAgent.last_run["user_message"], list)
    assert _CapturingAgent.last_run["persist_user_message"] == "[14:02]"


class _RecordingSessionDB:
    def __init__(self):
        self.messages = []

    def append_message(self, session_id, role, content, **kwargs):
        self.messages.append({
            "session_id": session_id,
            "role": role,
            "content": content,
            **kwargs,
        })


def test_persist_user_message_override_does_not_strip_openai_image_url_parts(monkeypatch):
    from run_agent import AIAgent

    captured_api_messages = []
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        enabled_toolsets=[],
        session_id="session-vision-openai",
    )
    agent._session_db = _RecordingSessionDB()
    agent._session_db_created = True
    agent._save_session_log = lambda messages: None
    agent._cached_system_prompt = "system"
    agent._disable_streaming = True
    monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)

    def fake_api_call(api_kwargs):
        captured_api_messages.extend(api_kwargs["messages"])
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="saw it", tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", fake_api_call)

    image_only_parts = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
        }
    ]
    result = agent.run_conversation(
        image_only_parts,
        persist_user_message="",
        conversation_history=[],
    )

    assert result["final_response"] == "saw it"
    api_user_messages = [m for m in captured_api_messages if m.get("role") == "user"]
    assert isinstance(api_user_messages[-1]["content"], list)
    assert any(p.get("type") == "image_url" for p in api_user_messages[-1]["content"])
    assert result["messages"][0]["content"] == image_only_parts
    persisted_users = [m for m in agent._session_db.messages if m["role"] == "user"]
    assert persisted_users
    assert persisted_users[0]["content"] == ""


def test_persist_user_message_override_does_not_strip_anthropic_image_parts(monkeypatch):
    from run_agent import AIAgent

    captured_api_messages = []
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="claude-test",
        api_mode="anthropic_messages",
        max_iterations=1,
        quiet_mode=True,
        enabled_toolsets=[],
        session_id="session-vision-anthropic",
    )
    agent._session_db = _RecordingSessionDB()
    agent._session_db_created = True
    agent._save_session_log = lambda messages: None
    agent._cached_system_prompt = "system"
    agent._disable_streaming = True
    monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)

    def fake_api_call(api_kwargs):
        captured_api_messages.extend(api_kwargs["messages"])
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="saw it")],
            stop_reason="end_turn",
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", fake_api_call)

    image_only_parts = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "ZmFrZQ==",
            },
        }
    ]
    result = agent.run_conversation(
        image_only_parts,
        persist_user_message="",
        conversation_history=[],
    )

    assert result["final_response"] == "saw it"
    api_user_messages = [m for m in captured_api_messages if m.get("role") == "user"]
    assert isinstance(api_user_messages[-1]["content"], list)
    assert any(p.get("type") == "image" for p in api_user_messages[-1]["content"])
    assert result["messages"][0]["content"] == image_only_parts
    persisted_users = [m for m in agent._session_db.messages if m["role"] == "user"]
    assert persisted_users
    assert persisted_users[0]["content"] == ""
