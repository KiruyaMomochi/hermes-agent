"""Native audio tool results must reach the main model on a valid wire role."""
import copy
import pytest

from agent.gemini_native_adapter import _build_gemini_contents
from agent.tool_audio import prepare_tool_audio_for_api
from agent.tool_dispatch_helpers import make_tool_result_message
from agent.vision_message_prep import VisionMessagePrepMixin


AUDIO = {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}


class _Agent(VisionMessagePrepMixin):
    _current_turn_timestamp = 1000.0
    ephemeral_system_prompt = None
    base_url = ""

    def __init__(self, provider="openai", model="gpt-audio"):
        self.provider, self.model = provider, model

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _history(agent, *, envelope=False, parallel=False):
    result = {"_multimodal": True,
              "content": [{"type": "text", "text": "Listen for the instrument."}, copy.deepcopy(AUDIO)],
              "text_summary": "Audio attached natively for the main model."}
    content = result if envelope else agent._tool_result_content_for_active_model("audio_listen", result)
    calls = [{"id": "listen", "type": "function", "function": {"name": "audio_listen", "arguments": "{}"}}]
    rows = [{"role": "user", "content": "What instrument is this?"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            make_tool_result_message("audio_listen", content, "listen")]
    if parallel:
        calls.append({"id": "other", "type": "function", "function": {"name": "terminal", "arguments": "{}"}})
        rows.append(make_tool_result_message("terminal", "other result", "other"))
    return rows


def _wire(agent, history):
    from agent.turn_context import build_api_messages
    return build_api_messages(agent, history, current_turn_user_idx=0,
                              ext_prefetch_cache="", plugin_user_context="", moa_config=None,
                              active_system_prompt="stable")[0]


@pytest.mark.parametrize("envelope", [False, True])
def test_openai_tool_audio_is_user_input_after_complete_tool_batch(envelope):
    agent = _Agent()
    history = _history(agent, envelope=envelope, parallel=True)
    original = copy.deepcopy(history)
    wire = _wire(agent, history)
    assert [row["role"] for row in wire] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert isinstance(wire[3]["content"], str)
    assert "Listen for the instrument." in wire[3]["content"]
    assert wire[-1]["content"][-1] == AUDIO
    assert "listen" in wire[-1]["content"][0]["text"]
    assert history == original
    assert _wire(agent, history) == wire


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3-pro-preview"])
def test_gemini_tool_audio_reaches_function_response_sibling(model):
    agent = _Agent("gemini", model)
    history = _history(agent)
    original = copy.deepcopy(history)
    wire = _wire(agent, history)
    assert [row["role"] for row in wire] == ["system", "user", "assistant", "tool"]
    contents, _ = _build_gemini_contents(wire, include_tool_call_ids=True, is_gemini3=model.startswith("gemini-3"))
    result = contents[-1]
    assert result["role"] == "user"
    assert result["parts"][0]["functionResponse"]["name"] == "audio_listen"
    assert result["parts"][1] == {"inlineData": {"mimeType": "audio/wav", "data": AUDIO["input_audio"]["data"]}}
    assert history == original


@pytest.mark.parametrize("envelope", [False, True])
def test_model_switch_to_non_audio_keeps_text_and_removes_audio(envelope):
    agent = _Agent()
    history = _history(agent, envelope=envelope)
    original = copy.deepcopy(history)
    agent.provider, agent.model = "openai", "gpt-4.1"
    wire = _wire(agent, history)
    assert [row["role"] for row in wire] == ["system", "user", "assistant", "tool"]
    assert isinstance(wire[-1]["content"], str)
    assert "Listen for the instrument." in wire[-1]["content"]
    assert AUDIO["input_audio"]["data"] not in str(wire)
    assert history == original


@pytest.mark.parametrize("provider,model", [("openai", "gpt-audio"), ("gemini", "gemini-2.5-flash"),
                                           ("openai", "gpt-4.1")])
@pytest.mark.parametrize("envelope", [False, True])
def test_helper_routes_audio_without_changing_tool_pairing(provider, model, envelope):
    agent = _Agent()
    history = _history(agent, envelope=envelope, parallel=True)
    wire = copy.deepcopy(history)
    agent.provider, agent.model = provider, model
    prepare_tool_audio_for_api(agent, wire)
    assert [row["tool_call_id"] for row in wire if row["role"] == "tool"] == ["listen", "other"]
    if provider == "gemini":
        assert wire[2]["content"][-1] == AUDIO
        assert len(wire) == len(history)
    elif model == "gpt-audio":
        assert isinstance(wire[2]["content"], str)
        assert wire[-1]["role"] == "user"
        assert wire[-1]["content"][-1] == AUDIO
    else:
        assert isinstance(wire[2]["content"], str)
        assert AUDIO["input_audio"]["data"] not in str(wire)
        assert len(wire) == len(history)
    first = copy.deepcopy(wire)
    prepare_tool_audio_for_api(agent, wire)
    assert wire == first


def test_helper_merges_audio_before_following_user_instead_of_adjacent_user_roles():
    agent = _Agent()
    wire = _history(agent)
    wire.append({"role": "user", "content": "Now identify the key."})
    prepare_tool_audio_for_api(agent, wire)
    assert [row["role"] for row in wire] == ["user", "assistant", "tool", "user"]
    assert wire[-1]["content"][-2] == AUDIO
    assert wire[-1]["content"][-1] == {"type": "text", "text": "Now identify the key."}
