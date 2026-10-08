"""Profile-local tool text stays out of the shared definitions and child agents."""
from types import SimpleNamespace

import pytest

from hermes_yaml import safe_dump

from tools.registry import prompt_override


def test_parameter_override_is_copy_on_write(tmp_path, monkeypatch):
    from tools.prompt_text import apply_tool_text

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({
        "tool_descriptions": {"demo": {"description": "custom", "params": {"submit": "custom param"}}},
    }))
    original = {"type": "function", "function": {"name": "demo", "description": "original",
        "parameters": {"properties": {"submit": {"type": "boolean", "description": "original param"}}}}}
    result = apply_tool_text([original])[0]["function"]
    assert result["description"] == "custom"
    assert result["parameters"]["properties"]["submit"]["description"] == "custom param"
    assert original["function"]["description"] == "original"
    assert original["function"]["parameters"]["properties"]["submit"]["description"] == "original param"


def test_cached_yaml_refreshes_and_profiles_stay_separate(tmp_path, monkeypatch):
    from pathlib import Path
    from unittest.mock import patch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "prompt_overrides.yaml"
    path.write_text("demo: first\n")
    with patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened:
        assert prompt_override("demo") == "first"
        assert prompt_override("demo") == "first"
        assert opened.call_count == 1
        path.write_text("demo: changed\n")
        assert prompt_override("demo") == "changed"
    other = tmp_path / "other"
    other.mkdir()
    (other / "prompt_overrides.yaml").write_text("demo: other\n")
    monkeypatch.setenv("HERMES_HOME", str(other))
    assert prompt_override("demo") == "other"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert prompt_override("demo") == "changed"


def test_child_defaults_profile_child_own_text_and_pinned_parent(tmp_path, monkeypatch):
    from agent.delegation_context import delegated_child_context
    from tools.prompt_text import agent_tool_text, apply_tool_text, tool_text
    from tools.delegate_tool_routes import delegated_profile_scope

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "prompt_overrides.yaml"
    path.write_text("tool_descriptions:\n  demo: parent\ntool_results:\n  demo: parent result\n")
    schema = [{"function": {"name": "demo", "description": "original"}}]
    parent = SimpleNamespace()
    assert apply_tool_text(schema, agent_tool_text(parent))[0]["function"]["description"] == "parent"
    with delegated_child_context():
        assert apply_tool_text(schema, agent_tool_text(SimpleNamespace()))[0]["function"]["description"] == "original"
        assert tool_text("tool_results.demo", "original result") == "original result"
    other = tmp_path / "other"
    other.mkdir()
    (other / "prompt_overrides.yaml").write_text("tool_descriptions:\n  demo: child\n")
    with delegated_profile_scope(other), delegated_child_context(profile_home=other):
        assert apply_tool_text(schema, agent_tool_text(SimpleNamespace()))[0]["function"]["description"] == "child"
    path.write_text("tool_descriptions:\n  demo: edited\n")
    assert apply_tool_text(schema, agent_tool_text(parent))[0]["function"]["description"] == "parent"
    assert prompt_override("tool_descriptions.demo") == "edited"


def test_result_template_override_and_original_bytes(tmp_path, monkeypatch):
    from agent.delegation_context import delegated_child_context
    from tools.terminal_hints import annotate_masked_success, _MASKING_SHAPES

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expected = _MASKING_SHAPES[0][1]
    assert annotate_masked_success("make | tail", "BUILD FAILED") == expected
    (tmp_path / "prompt_overrides.yaml").write_text(
        "tool_results:\n  terminal:\n    pipeline_masked_success: replacement\n")
    assert annotate_masked_success("make | tail", "BUILD FAILED") == "replacement"
    with delegated_child_context():
        assert annotate_masked_success("make | tail", "BUILD FAILED") == expected


def test_agent_surface_and_describe_share_pinned_text_not_shared_memo(tmp_path, monkeypatch):
    import json
    import model_tools
    from toolsets import agent_tool_surface
    from tools.prompt_text import tool_text_scope
    from tools.registry import registry
    from agent.delegation_context import delegated_child_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    name = "_prompt_text_deferred_demo"
    registry.register(name=name, toolset="prompt_text_demo", schema={"name": name,
        "description": "original description", "parameters": {"type": "object", "properties": {
            "submit": {"type": "boolean", "description": "original param"}}}}, handler=lambda args: "ok")
    (tmp_path / "config.yaml").write_text("tools:\n  tool_search:\n    enabled: on\n    defer: []\n")
    path = tmp_path / "prompt_overrides.yaml"
    path.write_text(safe_dump({"tool_descriptions": {name: {
        "description": "parent wording", "params": {"submit": "parent param"}},
        "tool_describe": {"params": {"names": "custom bridge param"}}}}))
    def fresh():
        return model_tools.get_tool_definitions(enabled_toolsets=["prompt_text_demo"],
            quiet_mode=True, skip_tool_search_assembly=True)
    parent = SimpleNamespace(platform="cli")
    raw = fresh()
    surface = agent_tool_surface(parent, raw)
    assert "parent wording" in json.dumps(surface)
    assert "custom bridge param" in json.dumps(surface)
    assert "parent wording" not in json.dumps(fresh())
    path.write_text("tool_descriptions:\n  _prompt_text_deferred_demo: edited\n")
    with tool_text_scope(parent):
        result = json.loads(model_tools.handle_function_call("tool_describe", {"names": [name]},
            enabled_toolsets=["prompt_text_demo"]))["tools"][name]
    assert result["description"] == "parent wording"
    assert result["parameters"]["properties"]["submit"]["description"] == "parent param"
    assert agent_tool_surface(parent, fresh()) == surface
    with delegated_child_context():
        child = SimpleNamespace(platform="subagent", _delegate_depth=1)
        child_surface = agent_tool_surface(child, fresh())
        with tool_text_scope(child):
            described = json.loads(model_tools.handle_function_call("tool_describe", {"names": [name]},
                enabled_toolsets=["prompt_text_demo"]))["tools"][name]
        assert "parent wording" not in json.dumps(child_surface)
        assert described["description"] == "original description"
        assert described["parameters"]["properties"]["submit"]["description"] == "original param"


def test_notification_template_keeps_evidence_and_footer_is_idempotent(tmp_path, monkeypatch):
    from tools.process_registry_notifications import format_process_notification
    from gateway.run_notifications import _mark_internal_notification, INTERNAL_NOTIFICATION_FOOTER

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    evt = {"session_id": "process-test", "exit_code": 0, "command": "printf '{hello}'", "output": "{output}"}
    expected = "[IMPORTANT: Background process process-test completed normally (exit code 0).\nCommand: printf '{hello}'\nOutput:\n{output}]"
    assert format_process_notification(evt) == expected
    assert _mark_internal_notification("body") == "body\n\n" + INTERNAL_NOTIFICATION_FOOTER
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({"notifications": {
        "process": {"completion": "{session_id}: {command} -> {output} ({exit_code})"},
        "gateway": {"internal_footer": "custom footer"}}}))
    assert format_process_notification(evt) == "process-test: printf '{hello}' -> {output} (0)"
    marked = _mark_internal_notification("body")
    assert marked == "body\n\ncustom footer"
    assert _mark_internal_notification(marked) == marked


def test_concurrent_describe_uses_initial_snapshot(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock, patch
    from run_agent import AIAgent
    from tools.registry import registry
    from agent.tool_executor import _ConcurrentBatch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    name = "_concurrent_prompt_demo"
    registry.register(name=name, toolset="concurrent_prompt_demo", schema={"name": name,
        "description": "original", "parameters": {"type": "object", "properties": {}}}, handler=lambda args: "ok")
    path = tmp_path / "prompt_overrides.yaml"
    path.write_text(safe_dump({"tool_descriptions": {name: "initial wording"}}))
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(model="test/model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
            enabled_toolsets=["concurrent_prompt_demo"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, max_iterations=3)
    agent.client = MagicMock()
    agent._cached_system_prompt = "Use the available tools."
    agent._use_prompt_caching = False
    agent.compression_enabled = agent.save_trajectories = False
    path.write_text(safe_dump({"tool_descriptions": {name: "edited wording"}}))
    calls = [SimpleNamespace(id=f"describe-{i}", type="function", function=SimpleNamespace(
        name="tool_describe", arguments=json.dumps({"names": [name] * (i + 1)}))) for i in range(2)]
    def response(calls):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="" if calls else "done", tool_calls=calls), finish_reason="tool_calls" if calls else "stop")],
            model="test/model", usage=None)
    agent.client.chat.completions.create.side_effect = [response(calls), response(None)]
    with patch.object(_ConcurrentBatch, "_dispatch_worker", autospec=True,
                      side_effect=_ConcurrentBatch._dispatch_worker) as worker:
        result = agent.run_conversation("Describe both tools.")
    assert worker.call_count == 2
    rows = [row for row in result["messages"] if row["role"] == "tool"]
    assert len(rows) == 2
    assert all(json.loads(row["content"])["tools"][name]["description"] == "initial wording" for row in rows)


def test_nested_parameter_description_and_children(tmp_path, monkeypatch):
    from tools.prompt_text import apply_tool_text
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({"tool_descriptions": {"demo": {"params": {
        "options": {"description": "options text", "params": {"submit": "submit text"}}}}}}))
    schema = [{"function": {"name": "demo", "parameters": {"properties": {"options": {
        "description": "old options", "properties": {"submit": {"description": "old submit"}}}}}}}]
    options = apply_tool_text(schema)[0]["function"]["parameters"]["properties"]["options"]
    assert options["description"] == "options text"
    assert options["properties"]["submit"]["description"] == "submit text"


def test_refresh_overlays_injected_tools_without_changing_prefix(tmp_path, monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "prompt_overrides.yaml"
    path.write_text("tool_descriptions:\n  injected_demo: injected wording\n")
    schema = {"name": "injected_demo", "description": "original", "parameters": {}}
    agent = SimpleNamespace(tools=[], valid_tool_names=set(), enabled_toolsets=["context_engine"],
        disabled_toolsets=None, context_compressor=SimpleNamespace(get_tool_schemas=lambda: [schema]))
    refresh_agent_mcp_tools(agent)
    assert agent.tools[0]["function"]["description"] == "injected wording"
    prefix = agent.tools.copy()
    path.write_text("tool_descriptions:\n  injected_demo: edited wording\n")
    refresh_agent_mcp_tools(agent, preserve_prefix=True)
    assert agent.tools == prefix
    assert schema["description"] == "original"


def test_tool_search_frame_override_keeps_live_listing(tmp_path, monkeypatch):
    from tools.prompt_text import apply_tool_text
    from tools.tool_search import bridge_tool_schemas

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({"tool_descriptions": {"tool_search": {
        "frame": {"intro": "有 {count} 個工具. ", "listed": "\n\n直接 `{describe}`。"},
        "params": {"limit": "每次幾條"}}}}))
    schemas = apply_tool_text(bridge_tool_schemas(2, listing="- a: x\n- b: y"))
    fn = {s["function"]["name"]: s["function"] for s in schemas}["tool_search"]
    assert fn["description"].startswith("有 2 個工具. ")
    assert "直接 `tool_describe`。" in fn["description"]
    assert fn["description"].endswith("- a: x\n- b: y")
    assert fn["parameters"]["properties"]["limit"]["description"] == "每次幾條"


@pytest.mark.parametrize("replacement", ["Bounds {image_width}x{image_height}", ""])
@pytest.mark.parametrize("mode", ["ax", "som"])
def test_computer_use_bounds_hint_override_is_main_only(tmp_path, monkeypatch, replacement, mode):
    import json
    from agent.delegation_context import delegated_child_context
    from tools.computer_use.backend import CaptureResult, UIElement
    from tools.computer_use import tool as cu_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cu_tool, "_should_route_through_aux_vision", lambda: False)
    monkeypatch.setattr(cu_tool, "_persist_capture_image", lambda cap: None)
    cap = CaptureResult(mode=mode, width=8, height=8,
        png_b64=("iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76L"
                 "AAAADUlEQVR4nGNgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="),
        elements=[UIElement(index=0, role="section", bounds=(0, 48, 1296, 1917))])

    def summary():
        result = cu_tool._capture_response(cap)
        return result["text_summary"] if isinstance(result, dict) else json.loads(result)["summary"]

    original = summary()
    assert "does not establish a scale factor" in original
    (tmp_path / "prompt_overrides.yaml").write_text(safe_dump({"tool_results": {
        "computer_use": {"bounds_note": replacement}}}))
    overridden = summary()
    assert "does not establish a scale factor" not in overridden
    assert "estimated scale" not in overridden
    if replacement:
        assert "Bounds 8x8" in overridden
    else:
        assert "  ()" not in overridden
    with delegated_child_context():
        assert summary() == original
