"""Window captures negotiate optional arguments with the connected driver's schema."""

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.cua_backend_session import _CuaDriverSession


_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nGNgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="


class _WindowSession(_CuaDriverSession):
    def __init__(self, modern=True, refetch=False):
        super().__init__(None)
        properties = {name: {} for name in ("pid", "window_id", "session", "max_elements")}
        if modern:
            properties.update({"tree_format": {"enum": ["markdown", "elements", "both"]},
                               "include_accessibility_tree": {"type": "boolean"},
                               "include_screenshot": {"type": "boolean"}})
        self._tool_schemas = {"get_window_state": {"properties": properties, "additionalProperties": False}}
        self.accepted_properties = properties
        self._tool_schemas["click"] = {"properties": {"element_token": {}}}
        self.calls = []
        self.refetch = refetch

    def _response(self, name, args, transport):
        self.calls.append((transport, name, args.copy()))
        if name != "get_window_state":
            return {"data": "", "isError": False, "structuredContent": {"ok": True}}
        assert not (args.keys() - self.accepted_properties.keys())
        if self.refetch and transport == "mcp":
            return {}
        out = {"data": 'Window\n[1] AXButton "Save"', "structuredContent": {}}
        if args.get("include_screenshot", True):
            out["images"] = [_PNG]
        if args.get("include_accessibility_tree", True) and args.get("tree_format") in {"elements", "both"}:
            out["structuredContent"]["elements"] = [{
                "element_index": 1, "role": "AXButton", "label": "Save",
                "frame": {"x": 12, "y": 24, "w": 80, "h": 30}, "element_token": "snap:1",
            }]
        return out

    def call_tool(self, name, args, **kwargs):
        return self._response(name, args, "mcp")

    def _call_tool_via_cli(self, name, args, timeout):
        return self._response(name, args, "cli")


@pytest.mark.parametrize("mode", ["ax", "som"])
@pytest.mark.parametrize("refetch", [False, True])
def test_structured_capture_retains_bounds_and_click_tokens(mode, refetch):
    backend = CuaDriverBackend()
    backend._session = session = _WindowSession(refetch=refetch)
    cap = backend.capture(mode=mode, pid=9, window_id=12)
    assert cap.elements[0].bounds == (12, 24, 80, 30)
    assert cap.elements[0].element_token == "snap:1"
    assert backend.click(element=1).ok
    assert session.calls[-1][2]["element_token"] == "snap:1"
    assert cap.png_b64 == (None if mode == "ax" else _PNG)
    assert all(args["session"] == backend._session_id for _, _, args in session.calls)


@pytest.mark.parametrize("refetch", [False, True])
def test_vision_skips_accessibility_walk_but_keeps_window_snapshot(refetch):
    backend = CuaDriverBackend()
    backend._session = session = _WindowSession(refetch=refetch)
    cap = backend.capture(mode="vision", pid=9, window_id=12)
    assert cap.png_b64 == _PNG
    assert cap.elements == []
    assert (backend._active_pid, backend._active_window_id) == (9, 12)
    for _, name, args in session.calls:
        assert name == "get_window_state"
        assert args["include_accessibility_tree"] is False
        assert "tree_format" not in args
        assert "include_screenshot" not in args
        assert "display_only" not in args
        assert args["session"] == backend._session_id


@pytest.mark.parametrize("mode", ["ax", "som", "vision"])
@pytest.mark.parametrize("discovered", [False, True])
def test_old_or_undiscovered_schema_omits_optional_capture_arguments(mode, discovered):
    backend = CuaDriverBackend()
    backend._session = session = _WindowSession(modern=False)
    if not discovered:
        session._tool_schemas = {}
    cap = backend.capture(mode=mode, pid=9, window_id=12)
    assert cap.png_b64 == _PNG
    assert not ({"tree_format", "include_screenshot", "include_accessibility_tree"} & session.calls[0][2].keys())
