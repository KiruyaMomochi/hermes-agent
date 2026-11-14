"""Capture coordinates must not be inferred from accessibility content extents."""

import json

import pytest

from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.tool import _capture_response


@pytest.mark.parametrize("bounds", [
    [(0, 48, 1296, 690)],
    [(0, 48, 1296, 690), (12, 173, 1261, 1917)],
    [(3730, 0, 69, 60)],
    [(-1920, -200, 1296, 690)],
])
def test_ax_extents_do_not_estimate_pixel_click_scale(bounds):
    cap = CaptureResult(mode="ax", width=1280, height=672, elements=[
        UIElement(index=i, role="frame" if i == 0 else "section", bounds=rect)
        for i, rect in enumerate(bounds)
    ])
    out = json.loads(_capture_response(cap))
    assert "bounds_scale" not in out
    assert "estimated scale" not in out["summary"]
    assert "coordinate= clicks expect the native" not in out["summary"]
    assert "screenshot pixels" in out["summary"]


@pytest.mark.parametrize("discovered", [True, False])
def test_vision_capture_owns_snapshot_for_following_pixel_click(discovered):
    from unittest.mock import MagicMock
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    session = MagicMock()
    session.capabilities_discovered = discovered
    session._has_tool.return_value = True
    snapshot_owner = None
    calls = []
    png = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nGNgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="

    def call_tool(name, args):
        nonlocal snapshot_owner
        calls.append((name, args))
        if name == "get_window_state":
            snapshot_owner = args["session"]
            assert (args["pid"], args["window_id"]) == (9, 12)
            return {"images": [png], "image_mime_types": ["image/png"],
                    "structuredContent": {"elements": [{"element_index": 1, "role": "button"}]}}
        if name == "screenshot":
            return {"images": [png], "image_mime_types": ["image/png"]}
        if name == "click":
            return {"data": "", "isError": snapshot_owner != args["session"],
                    "structuredContent": {"ok": snapshot_owner == args["session"],
                    "code": None if snapshot_owner == args["session"] else "screenshot_context_missing"}}
        raise AssertionError(name)

    session.call_tool.side_effect = call_tool
    backend._session = session
    backend._snapshot_tokens = {1: "previous-snapshot"}
    cap = backend.capture(mode="vision", pid=9, window_id=12)
    result = backend.click(x=3, y=4, delivery_mode="foreground")
    assert result.ok, result
    assert result.code is None
    assert cap.png_b64 == png
    assert (cap.width, cap.height) == (8, 8)
    assert cap.elements == []
    assert backend._snapshot_tokens == {}
    assert [name for name, _ in calls] == ["get_window_state", "click"]
    assert calls[-1][1]["x"] == 3
    assert calls[-1][1]["y"] == 4
    assert calls[-1][1]["delivery_mode"] == "foreground"
