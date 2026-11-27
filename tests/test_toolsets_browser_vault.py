"""Browser vault can be denied without disabling browser automation."""

import json
from types import SimpleNamespace

import pytest

from model_tools import _dispatch_bridge_tool, _select_tool_names, get_tool_definitions
from tools.delegate_tool_toolsets import _resolve_child_toolsets


@pytest.fixture
def available_browser(monkeypatch):
    # Exercise real schemas/selection without requiring a running CDP endpoint or vault.
    monkeypatch.setattr("tools.registry._memo_check", lambda fn, memo: True)


@pytest.mark.parametrize(
    "disabled, expected",
    [
        ([], {"browser_exec", "browser_cdp", "browser_dialog", "browser_vault"}),
        (["browser_vault"], {"browser_exec", "browser_cdp", "browser_dialog"}),
        (["browser"], set()),
    ],
)
def test_browser_toolset_subtraction(disabled, expected):
    names = _select_tool_names(["browser"], disabled, quiet_mode=True)
    assert {name for name in names if name.startswith("browser_")} == expected


@pytest.mark.parametrize("disabled", [[], ["browser_vault"], ["browser"]])
def test_browser_vault_deferred_catalog_and_hints(available_browser, disabled):
    enabled = ["browser", "terminal"]
    vault_enabled = not disabled
    definitions = get_tool_definitions(
        enabled, disabled, quiet_mode=True, skip_tool_search_assembly=True,
    )
    by_name = {td["function"]["name"]: td["function"] for td in definitions}
    assert ("browser_vault" in by_name) == vault_enabled
    if disabled != ["browser"]:
        assert {"browser_exec", "browser_cdp", "browser_dialog"} <= by_name.keys()
        assert ("call browser_vault" in by_name["browser_exec"]["description"]) == vault_enabled
    else:
        assert not any(name.startswith("browser_") for name in by_name)

    # The eager tool-search listing must agree with the actual bridge catalog.
    assembled = get_tool_definitions(enabled, disabled, quiet_mode=True)
    search_description = next(
        (td["function"]["description"] for td in assembled
         if td["function"]["name"] == "tool_search"), "",
    )
    assert ("browser_vault" in search_description) == vault_enabled
    search, _ = _dispatch_bridge_tool(
        "tool_search", {"queries": ["browser_vault"]}, enabled, disabled,
    )
    assert ("browser_vault" in json.loads(search)["tools"]) == vault_enabled
    described, _ = _dispatch_bridge_tool(
        "tool_describe", {"names": ["browser_vault"]}, enabled, disabled,
    )
    assert ("browser_vault" in json.loads(described)["tools"]) == vault_enabled
    result, target = _dispatch_bridge_tool(
        "tool_call", {"name": "browser_vault", "arguments": {"action": "list"}},
        enabled, disabled,
    )
    if vault_enabled:
        assert result is None
        assert target == ("browser_vault", {"action": "list"})
    else:
        assert target is None
        assert json.loads(result)["error"]


@pytest.mark.parametrize("disabled", [[], ["browser_vault"], ["browser"]])
@pytest.mark.parametrize("requested", [None, ["browser", "terminal"], ["browser_vault"]])
@pytest.mark.parametrize("role", ["leaf", "orchestrator"])
def test_child_inherits_browser_vault_denial(available_browser, disabled, requested, role):
    parent = SimpleNamespace(enabled_toolsets=["browser", "terminal"], disabled_toolsets=disabled)
    enabled, child_disabled = _resolve_child_toolsets(parent, requested, role)
    assert set(disabled) <= set(child_disabled)
    definitions = get_tool_definitions(
        enabled, child_disabled, quiet_mode=True, skip_tool_search_assembly=True,
    )
    names = {td["function"]["name"] for td in definitions}
    assert ("browser_vault" in names) == (not disabled)
    if requested != ["browser_vault"]:
        expected = {"browser_exec", "browser_cdp", "browser_dialog"}
        assert (expected <= names) == (disabled != ["browser"])
