"""Profile-local tool wording, applied after the shared schema memo.

Schema overrides are pinned per agent; result text is resolved at emission time.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Callable

from tools.registry import prompt_override

_schema_text: ContextVar[Callable | None] = ContextVar("tool_schema_text", default=None)


def tool_text(key, default):
    """Resolve a result template without changing missing/empty override semantics."""
    from agent.delegation_context import tool_prompt_overrides_allowed
    if not tool_prompt_overrides_allowed():
        return default
    value = prompt_override(key)
    return default if value is None else value


def _description(node, key, resolve):
    value = resolve(key)
    if value is not None:
        node["description"] = value


def _parameters(node, key, resolve):
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    for name, prop in (properties.items() if isinstance(properties, dict) else ()):
        if not isinstance(prop, dict):
            continue
        path = f"{key}.params.{name}"
        _description(prop, path, resolve)
        _description(prop, f"{path}.description", resolve)
        _parameters(prop, path, resolve)
    items = node.get("items")
    if isinstance(items, dict):
        _parameters(items, f"{key}.items", resolve)


def apply_tool_text(tools, resolve=None, *, vault_note=True):
    """Copy schemas before replacing descriptions, including nested object/array params."""
    resolve = resolve or _schema_text.get() or (lambda key: tool_text(key, None))
    result = deepcopy(tools)
    for tool in result:
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue
        key = f"tool_descriptions.{fn.get('name', '')}"
        value = resolve(key)
        if value is not None:
            fn["description"] = value
        _description(fn, f"{key}.description", resolve)
        _parameters(fn.get("parameters") or {}, key, resolve)
        note = resolve("VAULT_NO_PASSWORD_NOTE") if vault_note else None
        if note is not None:
            from model_tools import _VAULT_NO_PASSWORD_NOTE
            if _VAULT_NO_PASSWORD_NOTE in fn.get("description", ""):
                fn["description"] = fn["description"].replace(_VAULT_NO_PASSWORD_NOTE, note)
    return result


def agent_tool_text(agent):
    """Pin the profile's resolver once, including an empty snapshot for ordinary children."""
    if "_tool_prompt_resolver" not in vars(agent):
        from agent.delegation_context import tool_prompt_overrides_allowed
        from tools.registry import _prompt_override_data, _prompt_override_value
        data = _prompt_override_data() if tool_prompt_overrides_allowed() else {}
        agent._tool_prompt_resolver = lambda key: _prompt_override_value(data, key)
    return agent._tool_prompt_resolver


@contextmanager
def tool_text_scope(agent):
    """Carry the pinned catalog wording across bridge calls and tool worker threads."""
    token = _schema_text.set(agent_tool_text(agent))
    try:
        yield
    finally:
        _schema_text.reset(token)


def invoke_with_tool_text(agent, invoke, *args, **kwargs):
    """Bind the same schema snapshot in concurrent dispatch as sequential dispatch."""
    with tool_text_scope(agent):
        return invoke(*args, **kwargs)
