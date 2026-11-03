"""File-path skill hints, using the same catalog and load names as skill_view."""
from collections import OrderedDict
from functools import wraps
import json
import logging
from pathlib import Path
import re
import threading

from pathspec import PathSpec

logger = logging.getLogger(__name__)
_seen = OrderedDict()
_seen_lock = threading.Lock()


def matching_skills(paths, workspace, *, enabled_tools=None, enabled_toolsets=None):
    from agent.prompt_builder import _current_session_platform_hint, _skill_should_show
    from tools.skills_tool import _skill_catalog

    # Match logical paths, not symlink targets. Shared dotfiles/checkouts in both
    # local users' homes are intentional; /etc and /tmp stay outside this scope.
    roots = tuple(dict.fromkeys((Path(workspace), Path.home(), Path("/home/kyaru"))))
    relative = []
    for value in paths[:10]:
        path = Path(str(value)).expanduser()
        if ".." in path.parts:
            continue
        if path.is_absolute():
            for root in roots:
                if path.is_relative_to(root):
                    path = path.relative_to(root)
                    break
            else:
                continue
        relative.append(path.as_posix())
    found = []
    for skill in _skill_catalog():
        name = skill.get("load_name")
        if not name or not skill.get("covers"):
            continue
        if not _skill_should_show(skill.get("conditions", {}), enabled_tools,
                                  enabled_toolsets, _current_session_platform_hint()):
            continue
        patterns = PathSpec.from_lines("gitignore", skill["covers"])
        if any(patterns.match_file(path) for path in relative):
            found.append(name)
            if len(found) == 5:
                break
    return found


def with_skill_advisory(handler):
    """Keep hints inside JSON: RPC and strict-provider consumers parse these results."""
    @wraps(handler)
    def wrapped(args, **kw):
        result = handler(args, **kw)
        try:
            payload = json.loads(result)
            if not isinstance(payload, dict) or payload.get("error") or payload.get("success") is False:
                return result
            paths = [args.get("path") or args.get("file_path") or "."]
            paths += payload.get("files", [])
            paths += list(payload.get("counts", {}))
            paths += [m["path"] for m in payload.get("matches", []) if "path" in m]
            # This is SearchResult's documented path-grouped representation.
            paths += [line for line in payload.get("matches_text", "").splitlines()
                      if line and not line.startswith("  ")]
            paths += re.findall(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$",
                                args.get("patch") or "", re.MULTILINE)
            for old, new in re.findall(r"^\*\*\* Move File: (.+?) -> (.+)$",
                                       args.get("patch") or "", re.MULTILINE):
                paths.extend((old, new))
            from tools.file_tools_paths import _resolve_base_dir
            from hermes_constants import get_hermes_home
            workspace = str(_resolve_base_dir(kw.get("task_id") or "default"))
            names = matching_skills(paths, workspace, enabled_tools=kw.get("enabled_tools"),
                                    enabled_toolsets=kw.get("enabled_toolsets"))
            session = kw.get("session_id") or kw.get("task_id")
            if session:
                key = (str(get_hermes_home()), workspace, session)
                with _seen_lock:
                    seen = _seen.setdefault(key, set())
                    names = [name for name in names if name not in seen]
                    seen.update(names)
                    _seen.move_to_end(key)
                    while len(_seen) > 128:
                        _seen.popitem(last=False)
            if names:
                payload["_skill_advisory"] = (
                    f"Relevant skills available: {', '.join(names)}. "
                    "Load one with `skill_view` if applicable.")
                return json.dumps(payload, ensure_ascii=False)
        except Exception:
            logger.debug("Could not add file-path skill advisory", exc_info=True)
        return result
    return wrapped
