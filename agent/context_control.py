"""Runtime context injection control.

Reads ``~/.hermes/context_control.yaml`` (or ``$HERMES_HOME/context_control.yaml``)
and exposes the current settings to the conversation loop and memory providers.

The file is re-read on every turn so changes take effect immediately without
restarting the gateway.  The agent can modify the file via a tool to dynamically
adjust its own context injection behaviour.
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "enabled": True,
    "position": "before",  # before | after | system
    "min_score": 0.55,
    "top_k": 5,
    "max_items": 5,
}


@dataclass
class ContextControlSettings:
    """Current context injection settings."""

    enabled: bool = True
    position: str = "before"  # before | after | system
    min_score: float = 0.55
    top_k: int = 5
    max_items: int = 5

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "position": self.position,
            "min_score": self.min_score,
            "top_k": self.top_k,
            "max_items": self.max_items,
        }


_lock = threading.Lock()
_cached: Optional[ContextControlSettings] = None
_cached_mtime: float = 0.0


def _config_path() -> Path:
    """Resolve the context_control.yaml path."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(hermes_home) / "context_control.yaml"


def load_settings(force_reload: bool = False) -> ContextControlSettings:
    """Load settings from disk, with mtime-based caching.

    Re-reads only when the file has been modified since last load.
    Returns defaults if the file doesn't exist or is malformed.
    """
    global _cached, _cached_mtime

    path = _config_path()

    with _lock:
        if not path.exists():
            if _cached is None:
                _cached = ContextControlSettings()
            return _cached

        try:
            mtime = path.stat().st_mtime
        except OSError:
            if _cached is None:
                _cached = ContextControlSettings()
            return _cached

        if not force_reload and _cached is not None and mtime == _cached_mtime:
            return _cached

        try:
            import yaml  # noqa: F401 — available in hermes venv

            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            openviking = raw.get("openviking", raw)  # support both top-level and nested

            settings = ContextControlSettings(
                enabled=bool(openviking.get("enabled", _DEFAULTS["enabled"])),
                position=str(openviking.get("position", _DEFAULTS["position"])),
                min_score=float(openviking.get("min_score", _DEFAULTS["min_score"])),
                top_k=int(openviking.get("top_k", _DEFAULTS["top_k"])),
                max_items=int(openviking.get("max_items", _DEFAULTS["max_items"])),
            )

            # Validate position
            if settings.position not in ("before", "after", "system"):
                logger.warning(
                    "context_control: invalid position %r, falling back to 'before'",
                    settings.position,
                )
                settings.position = "before"

            _cached = settings
            _cached_mtime = mtime
            return settings

        except Exception as e:
            logger.warning("context_control: failed to load %s: %s", path, e)
            if _cached is None:
                _cached = ContextControlSettings()
            return _cached


def save_settings(settings: ContextControlSettings) -> None:
    """Write current settings to disk."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    data = {"openviking": settings.to_dict()}

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    # Invalidate cache so next load picks up the new mtime
    global _cached_mtime
    with _lock:
        _cached_mtime = 0.0
