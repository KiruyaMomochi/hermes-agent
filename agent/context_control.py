"""Per-turn controls for external memory context injection.

The optional ``context_control.yaml`` file is reloaded when its mtime changes,
allowing gateway users to tune recall without restarting a conversation.  Missing
or invalid files use conservative defaults and never interrupt a turn.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_POSITION = "after"


@dataclass(frozen=True)
class ContextControlSettings:
    enabled: bool = True
    position: str = _DEFAULT_POSITION
    min_score: float = 0.15
    top_k: int = 6
    max_items: int = 6

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
_cached_mtime: Optional[int] = None


def _config_path() -> Path:
    return get_hermes_home() / "context_control.yaml"


def load_settings(force_reload: bool = False) -> ContextControlSettings:
    """Return current controls, reloading the optional YAML file when changed."""
    global _cached, _cached_mtime
    path = _config_path()
    with _lock:
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            if _cached is None:
                _cached = ContextControlSettings()
            return _cached
        if not force_reload and _cached is not None and mtime == _cached_mtime:
            return _cached
        try:
            import yaml
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            values = raw.get("openviking", raw) if isinstance(raw, dict) else {}
            if not isinstance(values, dict):
                values = {}
            position = str(values.get("position", _DEFAULT_POSITION)).lower()
            if position not in {"before", "after", "system"}:
                logger.warning("context_control: invalid position %r; using %s", position, _DEFAULT_POSITION)
                position = _DEFAULT_POSITION
            settings = ContextControlSettings(
                enabled=bool(values.get("enabled", True)),
                position=position,
                min_score=max(0.0, min(1.0, float(values.get("min_score", 0.15)))),
                top_k=max(1, int(values.get("top_k", 6))),
                max_items=max(1, int(values.get("max_items", 6))),
            )
        except (OSError, TypeError, ValueError, ImportError) as exc:
            logger.warning("context_control: failed to load %s: %s", path, exc)
            settings = _cached or ContextControlSettings()
        _cached, _cached_mtime = settings, mtime
        return settings


def save_settings(settings: ContextControlSettings) -> None:
    """Persist controls for subsequent turns."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    path.write_text(yaml.safe_dump({"openviking": settings.to_dict()}, allow_unicode=True), encoding="utf-8")
    with _lock:
        global _cached_mtime
        _cached_mtime = None
