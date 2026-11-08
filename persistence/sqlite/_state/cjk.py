"""Load the optional native CJK-bigram tokenizer on each SQLite connection."""

import logging
import os
import sqlite3
from pathlib import Path

from sqlite_support import routed_sessions_setting

logger = logging.getLogger("hermes_state")

def fts5_cjk_so_path(home: Path) -> Path:
    """Location of the cjk_unicode61 loadable extension."""
    env = os.getenv("HERMES_FTS5_CJK_SO")
    return Path(env).expanduser() if env else home / "lib" / "libfts5_cjk.so"

def _cjk_fts_config_enabled() -> bool:
    """config.yaml ``sessions.cjk_fts`` (default on) for the profile being served."""
    value = routed_sessions_setting("cjk_fts", "HERMES_CJK_FTS")
    return value is None or str(value).strip().lower() not in ("0", "false", "off", "no")

def load_fts5_cjk_extension(conn: sqlite3.Connection, home: Path) -> bool:
    """Best-effort load of the cjk_unicode61 tokenizer; False (never raises) when
    the .so is absent, ``sessions.cjk_fts`` is off, or loading is compiled out."""
    path = fts5_cjk_so_path(home)
    if not _cjk_fts_config_enabled() or not path.exists():
        return False
    try:
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(path))
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:
        logger.warning("fts5_cjk extension load failed (%s)", path, exc_info=True)
        return False
