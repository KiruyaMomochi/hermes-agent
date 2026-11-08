"""Optional CJK index lifecycle; never retire a live index for a missing tokenizer."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from .cjk import fts5_cjk_so_path
from .._schema_ddl import FTS_CJK_TABLE_SQL, FTS_CJK_TRIGGER_SQL, FTS_CJK_STALE_KEY, _FTS_CJK_TRIGGERS

if TYPE_CHECKING:
    from .._holder import StoreHolder

logger = logging.getLogger("hermes_state")

def _ensure_fts_cjk_schema(holder: StoreHolder, cursor) -> None:
    """Create / repair / self-heal the CJK-bigram index (see the module comment).
    Sets ``_fts_cjk_available``; never raises. Loaded + absent → create (a
    populated DB gets backfill markers and is NOT served until optimize-storage
    backfills); loaded + present → ensure triggers, honour the stale breadcrumb;
    NOT loaded + live triggers → drop them (INSERTs must not fail at trigger time)."""
    try:
        cjk_present = bool(cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts_cjk'"
        ).fetchone())
        if not holder._fts_cjk_loaded:
            if cjk_present:
                live = [r[0] for r in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                    f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)})",
                    _FTS_CJK_TRIGGERS,
                ).fetchall()]
                if live:
                    # Breadcrumb FIRST (a crash between the two is merely conservative).
                    logger.warning(
                        "messages_fts_cjk triggers present but the "
                        "cjk_unicode61 tokenizer is unavailable (%s) — "
                        "dropping the cjk triggers so message writes keep "
                        "working. CJK search falls back to trigram/LIKE; "
                        "run `hermes sessions optimize-storage` on a host "
                        "with the extension to rebuild.",
                        fts5_cjk_so_path(holder.db_path.parent),
                    )
                    cursor.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = '1'",
                        (FTS_CJK_STALE_KEY,),
                    )
                    for trig in live:
                        cursor.execute(f"DROP TRIGGER IF EXISTS {trig}")
            holder._fts_cjk_available = False
            return
    except sqlite3.OperationalError:
        logger.warning(
            "messages_fts_cjk presence check failed; CJK search stays on "
            "trigram/LIKE", exc_info=True,
        )
        holder._fts_cjk_available = False
        return
    try:
        cursor.executescript(FTS_CJK_TABLE_SQL)
        if not cjk_present:
            # An old stale breadcrumb refers to a table that no longer exists.
            cursor.execute("DELETE FROM state_meta WHERE key = ?", (FTS_CJK_STALE_KEY,))
            # Empty DB: complete by construction, no markers. Populated DB: the
            # marker pair keeps the id-gated triggers correct until backfill.
            if cursor.execute("SELECT COUNT(*) FROM messages WHERE role <> 'tool'").fetchone()[0] > 0:
                hw = cursor.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
                for k, v in (
                    ("fts_cjk_rebuild_high_water", str(hw)), ("fts_cjk_rebuild_progress", "0"),
                ):
                    cursor.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (k, v),
                    )
        if cursor.execute("SELECT 1 FROM state_meta WHERE key = ?", (FTS_CJK_STALE_KEY,)).fetchone():
            # Gap of unknown extent: do NOT reinstall triggers (see module comment).
            holder._fts_cjk_available = False
            return
        cursor.executescript(FTS_CJK_TRIGGER_SQL)
        backfill_pending = cursor.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_cjk_rebuild_high_water' LIMIT 1"
        ).fetchone()
        holder._fts_cjk_available = not backfill_pending
    except sqlite3.OperationalError:  # incl. "no such tokenizer" after a failed registration
        logger.warning(
            "messages_fts_cjk ensure failed; CJK search stays on "
            "trigram/LIKE", exc_info=True,
        )
        holder._fts_cjk_available = False

def _quarantine_cjk_after_update_of_migration(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Fail closed after dropping the CJK UPDATE trigger mid-migration: clear availability,
    persist ``fts_cjk_stale``, drop any residual trigger so a later open cannot
    IF-NOT-EXISTS over a gap."""
    holder._fts_cjk_available = False
    try:
        holder.set_meta(FTS_CJK_STALE_KEY, "1", cursor=cursor)
    except Exception:
        logger.debug("Could not persist CJK FTS stale breadcrumb", exc_info=True)
    try:
        cursor.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
    except Exception:
        logger.debug("Could not drop residual CJK UPDATE trigger after quarantine", exc_info=True)
