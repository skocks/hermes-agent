"""External history store for the RLM context engine.

Deliberately separate from ~/.hermes/lcm.db — RLM never summarizes into
this store, it only appends. One row per archived message, addressable by
(session_id, insertion order). WAL mode + a busy timeout make it safe for
concurrent access from more than one process against the same file: this
class writes from the engine process, and the persistent REPL subprocess
(repl.py) reads from it via its own independent sqlite3 connection rather
than going through this class at all — WAL mode is exactly the mode that
makes that safe. A process-local lock additionally serializes writes from
this engine's own instance (SQLite's per-connection API isn't thread-safe
on its own even with check_same_thread=False).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

_LIKE_ESCAPE = "\\"


def _escape_like(term: str) -> str:
    return (
        term.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", _LIKE_ESCAPE + "%")
        .replace("_", _LIKE_ESCAPE + "_")
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS rlm_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_id     INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    ts          REAL NOT NULL,
    superseded  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_rlm_session_turn ON rlm_messages(session_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_rlm_session_ts ON rlm_messages(session_id, ts);
"""
# idx_rlm_session_superseded deliberately NOT in _SCHEMA above: on an
# existing (pre-N2) database the table already exists, so CREATE TABLE IF
# NOT EXISTS is a no-op and the superseded column doesn't exist yet --
# indexing a column that isn't there yet fails immediately, before
# _migrate_add_superseded_column() ever runs. Created after migration
# instead, once the column is guaranteed present either way (fresh CREATE
# TABLE or ALTER TABLE).

# N2 fix: a resync (shrink-guard or M6 verification failure) used to
# re-append the entire live transcript over rows already archived,
# duplicating them -- and those duplicates flowed straight into
# history()/context, so the model could see the same message several
# times and duplicates ate into the 5000-row context cap. A plain DELETE
# before resync isn't safe either: it would destroy archive-only history,
# which is exactly the situation that triggers a resync in the first
# place (see _archive_new's shrink-guard).
#
# Fix: tombstone instead of delete or duplicate. supersede_session() marks
# every existing row for a session as superseded=1 right before a resync
# re-inserts the full current transcript as fresh (superseded=0) rows.
# Nothing is ever deleted -- old rows stay in the table, physically
# recoverable -- but every read path below filters superseded=0, so
# history()/context/message_count only ever see the current, non-
# duplicated view.


class RLMStore:
    def __init__(self, db_path: str, busy_timeout_ms: int = 5000):
        self.db_path = db_path
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(Path(db_path).expanduser()),
            check_same_thread=False,
            timeout=busy_timeout_ms / 1000,
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate_add_superseded_column()
            # Safe unconditionally at this point: the column exists either
            # way (fresh CREATE TABLE included it, or the migration above
            # just ALTERed it in).
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rlm_session_superseded "
                "ON rlm_messages(session_id, superseded)"
            )
            self._conn.commit()

    def _migrate_add_superseded_column(self) -> None:
        """N2 fix: an existing ~/.hermes/rlm.db predates the superseded
        column (_SCHEMA's CREATE TABLE IF NOT EXISTS is a no-op against an
        already-existing table, so the column never gets added that way).
        Caller holds self._lock.
        """
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(rlm_messages)")}
        if "superseded" in cols:
            return
        self._conn.execute(
            "ALTER TABLE rlm_messages ADD COLUMN superseded INTEGER NOT NULL DEFAULT 0"
        )

    def append_messages(
        self, session_id: str, turn_id: int, messages: List[Dict[str, Any]]
    ) -> None:
        """Persist raw messages, tagged with the turn they belong to."""
        if not messages:
            return
        now = time.time()
        rows = [
            (session_id, turn_id, m.get("role", "unknown"), _stringify(m), now)
            for m in messages
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO rlm_messages (session_id, turn_id, role, content, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def search_any(self, session_id: str, keywords: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Engine-internal relevance check for auto-recall — NOT exposed to
        the model (rlm_repl's history()/context give it a much richer way
        to search). Cheap OR-of-LIKE match against a handful of keywords,
        used only to decide "does this turn's question plausibly need
        dropped history" before spending a digestion sub-call on it.
        """
        keywords = [k for k in (keywords or []) if k][:8]
        if not keywords:
            return []
        clauses = " OR ".join(["content LIKE ? ESCAPE '\\'"] * len(keywords))
        params = [f"%{_escape_like(k)}%" for k in keywords]
        limit = max(1, min(int(limit), 100))
        with self._lock:
            cur = self._conn.execute(
                f"SELECT turn_id, role, content, ts FROM rlm_messages "
                f"WHERE session_id = ? AND superseded = 0 AND ({clauses}) "
                f"ORDER BY id DESC LIMIT ?",
                (session_id, *params, limit),
            )
            rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]

    def message_count(self, session_id: str) -> int:
        """Count of the current, non-superseded view -- what history()/
        context actually show. This is also on_session_start()'s resume-
        watermark estimate (see engine.py's M6 fix), which is the whole
        reason superseded rows must be excluded here: counting them would
        reintroduce the inflation M6 exists to catch, just one layer up.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM rlm_messages WHERE session_id = ? AND superseded = 0",
                (session_id,),
            )
            return cur.fetchone()[0] or 0

    def raw_row_count(self, session_id: str) -> int:
        """Count of ALL physical rows for a session, superseded or not.
        Debug/test use -- confirms tombstoning preserves history rather
        than deleting it (superseded rows are never dropped, only hidden
        from the normal read paths above).
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM rlm_messages WHERE session_id = ?",
                (session_id,),
            )
            return cur.fetchone()[0] or 0

    def supersede_session(self, session_id: str) -> None:
        """Tombstone every current row for a session before a resync
        re-inserts the full live transcript as fresh rows. Never deletes --
        old rows stay physically in the table (see raw_row_count), just
        excluded from message_count/tail_content/search_any/history()
        going forward, so a resync stops duplicating what the model
        actually sees without losing archive-only history in the process.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE rlm_messages SET superseded = 1 WHERE session_id = ? AND superseded = 0",
                (session_id,),
            )
            self._conn.commit()

    def tail_content(self, session_id: str, n: int) -> List[str]:
        """Content of the last n archived rows (current view only, oldest-
        first) -- for M6's resume-watermark verification: compare against
        the corresponding tail of the live transcript to check
        message_count() actually still lines up with reality before
        trusting it as the resume position.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT content FROM rlm_messages WHERE session_id = ? AND superseded = 0 "
                "ORDER BY id DESC LIMIT ?",
                (session_id, max(0, int(n))),
            )
            rows = [r[0] for r in cur.fetchall()]
        return list(reversed(rows))

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()


def _stringify(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _row_to_dict(row) -> Dict[str, Any]:
    turn_id, role, content, ts = row
    return {"turn_id": turn_id, "role": role, "content": content, "ts": ts}
