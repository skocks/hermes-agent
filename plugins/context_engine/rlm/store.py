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
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

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
            self._fts_enabled = self._init_fts()

    # L2 fix: search_any's original LIKE '%kw%' scan is fine on a match --
    # ORDER BY id DESC LIMIT ? short-circuits as soon as it finds enough
    # rows. But a MISS (the common case: search_any is auto-recall's
    # prefilter, called on every provider request to decide whether a turn
    # plausibly needs dropped history) has no early exit -- it scans every
    # non-superseded row in the session, evaluating LIKE against each one.
    # Measured on a single session with realistic tool-result-sized rows
    # (8KB), synchronously on select_context()'s hot path: ~83ms/miss at
    # 5k rows, ~720ms/miss at 50k rows -- multiplied by however many
    # provider round trips one tool-heavy turn makes. An inverted index
    # (FTS5) turns a miss into an index lookup instead of a table scan,
    # independent of session size. Mirrored explicitly in Python (not SQL
    # triggers) rather than via CREATE VIRTUAL TABLE's own content-table
    # sync, because this class is the only writer (repl.py only reads, via
    # its own connection -- see module docstring), so explicit mirroring
    # at the two write sites (append_messages, supersede_session) is the
    # whole story, no hidden trigger behavior to reason about.
    def _init_fts(self) -> bool:
        """Best-effort: some sqlite3 builds omit FTS5. Caller holds
        self._lock. Falls back to the LIKE scan (_search_any_like) if this
        returns False -- never a hard dependency.
        """
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS rlm_search USING fts5("
                "content, session_id UNINDEXED, superseded UNINDEXED, "
                "turn_id UNINDEXED, role UNINDEXED, ts UNINDEXED)"
            )
        except sqlite3.OperationalError:
            logger.warning(
                "RLM: FTS5 unavailable in this sqlite3 build, "
                "search_any falls back to a LIKE scan"
            )
            return False
        # Backfill: rlm_search's rowid mirrors rlm_messages.id exactly (set
        # explicitly on every insert below), so a plain count comparison
        # tells us whether this is a pre-existing rlm_messages table being
        # upgraded onto FTS5 for the first time.
        total = self._conn.execute("SELECT COUNT(*) FROM rlm_messages").fetchone()[0]
        indexed = self._conn.execute("SELECT COUNT(*) FROM rlm_search").fetchone()[0]
        if indexed < total:
            self._conn.execute(
                "INSERT INTO rlm_search(rowid, content, session_id, superseded, "
                "turn_id, role, ts) "
                "SELECT id, content, session_id, superseded, turn_id, role, ts "
                "FROM rlm_messages WHERE id NOT IN (SELECT rowid FROM rlm_search)"
            )
            self._conn.commit()
        return True

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
            if self._fts_enabled:
                # Re-select the rows just inserted (scoped to session_id,
                # newest-first, capped to this batch's size) rather than
                # trying to reconstruct ids from cursor.lastrowid -- that
                # field's behavior across executemany() isn't reliable
                # enough to bet row identity on. The re-select is cheap:
                # it's the same index this table already leans on
                # (session_id-scoped, small LIMIT).
                just_inserted = self._conn.execute(
                    "SELECT id, content, session_id, superseded, turn_id, role, ts "
                    "FROM rlm_messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, len(rows)),
                ).fetchall()
                self._conn.executemany(
                    "INSERT INTO rlm_search(rowid, content, session_id, superseded, "
                    "turn_id, role, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    just_inserted,
                )
            self._conn.commit()

    def search_any(self, session_id: str, keywords: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """Engine-internal relevance check for auto-recall — NOT exposed to
        the model (rlm_repl's history()/context give it a much richer way
        to search). Used on every provider request (select_context's hot
        path) to decide "does this turn's question plausibly need dropped
        history" before spending a digestion sub-call on it -- so the miss
        case (no plausible match) is the COMMON path, not the edge case,
        and it needs to stay cheap regardless of session size. Routes
        through FTS5 (an index lookup) when available, falling back to the
        original LIKE scan (a table scan on a miss) otherwise.
        """
        keywords = [k for k in (keywords or []) if k][:8]
        if not keywords:
            return []
        limit = max(1, min(int(limit), 100))
        if self._fts_enabled:
            return self._search_any_fts(session_id, keywords, limit)
        return self._search_any_like(session_id, keywords, limit)

    def _search_any_fts(self, session_id: str, keywords: List[str], limit: int) -> List[Dict[str, Any]]:
        # Keywords are pre-sanitized by _extract_keywords() to
        # [A-Za-z0-9_]+ before they ever reach here (engine.py) -- no FTS5
        # query-syntax characters (quotes, spaces, boolean operators) can
        # appear, so no escaping is needed. Trailing '*' keeps prefix-match
        # behavior close to the old LIKE '%kw%' semantics (matches "test"
        # inside "testing"); it does NOT match a keyword as a mid-word or
        # suffix substring ("attest") the way LIKE did -- an accepted,
        # narrower trade for turning a scan into an index lookup on a
        # prefilter that's explicitly "cheap and approximate" by design.
        match_query = " OR ".join(f"{kw}*" for kw in keywords)
        with self._lock:
            cur = self._conn.execute(
                "SELECT turn_id, role, content, ts FROM rlm_search "
                "WHERE rlm_search MATCH ? AND session_id = ? AND superseded = 0 "
                "ORDER BY rowid DESC LIMIT ?",
                (match_query, session_id, limit),
            )
            rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]

    def _search_any_like(self, session_id: str, keywords: List[str], limit: int) -> List[Dict[str, Any]]:
        clauses = " OR ".join(["content LIKE ? ESCAPE '\\'"] * len(keywords))
        params = [f"%{_escape_like(k)}%" for k in keywords]
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
            if self._fts_enabled:
                self._conn.execute(
                    "UPDATE rlm_search SET superseded = 1 WHERE session_id = ? AND superseded = 0",
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
