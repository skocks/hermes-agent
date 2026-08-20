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

Retention: rlm.db has no independent retention policy and never will —
see sweep_orphaned_sessions(). It inherits state.db's session lifecycle
instead: a session's RLM archive (including its tombstoned/superseded
rows and its FTS5 index entries) is only ever deleted once that session
no longer exists in state.db's `sessions` table, via the SAME
`sessions.auto_prune` / `retention_days` / `min_interval_hours` /
`vacuum_after_prune` / `min_vacuum_interval_days` config keys state.db's
own SessionDB.maybe_auto_prune_and_vacuum already uses — no
`rlm.retention_days` or other RLM-specific age setting exists, or should
ever be added. If session pruning is off (the default), state.db never
loses a session_id, so this sweep deletes nothing. RLM follows state.db's
retention; it does not own or lead it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
CREATE TABLE IF NOT EXISTS rlm_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS rlm_seen_sessions (
    session_id       TEXT PRIMARY KEY,
    first_confirmed_ts REAL NOT NULL
);
"""
# rlm_seen_sessions -- round 20. Positive evidence a session_id was
# actually confirmed present in state.db at some point, not just "we
# have RLM archive rows for it". See sweep_orphaned_sessions's own
# docstring for why absence alone stopped being sufficient grounds for
# deletion: create_session() can silently fail (a pre-existing,
# documented hermes_state.py failure mode under write contention), so a
# session state.db never registered looks IDENTICAL to a session that
# was registered and later pruned -- "not in state.db" conflated "never
# existed here" with "existed and was released". A real user session
# lost this way had ~1.4MB of conversation whose ONLY surviving copy was
# this store, one sweep away from deletion. Now: a session_id is only
# ever eligible for the sweep once it has been POSITIVELY seen present
# in state.db at least once; a session_id absent from state.db that was
# never recorded here is left alone indefinitely, not swept "eventually"
# under some age rule -- because the failure mode being guarded against
# (create_session never succeeding) does not resolve itself with time.
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
        """Tombstone EVERY current row for a session, unconditionally.
        Never deletes -- old rows stay physically in the table (see
        raw_row_count), just excluded from message_count/tail_content/
        search_any/history() going forward.

        Round-18: this is a blunt primitive, not the resync fix. Its own
        docstring used to claim a resync using this "stops duplicating
        ... without losing archive-only history in the process" -- false,
        and the false claim went unverified for 8 rounds: if the live
        transcript being re-archived after this call is a DROPPED-MIDDLE
        view (the normal case -- select_context() drops the middle from
        every request once a session is long enough), the re-append never
        reproduces the rows this just tombstoned, and content that
        existed ONLY in the archive is gone from every read path while
        still physically present -- indistinguishable from data loss to
        the model, and it produced a fabricated case in a real user
        deliverable before being caught. See
        supersede_reproduced_rows() -- engine.py's resync path uses that,
        not this, for exactly that reason. This method is still correct
        and still used directly for cases where "hide literally
        everything for this session" IS the intent (round-9's orphan
        sweep does not call this at all -- it deletes; test fixtures that
        want a fully-hidden session for setup purposes are the real
        remaining callers).
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

    def supersede_reproduced_rows(self, session_id: str, messages: List[Dict[str, Any]]) -> int:
        """Round-18 fix: tombstone only the rows a resync's re-append will
        actually REPRODUCE, not every current row. supersede_session()'s
        blanket tombstone was wrong for exactly the case a resync always
        hits when it fires on a long session: the live transcript being
        re-archived has already had its middle dropped by select_context
        (the normal, expected state, not an edge case) -- so a blanket
        tombstone-then-reappend permanently hides everything the re-append
        doesn't cover, even though it was never deleted. That's what
        happened in production: 152 tombstoned rows in one session, 88 of
        them unique nowhere else, one of them the specific fact a later
        turn confidently reported as absent.

        Matches on (role, content) against `messages` (the messages about
        to be re-appended) -- only rows that match get superseded. Content
        the live transcript no longer contains is left superseded=0 and
        stays visible: that's precisely the archive-only history this
        store exists to hold, not a bug to route around.

        Returns the number of rows actually superseded, for the caller to
        log/verify.
        """
        if not messages:
            return 0
        pairs = {(m.get("role", "unknown"), _stringify(m)) for m in messages}
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content FROM rlm_messages "
                "WHERE session_id = ? AND superseded = 0",
                (session_id,),
            ).fetchall()
            to_supersede = [r[0] for r in rows if (r[1], r[2]) in pairs]
            if to_supersede:
                for i in range(0, len(to_supersede), 500):
                    batch = to_supersede[i : i + 500]
                    placeholders = ",".join("?" * len(batch))
                    self._conn.execute(
                        f"UPDATE rlm_messages SET superseded = 1 WHERE id IN ({placeholders})",
                        batch,
                    )
                    if self._fts_enabled:
                        self._conn.execute(
                            f"UPDATE rlm_search SET superseded = 1 WHERE rowid IN ({placeholders})",
                            batch,
                        )
                self._conn.commit()
        return len(to_supersede)

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

    # -- retention: inherited from state.db, never owned here -----------

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM rlm_meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO rlm_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def sweep_orphaned_sessions(
        self,
        state_db_path: str,
        current_session_id: Optional[str] = None,
        min_interval_hours: int = 24,
        vacuum_after_prune: bool = True,
        min_vacuum_interval_days: int = 30,
    ) -> Dict[str, Any]:
        """Delete every RLM row (live and tombstoned alike) for a session
        that WAS confirmed present in state.db's `sessions` table at some
        point and no longer is. This is the ONLY deletion path rlm.db has,
        and it deliberately has no clock or config of its own -- see the
        module docstring.

        Round 20: absence from state.db alone is NOT sufficient grounds
        for deletion anymore -- it used to be, and it deleted the only
        surviving copy of a real ~1.4MB user conversation whose
        create_session() had silently failed under write contention (a
        pre-existing, documented hermes_state.py failure mode). Absence
        can mean "existed and was released" (the intended case) or "never
        successfully registered in the first place" (the bug), and
        state.db cannot tell those apart after the fact -- there's no
        tombstone on ITS side. So this now requires POSITIVE evidence:
        every session currently found present in state.db is recorded
        into rlm_seen_sessions (a session_id is only ever a deletion
        candidate here once it's been seen present at least once), and
        the sweep target is `previously seen AND absent now` -- not just
        `absent now`. A session_id absent from state.db that was never
        recorded here is left alone indefinitely: the failure mode this
        guards against (registration never succeeding) doesn't resolve
        with age, so there is deliberately no age-based escape hatch
        either.

        Fail-open, same posture as select_context()'s "can't archive ->
        don't drop" rule: state.db unreadable, the query failing, or
        anything else going wrong here means nothing is deleted. A missed
        sweep costs disk; a wrong delete costs data that can't come back
        (unlike superseded rows, an orphan sweep is a real DELETE, not a
        tombstone -- there is nothing left standing this class could ever
        recover from a mistake here).

        `current_session_id` is an explicit extra guard on top of the
        state.db check, not a substitute for it: state.db's own
        `sessions` row is written by a try/except-guarded call at session
        start that can itself fail (see cli.py's `_session_db_created`
        flag) or simply not have landed yet the instant this sweep runs.
        Excluding the live session_id closes that race for the one
        session this process actually has open; it does not, and cannot,
        protect a same-moment race in another process's session -- state
        is the arbiter there, same as everywhere else in this file.

        Throttled like state.db's own maybe_auto_prune_and_vacuum(): at
        most once per min_interval_hours (tracked in this store's own
        rlm_meta, not state.db's state_meta -- RLM never writes to
        state.db), and VACUUM (which does not run inside the delete's own
        transaction -- SQLite disallows that) only when this sweep freed
        rows AND min_vacuum_interval_days has elapsed since the last one.
        """
        result: Dict[str, Any] = {
            "skipped": False, "sessions_pruned": 0, "rows_deleted": 0, "vacuumed": False,
        }
        try:
            now = time.time()
            last_raw = self.get_meta("last_orphan_sweep")
            if last_raw:
                try:
                    if now - float(last_raw) < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass  # corrupt meta -- treat as no prior run, sweep anyway

            with self._lock:
                candidates = {
                    r[0] for r in self._conn.execute(
                        "SELECT DISTINCT session_id FROM rlm_messages"
                    ).fetchall()
                }
            if current_session_id:
                candidates.discard(current_session_id)
            if not candidates:
                self.set_meta("last_orphan_sweep", str(now))
                return result

            # Read-only, and a real failure to open/query must abort the
            # whole sweep -- never treat "couldn't check state.db" as
            # "state.db has nothing", which would delete everything.
            existing = self._existing_state_db_sessions(state_db_path, candidates)
            # Round 20: record positive evidence BEFORE computing orphans,
            # so a session present right now is provably eligible for a
            # future sweep once it later disappears -- and so this sweep's
            # own orphan check below can require that evidence rather than
            # trusting bare absence.
            self._record_seen_sessions(existing)
            absent_now = candidates - existing
            previously_confirmed = self._filter_seen_sessions(absent_now)
            orphans = absent_now & previously_confirmed
            never_confirmed = absent_now - previously_confirmed
            if never_confirmed:
                logger.info(
                    "RLM: orphan sweep left %d session(s) untouched -- "
                    "absent from state.db but never confirmed present "
                    "there, so absence isn't trusted as proof of release "
                    "(round-20 guard)",
                    len(never_confirmed),
                )
            if not orphans:
                self.set_meta("last_orphan_sweep", str(now))
                return result

            orphans = list(orphans)
            deleted = 0
            with self._lock:
                for i in range(0, len(orphans), 500):
                    batch = orphans[i:i + 500]
                    placeholders = ",".join("?" * len(batch))
                    deleted += self._conn.execute(
                        f"SELECT COUNT(*) FROM rlm_messages WHERE session_id IN ({placeholders})",
                        batch,
                    ).fetchone()[0]
                    self._conn.execute(
                        f"DELETE FROM rlm_messages WHERE session_id IN ({placeholders})", batch
                    )
                    if self._fts_enabled:
                        self._conn.execute(
                            f"DELETE FROM rlm_search WHERE session_id IN ({placeholders})", batch
                        )
                    # The seen-evidence row is now moot -- the session it
                    # was evidence for no longer exists in either db.
                    self._conn.execute(
                        f"DELETE FROM rlm_seen_sessions WHERE session_id IN ({placeholders})", batch
                    )
                self._conn.commit()

            result["sessions_pruned"] = len(orphans)
            result["rows_deleted"] = deleted
            self.set_meta("last_orphan_sweep", str(now))

            last_vacuum_raw = self.get_meta("last_orphan_vacuum")
            vacuum_due = True
            if last_vacuum_raw:
                try:
                    vacuum_due = (now - float(last_vacuum_raw)) >= min_vacuum_interval_days * 86400
                except (TypeError, ValueError):
                    vacuum_due = True
            if vacuum_after_prune and deleted > 0 and vacuum_due:
                try:
                    with self._lock:
                        self._conn.execute("VACUUM")
                    result["vacuumed"] = True
                    self.set_meta("last_orphan_vacuum", str(now))
                except Exception as exc:
                    logger.warning("RLM: VACUUM after orphan sweep failed: %s", exc)

            logger.info(
                "RLM: orphan sweep removed %d row(s) across %d session(s) absent from state.db%s",
                deleted, len(orphans), " + VACUUM" if result["vacuumed"] else "",
            )
        except Exception as exc:
            logger.warning("RLM: orphan sweep failed, nothing deleted: %s", exc)
            result["error"] = str(exc)
        return result

    @staticmethod
    def _existing_state_db_sessions(state_db_path: str, candidates: set) -> set:
        """Which of `candidates` still have a row in state.db's `sessions`
        table. Opens state.db read-only via a plain sqlite3 URI connection
        -- deliberately NOT importing hermes_state.SessionDB, which is a
        heavy, write-capable class with its own production-DB safety
        checks meant for a different caller. This only ever reads.
        """
        uri = f"file:{Path(state_db_path).expanduser()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            found: set = set()
            candidates = list(candidates)
            for i in range(0, len(candidates), 500):
                batch = candidates[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT id FROM sessions WHERE id IN ({placeholders})", batch
                ).fetchall()
                found.update(r[0] for r in rows)
            return found
        finally:
            conn.close()

    def _record_seen_sessions(self, session_ids: set) -> None:
        """Round 20: persist positive evidence a session_id was confirmed
        present in state.db. INSERT OR IGNORE -- first confirmation wins,
        never overwritten (this is a "was it ever seen" record, not a
        "when was it last seen" one; the first sighting is all the
        deletion decision needs).
        """
        if not session_ids:
            return
        now = time.time()
        rows = [(sid, now) for sid in session_ids]
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO rlm_seen_sessions (session_id, first_confirmed_ts) "
                "VALUES (?, ?)",
                rows,
            )
            self._conn.commit()

    def _filter_seen_sessions(self, session_ids: set) -> set:
        """Which of `session_ids` have a rlm_seen_sessions record -- i.e.
        were confirmed present in state.db at some prior point, whether or
        not they still are now.
        """
        if not session_ids:
            return set()
        found: set = set()
        session_ids = list(session_ids)
        with self._lock:
            for i in range(0, len(session_ids), 500):
                batch = session_ids[i : i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT session_id FROM rlm_seen_sessions WHERE session_id IN ({placeholders})",
                    batch,
                ).fetchall()
                found.update(r[0] for r in rows)
        return found

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
