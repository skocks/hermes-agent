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
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rlm_session_turn ON rlm_messages(session_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_rlm_session_ts ON rlm_messages(session_id, ts);
"""


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
            self._conn.commit()

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
                f"WHERE session_id = ? AND ({clauses}) ORDER BY id DESC LIMIT ?",
                (session_id, *params, limit),
            )
            rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]

    def message_count(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM rlm_messages WHERE session_id = ?",
                (session_id,),
            )
            return cur.fetchone()[0] or 0

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
