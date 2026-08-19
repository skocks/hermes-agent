"""A persistent Python REPL subprocess, one per session — this is the part
that makes the engine an actual RLM implementation rather than a tool-
calling approximation of one.

Faithful to the paper's shape (confirmed against alexzhang13/rlm's own
README, which explicitly states the design moves AWAY from JSON tool-
calling): the context (here, this session's archived history) is bound as
data the model can query from inside the REPL, and there is one bound
function — ``rlm_query`` — that lets code call the language model itself
recursively, mid-script. State persists across calls: a variable set in
one ``rlm_repl`` tool call is still there in the next one, in the same
session. Root only ever sees the tool's captured stdout, hard-capped.

Where this still isn't the paper's exact mechanism: hermes-agent's own
agent loop is tool-calling at the framework level (that's how EVERY
action in hermes works, not something a context-engine plugin can
change), so there is necessarily one tool wrapping REPL access — the
paper's harness intercepts raw code blocks from the completion stream
directly, no tool-calling layer at all. This is the closest approximation
achievable without rewriting hermes' core conversation loop.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Child process bootstrap. Deliberately stdlib-only (sqlite3, urllib,
# json) — no new dependency, matches the "don't add deps without asking"
# rule and keeps the child trivially auditable. Runs as a normal
# subprocess (same isolation class hermes' own code_execution tool uses —
# a process boundary, not in-process exec — see the conversation that led
# here for why that boundary matters).
_BOOTSTRAP_TEMPLATE = '''
import sys, io, json, contextlib, traceback, sqlite3, urllib.request

DB_PATH = {db_path!r}
SESSION_ID = {session_id!r}
BASE_URL = {base_url!r}
MODEL = {model!r}
API_KEY = {api_key!r}

_ns = {{}}

def history(where="1=1", limit=100, order_by="id DESC"):
    """Query this session's archived messages. where/order_by are raw SQL
    fragments over columns turn_id, role, content, ts -- scoped to this
    session automatically, you cannot see another session's rows."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT turn_id, role, content, ts FROM rlm_messages "
            "WHERE session_id = ? AND (" + where + ") "
            "ORDER BY " + order_by + " LIMIT ?",
            (SESSION_ID, int(limit)),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [dict(zip(("turn_id", "role", "content", "ts"), r)) for r in rows]

def rlm_query(prompt, system="You are a focused sub-agent. Answer concisely.", max_tokens=500):
    """Recursive call: ask the language model itself something, e.g. to
    digest a large chunk of history before you print it. This is the
    actual recursion in Recursive Language Models -- a language model
    call made from inside code the root model wrote."""
    body = json.dumps({{
        "model": MODEL,
        "messages": [
            {{"role": "system", "content": system}},
            {{"role": "user", "content": prompt}},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }}).encode("utf-8")
    headers = {{"Content-Type": "application/json"}}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    url = BASE_URL.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]

_ns["history"] = history
_ns["rlm_query"] = rlm_query
_ns["sqlite3"] = sqlite3

# THE part that makes this faithful to the paper rather than a query API
# wearing a REPL costume: the paper binds the prompt/context as a plain
# variable the model slices with ordinary Python (indexing, regex, list
# comprehensions) -- it does not require the model to know SQL. `history()`
# above is a SQL-shaped convenience for re-querying after context was
# loaded (e.g. to pick up messages archived after this snapshot); `context`
# here is the actual paper mechanism -- a real Python list, sliced/
# filtered/searched with real Python, no query language needed.
context = history(limit=5000)
_ns["context"] = context

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    code = req.get("code", "")
    buf = io.StringIO()
    result = {{"stdout": "", "error": None}}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(code, "<rlm_repl>", "exec"), _ns)
    except Exception:
        result["error"] = traceback.format_exc(limit=6)
    result["stdout"] = buf.getvalue()
    sys.stdout.write(json.dumps(result) + "\\n")
    sys.stdout.flush()
'''


class PersistentREPL:
    """One subprocess per session, state alive for the session's lifetime."""

    def __init__(
        self,
        db_path: str,
        session_id: str,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 15.0,
        max_output_chars: int = 8000,
    ):
        self.db_path = db_path
        self.session_id = session_id
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _bootstrap_script(self) -> str:
        return _BOOTSTRAP_TEMPLATE.format(
            db_path=self.db_path, session_id=self.session_id,
            base_url=self.base_url, model=self.model, api_key=self.api_key or "",
        )

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", self._bootstrap_script()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def exec(self, code: str) -> Dict[str, Any]:
        """Run *code* in the persistent namespace. Returns {stdout, error}
        (error is None on success). Hard-caps stdout length -- the model is
        expected to digest large results itself via rlm_query() before
        printing, but a cap exists regardless so a forgetful/failed digest
        can't blow root's context, matching the same backstop principle
        applied everywhere else in this engine.
        """
        with self._lock:
            try:
                self._ensure_started()
                self._proc.stdin.write(json.dumps({"code": code}) + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                logger.warning("RLM REPL: write failed, restarting (%s)", e)
                self._restart_locked()
                return {"stdout": "", "error": f"REPL process died and was restarted — state was lost, retry: {e}"}

            line = self._read_with_timeout(self.timeout)
            if line is None:
                logger.warning("RLM REPL: call timed out after %.1fs, restarting", self.timeout)
                self._restart_locked()
                return {
                    "stdout": "",
                    "error": (
                        f"REPL call timed out after {self.timeout}s — process "
                        "restarted, all prior state (variables, imports) is "
                        "lost. Retry with smaller/faster code."
                    ),
                }
            try:
                result = json.loads(line)
            except Exception:
                logger.warning("RLM REPL: unparseable output, restarting")
                self._restart_locked()
                return {"stdout": "", "error": "REPL produced unparseable output — process restarted, state lost"}

        out = result.get("stdout", "")
        if len(out) > self.max_output_chars:
            omitted = len(out) - self.max_output_chars
            result["stdout"] = (
                out[: self.max_output_chars]
                + f"\n...[truncated, {omitted} more chars omitted — digest "
                "large output yourself with rlm_query() before printing it, "
                "rather than printing raw data]"
            )
            result["truncated"] = True
        return result

    def _read_with_timeout(self, timeout: float) -> Optional[str]:
        box = [None]

        def _reader():
            try:
                box[0] = self._proc.stdout.readline()
            except Exception:
                box[0] = None

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive() or not box[0]:
            return None
        return box[0]

    def _restart_locked(self) -> None:
        """Caller must hold self._lock."""
        self._close_locked()
        self._ensure_started()

    def _close_locked(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()
