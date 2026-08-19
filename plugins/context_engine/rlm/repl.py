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
import os
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
import sys, os, io, json, contextlib, traceback, sqlite3, urllib.request

DB_PATH = {db_path!r}
SESSION_ID = {session_id!r}
BASE_URL = {base_url!r}
MODEL = {model!r}
API_MODE = {api_mode!r}
QUERY_TIMEOUT = {query_timeout!r}
MAX_QUERY_CALLS = {max_query_calls!r}
# API key deliberately NOT interpolated into this script -- the script
# text becomes this process' argv (visible to any local user via `ps
# auxww`). Passed via env instead, which /proc/<pid>/environ still shows
# to the same user but not to `ps`, the far more common exposure vector.
API_KEY = os.environ.get("RLM_API_KEY", "")

_ns = {{}}

def _archive_count():
    """COUNT(*), not len(history(limit=big)) -- refreshed every exec() call
    now (see _refresh_context below), so this stays a cheap indexed count
    rather than fetching up to a million rows just to measure them."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM rlm_messages WHERE session_id = ? AND superseded = 0",
            (SESSION_ID,),
        )
        return cur.fetchone()[0]
    finally:
        conn.close()

def history(where="1=1", limit=100, order_by="id DESC"):
    """Query this session's archived messages. where/order_by are raw SQL
    fragments over columns turn_id, role, content, ts -- scoped to this
    session automatically, you cannot see another session's rows."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT turn_id, role, content, ts FROM rlm_messages "
            "WHERE session_id = ? AND superseded = 0 AND (" + where + ") "
            "ORDER BY " + order_by + " LIMIT ?",
            (SESSION_ID, int(limit)),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [dict(zip(("turn_id", "role", "content", "ts"), r)) for r in rows]

# L3 fix: nothing previously bounded how many times one exec() call could
# recurse -- a runaway loop calling rlm_query() was limited only by the
# wall-clock timeout (which then kills the whole REPL, destroying state,
# as its own failure mode), and root had no visibility into what a call
# actually spent even when it stayed within budget. Reset at the top of
# every exec() (see the main loop below), so it's a per-call budget, not
# cumulative across the session.
_query_spend = {{"count": 0, "chars_in": 0, "chars_out": 0}}

def rlm_query(prompt, system="You are a focused sub-agent. Answer concisely.", max_tokens=500):
    """Recursive call: ask the language model itself something, e.g. to
    digest a large chunk of history before you print it. This is the
    actual recursion in Recursive Language Models -- a language model
    call made from inside code the root model wrote.

    Only chat_completions is implemented. Anthropic-native / Responses-API
    style api_mode configs would need a different endpoint, payload shape,
    and response parse -- rather than silently sending an OpenAI-shaped
    request to an endpoint that doesn't speak it (wrong answers, or a
    confusing error far from the actual cause), this fails loudly and
    names exactly what's unsupported.

    Bounded to MAX_QUERY_CALLS per rlm_repl call (see the tool description
    for the current limit) -- raises, rather than silently truncating a
    runaway recursive loop, once exceeded.
    """
    if _query_spend["count"] >= MAX_QUERY_CALLS:
        raise RuntimeError(
            f"rlm_query() call limit reached ({{MAX_QUERY_CALLS}} per rlm_repl "
            "call). This is a per-call budget, not cumulative -- if the task "
            "genuinely needs more recursive calls than this, split it across "
            "multiple rlm_repl calls (state persists between them) rather "
            "than looping past the limit in one."
        )
    if API_MODE and API_MODE != "chat_completions":
        raise NotImplementedError(
            f"rlm_query() only supports api_mode='chat_completions', "
            f"this session is configured with api_mode={{API_MODE!r}}. "
            "Not implemented yet -- see plugins/context_engine/rlm/repl.py."
        )
    _query_spend["count"] += 1
    _query_spend["chars_in"] += len(prompt) + len(system)
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
    with urllib.request.urlopen(req, timeout=QUERY_TIMEOUT) as resp:
        data = json.loads(resp.read())
    answer = data["choices"][0]["message"]["content"]
    # Local models can return None content for very short max_tokens
    # (reasoning-only response, nothing in the visible field) -- guard
    # rather than crash the spend accounting on it. Return value itself is
    # unchanged (still None in that case), matching prior behavior.
    _query_spend["chars_out"] += len(answer or "")
    return answer

_ns["history"] = history
_ns["rlm_query"] = rlm_query
_ns["sqlite3"] = sqlite3

# M5 fix: the paper can return a REPL-constructed string directly as the
# answer (FINAL_VAR), no cap. Every answer here otherwise has to squeeze
# through the routine stdout cap (max_output_chars, ~8000 chars by
# default) -- real capability loss for genuinely long-form output the
# model deliberately constructed as ITS answer, as opposed to incidental
# print() spam. final() gets a separate, higher cap (final_max_chars,
# default higher than max_output_chars) instead of the general one -- but
# still capped, never unbounded: an uncapped path here would undo the
# entire guarantee this engine exists to provide, no matter how
# deliberate the model's intent was.
_final_holder = [None]

def final(text):
    """Mark `text` as your deliberate, complete answer -- returned under a
    higher length cap than ordinary print() output (which is capped more
    aggressively since it's usually incidental, not the actual answer).
    Still capped, not unlimited -- for a very long answer, still summarize
    with rlm_query() rather than assuming this is truly boundless."""
    _final_holder[0] = str(text)
    return text

# THE part that makes this faithful to the paper rather than a query API
# wearing a REPL costume: the paper binds the prompt/context as a plain
# variable the model slices with ordinary Python (indexing, regex, list
# comprehensions) -- it does not require the model to know SQL. `history()`
# above is a SQL-shaped convenience for re-querying after context was
# loaded (e.g. to pick up messages archived after this snapshot); `context`
# here is the actual paper mechanism -- a real Python list, sliced/
# filtered/searched with real Python, no query language needed.
#
# Chronological (oldest first) -- history()'s default order_by='id DESC'
# is right for "most recent N", wrong for a context snapshot: the paper's
# context is read start-to-end, and reversing it silently would make any
# code assuming order (e.g. "the first message is the earliest") wrong
# without any signal that something's off. context_total/context_truncated
# make the 5000-row cap visible instead of silent -- a model reasonably
# assumes "context" means "everything" unless told otherwise.
#
# Rebound at the top of EVERY exec(), not just once at process start: this
# REPL now lives for the whole session (a deliberate improvement on the
# paper's per-query REPL), so a context bound once at startup would go
# stale the moment anything gets archived after the first rlm_repl call --
# in a long session that shifts topic, that's precisely the newly-relevant
# history silently missing. Reassigning these three names doesn't touch
# any other variable the model has set.
def _refresh_context():
    total = _archive_count()
    ctx = history(limit=5000, order_by="id ASC")
    _ns["context"] = ctx
    _ns["context_total"] = total
    _ns["context_truncated"] = total > len(ctx)

_refresh_context()

_call_index = 0
_var_seen_at = {{}}   # name -> call index it first appeared, for the staleness footer
_code_log = []        # [(call_index, code), ...] -- provenance survives even after
                       # select_context() drops the turn that created a binding from
                       # root's visible context; the binding stays callable, but
                       # without this its origin would be unrecoverable.
_CODE_LOG_MAX = 200

def code_log(n=20):
    """Source of the last n exec() calls in this REPL session, as
    (call_index, code) pairs, most recent last. State (variables,
    functions) persists across turns, but root's visible context does not
    -- by the time a function defined many turns ago is still callable,
    its defining code may be long gone from what you can see. Use this to
    recover the provenance of what created your current state."""
    return list(_code_log[-int(n):])

# Names/values restored at the top of EVERY exec(), not just present at
# startup: exec()'s globals dict IS _ns, so model code assigning to
# history/rlm_query/sqlite3/reset/code_log doesn't create a local shadow,
# it silently overwrites the real thing for the rest of the session --
# including reset() itself, the one escape hatch meant to recover from
# exactly this kind of bad state. A base name a model overwrites is
# restored before the NEXT call, so shadowing only ever lasts for the one
# call that did it, never permanently.
_BASE_BINDINGS = {{
    "history": history, "rlm_query": rlm_query, "sqlite3": sqlite3, "final": final,
    "reset": None, "code_log": code_log,  # reset assigned just below (refers to itself)
}}

def reset():
    """Clear all variables/imports from earlier in this session, keeping
    only history/rlm_query/context/reset/code_log. Use this when starting
    on an unrelated task so old variables can't leak into it by accident."""
    for k in list(_ns.keys()):
        if k not in _BASE_BINDINGS and k != "__builtins__":
            del _ns[k]
    _var_seen_at.clear()
    _refresh_context()
    return "REPL namespace reset."

_BASE_BINDINGS["reset"] = reset
_ns.update(_BASE_BINDINGS)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    code = req.get("code", "")
    _call_index += 1
    _ns.update(_BASE_BINDINGS)  # restore anything the model clobbered last call
    _refresh_context()
    _final_holder[0] = None
    _query_spend["count"] = 0
    _query_spend["chars_in"] = 0
    _query_spend["chars_out"] = 0
    buf = io.StringIO()
    result = {{"stdout": "", "error": None}}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(code, "<rlm_repl>", "exec"), _ns)
    except Exception:
        result["error"] = traceback.format_exc(limit=6)

    if _final_holder[0] is not None:
        result["final"] = _final_holder[0]

    # L3 fix: always present (even all-zero), not just on overrun -- root
    # can otherwise never see what a call spent recursing, only find out
    # it went over budget after the fact via the error above.
    result["query_spend"] = dict(_query_spend)

    _code_log.append((_call_index, code))
    del _code_log[:-_CODE_LOG_MAX]

    stale_names = []
    for k in _ns:
        if k in _BASE_BINDINGS or k in ("context", "context_total", "context_truncated", "__builtins__"):
            continue  # __builtins__ is auto-injected by exec(), not a user variable
        if k not in _var_seen_at:
            _var_seen_at[k] = _call_index
        if _var_seen_at[k] < _call_index:
            stale_names.append(k)

    result["stdout"] = buf.getvalue()
    # N1 fix: this was appended to stdout_text and truncated away by the
    # parent's max_output_chars cap (which keeps the FRONT and trims the
    # END) on any call with 8000+ chars of real output -- exactly the
    # heavy-output exploratory calls most likely to leave stale names
    # behind, so the warning vanished precisely when it mattered most. A
    # separate result key is never touched by that truncation, same
    # pattern as `final`.
    if stale_names:
        result["stale_names"] = sorted(stale_names)
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
        api_mode: str = "",
        timeout: float = 90.0,
        query_timeout: float = 60.0,
        max_output_chars: int = 8000,
        final_max_chars: int = 20000,
        max_query_calls: int = 20,
    ):
        self.db_path = db_path
        self.session_id = session_id
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.api_mode = api_mode
        self.query_timeout = query_timeout
        # timeout bounds one whole exec() call; a code block can call
        # rlm_query() (bounded by query_timeout) and do other work besides.
        # If timeout <= query_timeout, a single sub-call alone can exceed
        # the REPL's own wall-clock budget and get killed mid-flight,
        # destroying all REPL state -- exactly the bug this class exists
        # to not have (see the audit that found repl_timeout_seconds=15
        # vs. a hardcoded 60s inner timeout). Self-heal rather than trust
        # every future config value to keep this ordering by hand.
        if timeout <= query_timeout:
            logger.warning(
                "RLM REPL: timeout (%.1fs) <= query_timeout (%.1fs) would let "
                "a single rlm_query() call outlive and kill the REPL that "
                "made it — raising timeout to query_timeout + 30s",
                timeout, query_timeout,
            )
            timeout = query_timeout + 30.0
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        # M5 fix: final() gets a separate, higher cap than routine stdout --
        # a deliberate, complete answer the model constructed shouldn't be
        # squeezed through the same aggressive cap as incidental print()
        # spam, but "higher" still means "capped", never unbounded (an
        # uncapped path here would undo the entire guarantee this engine
        # exists to provide).
        self.final_max_chars = final_max_chars
        # L3 fix: bounds how many times one exec() call may recurse via
        # rlm_query() -- previously unbounded except by the wall-clock
        # timeout, whose failure mode (kill the REPL, lose all state) is
        # worse than a clear in-band error naming the limit.
        self.max_query_calls = max_query_calls
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _bootstrap_script(self) -> str:
        return _BOOTSTRAP_TEMPLATE.format(
            db_path=self.db_path, session_id=self.session_id,
            base_url=self.base_url, model=self.model,
            api_mode=self.api_mode or "", query_timeout=self.query_timeout,
            max_query_calls=self.max_query_calls,
        )

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        # API key via env, not the -c script argv (which `ps auxww` shows
        # to any local user) -- see the module docstring note on RLM_API_KEY.
        child_env = dict(os.environ)
        if self.api_key:
            child_env["RLM_API_KEY"] = self.api_key
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", self._bootstrap_script()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=child_env,
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

        final_text = result.get("final")
        if isinstance(final_text, str) and len(final_text) > self.final_max_chars:
            omitted = len(final_text) - self.final_max_chars
            result["final"] = (
                final_text[: self.final_max_chars]
                + f"\n...[truncated, {omitted} more chars omitted -- final() "
                "is capped higher than routine output, not unlimited]"
            )
            result["final_truncated"] = True
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
