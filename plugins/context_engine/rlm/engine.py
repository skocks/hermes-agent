"""RLM (Recursive Language Model) context engine.

Mechanism (see agent/context_engine.py's module docstring for the two
verbs this ABC exposes):

  - select_context() keeps the ROOT model's per-turn context small and
    bounded instead of letting it grow. Once the live transcript passes
    protect_first_n + protect_last_n messages, the middle is DROPPED from
    the outgoing request — never summarized, never lost — because
    on_turn_complete() has already archived every message verbatim into
    an external SQLite store (store.py) by the time it would be dropped.
  - The model recovers dropped history through ONE tool, rlm_repl — a
    persistent Python REPL (repl.py), one subprocess per session, state
    alive for the session's lifetime. This is the actual paper mechanism,
    not an approximation of it: confirmed against alexzhang13/rlm's own
    README, RLM explicitly moves AWAY from JSON tool-calling schemas in
    favor of code the model writes against the context bound as data.
    Inside that REPL: `history(where, order_by, limit)` queries this
    session's archive (session-scoped automatically), and `rlm_query
    (prompt)` is the actual recursion — a language model call made from
    inside code the root model wrote, exactly the paper's mechanism,
    not engine-side digestion pretending to be that.
  - Where this still isn't identical to the paper: hermes-agent's agent
    loop is tool-calling at the framework level for EVERY action, not
    something a context-engine plugin can change — so there's necessarily
    one tool wrapping REPL access, where the paper's harness intercepts
    raw code blocks from the completion stream with no tool-calling layer
    at all. This is the ceiling of fidelity achievable without rewriting
    hermes' core conversation loop. rlm_repl's own output is still
    hard-capped (repl.py) as a backstop regardless of whether the model
    remembers to digest large results itself.
  - should_compress()/compress() exist only as an ABC-mandated safety
    net for the case a single turn is so large select_context() can't
    help (e.g. one giant tool result). They archive-then-trim, so even
    the safety net never loses data — it only ever loses it from the
    LIVE prompt, same as the normal path.

Enable/disable: this engine activates only when `context.engine: rlm` is
set in config.yaml — same switch every other engine (compressor, lcm)
uses. There is deliberately no second `rlm.enabled` flag: emulating the
built-in ContextCompressor's full construction (per-model threshold
overrides, Codex autoraise, tail_mode, ...) inside this plugin would
duplicate logic hermes-agent's own agent_init.py already owns and
maintains, and that copy would silently drift out of sync on every core
update. Flip `context.engine` back to `compressor` (or `lcm`) to get the
real, fully-maintained fallback instead of a reimplemented one.

Failure posture: every method is fail-open. If the SQLite store can't be
opened, the engine logs and runs as a pure passthrough (select_context
returns None, no tools exposed) rather than raising — a broken RLM store
must never be able to crash agent startup or a turn.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine
from agent.model_metadata import estimate_tokens_rough

from .repl import PersistentREPL
from .store import RLMStore

logger = logging.getLogger(__name__)

_DEFAULT_PROTECT_FIRST_N = 3
_DEFAULT_PROTECT_LAST_N = 25
_DEFAULT_TAIL_TOKEN_BUDGET_FRACTION = 0.5  # of context_length, for the tail slice
_MARKER_BUCKET = 5  # round the "N messages dropped" count to this, for cache stability

RLM_REPL_SCHEMA = {
    "name": "rlm_repl",
    "description": (
        "Run Python in a persistent REPL to recover conversation history "
        "the RLM context engine dropped from the visible context to keep "
        "the prompt small. History is archived, never deleted — use this "
        "before assuming something from earlier in the conversation is "
        "gone. State persists across calls within this session (a "
        "variable you set now is still there next time you call this) "
        "but is NOT reliable long-term: a timeout, a crash, or a /model "
        "switch silently wipes it. Write code that can re-derive what it "
        "needs (e.g. from context/history()) rather than code that "
        "assumes an earlier variable is still set.\n"
        "Pre-loaded in the REPL namespace, refreshed at the start of "
        "every call (so context always reflects the current archive, "
        "not a stale snapshot from when the REPL first started):\n"
        "- context: a plain Python list of dicts (turn_id, role, content, "
        "ts), oldest-first (chronological). Slice/filter/search it with "
        "ordinary Python — indexing, list comprehensions, regex — no "
        "query language needed.\n"
        "- context_total: how many messages are actually archived. "
        "context_truncated: True if context holds fewer than that (capped "
        "at 5000) — check this before assuming context is everything.\n"
        "- history(where='1=1', order_by='id DESC', limit=100) -> same "
        "shape as context. Use this for anything context's defaults don't "
        "cover — a different order, a smaller/targeted slice, a SQL "
        "predicate. where/order_by are SQL fragments over turn_id, role, "
        "content, ts — scoped to this session automatically, you cannot "
        "see another session's rows.\n"
        "- rlm_query(prompt, system=..., max_tokens=500) -> str. Calls "
        "the language model itself — use this to digest/summarize a "
        "large result BEFORE printing it, so your own output (which is "
        "capped) stays useful instead of getting truncated.\n"
        "- reset() -> clears every variable you've set (keeping context/"
        "history/rlm_query/code_log/reset). Call this when starting an "
        "unrelated task so leftover variables from an earlier one (chunks, "
        "summary, results, ...) can't silently leak into it. If a call's "
        "output ends with a '[REPL: N name(s) from earlier call(s) still "
        "live: ...]' line, that's this happening — decide whether those "
        "names still matter or call reset().\n"
        "- code_log(n=20) -> the actual Python source of your last n calls "
        "in this REPL, most recent last. Variables and functions persist "
        "across turns, but root's visible chat context does not (older "
        "turns get dropped to keep the prompt small) — by the time a "
        "function from many turns ago is still callable, the turn that "
        "defined it may be gone from what you can see. Use this to recover "
        "what actually created your current state.\n"
        "Write ordinary Python: loops, filtering, string/regex work, "
        "whatever the task needs — this is not a fixed set of query "
        "modes, decompose the problem however fits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute."},
        },
        "required": ["code"],
    },
}


class RLMContextEngine(ContextEngine):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        full_cfg = config if config is not None else self._load_config()
        cfg = full_cfg.get("rlm", {}) or {}
        _delegation_cfg = full_cfg.get("delegation", {}) or {}
        self.protect_first_n: int = int(cfg.get("protect_first_n", _DEFAULT_PROTECT_FIRST_N))
        self.protect_last_n: int = int(cfg.get("protect_last_n", _DEFAULT_PROTECT_LAST_N))
        self._tail_token_fraction: float = float(
            cfg.get("tail_token_budget_fraction", _DEFAULT_TAIL_TOKEN_BUDGET_FRACTION)
        )
        self._marker_role: str = cfg.get("marker_role", "system")
        self._repl_timeout: float = float(cfg.get("repl_timeout_seconds", 90.0))
        self._repl_query_timeout: float = float(cfg.get("repl_query_timeout_seconds", 60.0))
        self._repl_max_output_chars: int = int(cfg.get("repl_max_output_chars", 8000))
        # Default the recursive sub-call to hermes' own delegation model —
        # config.yaml already has a `delegation:` block precisely for
        # "cheaper model for delegated sub-work" (mirrors the paper's cost-
        # parity setup: expensive root, cheap child). Reusing it beats
        # defaulting to the root model, which was the audit-flagged bug:
        # every recursive call cost the same as a root call, for no reason.
        # Explicit rlm.repl_query_model still overrides both.
        self._repl_query_model: str = cfg.get("repl_query_model") or _delegation_cfg.get("model", "")
        self._repl_query_base_url: str = cfg.get("repl_query_base_url") or _delegation_cfg.get("base_url", "")
        # Introduced by the M2 fix, caught in round-2 audit (R5): model and
        # base_url can come from delegation:, but api_mode was still always
        # root's -- if delegation points at a differently-shaped endpoint
        # than root, rlm_query() would wrongly inherit root's api_mode and
        # either mis-send or hit the H2 NotImplementedError for no reason.
        self._repl_query_api_mode: str = cfg.get("repl_query_api_mode") or _delegation_cfg.get("api_mode", "")
        # Auto-recall: forced (not voluntary) recovery. Every turn past the
        # drop threshold, select_context() itself checks whether the
        # incoming user message plausibly needs dropped history (cheap
        # keyword match against the archive) and, if so, injects a
        # digested-and-capped snippet directly into the request — no
        # dependence on the model noticing the marker or choosing to call
        # rlm_repl. Root is protected regardless of digestion outcome: a
        # failed/oversized digest is hard-truncated, never raw, never
        # unbounded — same backstop discipline as the rest of this engine.
        self._auto_recall: bool = bool(cfg.get("auto_recall", True))
        self._auto_recall_min_keyword_len: int = int(cfg.get("auto_recall_min_keyword_len", 4))
        self._auto_recall_max_keywords: int = int(cfg.get("auto_recall_max_keywords", 6))
        self._auto_recall_min_keywords: int = int(cfg.get("auto_recall_min_keywords", 2))
        self._auto_recall_digest_threshold_tokens: int = int(cfg.get("auto_recall_digest_threshold_tokens", 400))
        self._auto_recall_max_tokens: int = int(cfg.get("auto_recall_max_tokens", 300))
        self._db_path: str = cfg.get("db_path") or self._default_db_path()

        self._session_id: str = "unknown"
        self._persisted_count: int = 0  # how many of the live `messages` we've archived
        # Best-effort turn_id for the M4 mid-turn-archiving fix: messages
        # archived from select_context() (before the turn finishes) don't
        # have a real turn_id yet -- on_turn_complete() provides the real
        # one, but by the time it fires those messages are usually already
        # archived (nothing new left to backfill it onto). This tracks
        # "one past the last CONFIRMED turn_id" as the best guess for
        # in-progress-turn messages; imprecise during that one turn, never
        # wrong by more than one, and never a data-loss issue either way —
        # content is fully archived and searchable regardless of this tag.
        self._next_turn_id: int = 0
        self._store: Optional[RLMStore] = None
        self._store_error: Optional[str] = None
        self._runtime: Dict[str, str] = {}  # captured in update_model(), used to spawn the REPL
        self._repl: Optional[PersistentREPL] = None

        self._open_store()

    def _open_store(self) -> None:
        """(Re)open the archive. Safe to call when already open (no-op)."""
        if self._store is not None:
            return
        try:
            self._store = RLMStore(self._db_path)
            self._store_error = None
        except Exception as e:
            self._store = None
            self._store_error = str(e)
            logger.exception("RLM: failed to open store at %s — running as passthrough", self._db_path)

    # -- setup helpers ---------------------------------------------------

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        try:
            from hermes_cli.config import load_config_readonly
            return load_config_readonly()
        except Exception:
            logger.exception("RLM: failed to load config.yaml, using defaults")
            return {}

    @staticmethod
    def _default_db_path() -> str:
        try:
            from hermes_cli.config import get_config_path
            return str(get_config_path().parent / "rlm.db")
        except Exception:
            return str(Path.home() / ".hermes" / "rlm.db")

    def is_available(self) -> bool:
        return self._store is not None

    # -- identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "rlm"

    # -- model switch: capture runtime for the REPL's rlm_query() -----------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        # Base impl sets self.context_length / threshold_tokens correctly —
        # keep that behavior, just also remember the connection details so
        # the REPL child process can make its own recursive calls without
        # needing the live agent object (which it never receives).
        super().update_model(
            model=model, context_length=context_length, base_url=base_url,
            api_key=api_key, provider=provider, api_mode=api_mode,
        )
        self._runtime = {
            "model": self._repl_query_model or model,
            # If a delegation/override model is in play, its base_url must
            # travel with it — a different model can mean a different
            # endpoint entirely, not just a cheaper name on the same one.
            "base_url": (self._repl_query_model and self._repl_query_base_url) or base_url or "",
            # NOTE (known limitation, not silently swept under the rug):
            # api_key still follows ROOT's resolved key, not a separately-
            # resolved delegation key. hermes' real API-key resolution
            # (auth.json / provider config / env) is core machinery this
            # plugin doesn't have a clean way to invoke standalone. Correct
            # on this box (local server, no key needed either way); would
            # be wrong if delegation pointed at a provider needing its own
            # key different from root's.
            "api_key": api_key or "",
            # R5 fix: follows the delegation endpoint when the model/base_url
            # above did too, same "a different endpoint can be shaped
            # differently" reasoning as base_url. Falls back to root's
            # api_mode only when no delegation override is active.
            "api_mode": (self._repl_query_model and self._repl_query_api_mode) or api_mode or "",
        }
        # A running REPL was bootstrapped with the OLD model/base_url —
        # restart it so a mid-session /model switch takes effect. Losing
        # REPL state on a model switch is an acceptable tradeoff; silently
        # recursing against a stale/wrong endpoint is not.
        if self._repl is not None:
            self._repl.close()
            self._repl = None

    # -- token tracking ------------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", self.last_prompt_tokens)
        self.last_completion_tokens = usage.get("completion_tokens", self.last_completion_tokens)
        self.last_total_tokens = usage.get("total_tokens", self.last_total_tokens)

    # -- compaction: safety net only, select_context does the real work ------

    def should_compress(self, prompt_tokens: int = None) -> bool:
        # select_context() keeps the live request small; this only trips if
        # something slipped through (e.g. one oversized turn select_context
        # can't shrink because it operates on whole messages, not content).
        if prompt_tokens and self.context_length:
            return prompt_tokens > self.context_length * 0.95
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        logger.warning(
            "RLM: compress() safety net triggered (session=%s, %d messages) — "
            "select_context() should have prevented this; investigate.",
            self._session_id, len(messages),
        )
        if len(messages) <= self.protect_first_n + self.protect_last_n + 1:
            return messages

        # Archive first so the safety net never loses data either — it only
        # ever drops from the LIVE prompt, exactly like the normal path.
        # Unlike select_context() (optional, safe to no-op), compress()
        # exists precisely because the request must shrink or the provider
        # 400s on overflow — refusing to drop here would trade silent data
        # loss for a guaranteed hard failure, which is worse. If the store
        # is unavailable, dropping still has to happen; make that loud.
        if not self._store:
            logger.error(
                "RLM: compress() must drop content but the archive store is "
                "unavailable (%s) — this drop is UNRECOVERABLE, not the "
                "normal archived-and-recoverable path.",
                self._store_error or "not open",
            )
        self._archive_new(messages)

        system = messages[:1] if messages and messages[0].get("role") == "system" else []
        rest = messages[1:] if system else messages
        head = rest[: self.protect_first_n]
        tail = rest[-self.protect_last_n :] if self.protect_last_n else []
        dropped = len(rest) - len(head) - len(tail)
        if dropped <= 0:
            return messages
        return system + head + [self._dropped_marker(dropped)] + tail

    # -- the actual mechanism --------------------------------------------------

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,
        incoming_message: Dict[str, Any] = None,
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        if not self._store:
            # Never drop content we can't archive. Without an open store,
            # "drop the middle" isn't a bounded-context trick, it's data
            # loss with no recovery path (rlm_repl needs the store too) —
            # fail open to the full transcript instead. This must come
            # before anything else in this method.
            return None
        convo = conversation_messages or []

        # M4 fix: archive here too, not just at on_turn_complete(). This
        # method runs on every provider request, including mid-turn
        # tool-calling round trips -- on_turn_complete only fires once the
        # WHOLE turn finishes. Without this, a message dropped from the
        # request partway through a long tool-calling turn genuinely isn't
        # in SQLite yet: rlm_repl can't retrieve it even though the marker
        # promises it can. _archive_new() is idempotent (only appends what's
        # new since the last call), so calling it here every request is
        # safe, not a growing duplicate cost.
        self._archive_new(convo, turn_id=self._next_turn_id)

        non_system = [m for m in convo if m.get("role") != "system"]
        if len(non_system) <= self.protect_first_n + self.protect_last_n:
            return None  # nothing to drop yet — leave the request untouched

        tail = self._token_bounded_tail(non_system, budget_tokens)
        head = non_system[: self.protect_first_n]
        dropped = len(non_system) - len(head) - len(tail)
        if dropped <= 0:
            return None

        system = [m for m in request_messages if m.get("role") == "system"][:1]
        marker = self._dropped_marker(dropped)

        if self._auto_recall and self._store:
            recall = self._auto_recall_snippet(incoming_message)
            if recall is not None:
                marker = {"role": self._marker_role, "content": marker["content"] + "\n\n" + recall}

        return system + head + [marker] + tail

    # -- forced recovery: doesn't wait to be asked ----------------------------

    def _auto_recall_snippet(self, incoming_message: Optional[Dict[str, Any]]) -> Optional[str]:
        """Check whether the current turn plausibly needs dropped history,
        and if so, return a short, hard-capped, digested snippet to inject —
        or None if nothing relevant matched. Never returns raw/unbounded
        content: a failed or oversized digest is truncated, not passed
        through, so root is protected regardless of how this turns out.
        """
        question = _message_text(incoming_message)
        if not question:
            return None
        keywords = _extract_keywords(question, self._auto_recall_min_keyword_len, self._auto_recall_max_keywords)
        if len(keywords) < self._auto_recall_min_keywords:
            # A short/generic turn ("yes update") extracting to one weak
            # keyword isn't enough signal — validated against real history
            # (session 20260818_202325_d5a052, turn 31): a single generic
            # keyword matched unrelated content. Require real signal before
            # spending a search + possible digestion call on it.
            return None

        try:
            matches = self._store.search_any(self._session_id, keywords, limit=20)
        except Exception:
            logger.exception("RLM: auto-recall search failed")
            return None
        if not matches:
            return None

        raw_text = "\n".join(f"[{m['role']}] {m['content']}" for m in matches)
        raw_tokens = estimate_tokens_rough(raw_text)
        prefix = f"[RLM auto-recall: {len(matches)} possibly-relevant archived message(s) found]\n"

        if raw_tokens <= self._auto_recall_digest_threshold_tokens:
            return prefix + raw_text

        digested = self._digest_for_recall(raw_text, question)
        if digested is not None:
            return prefix + digested

        # Digestion failed — hard-cap rather than pass raw content through.
        # This is the guarantee: root is protected from an unbounded
        # subagent/lookup result no matter how this path resolves.
        logger.warning("RLM: auto-recall digestion failed, falling back to capped raw")
        return prefix + _truncate_to_tokens(raw_text, self._auto_recall_max_tokens)

    def _digest_for_recall(self, raw_text: str, question: str) -> Optional[str]:
        if not self._runtime.get("model"):
            return None
        try:
            from agent.auxiliary_client import call_llm
        except Exception:
            logger.exception("RLM: could not import call_llm for auto-recall digestion")
            return None

        raw_text = _truncate_to_tokens(raw_text, 20_000)  # bound the sub-call's own input too
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a context-recall assistant for another AI agent. "
                    "Given archived conversation history and the question that "
                    "agent is currently trying to answer, extract ONLY the "
                    "information relevant to that question, as concisely as "
                    "possible. If nothing is relevant, say so in one line. No "
                    "commentary, no preamble."
                ),
            },
            {
                "role": "user",
                "content": f"## Current question\n{question}\n\n## Archived history\n{raw_text}",
            },
        ]
        try:
            response = call_llm(
                task="compression",
                messages=messages,
                max_tokens=self._auto_recall_max_tokens,
                temperature=0.1,
                model=self._runtime["model"],
                base_url=self._runtime.get("base_url") or None,
                api_key=self._runtime.get("api_key") or None,
                api_mode=self._runtime.get("api_mode") or None,
            )
        except Exception:
            logger.exception("RLM: auto-recall digestion sub-call raised")
            return None

        message = response.choices[0].message
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            return None
        # Hard cap even a successful digest — a sub-call ignoring
        # max_tokens (provider quirk, bad model behavior) must never be
        # the thing that reopens root's context to an unbounded subagent
        # result. This is non-negotiable, not a best-effort measure.
        return _truncate_to_tokens(content.strip(), self._auto_recall_max_tokens)

    def _token_bounded_tail(
        self, non_system: List[Dict[str, Any]], budget_tokens: int
    ) -> List[Dict[str, Any]]:
        """Take up to protect_last_n trailing messages, capped by a token budget.

        Message-count protection alone can still blow the context if the
        tail happens to contain a few huge tool results. Walk backward and
        stop early if the token estimate exceeds the configured fraction of
        the model's context window — better a shorter-but-safe tail than a
        request that 400s on overflow.
        """
        candidates = non_system[-self.protect_last_n :] if self.protect_last_n else []
        if not candidates:
            return []
        token_cap = int((budget_tokens or self.context_length or 0) * self._tail_token_fraction)
        if token_cap <= 0:
            return candidates  # no budget info available — fall back to count-based

        kept: List[Dict[str, Any]] = []
        used = 0
        for m in reversed(candidates):
            cost = estimate_tokens_rough(_content_as_text(m))
            if kept and used + cost > token_cap:
                break
            if not kept and cost > token_cap:
                # The single most recent message alone exceeds the budget
                # (e.g. one huge tool/rlm_repl result). Unconditionally
                # admitting the first candidate before checking its size
                # would defeat the point of this budget — truncate instead
                # of shipping it whole or dropping it outright.
                kept.append(_truncate_message(m, token_cap))
                break
            used += cost
            kept.append(m)
        kept.reverse()
        return kept or [_truncate_message(candidates[-1], token_cap)]

    def _dropped_marker(self, dropped: int) -> Dict[str, Any]:
        bucket = max(_MARKER_BUCKET, (dropped // _MARKER_BUCKET) * _MARKER_BUCKET)
        return {
            "role": self._marker_role,
            "content": (
                f"[RLM: roughly {bucket} earlier message(s) omitted from this "
                "context to keep it small — archived, not deleted. Use "
                "rlm_repl (history()/rlm_query() are pre-loaded) if you "
                "need something from earlier in this conversation.]"
            ),
        }

    # -- archiving ---------------------------------------------------------

    def _archive_new(self, messages: List[Dict[str, Any]], turn_id: int = -1) -> None:
        if not self._store:
            return
        if len(messages) < self._persisted_count:
            # The live transcript got shorter than what we've archived —
            # e.g. a /reset, a manual /compress, or a session we haven't
            # seen state for. Don't silently drop future messages: re-sync
            # from zero and accept the (harmless, query-side-visible)
            # duplicate rows over the alternative of quietly losing data.
            logger.info(
                "RLM: transcript shrank under archived count (session=%s, "
                "have=%d archived=%d) — re-syncing from 0",
                self._session_id, len(messages), self._persisted_count,
            )
            self._persisted_count = 0
        new_messages = messages[self._persisted_count :]
        if not new_messages:
            return
        try:
            self._store.append_messages(self._session_id, turn_id, new_messages)
            self._persisted_count = len(messages)
        except Exception:
            logger.exception(
                "RLM: failed to archive %d message(s) for session=%s",
                len(new_messages), self._session_id,
            )

    def on_turn_complete(
        self,
        messages: List[Dict[str, Any]],
        usage: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:
        turn_id = kwargs.get("turn_id", self._next_turn_id)
        self._archive_new(messages, turn_id=turn_id)
        # Best-effort tracker for select_context()'s mid-turn archiving
        # (M4 fix) -- advance past whatever turn just confirmed-finished.
        try:
            if int(turn_id) >= self._next_turn_id:
                self._next_turn_id = int(turn_id) + 1
        except (TypeError, ValueError):
            self._next_turn_id += 1

    # -- session lifecycle ---------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or "unknown"
        # Self-healing: hermes-agent reuses ONE engine instance across /new
        # (cli.py calls on_session_end at that boundary, then rebinds this
        # same instance to the new session — it does not construct a fresh
        # engine). on_session_end() below closes the store, so without this
        # reopen, every session after the first /new would run with
        # self._store permanently None: select_context() would (if C1
        # weren't also fixed) drop history with nothing archived, silently
        # and permanently. Also retries a store that failed to open at
        # construction time (transient FS issue, etc).
        self._open_store()
        if self._store:
            try:
                # Resume-safe: if this session already has archived rows
                # (process restart, gateway reconnect), don't re-persist
                # what's already there on the next on_turn_complete().
                self._persisted_count = self._store.message_count(self._session_id)
            except Exception:
                logger.exception("RLM: message_count lookup failed on session start")
                self._persisted_count = 0
        else:
            self._persisted_count = 0
        self._next_turn_id = 0

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if self._repl is not None:
            self._repl.close()
            self._repl = None
        if not self._store:
            return
        try:
            self._archive_new(messages)  # catch anything since the last on_turn_complete
        finally:
            # Close but do NOT null self._store's slot permanently here —
            # on_session_start() reopens it. Nulling only the reference (not
            # a flag) is fine: is_available()/every store check treats "is
            # it currently open" via self._store, and the next
            # on_session_start() unconditionally calls _open_store() before
            # anything else touches it.
            self._store.close()
            self._store = None

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._persisted_count = 0
        self._next_turn_id = 0
        if self._repl is not None:
            self._repl.close()
            self._repl = None

    # -- tools: the persistent REPL -------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._store:
            return []
        return [RLM_REPL_SCHEMA]

    def _ensure_repl(self) -> Optional[PersistentREPL]:
        if self._repl is not None:
            return self._repl
        if not self._store or not self._runtime.get("model"):
            return None
        self._repl = PersistentREPL(
            db_path=self._store.db_path,
            session_id=self._session_id,
            base_url=self._runtime.get("base_url", ""),
            model=self._runtime.get("model", ""),
            api_key=self._runtime.get("api_key", ""),
            api_mode=self._runtime.get("api_mode", ""),
            timeout=self._repl_timeout,
            query_timeout=self._repl_query_timeout,
            max_output_chars=self._repl_max_output_chars,
        )
        return self._repl

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if name != "rlm_repl":
            return json.dumps({"error": f"Unknown context engine tool: {name}"})
        if not self._store:
            return json.dumps({"error": f"RLM store unavailable: {self._store_error or 'not open'}"})

        repl = self._ensure_repl()
        if repl is None:
            return json.dumps({"error": "RLM REPL not ready yet (model runtime not initialized)"})

        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return json.dumps({"error": "'code' must be a non-empty string"})

        result = repl.exec(code)
        return json.dumps(result)

    # -- status ---------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["engine"] = "rlm"
        status["store_ok"] = self._store is not None
        status["repl_running"] = self._repl is not None
        if self._store_error:
            status["store_error"] = self._store_error
        if self._store:
            try:
                status["archived_messages"] = self._store.message_count(self._session_id)
            except Exception:
                pass
        return status


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cheap char-based cap matching estimate_tokens_rough's ~4-chars/token
    ASCII rule closely enough for a safety-net truncation (not exact for
    dense CJK text, but this path only fires as a last resort — approximate-
    but-bounded beats unbounded).
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated, {len(text) - max_chars} more chars omitted]"


def _truncate_message(message: Dict[str, Any], token_cap: int) -> Dict[str, Any]:
    """Return a copy of *message* with its content capped to ~token_cap
    tokens. Used only when a single message alone exceeds the tail's token
    budget — flattens structured content to text in the process (acceptable
    here: this only fires as a last-resort safety net, not the normal path).
    """
    text = _content_as_text(message)
    truncated = _truncate_to_tokens(text, max(1, token_cap))
    out = dict(message)
    out["content"] = truncated + "\n[RLM: message truncated — exceeded the context tail's token budget on its own]"
    return out


_STOPWORDS = frozenset("""
the a an and or but if then else when where what which who whom this that
these those is are was were be been being have has had do does did will
would could should can may might must shall to of in on at for with by
from as it its it's i you he she we they them his her their our your my
me him us not no yes so than too very just about into over under again
please thanks thank can't don't didn't isn't wasn't aren't weren't
""".split())


def _message_text(message: Optional[Dict[str, Any]]) -> str:
    if not isinstance(message, dict):
        return ""
    return _content_as_text(message).strip()


def _extract_keywords(text: str, min_len: int, max_keywords: int) -> List[str]:
    """Cheap, deterministic significant-word extraction for the auto-recall
    relevance check — no LLM call, no dependency beyond stdlib. Not meant
    to be clever: a fast pre-filter deciding whether a real (LLM-backed)
    digestion call is worth spending at all, not the recall mechanism
    itself.
    """
    import re
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    seen: List[str] = []
    for w in words:
        if len(w) < min_len or w in _STOPWORDS or w in seen:
            continue
        seen.append(w)
        if len(seen) >= max_keywords:
            break
    return seen


def _content_as_text(message: Dict[str, Any]) -> str:
    """Best-effort plain-text rendering of a message, for token estimation only."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # text parts contribute their text; anything else (images,
                # tool_use blocks, ...) contributes its JSON so it still
                # counts toward the token estimate rather than vanishing.
                parts.append(part.get("text") or json.dumps(part, ensure_ascii=False))
        return " ".join(p for p in parts if p)
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)
