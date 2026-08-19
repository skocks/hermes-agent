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
  - Transport, not mechanism, is where this differs from the paper: the
    paper's harness intercepts raw code blocks straight from the
    completion stream, no tool-calling layer at all; hermes-agent wraps
    REPL access in one tool, rlm_repl. The paper's actual objection to
    tool-calling is that a rigid JSON schema forces the model to
    decompose its problem through a human-designed interface —
    RLM_REPL_SCHEMA has exactly one free-text field, `code`, which
    constrains nothing: the model writes arbitrary Python, context is
    bound as data inside it, rlm_query recurses from inside code the
    model itself wrote. Every mechanism the paper argues for is present;
    only whether the Python arrives JSON-escaped in a tool argument or
    raw in a fence differs, and that's not what the paper's argument
    rests on.
    Checked, not assumed, that closing even this transport gap isn't a
    plugin-reachable change: conversation_loop.py's
    `if assistant_message.tool_calls:` is the sole point deciding
    continue-vs-finalize each turn, so intercepting a fenced code block
    instead would need an else branch there — core loop surgery, not
    something a context-engine plugin can add. post_llm_call fires after
    the tool loop has already finished (turn_finalizer.py), so no plugin
    hook gets a chance to re-enter it either. There's also no fenced-block
    precedent already in the codebase to build on (`_strip_code_fences` in
    plugin_llm.py/title_generator.py unwraps structured output, it
    doesn't dispatch by fence tag), and an interception design would need
    a reserved fence tag anyway to avoid executing the ordinary python
    blocks models emit in normal answers — which is a tool call again,
    just one without schema validation, id correlation, or sanitizer
    coverage. Not implementing this was a deliberate decision on
    substance, not an unreached ceiling. rlm_repl's own output is still
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

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine
from agent.model_metadata import estimate_tokens_rough

from .repl import PersistentREPL
from .store import RLMStore, _stringify as _store_stringify

logger = logging.getLogger(__name__)

_DEFAULT_PROTECT_FIRST_N = 3
_DEFAULT_PROTECT_LAST_N = 25
_DEFAULT_TAIL_TOKEN_BUDGET_FRACTION = 0.5  # of context_length, for the tail slice
_DEFAULT_DROP_CHUNK_SIZE = 20  # round-15: quantized tail boundary step, see _select_tail()

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
        "capped) stays useful instead of getting truncated. Limited to "
        "20 calls per rlm_repl call (a fresh budget each time you call "
        "this tool) — raises a clear error if you exceed it rather than "
        "truncating silently; split genuinely larger recursive work "
        "across multiple rlm_repl calls instead of looping past it. "
        "Every response includes a query_spend field (count, chars_in, "
        "chars_out) so you can see what a call spent even under budget.\n"
        "- final(text) -> use this for your deliberate, complete answer "
        "instead of print() when it's long-form — it's capped higher than "
        "routine print() output (which is capped more aggressively since "
        "it's usually incidental). Still capped, not unlimited — for a "
        "very long answer, summarize with rlm_query() rather than "
        "assuming this is truly boundless.\n"
        "- reset() -> clears every variable you've set (keeping context/"
        "history/rlm_query/code_log/reset). Call this when starting an "
        "unrelated task so leftover variables from an earlier one (chunks, "
        "summary, results, ...) can't silently leak into it. If a "
        "response includes a 'stale_names' field, that's this happening — "
        "decide whether those names still matter or call reset().\n"
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
        # Round-15: protect_last_n is now a MINIMUM, not an exact trailing
        # count -- see _select_tail()'s docstring for why (prefix-cache
        # stability: a sliding exact-N window shifts by one message every
        # request, invalidating a strict prefix-matching KV cache almost
        # entirely on every turn). The actual tail size floats between
        # protect_last_n and protect_last_n + drop_chunk_size - 1.
        self.protect_last_n: int = int(cfg.get("protect_last_n", _DEFAULT_PROTECT_LAST_N))
        # How many messages the drop boundary advances by at a time,
        # instead of shifting by one every request. Bigger = more turns
        # share an identical prefix (cheaper prefill) but a larger tail
        # once the boundary does advance (one turn pays more, less often).
        self._drop_chunk_size: int = max(1, int(cfg.get("drop_chunk_size", _DEFAULT_DROP_CHUNK_SIZE)))
        # Persists across requests/turns within a session (NOT per-turn --
        # the whole point is to advance rarely). Absolute index into the
        # current session's non-system message list; reset on session
        # start/reset since it's meaningless against a different
        # conversation. See _select_tail().
        self._tail_boundary: int = 0
        self._tail_token_fraction: float = float(
            cfg.get("tail_token_budget_fraction", _DEFAULT_TAIL_TOKEN_BUDGET_FRACTION)
        )
        # Round-18: prune_tool_results_only()'s floor -- a raw tool-result
        # payload outside the protected tail must be at least this many
        # chars before it's worth replacing with a placeholder. Small
        # results aren't the problem (measured: raw web payloads were 50%
        # of one real session's visible context, 12,548 of 24,653 tokens).
        self._prune_min_result_chars: int = int(cfg.get("prune_min_result_chars", 2000))
        # Round-11 production break: the marker is inserted mid-conversation
        # (system + head + [marker] + tail -- see select_context()/
        # compress()), never at index 0. A role=='system' message anywhere
        # but index 0 is a hard 400 on strict OpenAI-compatible chat
        # templates ("System message must be at the beginning") -- a known
        # class in this codebase (title_generator.py #48338), not a
        # provider quirk. Default changed 'system' -> 'user': the marker is
        # informational text for the model, not a system instruction, and
        # every template accepts a mid-conversation user-role message.
        # Kept configurable (some future role, e.g. a template with its own
        # "note" role, is plausible) but 'system' specifically is refused
        # below -- it is never valid for THIS position, not merely
        # provider-dependent, so silently honoring it would reintroduce
        # this exact outage from config alone.
        _marker_role_cfg = cfg.get("marker_role", "user")
        if _marker_role_cfg == "system":
            logger.warning(
                "RLM: rlm.marker_role='system' in config, but the marker is "
                "inserted mid-conversation, never at index 0 -- strict chat "
                "templates reject that unconditionally, this is not a "
                "provider-specific tolerance. Coercing to 'user' this run; "
                "fix rlm.marker_role in config.yaml."
            )
            _marker_role_cfg = "user"
        self._marker_role: str = _marker_role_cfg
        self._repl_timeout: float = float(cfg.get("repl_timeout_seconds", 90.0))
        self._repl_query_timeout: float = float(cfg.get("repl_query_timeout_seconds", 60.0))
        self._repl_max_output_chars: int = int(cfg.get("repl_max_output_chars", 8000))
        # M5 fix: final() (see repl.py) gets this higher cap instead of the
        # routine one above -- a deliberate, complete answer shouldn't be
        # squeezed through the same aggressive cap as incidental print()
        # output, but still capped, never unbounded.
        self._repl_final_max_chars: int = int(cfg.get("repl_final_max_chars", 20000))
        # L3 fix: bounds one exec() call's rlm_query() recursion -- fails
        # loud with a clear limit-naming error instead of the only prior
        # bound (wall-clock timeout) which kills the whole REPL and its
        # state as its failure mode.
        self._repl_max_query_calls: int = int(cfg.get("repl_max_query_calls", 20))
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
        # Auto-recall: OFF by default (round 13). When enabled, every turn
        # past the drop threshold, select_context() checks whether the
        # incoming user message plausibly needs dropped history (cheap
        # keyword match against the archive) and, if so, injects a
        # digested-and-capped snippet directly into the request — forced
        # (not voluntary) recovery, no dependence on the model noticing
        # the marker or choosing to call rlm_repl itself. Root stays
        # protected regardless of digestion outcome when it's on: a
        # failed/oversized digest is hard-truncated, never raw, never
        # unbounded — same backstop discipline as the rest of this engine.
        #
        # Why off by default now, when round 1 argued for forcing it: that
        # argument was strong when the voluntary path (the model calling
        # rlm_repl itself) was unreliable by construction -- H1's
        # 15s-vs-60s timeout misconfiguration could kill the REPL before a
        # real rlm_query() call returned, and the REPL's context variable
        # went stale after the first call. Both fixed since. With the
        # voluntary path actually working, the case for spending a
        # blocking call_llm() digest on the request path to pre-empt the
        # model asking no longer holds -- it's a real, measured latency
        # cost (round 13) paid on every qualifying turn whether or not
        # the model would have needed it. Stays available and fully
        # supported (round 13 also memoizes it per-turn so anyone who
        # enables it doesn't pay for N identical digests per turn) --
        # just no longer the default a fresh install gets.
        self._auto_recall: bool = bool(cfg.get("auto_recall", False))
        self._auto_recall_min_keyword_len: int = int(cfg.get("auto_recall_min_keyword_len", 4))
        self._auto_recall_max_keywords: int = int(cfg.get("auto_recall_max_keywords", 6))
        self._auto_recall_min_keywords: int = int(cfg.get("auto_recall_min_keywords", 2))
        self._auto_recall_digest_threshold_tokens: int = int(cfg.get("auto_recall_digest_threshold_tokens", 400))
        self._auto_recall_max_tokens: int = int(cfg.get("auto_recall_max_tokens", 300))
        self._db_path: str = cfg.get("db_path") or self._default_db_path()
        # Round-9: retention for rlm.db is NOT an RLM-specific setting --
        # deliberately no rlm.retention_days key exists. RLM's archive
        # follows state.db's own session lifecycle (sessions.auto_prune /
        # retention_days / min_interval_hours / vacuum_after_prune /
        # min_vacuum_interval_days, the SAME keys state.db's own
        # maybe_auto_prune_and_vacuum already uses), so it just reads that
        # existing config block. See store.py's sweep_orphaned_sessions()
        # and its module docstring for the actual "RLM follows, never
        # leads" mechanism.
        self._sessions_cfg: Dict[str, Any] = full_cfg.get("sessions", {}) or {}
        self._state_db_path: str = self._default_state_db_path()

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
        # Round-13: select_context() runs on EVERY provider request within
        # a turn (M4's own premise), and incoming_message never changes
        # within a turn -- so an unmemoized auto-recall paid for the same
        # keyword search + digest sub-call once per request, up to N times
        # for an N-tool-call turn, all blocking on the critical path. One
        # slot, not a dict: only ever one turn "in flight" per engine
        # instance. Keyed on a hash of the question TEXT, not the message
        # dict itself, so this never holds a live reference into the
        # conversation. Round-14 added the third field: _persisted_count
        # at cache-write time, so a large enough mid-turn archive drift
        # (content rolling out of the live tail before the turn ends)
        # invalidates the entry instead of silently going stale. See
        # _cached_auto_recall_snippet() for the full reasoning.
        self._auto_recall_cache: Optional[tuple] = None  # (question_hash, snippet_or_None, persisted_count)
        # M6 fix: message_count(session_id) (a raw row total) is what
        # on_session_start uses to guess _persisted_count on resume -- but
        # it's inflatable by the shrink-guard's own past resyncs (each one
        # duplicates rows), so trusting it blindly can set _persisted_count
        # HIGHER than the true overlap point, silently skipping archival of
        # real content until the transcript organically grows past the
        # inflated number. Verified content-for-content on the first
        # _archive_new() call after a resume (the earliest point the live
        # transcript is actually visible); until then, treated as unverified.
        self._resume_verified: bool = True  # True until on_session_start sets it False
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

    @staticmethod
    def _default_state_db_path() -> str:
        try:
            from hermes_constants import get_hermes_home
            return str(get_hermes_home() / "state.db")
        except Exception:
            return str(Path.home() / ".hermes" / "state.db")

    def is_available(self) -> bool:
        return self._store is not None

    # -- round-16: accurate automatic-compaction status ---------------------

    def get_automatic_compaction_status_message(
        self,
        *,
        phase: str,
        default_message: str,
        **context: Any,
    ) -> Optional[str]:
        """The ABC's default text (both phases: "preflight" before a
        compaction pass, "compress" for the pass itself) describes LLM
        summarization -- "This may take a moment", "summarizing earlier
        conversation". Neither is true for this engine: select_context()
        drops from the outgoing REQUEST only (nothing summarized, nothing
        mutated), and compress() (the rare safety net -- see its own
        docstring) archives-then-trims an already-persisted transcript,
        no LLM call, no meaningful delay.

        Considered staying silent instead (emit_automatic_compaction_status
        = False): routine automatic passes are a defensible place for
        silence, and that's the honest reason to consider it here. Decided
        against it because this ISN'T rare on a long session -- preflight
        fires whenever the raw transcript (which select_context()
        deliberately never touches, see turn_context.py's preflight check)
        crosses 0.95 * context_length, which every sufficiently long
        session eventually does BY CONSTRUCTION, dropping the middle of
        the outgoing request cannot prevent it. A user seeing nothing
        during a real, recurring pause is worse than a short, true line --
        so replaced, not silenced.
        """
        approx_tokens = context.get("approx_tokens")
        tokens_str = f"~{approx_tokens:,} tokens " if approx_tokens else ""
        if phase == "preflight":
            return f"📦 RLM: trimming {tokens_str}from the request (already archived, nothing lost)."
        if phase == "compress":
            return "🗜️ RLM: trimming an already-archived transcript — not summarizing, nothing lost."
        return default_message

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
        # Round-16: this used to log a warning claiming select_context()
        # "should have prevented this" -- false, and misleading enough to
        # waste a future reader's time chasing a bug that doesn't exist.
        # select_context() shrinks the outgoing REQUEST; the preflight
        # check that calls compress() measures the raw TRANSCRIPT (see
        # turn_context.py), which select_context() deliberately never
        # touches (nothing may leak across turns via the live list). The
        # transcript therefore grows unbounded regardless of how
        # aggressively requests are trimmed, and every long session
        # crosses the preflight threshold eventually, by construction --
        # dropping sooner cannot prevent that. This IS the expected,
        # periodic maintenance this safety net exists for on a long
        # session, not a sign select_context() failed. Info, not warning.
        logger.info(
            "RLM: compress() safety-net trim on a long session (session=%s, "
            "%d transcript messages) -- expected periodic maintenance, not "
            "a select_context() failure (it bounds requests, not the "
            "transcript). Archiving then trimming; nothing is summarized "
            "or lost.",
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
        # Round-16/round-15 interaction: compress() just replaced the LIVE
        # transcript with a much shorter one (system + head + marker +
        # tail) -- self._tail_boundary is an absolute index into the OLD,
        # larger non-system list, and is now meaningless against the new
        # one. Left alone, min(boundary, n) in _select_tail() would clamp
        # it to the new (small) length, producing an EMPTY tail on the
        # very next select_context() call -- a silent desync, not a clean
        # reprefill. compress() already forces a full transcript reshape
        # (a real, unavoidable reprefill -- the safety net exists BECAUSE
        # the request must shrink right now), so resetting the boundary
        # here doesn't add a new cache cost, it just makes the ALREADY-
        # necessary reprefill happen cleanly instead of leaving a broken
        # boundary for the next call to trip over. 0 is correct, not just
        # convenient: it's the same "nothing dropped yet" state a brand
        # new session starts in, and _select_tail()'s own quantization
        # naturally recomputes the right thing from there against the new
        # (much smaller) transcript on the next call.
        self._tail_boundary = 0
        return _enforce_system_message_position(
            system + head + [self._dropped_marker()] + tail
        )

    # -- round-18: proactive tool-result prune ---------------------------------

    def prune_tool_results_only(
        self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
    ) -> tuple:
        """Deterministic, no-LLM transcript prune -- RLM didn't implement
        this ABC hook before round 18, so it inherited the safe no-op.

        Root cause this closes: RLM's tail is purely positional
        (_select_tail's quantized boundary) -- newest N messages win
        regardless of what they contain or what they cost to produce.
        Traced through real TabbyAPI turn logs: a user session spent
        roughly 217s of model time retrieving history via rlm_repl, and
        that retrieved content was completely gone from the visible
        window ~3.5 minutes later, displaced message-for-message by raw
        web-search payloads (5.7 KB -> 115.9 KB over the same window). The
        model then re-searched adjacent topics (59 web searches, 20
        near-duplicate pairs at 60%+ overlap) -- lacks earlier findings,
        retrieves them expensively, research output evicts them, lacks
        them again. Retrieval that gets evicted before the model has used
        it is pure waste.

        POLICY, stated explicitly: this is where the value-awareness
        lives, not select_context()/_select_tail(). Two ways to make the
        tail value-aware were on the table -- exempt rlm_repl results
        from _select_tail()'s positional window directly, or shrink raw
        tool payloads by kind before they ever compete for tail space.
        Chose the second, cheaper form: _select_tail()'s single quantized
        boundary is round 15's whole prefix-cache-stability mechanism (a
        measured 44x prefill reduction) -- splicing a second,
        content-dependent inclusion rule into that hot, per-request path
        risks reintroducing exactly the instability round 15 spent a
        round fixing. This hook runs on its own separate, lower-frequency
        trigger instead: every non-rlm_repl tool-result payload outside
        the protected tail (protect_last_n, reusing select_context's own
        notion of "recent enough to always keep"), above
        prune_min_result_chars, gets replaced by a short placeholder
        pointing at rlm_repl -- content already safely archived (this
        calls _archive_new() first, same discipline as compress()), never
        deleted, never unrecoverable. rlm_repl's OWN results are matched
        by tool_call_id back to the assistant message that called it and
        are NEVER pruned, at any position -- once a raw payload can no
        longer accumulate weight, it can no longer positionally outrank a
        retrieval result in _select_tail()'s token-budgeted tail either.
        The pathological case (fresh retrieval evicted before use) is
        prevented by construction: nothing this cheap for the model to
        re-fetch (a live web page) is ever allowed to outweigh something
        this expensive to have fetched (an rlm_repl result), because the
        cheap thing shrinks first.
        """
        if not self._store:
            return messages, 0
        if len(messages) <= self.protect_first_n + self.protect_last_n:
            return messages, 0

        # Archive first -- pruning must never be the operation that makes
        # content unrecoverable; it's only safe to prune BECAUSE it's
        # already archived. Idempotent (only appends what's new), so
        # calling this here every time this hook fires is not a growing cost.
        self._archive_new(messages)

        tool_name_by_call_id: Dict[str, str] = {}
        for m in messages:
            for tc in (m.get("tool_calls") or []):
                call_id = tc.get("id")
                fn = tc.get("function") or {}
                name = fn.get("name")
                if call_id and name:
                    tool_name_by_call_id[call_id] = name

        protected_tail_ids = (
            {id(m) for m in messages[-self.protect_last_n :]} if self.protect_last_n else set()
        )

        new_messages: List[Dict[str, Any]] = []
        pruned_count = 0
        for m in messages:
            if (
                m.get("role") == "tool"
                and id(m) not in protected_tail_ids
                and tool_name_by_call_id.get(m.get("tool_call_id")) != "rlm_repl"
            ):
                text = _content_as_text(m)
                if len(text) >= self._prune_min_result_chars:
                    pruned = dict(m)
                    pruned["content"] = self._prune_placeholder(len(text))
                    new_messages.append(pruned)
                    pruned_count += 1
                    continue
            new_messages.append(m)

        if not pruned_count:
            # Standard no-op caller contract (agent/context_engine.py):
            # hand back the INPUT object so callers can gate on
            # `result is not messages`.
            return messages, 0
        return new_messages, pruned_count

    @staticmethod
    def _prune_placeholder(orig_chars: int) -> str:
        return (
            f"[RLM: {orig_chars:,}-char tool result pruned from context to "
            "save space -- fully archived, not lost. Call rlm_repl to "
            "retrieve it verbatim if you need it again.]"
        )

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

        tail = self._select_tail(non_system, budget_tokens)
        head = non_system[: self.protect_first_n]
        dropped = len(non_system) - len(head) - len(tail)
        if dropped <= 0:
            return None

        system = [m for m in request_messages if m.get("role") == "system"][:1]
        marker = self._dropped_marker()

        if self._auto_recall and self._store:
            recall = self._cached_auto_recall_snippet(incoming_message)
            if recall is not None:
                marker = {"role": self._marker_role, "content": marker["content"] + "\n\n" + recall}

        return _enforce_system_message_position(system + head + [marker] + tail)

    # -- forced recovery: doesn't wait to be asked ----------------------------

    def _cached_auto_recall_snippet(self, incoming_message: Optional[Dict[str, Any]]) -> Optional[str]:
        """Round-13: select_context() runs once per provider request, not
        once per turn (M4's own premise), and incoming_message is the same
        object for every request within one turn -- without this, a
        10-tool-call turn paid for 10-11 identical keyword searches AND,
        worse, up to 10-11 identical blocking call_llm() digest sub-calls
        on the critical path before dispatch, one per request. Memoized
        for the life of one turn, keyed on a hash of the question TEXT
        (never the message dict -- this must not hold a live reference
        into the conversation). The negative result ("nothing relevant")
        is cached too: it's the common outcome, and re-running the search
        for it every request is exactly the waste this exists to remove.
        Cleared in on_turn_complete()/on_session_start()/on_session_reset()
        -- nothing survives past the turn or session it was computed in.

        Round-14: also invalidated mid-turn on archive drift, closing a
        real gap round 13 shipped documented-but-open. _archive_new()/M4
        mean the archive GROWS between requests within a turn, so a
        cached negative computed early in a turn could miss content
        archived moments later in the SAME turn. Safe while that new
        content is still in the live tail the model already sees
        (_token_bounded_tail always takes the current trailing
        protect_last_n messages -- auto-recall was never the mechanism
        that would have surfaced something already in context). NOT safe
        once it rolls out of that tail before the turn ends -- and with
        protect_last_n defaulting to 25 and each tool call contributing
        ~2 messages (assistant + tool result), ~13 tool calls in one turn
        crosses that threshold. That's an ordinary agentic turn here, not
        an edge case (round-13.md's initial "narrow" framing was wrong on
        frequency; corrected there). What made the gap acceptable to ship
        for one round was severity, not rarity: the content at risk is
        the model's own recent tool output, already seen once when
        produced, not what the user's question was about, and reachable
        via rlm_repl regardless -- caching only downgrades a best-effort
        pre-emptive hint, tail eviction is the actual data-visibility
        mechanism either way.

        Closed here at near-zero extra cost: no DB call needed, just
        comparing self._persisted_count (already updated by _archive_new,
        which runs before this every request thanks to M4) against the
        count captured when this cache entry was written. Once the drift
        exceeds protect_last_n, SOME already-archived content from this
        turn is guaranteed to have rolled out of the tail since the cache
        was populated, so a full recompute is triggered -- restoring
        exactly the uncached behavior in precisely the case where caching
        could have differed, while the common case (a turn adding a
        handful of messages, never tripping the threshold) keeps the full
        saving. Deliberately not a partial-refresh or incremental-merge
        scheme -- that's real complexity for a pre-emptive hint on an
        opt-in path, and would erode the reason memoization exists.
        """
        question = _message_text(incoming_message)
        key = hashlib.sha256(question.encode("utf-8")).hexdigest() if question else None
        if self._auto_recall_cache is not None:
            cached_key, cached_snippet, cached_persisted_count = self._auto_recall_cache
            drift = self._persisted_count - cached_persisted_count
            if cached_key == key and drift <= self.protect_last_n:
                return cached_snippet
        snippet = self._auto_recall_snippet(incoming_message)
        self._auto_recall_cache = (key, snippet, self._persisted_count)
        return snippet

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

    def _select_tail(
        self, non_system: List[Dict[str, Any]], budget_tokens: int
    ) -> List[Dict[str, Any]]:
        """Round-15: WHICH messages form the tail -- quantized so
        consecutive requests share an identical prefix instead of sliding
        by one message every time.

        The prior behavior (non_system[-protect_last_n:]) is an exact
        trailing window: by construction it shifts by however many
        messages were added since the last call, EVERY single request
        (select_context runs once per provider request, per M4 -- not
        once per turn). Measured against real TabbyAPI turn logs: RLM
        turns averaged 88% UNCACHED tokens and ~10x the prefill time of
        non-RLM turns on the same box, because a strict-prefix-matching
        KV cache invalidates from the first byte that differs onward --
        and with a sliding window, that first differing byte is near the
        very front of the tail almost every request. We were trading
        context length for prefill compute, and on this hardware that
        trade loses badly. (An earlier attempt at this, _MARKER_BUCKET,
        rounded the dropped-count string in the marker to reduce churn --
        insufficient on its own, since the tail itself still moved every
        request regardless of what the marker said; removed in this same
        round in favor of a constant marker, see _dropped_marker().)

        Fix: the boundary (index into non_system where the tail begins)
        only advances in steps of self._drop_chunk_size, and only ever
        forward (self._tail_boundary persists across requests within a
        session). Between advances, non_system[boundary:] for request K+1
        is exactly non_system[boundary:] for request K PLUS whatever new
        messages arrived since -- a pure append, exactly what prefix
        caching wants. N-1 out of N requests (chunk_size-ish) become
        near-free; one pays a reprefill when the boundary steps. This is
        also why protect_last_n is now a MINIMUM: the actual tail floats
        between protect_last_n and protect_last_n + drop_chunk_size - 1
        messages, only ever snapping back down to protect_last_n-ish
        right after the boundary advances.

        Hitting the token cap (see _bound_tail_tokens) is itself treated
        as a legitimate reason to advance the boundary early, not just a
        per-request truncation: without that, an oversized tail would get
        re-trimmed (and re-shifted) identically on every subsequent
        request, which is the exact instability this method exists to
        remove. So a cap-forced trim permanently advances the boundary to
        where the trim actually starts.
        """
        n = len(non_system)
        if n == 0 or self.protect_last_n <= 0:
            self._tail_boundary = n
            return []
        min_boundary = max(0, n - self.protect_last_n)
        quantized = (min_boundary // self._drop_chunk_size) * self._drop_chunk_size
        boundary = min(max(self._tail_boundary, quantized), n)
        candidates = non_system[boundary:]
        trimmed = self._bound_tail_tokens(candidates, budget_tokens)
        if len(trimmed) < len(candidates):
            boundary = n - len(trimmed)
        self._tail_boundary = boundary
        return trimmed

    def _bound_tail_tokens(
        self, candidates: List[Dict[str, Any]], budget_tokens: int
    ) -> List[Dict[str, Any]]:
        """Cap an already-selected tail slice by token budget.

        Message-count protection alone can still blow the context if the
        tail happens to contain a few huge tool results. Walk backward and
        stop early if the token estimate exceeds the configured fraction of
        the model's context window — better a shorter-but-safe tail than a
        request that 400s on overflow. Takes the candidate slice directly
        (from _select_tail's quantized boundary) rather than slicing
        non_system itself -- WHICH messages form the tail and HOW MANY of
        them fit the token budget are separate concerns now.
        """
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

    def _dropped_marker(self) -> Dict[str, Any]:
        """Round-15: constant text, no embedded count. It used to report a
        bucketed dropped-message count (_MARKER_BUCKET, rounded to 5) --
        removed, not just left as-is, because the marker sits near the
        front of the request (system + head + [marker] + tail) where a
        strict-prefix-matching KV cache is most sensitive: ANY change here
        invalidates the marker itself plus the entire tail after it,
        regardless of whether the tail's own content actually changed.
        Bucketing to 5 reduced how often that happened; it didn't stop it,
        and round-15's actual measurement (real TabbyAPI turn logs, not a
        guess) is what showed a "reduced" cache break was still a
        devastating one on this hardware. The count was never something
        the model acted on -- it's not worth a cache break at any
        granularity, so it's gone, not bucketed further.

        Round-17: reworded from informational to directive. Measured
        against 5,458 real production turns: 97 carried this marker, 0
        ever led to an rlm_repl call -- yet a manually-pointed real
        session (same model, same tool list) used rlm_repl competently,
        multi-call, self-refining, and got a correct, verbatim-recovered
        answer the moment it was told to look. Not a capability gap, not
        a tool-description gap -- a discovery gap: the old text
        ("...Use rlm_repl ... if you need something from earlier") made
        retrieval conditional on the model first judging, unprompted,
        that it lacks something -- precisely the judgment it wasn't
        making. The rewrite makes checking the default action instead of
        a maybe: not being sure whether earlier context matters is itself
        the trigger, not a reason to skip it. Kept short and still
        constant text -- this sits on the same cache-sensitive prefix
        round 15 stabilized, so its length matters as much as the old
        bucketed count did.
        """
        return {
            "role": self._marker_role,
            "content": (
                "[RLM: earlier context was dropped here — archived, not "
                "deleted. Call rlm_repl FIRST before answering anything "
                "that might depend on it. Being unsure whether it matters "
                "IS the reason to check — don't guess or claim you can't "
                "recall without checking.]"
            ),
        }

    # -- archiving ---------------------------------------------------------

    def _archive_new(self, messages: List[Dict[str, Any]], turn_id: int = -1) -> None:
        if not self._store:
            return
        if not self._resume_verified:
            self._verify_resume_watermark(messages)
        if len(messages) < self._persisted_count:
            # The live transcript got shorter than what we've archived —
            # e.g. a /reset, a manual /compress, or a session we haven't
            # seen state for. Don't silently drop future messages: re-sync
            # from zero.
            logger.info(
                "RLM: transcript shrank under archived count (session=%s, "
                "have=%d archived=%d) — re-syncing from 0",
                self._session_id, len(messages), self._persisted_count,
            )
            self._trigger_resync()
        new_messages = messages[self._persisted_count :]
        if not new_messages:
            return
        # Round-18 fix: a resync used to tombstone EVERY current row
        # up front (_trigger_resync, before this method ever saw the
        # messages actually being re-appended), then re-append whatever
        # the live transcript happened to contain at that moment. If the
        # live transcript's middle was already dropped by select_context()
        # -- the normal state once a session is long enough, not an edge
        # case -- the re-append never reproduced everything the blanket
        # tombstone just hid, and that archive-only content vanished from
        # every read path (search_any, history(), context) while still
        # physically present. Production impact: 152 tombstoned rows in
        # one real session, 88 of them unique nowhere else, one of them a
        # specific fact a later turn confidently reported as absent and
        # then fabricated a substitute for.
        #
        # Fix: supersede only AFTER new_messages is known, and only rows
        # this exact re-append reproduces (content-matched, see
        # supersede_reproduced_rows's own docstring). Runs on every
        # append, not just post-resync ones -- harmless no-op the rest of
        # the time (new_messages on a normal incremental append is brand
        # new content that can't match any existing row), and doing it
        # unconditionally here (instead of gating on "was this call
        # preceded by a resync") is simpler and can't drift out of sync
        # with _trigger_resync's own bookkeeping.
        try:
            superseded_count = self._store.supersede_reproduced_rows(self._session_id, new_messages)
            if superseded_count:
                logger.info(
                    "RLM: resync superseded %d row(s) reproduced by the "
                    "re-append (session=%s) -- archive-only content not in "
                    "new_messages was left visible, not tombstoned",
                    superseded_count, self._session_id,
                )
        except Exception:
            logger.exception(
                "RLM: supersede_reproduced_rows failed for session=%s -- "
                "proceeding with the append anyway; duplicates may be "
                "visible until this is retried successfully, which is "
                "safer than the alternative (skipping the append and "
                "losing new content)",
                self._session_id,
            )
        try:
            self._store.append_messages(self._session_id, turn_id, new_messages)
            self._persisted_count = len(messages)
        except Exception:
            logger.exception(
                "RLM: failed to archive %d message(s) for session=%s",
                len(new_messages), self._session_id,
            )

    def _trigger_resync(self) -> None:
        """Reset the archive cursor to 0 so the caller's subsequent
        append_messages() re-inserts the full live transcript. Round-18:
        no longer tombstones here -- _archive_new() does that AFTER it
        knows what new_messages actually is, matching content rather than
        blanket-hiding everything (see supersede_reproduced_rows). Still
        centralized so every resync trigger (shrink-guard, watermark
        verification failure or error) goes through the same reset.
        """
        self._persisted_count = 0

    def _verify_resume_watermark(self, messages: List[Dict[str, Any]]) -> None:
        """M6 fix: on_session_start()'s _persisted_count estimate comes
        from message_count() (a raw row total), inflatable by the shrink-
        guard's own past resyncs -- each duplicates rows, so the total can
        end up HIGHER than the true overlap point between what's archived
        and what's live. Trusting that blindly means messages[_persisted_
        count:] silently skips real, never-archived content until the live
        transcript organically grows past the inflated number.

        Runs once, on the first _archive_new() call after a resume (the
        earliest point the live transcript is actually visible to this
        engine -- on_session_start() doesn't receive it). Compares the
        stored content of the last few archived rows against the
        corresponding tail of the live transcript; on any mismatch (or if
        there isn't enough live transcript to check against), falls back
        to the existing safe default -- full resync from 0 -- rather than
        trusting an unverifiable number.
        """
        self._resume_verified = True  # only ever attempt this once per resume
        check_n = min(5, self._persisted_count, len(messages))
        if check_n <= 0:
            if self._persisted_count > len(messages):
                # Round-5 review: a bare persisted_count reset here, without
                # going through _trigger_resync(), used to leave the NEXT
                # on_turn_complete's re-append undedupe'd against old rows
                # -- duplicating like N2. _trigger_resync() (now just a
                # cursor reset -- round 18 moved the actual superseding into
                # _archive_new(), content-matched against what's really
                # being re-appended) is the safe default regardless of
                # whether there's anything to re-append on THIS call.
                self._trigger_resync()
            return
        try:
            archived_tail = self._store.tail_content(self._session_id, check_n)
        except Exception:
            logger.exception("RLM: resume watermark verification failed, resyncing from 0")
            self._trigger_resync()
            return
        live_tail = [
            _store_stringify(m) for m in messages[self._persisted_count - check_n : self._persisted_count]
        ]
        if archived_tail != live_tail:
            logger.warning(
                "RLM: resume watermark for session=%s did not verify "
                "(message_count=%d likely inflated by an earlier resync) "
                "— resyncing from 0 instead of trusting it",
                self._session_id, self._persisted_count,
            )
            self._trigger_resync()

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
        # Round-13: the turn this cache was scoped to just ended -- clear
        # it so the next turn's (possibly identical-looking) question
        # recomputes rather than reusing this turn's answer.
        self._auto_recall_cache = None

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
                # what's already there on the next on_turn_complete(). This
                # is a provisional estimate -- _archive_new() verifies it
                # content-for-content on first real use (M6 fix) rather
                # than trusting it blindly for the rest of the session.
                self._persisted_count = self._store.message_count(self._session_id)
                self._resume_verified = self._persisted_count == 0  # nothing to verify against
            except Exception:
                logger.exception("RLM: message_count lookup failed on session start")
                # Round-5 review: same counterexample as the check_n<=0
                # branch above -- a bare reset here would leave real old
                # rows undeduped against the next append. We don't know if
                # there ARE old rows (the query that would tell us just
                # failed), so _trigger_resync() (a cursor reset only, since
                # round 18 -- the actual content-matched supersede happens
                # in _archive_new() once real messages are known) is the
                # safe default regardless.
                self._trigger_resync()
                self._resume_verified = True
            # Round-9: throttled internally (rlm_meta, not this call site),
            # so calling it on every on_session_start -- including every
            # /new, since this engine instance is reused rather than
            # reconstructed -- is safe and normally a no-op. Never allowed
            # to block session start: sweep_orphaned_sessions() itself
            # fails open and never raises, but this session must start
            # regardless of what happens here either way.
            try:
                self._store.sweep_orphaned_sessions(
                    self._state_db_path,
                    current_session_id=self._session_id,
                    min_interval_hours=int(self._sessions_cfg.get("min_interval_hours", 24)),
                    vacuum_after_prune=bool(self._sessions_cfg.get("vacuum_after_prune", True)),
                    min_vacuum_interval_days=int(self._sessions_cfg.get("min_vacuum_interval_days", 30)),
                )
            except Exception:
                logger.exception("RLM: orphan sweep call site failed")
        else:
            self._persisted_count = 0
            self._resume_verified = True
        self._next_turn_id = 0
        # Round-13: a new session must never inherit the previous
        # session's auto-recall answer -- same reasoning as
        # on_turn_complete(), one level up (this engine instance is reused
        # across /new, so without this a stale cross-session cache entry
        # is a real, not theoretical, risk).
        self._auto_recall_cache = None
        # Round-15: the tail boundary is an absolute index into THIS
        # session's non-system message list -- meaningless (and, since it
        # persists across requests by design, dangerous: min(...,n)
        # clamps out-of-range but a stale large value would otherwise
        # start a new session with an artificially empty tail) against a
        # different session's conversation.
        self._tail_boundary = 0

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
        self._resume_verified = True  # count is 0, nothing to verify
        self._next_turn_id = 0
        self._auto_recall_cache = None
        self._tail_boundary = 0
        if self._repl is not None:
            self._repl.close()
            self._repl = None

    # -- tools: the persistent REPL -------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # L4 fix: agent_init.py reads this ONCE at agent construction. If
        # the store wasn't open at that exact moment (a transient FS
        # error), gating this on self._store meant rlm_repl was never
        # registered even after _open_store() later succeeded on the next
        # on_session_start() -- yet the dropped-content marker still tells
        # the model to call rlm_repl, pointing it at a tool that doesn't
        # exist. Register unconditionally; handle_tool_call() below already
        # reports a clean, specific error when the store really is down.
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
            final_max_chars=self._repl_final_max_chars,
            max_query_calls=self._repl_max_query_calls,
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


def _enforce_system_message_position(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Round-11 production break: strict OpenAI-compatible chat templates
    reject a request where a role=='system' message isn't the very first
    message ("System message must be at the beginning") -- a known class
    in this codebase (title_generator.py's #48338 finding), not a
    provider-specific quirk. select_context()/compress() build
    system + head + [marker] + tail themselves, so nothing but index 0
    should ever be role=='system' -- checked here structurally rather
    than trusted, because a malformed list on this path is a hard 400
    that aborts the user's turn outright, not a degraded-but-working
    response the model could route around.

    Coerces any offending message's role to 'user' (copying the dict,
    never mutating the original -- these are shared with the live
    conversation/archive) rather than only logging: shipping the broken
    request anyway after detecting it defeats the point of the check.
    Should never actually trigger post-fix (marker_role's own 'system'
    guard in __init__ is the primary fix); this is the backstop for
    anything that reaches this point some other way in the future.
    """
    fixed = []
    for i, m in enumerate(messages):
        if i > 0 and isinstance(m, dict) and m.get("role") == "system":
            logger.error(
                "RLM: a non-first message had role='system' (index %d of %d) "
                "-- coerced to 'user' to avoid a hard 400 from strict chat "
                "templates. This should never happen; investigate.",
                i, len(messages),
            )
            m = {**m, "role": "user"}
        fixed.append(m)
    return fixed


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
