"""Regression tests for the RLM context engine.

Covers the exact bugs found by an independent fidelity audit
(scratchpad/rlm-audit.md, 2026-08-19) — C1/C2 (data loss) and H1 (the
paper's core recursion mechanism silently broken by a timeout
misconfiguration) are the audit's own "highest-value first" picks, so
they're first here too. No live model/network calls — those are verified
manually against the running local server, not in CI.
"""

from __future__ import annotations

import json

import pytest

from agent.model_metadata import estimate_tokens_rough
from plugins.context_engine.rlm.engine import RLMContextEngine, _content_as_text
from plugins.context_engine.rlm.repl import PersistentREPL


def _make_engine(tmp_path, **rlm_overrides):
    cfg = {"rlm": {"db_path": str(tmp_path / "rlm.db"), **rlm_overrides}}
    return RLMContextEngine(config=cfg)


def _convo(n=60):
    return [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(n)
    ]


# ---------------------------------------------------------------------------
# C1 — select_context() must never drop content it can't archive
# ---------------------------------------------------------------------------

def test_select_context_noop_when_store_unavailable(tmp_path):
    # A path no test process can create -> RLMStore.__init__ raises,
    # self._store stays None.
    unwritable = "/root/definitely-not-writable/rlm.db"
    engine = RLMContextEngine(config={"rlm": {"db_path": unwritable}})
    assert engine._store is None

    convo = _convo()
    selected = engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)

    assert selected is None, "must fail open (no drop) when nothing can archive the dropped content"


def test_select_context_drops_normally_when_store_available(tmp_path):
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo()
    selected = engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)

    assert selected is not None
    assert len(selected) < len(convo)


# ---------------------------------------------------------------------------
# C2 — the engine instance is reused across /new; the store must survive it
# ---------------------------------------------------------------------------

def test_store_reopens_across_session_end_then_start(tmp_path):
    """Mirrors cli.py's real /new sequence: on_session_end() fires on the
    SAME engine instance, then on_session_start() rebinds it to a new
    session — the engine is never reconstructed.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("session-A")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine.on_turn_complete([{"role": "user", "content": "hello A"}], turn_id=1)
    assert engine._store.message_count("session-A") == 1

    engine.on_session_end("session-A", [{"role": "user", "content": "hello A"}])
    assert engine._store is None, "store is expected closed immediately after on_session_end"

    engine.on_session_start("session-B")
    assert engine._store is not None, "store must reopen on the next session start"

    engine.on_turn_complete([{"role": "user", "content": "hello B"}], turn_id=1)
    assert engine._store.message_count("session-B") == 1


def test_select_context_still_safe_immediately_after_new(tmp_path):
    engine = _make_engine(tmp_path)
    engine.on_session_start("session-A")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine.on_session_end("session-A", [])
    engine.on_session_start("session-B")

    convo = _convo()
    selected = engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)
    assert selected is not None and len(selected) < len(convo), (
        "archiving must work again after /new, not just fail open forever"
    )


def test_second_new_also_reopens(tmp_path):
    """Not just the first /new — every subsequent one too."""
    engine = _make_engine(tmp_path)
    for i, sid in enumerate(["s1", "s2", "s3"]):
        engine.on_session_start(sid)
        engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
        engine.on_turn_complete([{"role": "user", "content": f"hi {i}"}], turn_id=1)
        assert engine._store is not None
        assert engine._store.message_count(sid) == 1
        engine.on_session_end(sid, [])


# ---------------------------------------------------------------------------
# H1 — REPL timeout must never be shorter than rlm_query's own HTTP timeout
# ---------------------------------------------------------------------------

def test_repl_timeout_self_heals_when_shorter_than_query_timeout(caplog):
    repl = PersistentREPL(
        db_path="/tmp/unused.db", session_id="s", base_url="http://x/v1", model="m",
        timeout=15.0, query_timeout=60.0,
    )
    assert repl.timeout > repl.query_timeout, (
        "a single rlm_query() call must not be able to outlive and kill the "
        "REPL that made it — this was the audit's H1 finding at the shipped "
        "defaults (15s vs 60s)"
    )


def test_repl_timeout_left_alone_when_already_sane():
    repl = PersistentREPL(
        db_path="/tmp/unused.db", session_id="s", base_url="http://x/v1", model="m",
        timeout=90.0, query_timeout=60.0,
    )
    assert repl.timeout == 90.0


def test_default_repl_timeout_exceeds_default_query_timeout(tmp_path):
    """The shipped config_defaults.py values, not just the self-heal path."""
    engine = _make_engine(tmp_path)
    assert engine._repl_timeout > engine._repl_query_timeout


# ---------------------------------------------------------------------------
# M1 — API key must not be interpolated into the child process argv
# ---------------------------------------------------------------------------

def test_api_key_not_in_bootstrap_script_text(tmp_path):
    repl = PersistentREPL(
        db_path=str(tmp_path / "x.db"), session_id="s", base_url="http://x/v1",
        model="m", api_key="super-secret-value",
    )
    script = repl._bootstrap_script()
    assert "super-secret-value" not in script, (
        "the API key must travel via env (RLM_API_KEY), not get baked into "
        "the -c script text, which becomes this process' argv (visible to "
        "any local user via `ps auxww`)"
    )


# ---------------------------------------------------------------------------
# H2 — auto-recall digestion must propagate api_mode (not silently ignore it)
# ---------------------------------------------------------------------------

def test_update_model_captures_api_mode(tmp_path):
    engine = _make_engine(tmp_path)
    engine.update_model(
        model="m", context_length=131072, base_url="http://x/v1",
        api_key="k", provider="custom", api_mode="chat_completions",
    )
    assert engine._runtime.get("api_mode") == "chat_completions"


# ---------------------------------------------------------------------------
# M2 — recursive sub-call model should default to delegation config, not root
# ---------------------------------------------------------------------------

def test_repl_query_model_defaults_to_delegation_model(tmp_path):
    cfg = {
        "rlm": {"db_path": str(tmp_path / "rlm.db")},
        "delegation": {"model": "cheap-delegate-model", "base_url": "http://delegate/v1"},
    }
    engine = RLMContextEngine(config=cfg)
    assert engine._repl_query_model == "cheap-delegate-model"
    assert engine._repl_query_base_url == "http://delegate/v1"


def test_repl_query_model_falls_back_to_root_when_no_delegation_configured(tmp_path):
    engine = _make_engine(tmp_path)  # no "delegation" key at all
    engine.update_model(model="root-model", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    assert engine._runtime["model"] == "root-model"


def test_explicit_repl_query_model_overrides_delegation_default(tmp_path):
    cfg = {
        "rlm": {"db_path": str(tmp_path / "rlm.db"), "repl_query_model": "explicit-override"},
        "delegation": {"model": "cheap-delegate-model"},
    }
    engine = RLMContextEngine(config=cfg)
    assert engine._repl_query_model == "explicit-override"


# ---------------------------------------------------------------------------
# Round-2 audit findings (rlm-audit.md, "Re-check (round 2)")
# ---------------------------------------------------------------------------

# R5 — api_mode must follow delegation the same way model/base_url do
def test_api_mode_follows_delegation_override(tmp_path):
    cfg = {
        "rlm": {"db_path": str(tmp_path / "rlm.db")},
        "delegation": {"model": "cheap-model", "base_url": "http://cheap/v1", "api_mode": "chat_completions"},
    }
    engine = RLMContextEngine(config=cfg)
    engine.update_model(model="root-model", context_length=131072, base_url="http://root/v1", api_mode="responses")
    assert engine._runtime["api_mode"] == "chat_completions", (
        "a delegation override with a differently-shaped endpoint must not "
        "silently inherit root's api_mode"
    )


def test_api_mode_falls_back_to_root_without_delegation_override(tmp_path):
    engine = _make_engine(tmp_path)
    engine.update_model(model="root-model", context_length=131072, base_url="http://root/v1", api_mode="responses")
    assert engine._runtime["api_mode"] == "responses"


# M4 — select_context() must archive too, not just on_turn_complete()
def test_select_context_archives_mid_turn(tmp_path):
    """A message dropped from the request partway through a long
    tool-calling turn must already be in the archive -- on_turn_complete()
    only fires once the whole turn finishes, which is too late for
    rlm_repl to retrieve it during that same turn.
    """
    engine = _make_engine(tmp_path, protect_first_n=1, protect_last_n=3)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"mid-turn msg {i}"}
        for i in range(20)
    ]
    # Simulates a provider round trip mid-turn: on_turn_complete has NOT fired.
    engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)

    assert engine._store.message_count("s1") > 0, (
        "content must be archived from select_context() alone, before "
        "on_turn_complete ever fires"
    )


def test_select_context_archiving_is_idempotent(tmp_path):
    """Calling select_context() repeatedly within one turn (multiple
    tool-calling round trips) must not duplicate archive rows.
    """
    engine = _make_engine(tmp_path, protect_first_n=1, protect_last_n=3)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(10)
    ]
    engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)
    first_count = engine._store.message_count("s1")
    engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)  # same convo again
    second_count = engine._store.message_count("s1")

    assert first_count == second_count == len(convo)  # 1 system + 10 user, archived once


# ---------------------------------------------------------------------------
# R2/R3 — real REPL subprocess tests (no network: only history()/context/
# reset() are exercised, never rlm_query()).
# ---------------------------------------------------------------------------

def _archive_and_repl(tmp_path, messages, turn_id=1, **repl_overrides):
    engine = _make_engine(tmp_path)
    engine.on_session_start("repl-test")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine.on_turn_complete(messages, turn_id=turn_id)
    repl_kwargs = {"base_url": "http://x/v1", "model": "m"}
    repl_kwargs.update(repl_overrides)
    repl = PersistentREPL(db_path=engine._store.db_path, session_id="repl-test", **repl_kwargs)
    return engine, repl


@pytest.fixture(autouse=True)
def _cleanup_repl():
    repls = []
    yield repls
    for r in repls:
        r.close()


def test_context_refreshes_across_calls_not_stale(tmp_path, _cleanup_repl):
    engine, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "first"}], turn_id=1)
    _cleanup_repl.append(repl)

    r1 = repl.exec("print(len(context))")
    assert r1["error"] is None
    before = int(r1["stdout"].strip())

    # Archive more AFTER the REPL process already started.
    engine.on_turn_complete(
        [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}], turn_id=2,
    )

    r2 = repl.exec("print(len(context))")
    assert r2["error"] is None
    after = int(r2["stdout"].strip())

    assert after > before, (
        "context must reflect newly archived content on every call -- a "
        "REPL that lives for the whole session cannot bind context once "
        "at startup, or it goes stale the moment anything new is archived"
    )


def test_context_is_chronological(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(
        tmp_path,
        [{"role": "user", "content": "oldest"}, {"role": "user", "content": "newest"}],
    )
    _cleanup_repl.append(repl)
    r = repl.exec("print(context[0]['content'])")
    assert r["stdout"].strip() == "oldest"


def test_context_total_and_truncated_exposed(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)
    r = repl.exec("print(context_total, context_truncated)")
    assert r["error"] is None
    assert r["stdout"].strip() == "1 False"


def test_reset_clears_user_vars_keeps_builtins(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("leftover = 'should not survive reset'")
    reset_result = repl.exec("print(reset())")
    assert reset_result["error"] is None

    check = repl.exec("print('leftover' in dir())\nprint(callable(history))\nprint(callable(rlm_query))")
    assert check["stdout"].strip() == "False\nTrue\nTrue"


# ---------------------------------------------------------------------------
# Round-3 audit findings: base names clobberable, staleness invisible,
# no provenance for bindings created by turns no longer in visible context.
# ---------------------------------------------------------------------------

def test_clobbered_base_name_self_heals_next_call(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("history = 42")  # clobber
    r = repl.exec("print(callable(history))")
    assert r["stdout"].strip() == "True", "a clobbered base name must self-heal by the next call"


def test_clobbered_reset_itself_self_heals(tmp_path, _cleanup_repl):
    """reset() is the one escape hatch from bad state -- it must not be
    permanently killable by the model shadowing it.
    """
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("def reset(): return 'fake'")
    r = repl.exec("print(reset())")
    assert "REPL namespace reset" in r["stdout"], "reset() must self-heal even after being shadowed"


def test_staleness_reported_via_dedicated_field(tmp_path, _cleanup_repl):
    """N1 fix: this used to be an inline stdout footer, which the parent's
    max_output_chars truncation (keeps the front, trims the end) silently
    ate on any call with 8000+ chars of real output -- exactly the heavy-
    output exploratory calls most likely to leave stale names behind. Now
    a dedicated result key, immune to that truncation.
    """
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("chunks = [1, 2, 3]")
    r = repl.exec("print(1)")  # does not touch chunks -- must be flagged as stale
    assert r.get("stale_names") == ["chunks"]
    assert "chunks" not in r["stdout"], "stale-name reporting must not live in stdout at all anymore"

    r_same_call = repl.exec("just_set = True\nprint(1)")
    assert "just_set" not in (r_same_call.get("stale_names") or []), (
        "a variable set in THIS call must not be flagged as stale yet"
    )


def test_staleness_field_absent_after_reset(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("chunks = [1, 2, 3]")
    repl.exec("reset()")
    r = repl.exec("print(1)")
    assert "stale_names" not in r


def test_staleness_field_survives_heavy_stdout_truncation(tmp_path, _cleanup_repl):
    """N1's exact regression case: a call whose real print() output alone
    exceeds max_output_chars must still surface the drift warning.
    """
    _, repl = _archive_and_repl(
        tmp_path, [{"role": "user", "content": "x"}], max_output_chars=100,
    )
    _cleanup_repl.append(repl)

    repl.exec("chunks = [1, 2, 3]")
    r = repl.exec("print('X' * 9000)")
    assert r.get("truncated") is True, "sanity check: this call must actually hit the stdout cap"
    assert r.get("stale_names") == ["chunks"], (
        "drift warning must survive even when stdout itself gets truncated"
    )


def test_code_log_recovers_provenance(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("def my_func(): return 1")
    r = repl.exec("print(code_log(5))")
    assert "my_func" in r["stdout"], "code_log() must recover the source of an earlier call"


def test_code_log_survives_reset(tmp_path, _cleanup_repl):
    """reset() clears variables, but the record of what created them should
    still be inspectable -- otherwise reset() itself destroys the
    provenance code_log() exists to preserve.
    """
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("def my_func(): return 1")
    repl.exec("reset()")
    r = repl.exec("print(code_log(10))")
    assert "my_func" in r["stdout"]


# ---------------------------------------------------------------------------
# M6 — resume watermark inflation must not silently skip real content
# ---------------------------------------------------------------------------

def test_m6_inflated_resume_count_verified_and_corrected(tmp_path):
    """The narrow case a plain len(messages) < persisted_count shrink-guard
    misses: an inflated message_count() LOWER than the live transcript's
    length at resume. Without content verification this silently skips
    real archived content forever, because the count "looks" consistent.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    # Plant an inflated archive: 2 real rows + 5 unrelated "stale" rows,
    # simulating rows left behind by an earlier shrink+resync duplication.
    engine._store.append_messages("s1", 1, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    engine._store.append_messages("s1", 1, [{"role": "user", "content": f"stale-{i}"} for i in range(5)])
    assert engine._store.message_count("s1") == 7

    # True resume: a fresh engine instance, live transcript continuing from
    # the real 2-message base, now at 9 messages -- LONGER than the
    # inflated count, so the existing shrink-guard alone would not fire.
    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    assert engine2._persisted_count == 7  # provisional, unverified
    assert engine2._resume_verified is False

    live = [{"role": "user", "content": c} for c in "abcdefghi"]
    engine2.on_turn_complete(live, turn_id=2)

    import sqlite3
    rows = [
        r[0] for r in sqlite3.connect(str(tmp_path / "rlm.db"))
        .execute("SELECT content FROM rlm_messages WHERE session_id='s1' ORDER BY id").fetchall()
    ]
    missing = [c for c in "cdefg" if c not in rows]
    assert not missing, f"content silently skipped: {missing}"
    assert engine2._resume_verified is True


def test_m6_verified_resume_does_not_resync_unnecessarily(tmp_path):
    """The common case -- a real, non-inflated resume -- must not pay a
    full resync (duplicate rows) just because verification now runs.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine.on_turn_complete([{"role": "user", "content": c} for c in "abcde"], turn_id=1)
    assert engine._store.message_count("s1") == 5

    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    live = [{"role": "user", "content": c} for c in "abcdef"]  # one new message: 'f'
    engine2.on_turn_complete(live, turn_id=2)

    assert engine2._store.message_count("s1") == 6, "must not duplicate a/b/c/d/e when the count was accurate"


# ---------------------------------------------------------------------------
# M5 — final() gets a higher cap than routine stdout, but still a cap
# ---------------------------------------------------------------------------

def test_final_passes_through_under_its_cap(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(
        tmp_path, [{"role": "user", "content": "x"}],
        max_output_chars=100, final_max_chars=500,
    )
    _cleanup_repl.append(repl)
    r = repl.exec("final('X' * 300)")
    assert r.get("final") == "X" * 300
    assert not r.get("final_truncated")


def test_final_still_capped_above_its_limit(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(
        tmp_path, [{"role": "user", "content": "x"}],
        max_output_chars=100, final_max_chars=500,
    )
    _cleanup_repl.append(repl)
    r = repl.exec("final('Y' * 1000)")
    assert r.get("final_truncated") is True
    assert len(r["final"]) < 1000, "final() must never be truly unbounded"


def test_final_cap_is_higher_than_routine_stdout_cap(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(
        tmp_path, [{"role": "user", "content": "x"}],
        max_output_chars=100, final_max_chars=500,
    )
    _cleanup_repl.append(repl)
    r_stdout = repl.exec("print('Z' * 300)")
    assert r_stdout.get("truncated") is True, "routine print() must use the smaller, more aggressive cap"

    r_final = repl.exec("final('Z' * 300)")
    assert not r_final.get("final_truncated"), "the same length must pass through final()'s higher cap"


def test_final_absent_when_not_called(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)
    r = repl.exec("print('no final call')")
    assert "final" not in r


# ---------------------------------------------------------------------------
# N2 — resync must not duplicate content into history()/context, and must
# not destroy archive-only history either (tombstone, not delete/duplicate).
# ---------------------------------------------------------------------------

def test_resync_does_not_duplicate_visible_content(tmp_path):
    """The exact case flagged in round-5 review: a resync used to leave
    duplicated content directly visible through message_count() (and, via
    the real REPL, through history()/context) -- not just extra physical
    rows, but rows the model would actually see and could see more than
    once.

    Round-18 rewrite: the original version of this test asserted
    message_count() == 9 (the live transcript's own length) after resync
    -- which was only correct BECAUSE the old fixture's "stale-N" rows
    happened to also get blanket-tombstoned by the pre-round-18 resync,
    silently confirming the very bug round 18 fixed (archive-only content
    disappearing). The correct invariant is narrower: content the new
    live transcript ACTUALLY REPRODUCES ("a", "b") must not be visible
    twice; content it does NOT reproduce ("stale-0".."stale-4") must stay
    visible, not vanish. Both checked explicitly here now.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    # Plant an inflated archive: "a"/"b" (which the resync below WILL
    # reproduce) plus 5 unrelated stale rows it will NOT.
    engine._store.append_messages("s1", 1, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    engine._store.append_messages("s1", 1, [{"role": "user", "content": f"stale-{i}"} for i in range(5)])
    assert engine._store.raw_row_count("s1") == 7

    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    live = [{"role": "user", "content": c} for c in "abcdefghi"]  # 9 messages, starts with a, b
    engine2.on_turn_complete(live, turn_id=2)

    assert engine2._store.message_count("s1") == 14, (
        "9 from the fresh re-append + 5 preserved 'stale-N' rows the live "
        "transcript never reproduced -- 'a'/'b' must not be double-counted, "
        "but 'stale-N' must not have vanished either"
    )
    contents = {row["content"] for row in engine2._store.search_any("s1", ["stale"], limit=20)}
    assert len(contents) == 5, "the preserved stale rows must still be findable, not just counted"


def test_resync_preserves_archive_only_history_physically(tmp_path):
    """A plain DELETE before resync would destroy archive-only history --
    exactly the situation that triggers a resync in the first place.
    Tombstoning must never delete: old rows stay physically present.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    engine._store.append_messages("s1", 1, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    engine._store.append_messages("s1", 1, [{"role": "user", "content": f"stale-{i}"} for i in range(5)])

    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    live = [{"role": "user", "content": c} for c in "abcdefghi"]
    engine2.on_turn_complete(live, turn_id=2)

    assert engine2._store.raw_row_count("s1") == 16, (
        "old rows (7) plus the fresh re-insert (9) must both still exist "
        "physically -- nothing was deleted, only hidden from the current view"
    )


def test_resync_does_not_fire_on_healthy_resume(tmp_path):
    """The common, non-inflated case must not pay a supersede+full-resync
    just because the verification machinery now exists.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine.on_turn_complete([{"role": "user", "content": c} for c in "abcde"], turn_id=1)
    assert engine._store.raw_row_count("s1") == 5

    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    engine2.on_turn_complete([{"role": "user", "content": c} for c in "abcdef"], turn_id=2)

    assert engine2._store.raw_row_count("s1") == 6, "must not tombstone+reinsert when the resume estimate was accurate"
    assert engine2._store.message_count("s1") == 6


def test_context_shows_no_duplicates_after_resync(tmp_path, _cleanup_repl):
    """End-to-end through the real REPL: the model-visible `context`
    variable must not contain the same message more than once after a
    resync (reproduced content deduped) -- AND must still include
    archive-only content the resync's re-append didn't reproduce (round
    18: this is the actual production-bug reproduction, through the real
    REPL rather than the store API directly).
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine._store.append_messages("s1", 1, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    engine._store.append_messages("s1", 1, [{"role": "user", "content": f"stale-{i}"} for i in range(5)])

    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    live = [{"role": "user", "content": c} for c in "abcdefghi"]
    engine2.on_turn_complete(live, turn_id=2)

    repl = PersistentREPL(db_path=engine2._store.db_path, session_id="s1", base_url="http://x/v1", model="m")
    _cleanup_repl.append(repl)
    r = repl.exec("contents = [m['content'] for m in context]\nprint(len(contents), len(set(contents)))")
    total, unique = r["stdout"].split()
    assert total == unique == "14", (
        "9 live messages + 5 preserved stale rows, all distinct -- no "
        "duplicates from 'a'/'b' being reproduced, and no archive-only "
        "content missing from what the REPL can see"
    )


def test_migration_adds_superseded_column_to_existing_db(tmp_path):
    """A ~/.hermes/rlm.db from before N2 has no superseded column at all --
    opening it must migrate cleanly, not crash, and preserve existing data.
    """
    import sqlite3
    db_path = str(tmp_path / "pre_n2.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE rlm_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO rlm_messages (session_id, turn_id, role, content, ts) "
        "VALUES ('old', 1, 'user', 'pre-migration', 1.0)"
    )
    conn.commit()
    conn.close()

    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(db_path)  # must not raise
    assert store.message_count("old") == 1
    store.close()

    # Idempotent: opening a second time (column already present) must also work.
    store2 = RLMStore(db_path)
    assert store2.message_count("old") == 1
    store2.close()


# ---------------------------------------------------------------------------
# Round-6: two paths that reset _persisted_count = 0 without going through
# _trigger_resync() -- _trigger_resync's own docstring claims every resync
# trigger is centralized through it; these were the counterexamples.
# ---------------------------------------------------------------------------

def test_verify_watermark_empty_transcript_branch_resyncs_cleanly(tmp_path):
    """engine.py's check_n<=0 branch in _verify_resume_watermark: fires
    when the live transcript is empty but the resume estimate is not.
    Harmless on THAT call (nothing to re-append yet), but without
    _trigger_resync() the cursor stays inflated and the NEXT real
    on_turn_complete would skip archiving real new content, thinking it
    was already covered.

    Round-18 rewrite: the original assertion here (message_count == 1)
    was actually pinning the OLD bug -- old1/old2 have nothing in common
    with new1, so a correct resync must leave them VISIBLE, not hide
    them. Renamed off "_tombstones" accordingly.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine._store.append_messages("s1", 1, [{"role": "user", "content": "old1"}, {"role": "user", "content": "old2"}])

    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    engine2.on_turn_complete([], turn_id=1)  # empty transcript -- hits the flagged branch
    engine2.on_turn_complete([{"role": "user", "content": "new1"}], turn_id=2)  # real content arrives

    assert engine2._store.message_count("s1") == 3, (
        "old1/old2 must stay visible (never reproduced by new1, so never "
        "reproduced-and-deduped) and new1 must be archived -- not old1/old2 "
        "silently disappearing, and not new1 silently skipped"
    )


def test_message_count_lookup_failure_resyncs_cleanly(tmp_path):
    """engine.py's on_session_start exception handler when message_count()
    itself raises: same counterexample, different trigger. See the
    empty-transcript-branch test above for why this asserts preservation,
    not tombstoning, post round-18.
    """
    from unittest import mock

    engine = _make_engine(tmp_path)
    engine._store.append_messages("s2", 1, [{"role": "user", "content": "old-b"}])

    with mock.patch.object(engine._store, "message_count", side_effect=RuntimeError("boom")):
        engine.on_session_start("s2")
    assert engine._resume_verified is True
    assert engine._persisted_count == 0

    engine.on_turn_complete([{"role": "user", "content": "new-b"}], turn_id=1)
    assert engine._store.message_count("s2") == 2, (
        "old-b must stay visible (not reproduced by new-b) and new-b must "
        "be archived"
    )


# ---------------------------------------------------------------------------
# Round-7 (L2/L3/L4)
# ---------------------------------------------------------------------------
#
# L2 (round 7, measured only): search_any/message_count/tail_content are
# all session_id-scoped and hit the (session_id, ...) composite indexes.
# The round-7 measurement used matching keywords only, which let
# search_any's ORDER BY id DESC LIMIT ? short-circuit as soon as it found
# enough rows -- the sub-millisecond numbers above are real, but only for
# the hit case. Correctly flagged in round-8 review: search_any is
# auto-recall's prefilter, called on every provider request, and a MISS
# (the common case -- most turns don't need dropped history) has no early
# exit, so it scanned every non-superseded row in the session. Re-measured
# with the miss case and realistic row sizes (2-8KB, not 100-byte
# synthetic rows): 50k rows/session, 8KB rows -> ~720ms/miss, synchronous
# on select_context's hot path, paid once per provider round trip. See
# round-8's FTS5 fix below and store.py's _init_fts()/search_any() for the
# actual implementation.

def test_query_spend_present_even_when_rlm_query_never_called(tmp_path, _cleanup_repl):
    """L3: root must be able to see recursion spend on every call, not
    just discover it after the fact when a limit is hit.
    """
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)
    r = repl.exec("print(1)")
    assert r.get("query_spend") == {"count": 0, "chars_in": 0, "chars_out": 0}


def test_query_call_limit_enforced_and_counts_failed_attempts(tmp_path, _cleanup_repl):
    """L3: the limit must fire after exactly max_query_calls attempts, with
    a clear error naming the limit -- not a silent truncation, and not
    dependent on a real model (base_url points at a port nothing listens
    on, so every attempt fails fast and deterministically; the limit check
    itself runs before the network call, so this still exercises it).
    """
    _, repl = _archive_and_repl(
        tmp_path, [{"role": "user", "content": "x"}],
        base_url="http://127.0.0.1:1/v1", max_query_calls=2,
    )
    _cleanup_repl.append(repl)
    r = repl.exec(
        "attempts = 0\n"
        "blocked = None\n"
        "for i in range(5):\n"
        "    attempts += 1\n"
        "    try:\n"
        "        rlm_query('x')\n"
        "    except RuntimeError as e:\n"
        "        blocked = str(e)\n"
        "        break\n"
        "    except Exception:\n"
        "        pass\n"
        "print(attempts, bool(blocked))\n"
        "print(blocked)\n"
    )
    assert r["error"] is None, r
    lines = r["stdout"].splitlines()
    attempts, was_blocked = lines[0].split()
    assert int(attempts) == 3, "2 real (failed) attempts, then blocked on the 3rd"
    assert was_blocked == "True"
    assert "limit reached (2" in lines[1]
    assert r["query_spend"]["count"] == 2, "blocked calls must not count against spend"


def test_query_call_limit_resets_per_exec_call(tmp_path, _cleanup_repl):
    """The budget is per rlm_repl call, not cumulative across the session."""
    _, repl = _archive_and_repl(
        tmp_path, [{"role": "user", "content": "x"}],
        base_url="http://127.0.0.1:1/v1", max_query_calls=1,
    )
    _cleanup_repl.append(repl)

    def one_attempt():
        return repl.exec(
            "try:\n    rlm_query('x')\nexcept RuntimeError as e:\n    print('BLOCKED')\nexcept Exception:\n    print('NETERR')"
        )

    r1 = one_attempt()
    assert r1["stdout"].strip() == "NETERR"  # the one allowed attempt, fails on the network
    r2 = one_attempt()
    assert r2["stdout"].strip() == "NETERR", "a fresh call must get a fresh budget, not stay blocked"


# L4 — schema must register even when the store failed to open
def test_schema_present_when_store_unavailable(tmp_path):
    engine = RLMContextEngine(config={"rlm": {"db_path": "/root/no-permission/rlm.db"}})
    assert engine._store is None
    schemas = engine.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "rlm_repl"


def test_handle_tool_call_reports_store_error_when_schema_present_but_store_down(tmp_path):
    import json as _json
    engine = RLMContextEngine(config={"rlm": {"db_path": "/root/no-permission/rlm.db"}})
    result = _json.loads(engine.handle_tool_call("rlm_repl", {"code": "print(1)"}))
    assert "RLM store unavailable" in result["error"]


# ---------------------------------------------------------------------------
# Round-8 (L2 revisited): search_any moved onto FTS5 to fix the miss-case
# scan flagged in review. See store.py's _init_fts()/_search_any_fts().
# ---------------------------------------------------------------------------

def test_search_any_uses_fts_and_matches_prefix(tmp_path):
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    assert store._fts_enabled, "this sqlite3 build has FTS5; test assumes it"
    store.append_messages("s1", 1, [{"role": "user", "content": "talking about bananas today"}])
    store.append_messages("s1", 1, [{"role": "user", "content": "nothing relevant here"}])

    hits = store.search_any("s1", ["banana"])  # prefix match against "bananas"
    assert len(hits) == 1
    assert "bananas" in hits[0]["content"]

    assert store.search_any("s1", ["zzznotfound"]) == []
    store.close()


def test_search_any_fts_respects_session_and_tombstone_scoping(tmp_path):
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("s1", 1, [{"role": "user", "content": "shared keyword apples"}])
    store.append_messages("s2", 1, [{"role": "user", "content": "shared keyword apples"}])

    assert len(store.search_any("s1", ["apples"])) == 1, "must not leak another session's rows"

    store.supersede_session("s1")
    assert store.search_any("s1", ["apples"]) == [], "tombstoned rows must not surface as matches"
    assert store.raw_row_count("s1") == 1, "tombstoning must not delete the row"
    store.close()


def test_search_any_fts_backfills_from_pre_fts_database(tmp_path):
    """A database created before this fix has rlm_messages (with the
    superseded column) but no rlm_search table -- opening it must index
    the existing rows, not just new ones going forward.
    """
    import sqlite3
    db_path = str(tmp_path / "pre_fts.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE rlm_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, turn_id INTEGER NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, ts REAL NOT NULL, superseded INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        "INSERT INTO rlm_messages (session_id, turn_id, role, content, ts) "
        "VALUES ('old', 1, 'user', 'a pre-existing archived apricot', 1.0)"
    )
    conn.commit()
    conn.close()

    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(db_path)
    hits = store.search_any("old", ["apricot"])
    assert len(hits) == 1, "backfilled row must be searchable, not just rows appended after upgrade"
    store.close()

    # Idempotent: reopening must not double-index (would surface as
    # duplicate/garbled results or a UNIQUE-ish blowup on rowid reuse).
    store2 = RLMStore(db_path)
    assert len(store2.search_any("old", ["apricot"])) == 1
    store2.close()


def test_search_any_falls_back_to_like_when_fts_disabled(tmp_path):
    """If sqlite3 lacks FTS5 (or CREATE VIRTUAL TABLE fails for any other
    reason), search_any must still work correctly -- just without the
    speedup. _fts_enabled forced False here to exercise that path directly
    rather than depending on the local sqlite3 build's compile flags.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store._fts_enabled = False
    store.append_messages("s1", 1, [{"role": "user", "content": "talking about bananas today"}])

    hits = store.search_any("s1", ["banana"])
    assert len(hits) == 1
    assert store.search_any("s1", ["zzznotfound"]) == []
    store.close()


# ---------------------------------------------------------------------------
# Round-9: retention. rlm.db has no clock/config of its own -- an archived
# session is only ever deleted once it's absent from state.db's `sessions`
# table (sweep_orphaned_sessions). No rlm.retention_days key exists or
# should exist; sessions.* config is reused verbatim.
# ---------------------------------------------------------------------------

def _make_state_db(tmp_path, session_ids):
    """A minimal state.db stand-in: just enough of the real `sessions`
    table shape (id is all sweep_orphaned_sessions/_existing_state_db_sessions
    ever reads) for the sweep to query against.
    """
    import sqlite3
    path = str(tmp_path / "state.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL)")
    conn.executemany(
        "INSERT INTO sessions (id, started_at) VALUES (?, 0)",
        [(sid,) for sid in session_ids],
    )
    conn.commit()
    conn.close()
    return path


def _remove_from_state_db(state_db_path, session_id):
    """Simulate a session that WAS registered and has since been pruned/
    removed from state.db -- as distinct from one that was never
    registered at all (round 20's whole distinction)."""
    import sqlite3
    conn = sqlite3.connect(state_db_path)
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def _add_to_state_db(state_db_path, session_id):
    """Insert into an ALREADY-CREATED state.db stand-in (see
    _make_state_db) -- for tests that need to add a session after the
    fact rather than seed it up front."""
    import sqlite3
    conn = sqlite3.connect(state_db_path)
    conn.execute("INSERT INTO sessions (id, started_at) VALUES (?, 0)", (session_id,))
    conn.commit()
    conn.close()


def test_sweep_deletes_orphaned_session_rows_and_fts_entries(tmp_path):
    """A session_id CONFIRMED PRESENT in state.db at some point and later
    absent is an orphan: its rows -- live AND tombstoned -- and its FTS5
    entries are all removed. This is a real DELETE, unlike
    supersede_session's tombstone -- by design, this is rlm.db's only
    deletion path.

    Round 20: two-phase fixture now, not one -- absence alone no longer
    authorises deletion (see test_sweep_never_deletes_a_session_never_seen_
    present below for that exact guard). This test must first let the
    sweep CONFIRM the session present, then remove it from state.db, to
    exercise the actual "was seen, now gone" deletion path rather than
    accidentally re-testing the old (buggy) "absent from the start" case.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    assert store._fts_enabled
    store.append_messages("orphan", 1, [{"role": "user", "content": "will be swept"}])
    store.append_messages("orphan", 2, [{"role": "user", "content": "also swept"}])
    store.supersede_session("orphan")  # tombstone one round, so both live+dead rows exist
    store.append_messages("orphan", 3, [{"role": "user", "content": "swept too"}])
    assert store.raw_row_count("orphan") == 3

    state_db = _make_state_db(tmp_path, ["orphan"])  # present -- gets confirmed-seen
    first = store.sweep_orphaned_sessions(state_db, min_interval_hours=0)
    assert first["sessions_pruned"] == 0, "still present -- must not be swept yet"
    assert store.raw_row_count("orphan") == 3

    _remove_from_state_db(state_db, "orphan")  # now genuinely released
    result = store.sweep_orphaned_sessions(state_db, min_interval_hours=0)

    assert result["sessions_pruned"] == 1
    assert result["rows_deleted"] == 3
    assert store.raw_row_count("orphan") == 0
    fts_rows = store._conn.execute(
        "SELECT COUNT(*) FROM rlm_search WHERE session_id = ?", ("orphan",)
    ).fetchone()[0]
    assert fts_rows == 0, "FTS mirror must be swept too, not just rlm_messages"
    store.close()


def test_sweep_preserves_session_present_in_state_db_even_fully_superseded(tmp_path):
    """Existing in state.db is the whole test -- a session with every row
    tombstoned must still be kept if state.db still knows about it.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("kept", 1, [{"role": "user", "content": "still tracked"}])
    store.supersede_session("kept")  # message_count("kept") == 0, but state.db has the row

    state_db = _make_state_db(tmp_path, ["kept"])
    result = store.sweep_orphaned_sessions(state_db, min_interval_hours=0)

    assert result["sessions_pruned"] == 0
    assert result["rows_deleted"] == 0
    assert store.raw_row_count("kept") == 1
    store.close()


def test_sweep_never_deletes_current_session_even_if_absent_from_state_db(tmp_path):
    """Guards the race between session start and state.db's own (fallible)
    create_session() call -- current_session_id is excluded regardless of
    what state.db shows.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("live", 1, [{"role": "user", "content": "just started"}])

    state_db = _make_state_db(tmp_path, [])  # state.db hasn't caught up yet
    result = store.sweep_orphaned_sessions(state_db, current_session_id="live", min_interval_hours=0)

    assert result["sessions_pruned"] == 0
    assert store.raw_row_count("live") == 1
    store.close()


def test_sweep_deletes_nothing_when_state_db_missing(tmp_path):
    """Fail-open: an unreadable/missing state.db must never be treated as
    'state.db has no sessions' -- that would delete everything. Nothing is
    deleted and the failure is reported, not swallowed silently.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("s1", 1, [{"role": "user", "content": "x"}])

    result = store.sweep_orphaned_sessions(
        str(tmp_path / "does-not-exist.db"), min_interval_hours=0
    )
    assert result["sessions_pruned"] == 0
    assert result["rows_deleted"] == 0
    assert "error" in result
    assert store.raw_row_count("s1") == 1
    store.close()


def test_sweep_no_op_when_no_session_missing(tmp_path):
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("s1", 1, [{"role": "user", "content": "x"}])
    state_db = _make_state_db(tmp_path, ["s1"])

    result = store.sweep_orphaned_sessions(state_db, min_interval_hours=0)
    assert result == {"skipped": False, "sessions_pruned": 0, "rows_deleted": 0, "vacuumed": False}
    assert store.raw_row_count("s1") == 1
    store.close()


def test_sweep_throttled_by_min_interval_hours(tmp_path):
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("orphan", 1, [{"role": "user", "content": "x"}])
    state_db = _make_state_db(tmp_path, ["orphan"])
    store.sweep_orphaned_sessions(state_db, min_interval_hours=0)  # confirm-seen pass
    _remove_from_state_db(state_db, "orphan")
    store.set_meta("last_orphan_sweep", "0")  # simulate time passing since the confirm pass

    first = store.sweep_orphaned_sessions(state_db, min_interval_hours=24)
    assert first["skipped"] is False
    assert first["sessions_pruned"] == 1

    store.append_messages("orphan2", 1, [{"role": "user", "content": "y"}])
    second = store.sweep_orphaned_sessions(state_db, min_interval_hours=24)
    assert second["skipped"] is True
    assert store.raw_row_count("orphan2") == 1, "throttled sweep must not run at all"
    store.close()


def test_sweep_vacuum_gated_by_deleted_rows_and_min_vacuum_interval(tmp_path):
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    state_db = _make_state_db(tmp_path, [])

    # Nothing to delete -> no vacuum, even though vacuum_after_prune=True.
    r1 = store.sweep_orphaned_sessions(state_db, min_interval_hours=0, vacuum_after_prune=True)
    assert r1["sessions_pruned"] == 0
    assert r1["vacuumed"] is False

    store.append_messages("orphan", 1, [{"role": "user", "content": "x"}])
    _add_to_state_db(state_db, "orphan")  # now present
    store.sweep_orphaned_sessions(state_db, min_interval_hours=0)  # confirm-seen pass
    _remove_from_state_db(state_db, "orphan")
    r2 = store.sweep_orphaned_sessions(
        state_db, min_interval_hours=0, vacuum_after_prune=True, min_vacuum_interval_days=30
    )
    assert r2["sessions_pruned"] == 1
    assert r2["vacuumed"] is True

    store.append_messages("orphan2", 1, [{"role": "user", "content": "y"}])
    _add_to_state_db(state_db, "orphan2")
    store.sweep_orphaned_sessions(state_db, min_interval_hours=0)  # confirm-seen pass
    _remove_from_state_db(state_db, "orphan2")
    r3 = store.sweep_orphaned_sessions(
        state_db, min_interval_hours=0, vacuum_after_prune=True, min_vacuum_interval_days=30
    )
    assert r3["sessions_pruned"] == 1
    assert r3["vacuumed"] is False, "min_vacuum_interval_days must throttle even with fresh deletes"
    store.close()


# ---------------------------------------------------------------------------
# Round-20: absence from state.db alone stopped being sufficient grounds
# for deletion -- it deleted the only surviving copy of a real ~1.4MB user
# conversation whose create_session() had silently failed under write
# contention. A session_id is now only ever a sweep candidate once it has
# been positively confirmed present in state.db at least once.
# ---------------------------------------------------------------------------

def test_sweep_never_deletes_a_session_never_seen_present(tmp_path):
    """THE invariant this round exists for: a session absent from
    state.db from the very first sweep it's ever checked in -- exactly
    the create_session()-silently-failed shape -- must never be deleted,
    no matter how many sweeps run or how much time passes. No age-based
    escape hatch either: run several sweeps, still not deleted.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("never-registered", 1, [{"role": "user", "content": "real conversation"}])
    state_db = _make_state_db(tmp_path, [])  # never present, not even once

    for _ in range(3):
        result = store.sweep_orphaned_sessions(state_db, min_interval_hours=0)
        assert result["sessions_pruned"] == 0
        assert store.raw_row_count("never-registered") == 1

    hits = store.search_any("never-registered", ["conversation"])
    assert len(hits) == 1, "content must still be fully intact and findable, not just uncounted"
    store.close()


def test_sweep_still_deletes_a_session_confirmed_then_genuinely_released(tmp_path):
    """The other half of the invariant: retention must keep working for
    the real case -- a session that WAS registered and later legitimately
    aged out of state.db (round 9's original design intent) is still
    swept, not permanently grandfathered in by having existed once.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("released", 1, [{"role": "user", "content": "old conversation"}])
    state_db = _make_state_db(tmp_path, ["released"])

    confirm = store.sweep_orphaned_sessions(state_db, min_interval_hours=0)
    assert confirm["sessions_pruned"] == 0, "present -- confirmed seen, not swept yet"

    _remove_from_state_db(state_db, "released")  # state.db's own retention released it
    result = store.sweep_orphaned_sessions(state_db, min_interval_hours=0)

    assert result["sessions_pruned"] == 1
    assert store.raw_row_count("released") == 0


def test_seen_sessions_record_cleaned_up_after_deletion(tmp_path):
    """rlm_seen_sessions must not grow unboundedly with dead entries --
    a swept session's seen-record is removed in the same pass, not left
    behind as permanent evidence for a session_id that no longer exists
    anywhere.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("s1", 1, [{"role": "user", "content": "x"}])
    state_db = _make_state_db(tmp_path, ["s1"])
    store.sweep_orphaned_sessions(state_db, min_interval_hours=0)
    assert store._filter_seen_sessions({"s1"}) == {"s1"}

    _remove_from_state_db(state_db, "s1")
    store.sweep_orphaned_sessions(state_db, min_interval_hours=0)

    assert store._filter_seen_sessions({"s1"}) == set(), (
        "the seen-record must be cleaned up once its session is actually deleted"
    )
    store.close()


# ---------------------------------------------------------------------------
# Round-11: production break. marker_role defaulted to 'system', but the
# marker lands mid-conversation (system + head + [marker] + tail) -- a
# role=='system' message anywhere but index 0 is a hard 400 on strict
# OpenAI-compatible chat templates ("System message must be at the
# beginning"), not a provider quirk (title_generator.py #48338). Needed
# > protect_first_n + protect_last_n (28 by default) messages before
# select_context drops anything -- _convo(60) clears that; every earlier
# round's fixtures didn't, which is why 10 rounds of tests missed this.
# ---------------------------------------------------------------------------

def _assert_no_midlist_system_role(messages):
    for i, m in enumerate(messages):
        if i > 0:
            assert m.get("role") != "system", (
                f"message at index {i} has role='system' -- strict chat "
                f"templates reject any system-role message that isn't first"
            )


def test_select_context_output_never_has_system_role_after_index_0(tmp_path):
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    selected = engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)
    assert selected is not None, "fixture must actually exercise the drop path"
    _assert_no_midlist_system_role(selected)


def test_select_context_auto_recall_marker_never_has_system_role(tmp_path):
    """The recall snippet rebuilds the marker at a second call site
    (marker = {"role": self._marker_role, ...}) -- same invariant, second
    construction site, must be covered independently of the plain-marker
    path above.
    """
    # auto_recall defaults off (round 13) -- must enable it explicitly to
    # actually exercise the recall-marker construction site this test is
    # named for, not just the plain-marker path already covered above.
    engine = _make_engine(tmp_path, auto_recall=True)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    engine._archive_new(convo, turn_id=1)
    # Force the recall branch: real keywords, and a store hit guaranteed
    # by seeding an archived row containing them.
    engine._store.append_messages("s1", 1, [{"role": "user", "content": "xylophone marmalade"}])
    selected = engine.select_context(
        convo, conversation_messages=convo,
        incoming_message={"role": "user", "content": "tell me about xylophone marmalade again"},
        budget_tokens=131072,
    )
    assert selected is not None
    _assert_no_midlist_system_role(selected)


def test_compress_output_never_has_system_role_after_index_0(tmp_path):
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    compressed = engine.compress(convo, force=True)
    assert len(compressed) < len(convo), "fixture must actually exercise the drop path"
    _assert_no_midlist_system_role(compressed)


# ---------------------------------------------------------------------------
# Round-16: (1) accurate automatic-compaction status, not borrowed
# summarization language; (2) compress()'s log no longer self-accuses
# select_context() of a failure it is structurally incapable of
# preventing (it bounds requests, not the transcript preflight measures);
# (3) compress() must reset _tail_boundary, or the request immediately
# after a safety-net trim is nearly empty -- reproduced live by the
# reviewing agent against the round-15-only code (200 msgs -> boundary
# 160 -> compress() to 30 msgs -> next select_context() tail collapsed to
# 3 messages via min(boundary, n) clamping a now-meaningless index).
# ---------------------------------------------------------------------------

def test_automatic_compaction_status_is_accurate_not_summarization_language(tmp_path):
    engine = _make_engine(tmp_path)
    for phase in ("preflight", "compress"):
        msg = engine.get_automatic_compaction_status_message(
            phase=phase, default_message="should not appear", approx_tokens=12345,
        )
        assert msg is not None, "silence was considered and rejected -- see the docstring"
        # Deliberately allows "not summarizing" (an accurate denial) --
        # what must never appear is the OLD claim that summarization is
        # what's happening.
        assert "summarizing earlier conversation" not in msg.lower(), (
            "RLM never summarizes -- must not claim to"
        )
        assert "archiv" in msg.lower() or "lost" in msg.lower(), (
            "must say something true about what actually happens (archived/not lost)"
        )
    # Deliberate choice, not the silence path -- must not be flipped off.
    assert engine.emit_automatic_compaction_status is True


def test_compress_still_archives_before_trimming(tmp_path):
    """The reworded log message must not have accidentally dropped the
    actual archiving behavior -- compress() must still persist everything
    before trimming the live list.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    compressed = engine.compress(convo, force=True)
    assert len(compressed) < len(convo)
    assert engine._store.message_count("s1") == len(convo), (
        "every message must be archived BEFORE compress() trims the live list"
    )


def test_compress_resets_boundary_so_the_next_request_still_has_a_real_tail(tmp_path):
    """End-to-end reproduction of the reviewer's exact finding: a stale
    _tail_boundary surviving compress()'s transcript rewrite doesn't
    crash (the existing min(boundary, n) clamp prevents that) but silently
    collapses the next request's tail to almost nothing -- precisely when
    the conversation is longest and the model needs recent context most.
    'Consistent' here means more than 'boundary is a valid index': it
    means the FIRST select_context() call against the post-compress
    transcript still returns a tail sized like a normal one (>=
    protect_last_n) and containing the transcript's actual latest
    message -- not merely that nothing crashes.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    # Advance the boundary well past 0 first, matching the reviewer's repro.
    convo = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(200)
    ]
    engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)
    assert engine._tail_boundary > 0, "fixture must actually advance the boundary first"

    compressed = engine.compress(convo, force=True)
    assert engine._tail_boundary == 0, (
        "compress() must reset the boundary -- it just rewrote the transcript "
        "the old boundary was an index into"
    )

    # Grow the (now much shorter) transcript again, past drop_chunk_size,
    # so this exercises a REAL drop after compress() -- not the case where
    # the compressed transcript alone is still small enough that nothing
    # needs dropping.
    grown = compressed + [{"role": "user", "content": f"new{i}"} for i in range(40)]
    selected = engine.select_context(grown, conversation_messages=grown, budget_tokens=131072)
    assert selected is not None, "fixture must actually exercise the drop path post-compress"

    marker_idx = next(i for i, m in enumerate(selected) if "[RLM:" in (m.get("content") or ""))
    tail = selected[marker_idx + 1 :]
    assert len(tail) >= engine.protect_last_n, (
        "the tail right after a compress() reset must be a REAL tail, not "
        "collapsed by a stale boundary clamped against the new, shorter transcript"
    )
    assert grown[-1] in tail, "the transcript's actual latest message must be reachable"


def test_persisted_count_resyncs_consistently_after_compress_shrinks_transcript(tmp_path):
    """compress() shrinks the live list; the NEXT _archive_new() call (via
    select_context's M4 archiving) must detect that shrink and resync
    rather than silently under- or over-counting what's archived. This is
    the existing shrink-guard (_trigger_resync, N2/M6) -- confirming it's
    the intended handler for compress()'s shrink specifically, not an
    accidental side effect.

    'Consistent' here means: after the resync, _persisted_count equals
    the length of the (post-compress) live transcript select_context was
    just given -- not the pre-compress count, and not left stale.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)
    assert engine._persisted_count == len(convo)

    compressed = engine.compress(convo, force=True)
    assert engine._persisted_count == len(convo), (
        "compress() itself archives the FULL pre-trim transcript before "
        "shrinking it -- persisted_count reflects that until the next call"
    )

    # The next request carries the shrunk transcript -- _archive_new must
    # notice len(messages) < _persisted_count and resync, not skip
    # archiving the "new" (actually just repositioned) messages.
    engine.select_context(compressed, conversation_messages=compressed, budget_tokens=131072)
    assert engine._persisted_count == len(compressed), (
        "must resync to the POST-compress transcript length, not stay "
        "stuck on the stale pre-compress count"
    )


def test_auto_recall_cache_not_incorrectly_reused_after_compress_shrink(tmp_path):
    """The round-14 drift guard compares _persisted_count against a
    cached value -- after compress()'s shrink-triggered resync,
    _persisted_count drops (often below the cached value, making drift
    negative). Confirms this is harmless, not merely unconsidered: a
    negative/small drift still passes the '<= protect_last_n' check, but
    that's fine here specifically because a resync doesn't evict anything
    a recall would have needed to find -- it just re-establishes the same
    content under a fresh archive-cursor position.
    """
    from unittest import mock

    engine = _make_engine(tmp_path, auto_recall=True)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    _seed_recall_match(engine, "s1", "xylophone")
    incoming = {"role": "user", "content": "tell me about the xylophone marmalade situation"}

    with mock.patch(
        "agent.auxiliary_client.call_llm", return_value=_fake_llm_response("digested answer")
    ) as mocked:
        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 1
        cached_persisted_count = engine._auto_recall_cache[2]

        compressed = engine.compress(convo, force=True)
        # compress() itself doesn't call _cached_auto_recall_snippet, so
        # the cache entry (and its stale, larger persisted_count) survives
        # compress() untouched -- the resync happens on the NEXT call.
        assert engine._auto_recall_cache[2] == cached_persisted_count

        # Must not raise. Per the reviewer's confirmed reasoning: the
        # resync this triggers makes _persisted_count drop (often below
        # cached_persisted_count, so drift goes negative) -- harmless,
        # not unconsidered, because a resync doesn't evict anything a
        # recall would have needed: it re-archives the same content under
        # a fresh cursor, it doesn't lose rows. Negative drift still
        # passes "<= protect_last_n", so the (still-valid) cached answer
        # is served rather than recomputed -- confirmed here as the
        # actual, intended behavior, not merely "didn't crash".
        engine.select_context(
            compressed, conversation_messages=compressed, incoming_message=incoming, budget_tokens=131072
        )
        assert mocked.call_count == 1, (
            "a resync-induced negative drift must not force an unnecessary "
            "recompute -- the archived content itself didn't change"
        )


def test_marker_role_configured_as_system_is_coerced_to_user(tmp_path, caplog):
    """The dangerous config value itself must be refused, not just papered
    over downstream -- marker_role: system in config.yaml must not be
    able to reintroduce this outage.
    """
    import logging
    with caplog.at_level(logging.WARNING, logger="plugins.context_engine.rlm.engine"):
        engine = _make_engine(tmp_path, marker_role="system")
    assert engine._marker_role == "user"
    assert any("marker_role" in r.message for r in caplog.records), (
        "coercion must be logged, not silent"
    )

    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    convo = _convo(60)
    selected = engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)
    assert selected is not None
    _assert_no_midlist_system_role(selected)


def test_marker_role_user_configured_explicitly_is_respected(tmp_path):
    """Coercion must be specific to 'system', not a blanket override --
    a deliberately configured non-default-but-safe role still applies.
    """
    engine = _make_engine(tmp_path, marker_role="user")
    assert engine._marker_role == "user"


def test_enforce_system_message_position_does_not_mutate_original_dict(tmp_path):
    """The fix copies the offending message rather than mutating it in
    place -- these dicts are shared with the live conversation/archive,
    so an in-place role flip would corrupt state well beyond this list.
    """
    from plugins.context_engine.rlm.engine import _enforce_system_message_position

    offender = {"role": "system", "content": "should never happen"}
    original = dict(offender)
    fixed = _enforce_system_message_position([{"role": "system", "content": "ok"}, offender])

    assert fixed[1]["role"] == "user"
    assert offender == original, "the original message dict must be untouched"


# ---------------------------------------------------------------------------
# Round-13: auto_recall defaults off (round-1's "forced recovery" case was
# strong when rlm_repl's voluntary path was unreliable -- H1's timeout bug
# and repl.py's context-goes-stale-after-first-call bug, both fixed since).
# Stays available, opt-in, and memoized per-turn: select_context() runs
# once per provider request (M4's own premise) with the SAME
# incoming_message all turn, so an unmemoized auto-recall paid for the
# same search + digest sub-call once per request.
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _fake_llm_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _seed_recall_match(engine, session_id, keyword):
    # Large enough (several rows, well past auto_recall_digest_threshold_tokens
    # default 400) to force the actual digest sub-call, not the inline-raw
    # short-circuit.
    body = f"{keyword} filler word " * 40
    rows = [{"role": "user", "content": f"{keyword} archived detail {i}: {body}"} for i in range(4)]
    engine._store.append_messages(session_id, 1, rows)


def test_auto_recall_default_is_off(tmp_path):
    engine = _make_engine(tmp_path)
    assert engine._auto_recall is False


def test_auto_recall_can_still_be_enabled_explicitly(tmp_path):
    engine = _make_engine(tmp_path, auto_recall=True)
    assert engine._auto_recall is True


def test_auto_recall_digest_memoized_within_one_turn(tmp_path):
    """N requests within one turn (same incoming_message) must cost
    exactly one digest sub-call, not N -- the actual production bug: a
    10-tool-call turn was paying for up to 10-11 identical blocking
    call_llm() calls on the request path.
    """
    from unittest import mock

    engine = _make_engine(tmp_path, auto_recall=True)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    engine._archive_new(convo, turn_id=1)
    _seed_recall_match(engine, "s1", "xylophone")
    incoming = {"role": "user", "content": "tell me about the xylophone marmalade situation"}

    with mock.patch(
        "agent.auxiliary_client.call_llm", return_value=_fake_llm_response("digested answer")
    ) as mocked:
        for _ in range(5):
            selected = engine.select_context(
                convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072
            )
            assert selected is not None
        assert mocked.call_count == 1, "5 requests in one turn must cost exactly 1 digest call"


def test_auto_recall_cache_cleared_by_on_turn_complete(tmp_path):
    from unittest import mock

    engine = _make_engine(tmp_path, auto_recall=True)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    engine._archive_new(convo, turn_id=1)
    _seed_recall_match(engine, "s1", "xylophone")
    incoming = {"role": "user", "content": "tell me about the xylophone marmalade situation"}

    with mock.patch(
        "agent.auxiliary_client.call_llm", return_value=_fake_llm_response("digested answer")
    ) as mocked:
        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 1

        engine.on_turn_complete(convo, turn_id=1)

        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 2, (
            "next turn (even with the same question text) must recompute, "
            "not reuse the prior turn's cached answer"
        )


def test_auto_recall_cache_not_served_to_a_different_question_same_turn(tmp_path):
    from unittest import mock

    engine = _make_engine(tmp_path, auto_recall=True)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    engine._archive_new(convo, turn_id=1)
    _seed_recall_match(engine, "s1", "xylophone")
    _seed_recall_match(engine, "s1", "marmalade")

    q1 = {"role": "user", "content": "tell me about the xylophone situation please"}
    q2 = {"role": "user", "content": "tell me about the marmalade situation please"}

    with mock.patch(
        "agent.auxiliary_client.call_llm", return_value=_fake_llm_response("digested answer")
    ) as mocked:
        engine.select_context(convo, conversation_messages=convo, incoming_message=q1, budget_tokens=131072)
        engine.select_context(convo, conversation_messages=convo, incoming_message=q2, budget_tokens=131072)
        assert mocked.call_count == 2, (
            "a different question within the same turn must not be served "
            "the first question's cached answer"
        )


def test_auto_recall_negative_result_cached_too(tmp_path):
    """The common outcome ('nothing relevant') must not re-run the search
    every request either -- not just the (more expensive) digest.
    """
    from unittest import mock

    engine = _make_engine(tmp_path, auto_recall=True)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    engine._archive_new(convo, turn_id=1)
    incoming = {"role": "user", "content": "something entirely unrelated and unmatched"}

    with mock.patch.object(engine._store, "search_any", wraps=engine._store.search_any) as spy:
        for _ in range(4):
            selected = engine.select_context(
                convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072
            )
            assert selected is not None
        assert spy.call_count == 1, "a cached negative must not re-run search_any on every request"


def test_auto_recall_cache_does_not_leak_across_sessions(tmp_path):
    from unittest import mock

    engine = _make_engine(tmp_path, auto_recall=True)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    convo = _convo(60)
    engine._archive_new(convo, turn_id=1)
    _seed_recall_match(engine, "s1", "xylophone")
    incoming = {"role": "user", "content": "tell me about the xylophone situation please"}

    with mock.patch(
        "agent.auxiliary_client.call_llm", return_value=_fake_llm_response("digested answer")
    ) as mocked:
        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 1

        # Same engine instance, new session -- reused across /new, not
        # reconstructed (see on_session_start's own docstring).
        engine.on_session_start("s2")
        engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
        engine._archive_new(convo, turn_id=1)
        _seed_recall_match(engine, "s2", "xylophone")

        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 2, (
            "a new session must not inherit the previous session's cached answer, "
            "even for byte-identical question text"
        )


# ---------------------------------------------------------------------------
# Round-14: the cache must invalidate on its own when mid-turn archive
# growth (M4) exceeds protect_last_n, restoring exactly the uncached
# behavior in the one case round 13 left open (content rolling out of the
# live tail before the turn ends). Cheap check: _persisted_count drift,
# no DB call.
# ---------------------------------------------------------------------------

def test_auto_recall_cache_invalidated_by_archive_drift_past_protect_last_n(tmp_path):
    from unittest import mock

    engine = _make_engine(tmp_path, auto_recall=True, protect_first_n=1, protect_last_n=5)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(30)
    engine._archive_new(convo, turn_id=1)
    _seed_recall_match(engine, "s1", "xylophone")
    incoming = {"role": "user", "content": "tell me about the xylophone marmalade situation"}

    with mock.patch(
        "agent.auxiliary_client.call_llm", return_value=_fake_llm_response("digested answer")
    ) as mocked:
        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 1

        # Same turn (no on_turn_complete) -- but the archive grows past
        # protect_last_n's worth of new messages, simulating enough
        # mid-turn tool round trips (protect_last_n=5 here) that some
        # content archived before this point has rolled out of the live
        # tail.
        convo = convo + [{"role": "user", "content": f"extra{i}"} for i in range(8)]
        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 2, "drift past protect_last_n within one turn must force a recompute"

        # No further growth -> the freshly-recomputed entry is reused.
        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 2, "without further drift, the new cache entry must still be served"


def test_auto_recall_cache_survives_growth_under_the_threshold(tmp_path):
    """The common case round 13 optimized for -- a turn that adds a
    handful of messages and never trips the drift threshold -- must keep
    the full saving, not recompute on every small archive change.
    """
    from unittest import mock

    engine = _make_engine(tmp_path, auto_recall=True, protect_first_n=1, protect_last_n=5)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(30)
    engine._archive_new(convo, turn_id=1)
    _seed_recall_match(engine, "s1", "xylophone")
    incoming = {"role": "user", "content": "tell me about the xylophone marmalade situation"}

    with mock.patch(
        "agent.auxiliary_client.call_llm", return_value=_fake_llm_response("digested answer")
    ) as mocked:
        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 1

        convo = convo + [{"role": "user", "content": "extra1"}, {"role": "user", "content": "extra2"}]
        engine.select_context(convo, conversation_messages=convo, incoming_message=incoming, budget_tokens=131072)
        assert mocked.call_count == 1, "growth under protect_last_n must not force a premature recompute"


# ---------------------------------------------------------------------------
# Round-15: quantized tail boundary for prefix-cache stability. Real
# TabbyAPI turn logs (592 turns) showed RLM turns averaging 88% UNCACHED
# tokens and ~10x the prefill time of non-RLM turns on the same box --
# the old exact-N sliding tail window shifted by ~1 message every
# provider request, breaking a strict-prefix-matching KV cache almost
# every time.
# ---------------------------------------------------------------------------

def _serialize(messages):
    return [json.dumps(m, sort_keys=True) for m in messages]


def test_select_context_output_is_a_pure_append_when_boundary_unchanged(tmp_path):
    """The actual point of round 15: consecutive requests within the same
    chunk window must produce byte-identical PREFIXES, not just similar
    lengths -- out2 must equal out1 plus newly-appended messages, nothing
    reordered or rewritten in between.
    """
    engine = _make_engine(tmp_path, protect_first_n=2, protect_last_n=5, drop_chunk_size=20)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(30)
    out1 = engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)
    assert out1 is not None

    # One more provider round trip within the same turn -- a couple new
    # messages, nowhere near drop_chunk_size=20.
    convo2 = convo + [
        {"role": "assistant", "content": "calling a tool"},
        {"role": "user", "content": "tool result here"},
    ]
    out2 = engine.select_context(convo2, conversation_messages=convo2, budget_tokens=131072)
    assert out2 is not None

    s1, s2 = _serialize(out1), _serialize(out2)
    assert len(s2) > len(s1), "the fixture must actually grow between calls"
    assert s2[: len(s1)] == s1, (
        "out2 must be out1 with new messages appended -- any reordering or "
        "rewriting of the shared prefix defeats prefix caching entirely"
    )


def test_select_context_boundary_advances_past_chunk_size(tmp_path):
    """Growth that crosses drop_chunk_size worth of new messages must
    advance the boundary -- protect_last_n is a minimum, not a ceiling,
    so the tail is allowed to grow for a while, but not indefinitely.
    """
    engine = _make_engine(tmp_path, protect_first_n=2, protect_last_n=5, drop_chunk_size=10)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(30)
    engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)
    boundary_before = engine._tail_boundary

    convo2 = convo + [{"role": "user", "content": f"extra{i}"} for i in range(15)]  # > drop_chunk_size
    out2 = engine.select_context(convo2, conversation_messages=convo2, budget_tokens=131072)
    assert out2 is not None
    assert engine._tail_boundary > boundary_before, (
        "growth past drop_chunk_size must advance the boundary, not let "
        "the tail grow forever"
    )
    # protect_last_n is a floor: the tail must never end up shorter than it,
    # even right after an advance.
    non_system = [m for m in convo2 if m.get("role") != "system"]
    tail_len = len(non_system) - engine._tail_boundary
    assert tail_len >= engine.protect_last_n


def test_select_context_token_cap_forces_early_boundary_advance(tmp_path):
    """Hitting the token cap is itself a legitimate reason to advance the
    boundary early (per the spec) -- not just a one-off per-request trim
    that would otherwise re-trim (and re-shift) the same oversized head
    on every subsequent request.
    """
    engine = _make_engine(tmp_path, protect_first_n=1, protect_last_n=10, drop_chunk_size=20)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    huge = "x " * 2000  # ~500 tokens/message at the ~4-char/token estimate
    convo = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"{huge} {i}"} for i in range(15)
    ]
    # Tiny budget: protect_last_n=10 candidates can't possibly all fit.
    out = engine.select_context(convo, conversation_messages=convo, budget_tokens=600)
    assert out is not None

    non_system = [m for m in convo if m.get("role") != "system"]
    assert engine._tail_boundary > len(non_system) - engine.protect_last_n, (
        "the cap-forced trim must permanently advance the boundary past "
        "where an untrimmed quantized boundary would have landed"
    )

    # Token budget must still actually bound what's sent, not just move
    # the boundary marker around.
    tail_tokens = sum(estimate_tokens_rough(_content_as_text(m)) for m in out if m is not out[0])
    assert tail_tokens < 20000  # generous upper bound -- must not be anywhere near unbounded


def test_bound_tail_tokens_still_caps_an_oversized_slice(tmp_path):
    """Direct unit coverage of the token-cap pass in isolation from the
    boundary-quantization pass -- the budget must still actually bound
    the tail regardless of which messages were selected to be in it.
    """
    engine = _make_engine(tmp_path)
    huge = "x " * 5000
    candidates = [{"role": "user", "content": f"{huge} {i}"} for i in range(5)]

    capped = engine._bound_tail_tokens(candidates, budget_tokens=1000)
    assert len(capped) < len(candidates), "an oversized slice must actually be trimmed"

    total_tokens = sum(estimate_tokens_rough(_content_as_text(m)) for m in capped)
    token_cap = int(1000 * engine._tail_token_fraction)
    assert total_tokens <= token_cap * 1.2  # slack: char-based token estimate, not exact


# ---------------------------------------------------------------------------
# Round-17: measured 5,458 real production turns -- 97 carried the drop
# marker, 0 ever led to an rlm_repl call, despite a manually-pointed real
# session proving the model uses the tool competently once told to. Not a
# capability gap -- a discovery gap: the old marker made checking
# conditional on the model noticing it lacked something. Reworded
# directive; still constant text (round 15's cache-stability invariant).
# ---------------------------------------------------------------------------

def test_dropped_marker_is_directive_not_merely_informational(tmp_path):
    engine = _make_engine(tmp_path)
    marker = engine._dropped_marker()
    content = marker["content"].lower()
    assert "call rlm_repl" in content or "call rlm_repl first" in content, (
        "must instruct the model to call the tool, not merely note that it exists"
    )
    assert "if you need" not in content, (
        "must not make checking conditional on the model first judging it needs "
        "something -- that judgment call is precisely what production measured "
        "the model failing to make (97 marker-bearing turns, 0 rlm_repl calls)"
    )


# ---------------------------------------------------------------------------
# Round-18 URGENT: supersede_session()'s blanket tombstone, used by every
# resync, buried archive-only content instead of only deduping reproduced
# content. Production impact: 152 tombstoned rows in one real session, 88
# unique nowhere else, one of them a specific fact rlm_repl reported as
# absent -- the model then fabricated a substitute in a user deliverable.
# This is the invariant round 9 believed it had (its own docstring: a
# plain DELETE would "drop archive-only history, which is exactly the
# situation that triggers a resync in the first place") and did not:
# tombstoning avoided deleting that history, then hid it just as
# completely. Reproduced here exactly: archive content, shrink the live
# transcript via compress() (the real mechanism that mutates it), trigger
# a resync, assert the now-archive-only content is STILL visible -- via
# search_any AND the real REPL's history()/context, not just row counts.
# ---------------------------------------------------------------------------

def test_resync_after_compress_preserves_dropped_only_content(tmp_path, _cleanup_repl):
    """The exact production sequence: a long session gets compress()'d
    (round 16 -- this really does shrink the live transcript, not just
    the outgoing request), the next _archive_new() sees a shorter
    transcript than what's archived and triggers a resync. Content that
    existed ONLY in the archive before compress() -- the genuinely
    dropped middle -- must survive that resync, findable both by the
    engine's own search and by the model's real retrieval tool.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    convo = _convo(60)
    engine.select_context(convo, conversation_messages=convo, budget_tokens=131072)  # archives everything
    assert engine._store.message_count("s1") == len(convo)

    compressed = engine.compress(convo, force=True)
    assert len(compressed) < len(convo), "fixture must actually shrink the transcript"
    dropped_content = {
        m["content"] for m in convo
        if m.get("content") not in {c.get("content") for c in compressed}
    }
    assert dropped_content, "fixture must actually have dropped-only content to test"

    # The next real request carries the shrunk transcript -- triggers the
    # shrink-guard -> resync path this round-18 fix touches.
    engine.select_context(compressed, conversation_messages=compressed, budget_tokens=131072)

    sample_dropped = next(iter(dropped_content))
    keyword = sample_dropped.split()[0]  # e.g. "m17" from "m17" -- _convo's own content shape
    hits = engine._store.search_any("s1", [keyword])
    assert any(sample_dropped in h["content"] for h in hits), (
        "dropped-only content must still be findable via search_any -- "
        "not buried by the resync that just ran"
    )

    repl = PersistentREPL(db_path=engine._store.db_path, session_id="s1", base_url="http://x/v1", model="m")
    _cleanup_repl.append(repl)
    r = repl.exec(
        f"matches = [m for m in context if m['content'] == {sample_dropped!r}]\n"
        "print(len(matches))"
    )
    assert r["stdout"].strip() == "1", (
        "the REPL's own context/history() -- what the model actually uses "
        "via rlm_repl -- must also still see the dropped-only content"
    )


# ---------------------------------------------------------------------------
# Round-18 items 1/2: prune_tool_results_only(). RLM's tail is purely
# positional -- traced through real turn logs, retrieved rlm_repl results
# (60.1 KB) got fully evicted ~3.5 minutes after being fetched, displaced
# by raw web-search payloads that grew from 5.7 KB to 115.9 KB over the
# same window. Policy: shrink raw tool payloads by kind, before position
# -- never rlm_repl's own results, regardless of position.
# ---------------------------------------------------------------------------

def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _assistant_tool_call(call_id, tool_name):
    return {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "function": {"name": tool_name}}],
    }


def test_prune_tool_results_only_shrinks_payloads_and_reports_correct_count(tmp_path):
    engine = _make_engine(tmp_path, protect_first_n=1, protect_last_n=2)
    engine.on_session_start("s1")
    huge = "x" * 5000

    messages = [{"role": "system", "content": "sys"}]
    for i in range(5):
        messages.append({"role": "user", "content": f"search for topic {i}"})
        messages.append(_assistant_tool_call(f"call{i}", "web_search"))
        messages.append(_tool_result(f"call{i}", huge))
    # Protected tail: the last 2 messages must survive untouched even
    # though they'd otherwise qualify.
    before_tail = messages[-2:]

    pruned, count = engine.prune_tool_results_only(messages)
    assert pruned is not messages, "must return a new list object, not mutate in place"
    assert count == 4, "5 web_search results, 1 protected by the tail -> 4 pruned"

    pruned_tool_msgs = [m for m in pruned if m.get("role") == "tool"]
    shrunk = [m for m in pruned_tool_msgs if len(m["content"]) < len(huge)]
    assert len(shrunk) == 4
    for m in shrunk:
        assert "rlm_repl" in m["content"], "placeholder must point at the recovery path"
    assert pruned[-2:] == before_tail, "the protected tail must be untouched, not just shorter"


def test_prune_tool_results_only_never_touches_rlm_repl_results(tmp_path):
    """The actual point: raw tool output shrinks, rlm_repl's own results
    never do, regardless of position -- so a raw payload can no longer
    positionally outrank a retrieval result the model already paid for.
    """
    engine = _make_engine(tmp_path, protect_first_n=1, protect_last_n=1)
    engine.on_session_start("s1")
    huge = "y" * 5000

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look something up in history"},
        _assistant_tool_call("repl1", "rlm_repl"),
        _tool_result("repl1", huge),  # rlm_repl's own result -- must survive whole
        {"role": "user", "content": "now search the web"},
        _assistant_tool_call("web1", "web_search"),
        _tool_result("web1", huge),  # raw payload -- eligible to shrink
        {"role": "user", "content": "final unrelated turn"},  # keeps web1 out of the tail
    ]

    pruned, count = engine.prune_tool_results_only(messages)
    assert count == 1, "only the web_search result qualifies -- rlm_repl's is exempt"

    repl_result = next(m for m in pruned if m.get("tool_call_id") == "repl1")
    web_result = next(m for m in pruned if m.get("tool_call_id") == "web1")
    assert repl_result["content"] == huge, "rlm_repl's own result must be untouched, in full"
    assert len(web_result["content"]) < len(huge), "the raw web payload must have been shrunk"


def test_prune_tool_results_only_loses_nothing_from_the_archive(tmp_path):
    """Pruning must never be the operation that makes content
    unrecoverable -- everything pruned must already be archived, in full,
    before the placeholder replaces it in the live transcript.
    """
    engine = _make_engine(tmp_path, protect_first_n=1, protect_last_n=1)
    engine.on_session_start("s1")
    huge = "z" * 5000

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "search"},
        _assistant_tool_call("web1", "web_search"),
        _tool_result("web1", huge),
        {"role": "user", "content": "final unrelated turn"},
    ]
    pruned, count = engine.prune_tool_results_only(messages)
    assert count == 1

    hits = engine._store.search_any("s1", ["z" * 4])
    assert any(huge in h["content"] for h in hits), (
        "the full, unpruned content must be findable in the archive after pruning"
    )


def test_prune_tool_results_only_noop_below_threshold(tmp_path):
    engine = _make_engine(tmp_path, protect_first_n=1, protect_last_n=1, prune_min_result_chars=2000)
    engine.on_session_start("s1")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "search"},
        _assistant_tool_call("web1", "web_search"),
        _tool_result("web1", "short result"),
        {"role": "user", "content": "final unrelated turn"},
    ]
    pruned, count = engine.prune_tool_results_only(messages)
    assert count == 0
    assert pruned is messages, "no-op contract: must hand back the SAME object when nothing qualifies"


# ---------------------------------------------------------------------------
# Round-21 item 1: rotation-aware archiving. A gateway restart mints a
# fresh session_id but restores the same conversation; archiving used to
# treat that as brand new and re-archive the whole restored transcript
# from zero -- traced as the upstream cause of the duplication N2's
# tombstoning was invented to clean up, never traced to its source until
# now. One real pair shared 59 of 61 distinct messages, one genuinely new.
# ---------------------------------------------------------------------------

def test_detect_rotation_predecessor_finds_a_real_continuation(tmp_path):
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    old = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"old msg {i}"} for i in range(10)]
    store.append_messages("session-A", 1, old)

    # session-B's opening transcript = session-A's full content restored,
    # plus one genuinely new message after the restart.
    restored = old + [{"role": "user", "content": "one new message after restart"}]
    predecessor = store.detect_rotation_predecessor("session-B", restored)
    assert predecessor == "session-A"
    store.close()


def test_detect_rotation_predecessor_none_when_no_match(tmp_path):
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("session-A", 1, [{"role": "user", "content": f"unrelated {i}"} for i in range(10)])

    genuinely_new = [{"role": "user", "content": f"brand new topic {i}"} for i in range(5)]
    assert store.detect_rotation_predecessor("session-B", genuinely_new) is None
    store.close()


def test_detect_rotation_predecessor_none_when_ambiguous(tmp_path):
    """Two different sessions coincidentally share the same last message
    -- must not guess, a wrong link is worse than a missed one.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    shared_tail = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    store.append_messages("session-A", 1, shared_tail)
    store.append_messages("session-C", 1, shared_tail)  # same content, different session

    restored = shared_tail + [{"role": "user", "content": "new"}]
    assert store.detect_rotation_predecessor("session-B", restored) is None
    store.close()


def test_detect_rotation_predecessor_none_for_trivial_opening(tmp_path):
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    store.append_messages("session-A", 1, [{"role": "user", "content": "hi"}])
    assert store.detect_rotation_predecessor("session-B", [{"role": "user", "content": "hi"}]) is None
    store.close()


def test_archive_new_links_rotation_and_archives_only_the_new_tail(tmp_path):
    """End-to-end through the engine: session A's content must not be
    re-archived under session B -- only the genuinely new message.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("session-A")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    old = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"old msg {i}"} for i in range(10)]
    engine.on_turn_complete(old, turn_id=1)
    assert engine._store.message_count("session-A") == 10

    engine.on_session_start("session-B")  # same engine instance, reused across /new-like transitions
    restored = old + [{"role": "user", "content": "one new message after restart"}]
    engine.on_turn_complete(restored, turn_id=1)

    assert engine._store.message_count("session-B") == 1, (
        "only the genuinely new message should archive under the new id -- "
        "the restored 10 already exist under session-A"
    )
    assert engine._store.resolve_rotation_chain("session-B") == ["session-B", "session-A"]


def test_rotation_does_not_link_a_genuinely_new_session(tmp_path):
    engine = _make_engine(tmp_path)
    engine.on_session_start("session-A")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine.on_turn_complete([{"role": "user", "content": f"old {i}"} for i in range(10)], turn_id=1)

    engine.on_session_start("session-C")
    genuinely_new = [{"role": "user", "content": f"totally unrelated {i}"} for i in range(5)]
    engine.on_turn_complete(genuinely_new, turn_id=1)

    assert engine._store.message_count("session-C") == 5, "must archive normally, nothing to link"
    assert engine._store.resolve_rotation_chain("session-C") == ["session-C"]


def test_repl_retrieval_spans_the_full_rotation_chain(tmp_path, _cleanup_repl):
    """The actual point of item 1: the model's retrieval must reach the
    whole conversation regardless of which session_id its REPL instance
    is running under.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("session-A")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    old = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"old msg {i}"} for i in range(10)]
    engine.on_turn_complete(old, turn_id=1)

    engine.on_session_start("session-B")
    restored = old + [{"role": "user", "content": "one new message after restart"}]
    engine.on_turn_complete(restored, turn_id=1)

    repl = PersistentREPL(
        db_path=engine._store.db_path, session_id="session-B", base_url="http://x/v1", model="m"
    )
    _cleanup_repl.append(repl)
    r = repl.exec("print(context_total)")
    assert r["stdout"].strip() == "11", (
        "session-B's REPL must see all 10 of session-A's messages plus its "
        "own 1 new one -- the full conversation, not just what archived "
        "under session-B's own id"
    )


# ---------------------------------------------------------------------------
# Round-22: provenance instead of similarity. Round 21's rotation detection
# was compensating for the wrong root cause -- background review forks
# (agent/background_review.py's _spawn_background_review) construct their
# own AIAgent, on_session_start binds a fresh id, and RLM archived the
# FULL inherited parent transcript it was handed to review as if it were
# the fork's own conversation. Measured: 505 of 1,062 real archive rows
# (47%) were background agents re-archiving the user's own conversation.
# Fix: the host now tells the engine directly (agent_kind,
# inherited_message_count), no content comparison needed.
# ---------------------------------------------------------------------------

def test_on_session_start_seeds_persisted_count_from_inherited_count(tmp_path):
    engine = _make_engine(tmp_path)
    engine.on_session_start(
        "review-fork-1", agent_kind="background_review",
        parent_session_id="parent-session", inherited_message_count=42,
    )
    assert engine._persisted_count == 42
    assert engine._resume_verified is True
    assert engine._rotation_checked is True, (
        "provenance already answers the question rotation detection exists "
        "to answer -- it must not also run for a provenance-seeded session"
    )
    assert engine._agent_kind == "background_review"
    assert engine._parent_session_id == "parent-session"


def test_provenance_seeding_ignored_for_ordinary_conversation_sessions(tmp_path):
    """agent_kind='conversation' (the default -- an ordinary user session)
    must never have its persisted_count overridden by a stray
    inherited_message_count, even if one were somehow passed.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("normal-session", inherited_message_count=42)
    assert engine._persisted_count == 0
    assert engine._agent_kind == "conversation"


def test_background_review_archives_only_its_own_generated_content(tmp_path):
    """End-to-end: the actual point. A background review fork handed the
    parent's full transcript must archive only what it generates itself,
    not the inherited transcript a second time.
    """
    engine = _make_engine(tmp_path)
    parent_transcript = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"parent msg {i}"}
        for i in range(20)
    ]
    engine.on_session_start(
        "review-fork-2", agent_kind="background_review",
        parent_session_id="parent-session-2",
        inherited_message_count=len(parent_transcript),
    )
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    # The fork's own conversation = inherited transcript + its own turns.
    fork_conversation = parent_transcript + [
        {"role": "user", "content": "review prompt"},
        {"role": "assistant", "content": "updated skill X"},
    ]
    engine.on_turn_complete(fork_conversation, turn_id=1)

    assert engine._store.message_count("review-fork-2") == 2, (
        "only the fork's own 2 new messages should archive -- not the 20 "
        "inherited from the parent, which are already archived under the "
        "parent's own session_id"
    )


def test_background_review_does_not_pollute_parent_archive(tmp_path):
    """The inherited content isn't just skipped for the fork -- it must
    not leak into or duplicate the parent's own archive either (the fork
    and the parent are archived under different session_ids throughout).
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("parent-session-3")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    parent_transcript = [{"role": "user", "content": f"parent msg {i}"} for i in range(10)]
    engine.on_turn_complete(parent_transcript, turn_id=1)
    assert engine._store.message_count("parent-session-3") == 10

    engine.on_session_start(
        "review-fork-3", agent_kind="background_review",
        parent_session_id="parent-session-3",
        inherited_message_count=len(parent_transcript),
    )
    fork_conversation = parent_transcript + [{"role": "assistant", "content": "review done"}]
    engine.on_turn_complete(fork_conversation, turn_id=1)

    assert engine._store.message_count("parent-session-3") == 10, (
        "the parent's own archive must be untouched by the fork's activity"
    )
    assert engine._store.message_count("review-fork-3") == 1
