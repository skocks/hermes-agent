"""Regression tests for the RLM context engine.

Covers the exact bugs found by an independent fidelity audit
(scratchpad/rlm-audit.md, 2026-08-19) — C1/C2 (data loss) and H1 (the
paper's core recursion mechanism silently broken by a timeout
misconfiguration) are the audit's own "highest-value first" picks, so
they're first here too. No live model/network calls — those are verified
manually against the running local server, not in CI.
"""

from __future__ import annotations

import pytest

from plugins.context_engine.rlm.engine import RLMContextEngine
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
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")

    # Plant an inflated archive: 2 real rows + 5 unrelated stale rows,
    # simulating what a pre-N2 resync would have left behind.
    engine._store.append_messages("s1", 1, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    engine._store.append_messages("s1", 1, [{"role": "user", "content": f"stale-{i}"} for i in range(5)])
    assert engine._store.raw_row_count("s1") == 7

    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    live = [{"role": "user", "content": c} for c in "abcdefghi"]  # 9 messages
    engine2.on_turn_complete(live, turn_id=2)

    assert engine2._store.message_count("s1") == 9, (
        "the visible count must match the live transcript exactly, no "
        "duplicates from the old inflated rows"
    )


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
    resync, even though the physical table does (by design).
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
    assert total == unique == "9"


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

def test_verify_watermark_empty_transcript_branch_tombstones(tmp_path):
    """engine.py's check_n<=0 branch in _verify_resume_watermark: fires
    when the live transcript is empty but the resume estimate is not.
    Harmless on THAT call (nothing to re-append yet), but without
    _trigger_resync() the old rows stay untombstoned and the NEXT real
    on_turn_complete duplicates them.
    """
    engine = _make_engine(tmp_path)
    engine.on_session_start("s1")
    engine.update_model(model="m", context_length=131072, base_url="http://x/v1", api_mode="chat_completions")
    engine._store.append_messages("s1", 1, [{"role": "user", "content": "old1"}, {"role": "user", "content": "old2"}])

    engine2 = _make_engine(tmp_path)
    engine2.on_session_start("s1")
    engine2.on_turn_complete([], turn_id=1)  # empty transcript -- hits the flagged branch
    engine2.on_turn_complete([{"role": "user", "content": "new1"}], turn_id=2)  # real content arrives

    assert engine2._store.message_count("s1") == 1, "old1/old2 must be tombstoned, not duplicated onto new1"


def test_message_count_lookup_failure_tombstones(tmp_path):
    """engine.py's on_session_start exception handler when message_count()
    itself raises: same counterexample, different trigger.
    """
    from unittest import mock

    engine = _make_engine(tmp_path)
    engine._store.append_messages("s2", 1, [{"role": "user", "content": "old-b"}])

    with mock.patch.object(engine._store, "message_count", side_effect=RuntimeError("boom")):
        engine.on_session_start("s2")
    assert engine._resume_verified is True
    assert engine._persisted_count == 0

    engine.on_turn_complete([{"role": "user", "content": "new-b"}], turn_id=1)
    assert engine._store.message_count("s2") == 1, "old-b must be tombstoned, not duplicated onto new-b"


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


def test_sweep_deletes_orphaned_session_rows_and_fts_entries(tmp_path):
    """A session_id present in rlm.db but absent from state.db is an
    orphan: its rows -- live AND tombstoned -- and its FTS5 entries are
    all removed. This is a real DELETE, unlike supersede_session's
    tombstone -- by design, this is rlm.db's only deletion path.
    """
    from plugins.context_engine.rlm.store import RLMStore
    store = RLMStore(str(tmp_path / "rlm.db"))
    assert store._fts_enabled
    store.append_messages("orphan", 1, [{"role": "user", "content": "will be swept"}])
    store.append_messages("orphan", 2, [{"role": "user", "content": "also swept"}])
    store.supersede_session("orphan")  # tombstone one round, so both live+dead rows exist
    store.append_messages("orphan", 3, [{"role": "user", "content": "swept too"}])
    assert store.raw_row_count("orphan") == 3

    state_db = _make_state_db(tmp_path, [])  # state.db knows nothing of "orphan"
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
    state_db = _make_state_db(tmp_path, [])

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
    r2 = store.sweep_orphaned_sessions(
        state_db, min_interval_hours=0, vacuum_after_prune=True, min_vacuum_interval_days=30
    )
    assert r2["sessions_pruned"] == 1
    assert r2["vacuumed"] is True

    store.append_messages("orphan2", 1, [{"role": "user", "content": "y"}])
    r3 = store.sweep_orphaned_sessions(
        state_db, min_interval_hours=0, vacuum_after_prune=True, min_vacuum_interval_days=30
    )
    assert r3["sessions_pruned"] == 1
    assert r3["vacuumed"] is False, "min_vacuum_interval_days must throttle even with fresh deletes"
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
