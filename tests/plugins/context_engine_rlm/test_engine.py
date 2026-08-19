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
    repl = PersistentREPL(
        db_path=engine._store.db_path, session_id="repl-test",
        base_url="http://x/v1", model="m", **repl_overrides,
    )
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


def test_staleness_footer_lists_earlier_vars_not_current_call(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("chunks = [1, 2, 3]")
    r = repl.exec("print(1)")  # does not touch chunks -- must be flagged as stale
    assert "chunks" in r["stdout"]
    assert "__builtins__" not in r["stdout"], "exec()'s auto-injected __builtins__ is not a user variable"

    r_same_call = repl.exec("just_set = True\nprint(1)")
    assert "just_set" not in r_same_call["stdout"], "a variable set in THIS call must not be flagged as stale yet"


def test_staleness_footer_silent_after_reset(tmp_path, _cleanup_repl):
    _, repl = _archive_and_repl(tmp_path, [{"role": "user", "content": "x"}])
    _cleanup_repl.append(repl)

    repl.exec("chunks = [1, 2, 3]")
    repl.exec("reset()")
    r = repl.exec("print(1)")
    assert "REPL:" not in r["stdout"]


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
