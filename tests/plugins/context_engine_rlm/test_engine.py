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
