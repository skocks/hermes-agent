"""Round-20: create_session() failures at cli.py's /new call site used to
be swallowed by a bare ``except Exception: pass`` -- a session whose
state.db registration failed left zero trace anywhere. This is the
call site RLM's round-20 orphan-sweep fix depends on being visible:
without a log line, no one (human or a future audit) can tell a session
never registered from one that was legitimately pruned.

Confirms two things: the failure is now logged at warning with the
session_id, and -- unchanged, must stay this way -- a create_session()
failure still does not raise out of new_session(). A registration
failure must never crash session start.
"""

from __future__ import annotations

import importlib
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_cli(config_overrides=None):
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    if config_overrides:
        _clean_config.update(config_overrides)
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict("os.environ", {}, clear=False):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI()


@pytest.fixture(autouse=True)
def _reset_session_id_context():
    import os

    from gateway.session_context import _UNSET, _VAR_MAP

    yield
    os.environ.pop("HERMES_SESSION_ID", None)
    _VAR_MAP["HERMES_SESSION_ID"].set(_UNSET)


class _MinimalFakeAgent:
    """The create_session() try/except this test targets (cli.py's
    new_session) is nested inside `if self.agent:` -- a bare CLI instance
    has agent=None until one is attached, so the block under test is
    unreachable without a stand-in agent present, same as real usage
    after the first turn.
    """

    def __init__(self, session_id, session_start):
        self.session_id = session_id
        self.session_start = session_start
        self._session_db_created = False

    def reset_session_state(self):
        pass


def test_create_session_failure_is_logged_not_swallowed_silently(caplog):
    cli = _make_cli()
    cli.agent = _MinimalFakeAgent(cli.session_id, cli.session_start)
    cli._session_db = MagicMock()
    cli._session_db.create_session.side_effect = RuntimeError(
        "database is locked (another Hermes process held the state.db write lock)"
    )
    cli._confirm_destructive_slash = lambda *_a, **_kw: "once"

    with caplog.at_level(logging.WARNING, logger="cli"):
        cli.new_session(silent=True)  # must not raise

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("create_session failed" in r.message for r in warnings), (
        "a create_session failure must be logged, not pass silently -- "
        f"got: {[r.message for r in warnings]}"
    )
    assert any(cli.session_id in r.message for r in warnings), (
        "the log must name the session_id so the failure is traceable to a "
        "specific session, not just 'something failed somewhere'"
    )


def test_session_row_writes_already_use_a_generous_retry_budget():
    """Round-20 verification, not a new behavior: the reviewer's proposed
    fix was raising hermes_state.py's raw connect() timeout to 30s.
    Checked instead of applied -- create_session() -> _insert_session_row()
    already opens its connection with a deliberately SHORT raw timeout
    (1.0s, by design -- see hermes_state.py's own comment) paired with a
    60-second application-level jittered retry
    (_TRANSCRIPT_WRITE_PATIENCE_S) specifically for this write, meant to
    replace SQLite's own dumb linear busy-wait with staggered backoff.
    Raising the raw per-attempt timeout would make each retry iteration
    block longer for the SAME 60s total budget -- fewer, longer waits
    instead of many short jittered ones -- working against the documented
    intent, not fixing a gap. Declined the timeout change on this
    evidence; this test pins the evidence so it doesn't silently bit-rot
    if _TRANSCRIPT_WRITE_PATIENCE_S is ever changed without noticing it's
    the actual contention-tolerance mechanism for session registration.
    """
    import inspect

    from hermes_state import SessionDB

    assert SessionDB._TRANSCRIPT_WRITE_PATIENCE_S >= 60.0, (
        "session-row writes must keep a generous retry budget -- this is "
        "what actually tolerates state.db write contention, not a raw "
        "connect()-level timeout"
    )
    source = inspect.getsource(SessionDB._insert_session_row)
    assert "_TRANSCRIPT_WRITE_PATIENCE_S" in source, (
        "_insert_session_row (create_session's underlying write) must use "
        "the transcript-critical patience budget, not the default one"
    )


def test_create_session_failure_does_not_crash_new_session():
    """Must stay a swallow -- a registration failure can never be allowed
    to abort session start. Every downstream self-healing path
    (update_token_counts's own INSERT OR IGNORE) is built to tolerate
    _session_db_created being False.
    """
    cli = _make_cli()
    cli.agent = _MinimalFakeAgent(cli.session_id, cli.session_start)
    cli._session_db = MagicMock()
    cli._session_db.create_session.side_effect = RuntimeError("boom")
    cli._confirm_destructive_slash = lambda *_a, **_kw: "once"

    cli.new_session(silent=True)  # must complete without raising

    assert cli.session_id  # session start still produced a usable session
