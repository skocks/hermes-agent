"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    canonical_tool_args,
    classify_tool_failure,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)




def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2














def test_mutating_or_unknown_tools_warn_but_are_not_blocked_for_repeated_identical_success_output_by_default():
    # No-progress detection now covers mutating and unknown tools too, but
    # without hard_stop_enabled it only warns — it never blocks execution.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for tool_name, args in (
        ("write_file", {"path": "/tmp/x", "content": "x"}),
        ("custom_tool", {"x": 1}),
    ):
        assert controller.before_call(tool_name, args).action == "allow"
        first = controller.after_call(tool_name, args, "ok", failed=False)
        assert first.action == "allow"

        assert controller.before_call(tool_name, args).action == "allow"
        second = controller.after_call(tool_name, args, "ok", failed=False)
        assert second.action == "warn"
        assert second.code == "idempotent_no_progress_warning"
        assert second.count == 2

        # No hard stop by default: still not blocked and no halt decision.
        assert controller.before_call(tool_name, args).action == "allow"
        assert controller.halt_decision is None


def test_terminal_repeated_identical_success_output_warns_and_blocks_with_hard_stop():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
hard_stop_enabled=True,
no_progress_warn_after=2,
no_progress_block_after=3,
        )
    )
    args = {"command": "ls -la"}
    result = json.dumps({"exit_code": 0, "output": "same"})

    assert controller.before_call("terminal", args).action == "allow"
    assert controller.after_call("terminal", args, result, failed=False).action == "allow"

    assert controller.before_call("terminal", args).action == "allow"
    second = controller.after_call("terminal", args, result, failed=False)
    assert second.action == "warn"
    assert second.code == "idempotent_no_progress_warning"
    assert second.count == 2

    assert controller.before_call("terminal", args).action == "allow"
    third = controller.after_call("terminal", args, result, failed=False)
    assert third.action == "warn"
    assert third.count == 3

    blocked = controller.before_call("terminal", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"
    assert blocked.count == 3
    assert blocked.should_halt is True
    assert controller.halt_decision is not None






# ── Per-turn runaway-loop caps (Claude Code v2.1.212, Week 29) ──────────────

from agent.tool_guardrails import LoopCapConfig  # noqa: E402






def test_loop_cap_zero_disables_and_junk_falls_back():
    # 0 is a legitimate "unlimited" value; negatives / junk fall back to default.
    assert LoopCapConfig.from_mapping({"max_web_searches": 0}).max_web_searches == 0
    assert LoopCapConfig.from_mapping({"max_web_searches": -5}).max_web_searches == 50
    assert LoopCapConfig.from_mapping({"max_subagents": "nope"}).max_subagents == 50


def test_web_search_cap_blocks_after_limit_regardless_of_hard_stop():
    # Loop caps fire even with hard_stop_enabled=False (the per-turn loop
    # detector's flag). Each distinct query avoids the loop detector so we know
    # the block came from the loop cap, not exact-failure repetition.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(3):
        assert controller.before_call("web_search", {"query": f"q{i}"}).action == "allow"
    decision = controller.before_call("web_search", {"query": "q4"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"
    assert decision.should_halt is True












class TestToolClassificationCoverage:
    """Every tool on the root surface must be visible to the loop detector.

    The no-progress detector only evaluates tools listed in
    IDEMPOTENT_TOOL_NAMES or MUTATING_TOOL_NAMES; a tool in neither set is
    never examined. That silent default let a subagent spin on repeated
    skill_view calls for 182s on 2026-08-20 with hard stops enabled -- the
    detector never looked at it.
    """

    def test_read_only_skill_and_memory_tools_are_idempotent(self):
        from agent.tool_guardrails import IDEMPOTENT_TOOL_NAMES

        for name in (
            "skill_view",
            "skill_search",
            "skills_list",
            "hindsight_recall",
            "hindsight_reflect",
        ):
            assert name in IDEMPOTENT_TOOL_NAMES, name

    def test_writing_tools_are_mutating(self):
        from agent.tool_guardrails import MUTATING_TOOL_NAMES

        for name in ("hindsight_retain", "rlm_repl"):
            assert name in MUTATING_TOOL_NAMES, name

    def test_no_tool_is_in_both_sets(self):
        from agent.tool_guardrails import (
            IDEMPOTENT_TOOL_NAMES,
            MUTATING_TOOL_NAMES,
        )

        assert not (IDEMPOTENT_TOOL_NAMES & MUTATING_TOOL_NAMES)
