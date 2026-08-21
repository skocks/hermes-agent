#!/usr/bin/env python3
"""Round 24: delegate_task(agent=...) resolves a named agent profile
(tools/agent_profiles.py) into the child's toolsets/disabled_toolsets/
system_prompt_fragment/output_schema, instead of the child inheriting the
parent's full toolset unconditionally.

Coverage:
  - Omitting `agent` preserves today's behavior exactly (toolsets=None).
  - An unknown profile name fails the whole call loudly, before any child
    is built.
  - A malformed profile (AgentProfileError) fails the same way.
  - A resolved profile's enabled_toolsets/disabled_toolsets/
    system_prompt_fragment reach _build_child_preserving_parent_tools.
  - Per-task `agent` overrides the top-level one, mirroring `role`.
  - A profile's default output_schema is used as a fallback exactly like
    the existing single-task output_schema fallback.
  - DELEGATE_BLOCKED_TOOLS / role-based blocking still applies on top of
    a profile's enabled_toolsets (a profile cannot smuggle back tools
    that are always stripped for children).
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.agent_profiles import AgentProfileError, AgentProfileSpec
from tools.delegate_tool import _build_child_agent, delegate_task


def _make_mock_parent(depth=0, enabled_toolsets=None):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = enabled_toolsets
    return parent


_CREDS = {
    "provider": None,
    "model": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
}


def _completed(idx):
    return {
        "task_index": idx,
        "status": "completed",
        "summary": "ok",
        "api_calls": 1,
        "duration_seconds": 1.0,
        "_child_role": None,
    }


RESEARCHER_SPEC = AgentProfileSpec(
    skill_name="researcher",
    enabled_toolsets=["web", "file"],
    system_prompt_fragment="You are a research subagent.",
)


class TestUnknownOrMalformedProfileRejected(unittest.TestCase):
    def test_unknown_agent_profile_rejected_before_any_spawn(self):
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch("tools.delegate_tool._resolve_delegation_credentials", return_value=_CREDS),
            patch("tools.delegate_tool.agent_profile_for_installed", return_value=None),
            patch("tools.delegate_tool._run_single_child") as mock_run,
        ):
            out = delegate_task(
                goal="Check these five pages for a Sources section",
                agent="not-a-real-profile",
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        self.assertTrue(payload.get("error"))
        self.assertIn("not-a-real-profile", payload["error"])
        mock_run.assert_not_called()

    def test_malformed_profile_rejected(self):
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch("tools.delegate_tool._resolve_delegation_credentials", return_value=_CREDS),
            patch(
                "tools.delegate_tool.agent_profile_for_installed",
                side_effect=AgentProfileError("enabled_toolsets is required"),
            ),
            patch("tools.delegate_tool._run_single_child") as mock_run,
        ):
            out = delegate_task(
                goal="Do the thing",
                agent="broken-profile",
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        self.assertTrue(payload.get("error"))
        self.assertIn("malformed", payload["error"])
        mock_run.assert_not_called()


class TestProfileResolutionReachesChildBuild(unittest.TestCase):
    def test_omitted_agent_preserves_toolsets_none(self):
        captured = {}

        def _fake_build(**kwargs):
            captured.update(kwargs)
            child = MagicMock()
            child._delegate_output_schema = None
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch("tools.delegate_tool._resolve_delegation_credentials", return_value=_CREDS),
            patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_fake_build),
            patch("tools.delegate_tool._run_single_child", return_value=_completed(0)),
        ):
            delegate_task(goal="Refactor the login handler", parent_agent=_make_mock_parent())

        self.assertIsNone(captured.get("toolsets"))
        self.assertIsNone(captured.get("profile_disabled_toolsets"))
        self.assertIsNone(captured.get("profile_system_prompt_fragment"))

    def test_resolved_profile_passed_through_to_child_build(self):
        captured = {}

        def _fake_build(**kwargs):
            captured.update(kwargs)
            child = MagicMock()
            child._delegate_output_schema = None
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch("tools.delegate_tool._resolve_delegation_credentials", return_value=_CREDS),
            patch("tools.delegate_tool.agent_profile_for_installed", return_value=RESEARCHER_SPEC),
            patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_fake_build),
            patch("tools.delegate_tool._run_single_child", return_value=_completed(0)),
        ):
            delegate_task(
                goal="Check these five pages for a Sources section",
                agent="researcher",
                parent_agent=_make_mock_parent(),
            )

        self.assertEqual(captured.get("toolsets"), ["web", "file"])
        self.assertEqual(
            captured.get("profile_system_prompt_fragment"),
            "You are a research subagent.",
        )

    def test_per_task_agent_overrides_top_level(self):
        captured = []

        def _fake_build(**kwargs):
            captured.append(kwargs)
            child = MagicMock()
            child._delegate_output_schema = None
            return child

        file_auditor_spec = AgentProfileSpec(
            skill_name="file-auditor",
            enabled_toolsets=["file", "terminal"],
        )

        def _resolve(name):
            return {"researcher": RESEARCHER_SPEC, "file-auditor": file_auditor_spec}.get(name)

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch("tools.delegate_tool._resolve_delegation_credentials", return_value=_CREDS),
            patch("tools.delegate_tool.agent_profile_for_installed", side_effect=_resolve),
            patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_fake_build),
            patch(
                "tools.delegate_tool._run_single_child",
                side_effect=[_completed(0), _completed(1)],
            ),
        ):
            delegate_task(
                agent="researcher",
                tasks=[
                    {"goal": "Grep the codebase for deprecated call sites", "agent": "file-auditor"},
                    {"goal": "Check these pages for a Sources section"},
                ],
                parent_agent=_make_mock_parent(),
            )

        self.assertEqual(captured[0]["toolsets"], ["file", "terminal"])
        self.assertEqual(captured[1]["toolsets"], ["web", "file"])

    def test_profile_output_schema_used_as_default(self):
        schema = {"type": "object", "properties": {"found": {"type": "boolean"}}}
        spec = AgentProfileSpec(
            skill_name="researcher",
            enabled_toolsets=["web", "file"],
            output_schema=schema,
        )
        captured_child = {}

        def _fake_build(**kwargs):
            child = MagicMock()
            captured_child["child"] = child
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch("tools.delegate_tool._resolve_delegation_credentials", return_value=_CREDS),
            patch("tools.delegate_tool.agent_profile_for_installed", return_value=spec),
            patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_fake_build),
            patch("tools.delegate_tool._run_single_child", return_value=_completed(0)),
        ):
            delegate_task(
                goal="Check these pages for a Sources section",
                agent="researcher",
                parent_agent=_make_mock_parent(),
            )

        self.assertEqual(
            getattr(captured_child["child"], "_delegate_output_schema", None), schema
        )


class TestProfileToolsetsStillBlocked(unittest.TestCase):
    """A profile's enabled_toolsets is still intersected with the parent's
    and still passed through _strip_blocked_tools -- a profile cannot
    smuggle back DELEGATE_BLOCKED_TOOLS-only toolsets."""

    def test_delegation_toolset_still_stripped_for_leaf_child(self):
        parent = _make_mock_parent(enabled_toolsets=["web", "file", "delegation"])
        parent.valid_tool_names = []
        child = _build_child_agent(
            task_index=0,
            goal="test",
            context=None,
            toolsets=["web", "file", "delegation"],
            model=None,
            max_iterations=5,
            task_count=1,
            parent_agent=parent,
            role="leaf",
            profile_disabled_toolsets=None,
            profile_system_prompt_fragment="You are a research subagent.",
        )
        try:
            self.assertNotIn("delegation", child.enabled_toolsets)
            self.assertIn("web", child.enabled_toolsets)
            self.assertIn("file", child.enabled_toolsets)
            self.assertIn("research subagent", child.ephemeral_system_prompt)
        finally:
            try:
                child.close()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
