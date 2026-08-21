"""delegation.profile_grantable_toolsets — let vetted profiles grant beyond the parent.

_build_child_agent intersects a profile's enabled_toolsets with the parent's own
("a subagent must not gain tools the parent lacks"). That guard is right for an
arbitrary caller-supplied list, but it makes agent profiles unusable on a thinned
root: a root with only [clarify, delegation, memory, skills, todo] intersects a
researcher profile's [web, file] down to nothing, and the child silently falls
back to the parent's toolsets.

This allowlist is the operator's explicit statement that a *vetted profile* may
grant these specific toolsets even though root does not carry them. It does not
loosen anything for callers: only toolsets that arrived from a resolved profile
pass through it.
"""

from unittest.mock import patch

from tools.delegate_tool import _get_profile_grantable_toolsets


class TestProfileGrantableConfig:
    def test_absent_config_grants_nothing(self):
        """Default must preserve today's behaviour exactly."""
        with patch("tools.delegate_tool._load_config", return_value={}):
            assert _get_profile_grantable_toolsets() == set()

    def test_reads_list_from_config(self):
        with patch("tools.delegate_tool._load_config",
                   return_value={"profile_grantable_toolsets": ["web", "file"]}):
            assert _get_profile_grantable_toolsets() == {"web", "file"}

    def test_coerces_entries_to_strings(self):
        with patch("tools.delegate_tool._load_config",
                   return_value={"profile_grantable_toolsets": ["web", 7]}):
            assert _get_profile_grantable_toolsets() == {"web", "7"}

    def test_non_list_is_ignored_not_fatal(self):
        """A typo must not break delegation entirely -- warn and grant nothing."""
        with patch("tools.delegate_tool._load_config",
                   return_value={"profile_grantable_toolsets": "web"}):
            assert _get_profile_grantable_toolsets() == set()

    def test_empty_list_grants_nothing(self):
        with patch("tools.delegate_tool._load_config",
                   return_value={"profile_grantable_toolsets": []}):
            assert _get_profile_grantable_toolsets() == set()


# --- the intersection itself -------------------------------------------------

import unittest
from unittest.mock import MagicMock

from tests.tools.test_delegate_agent_profile import _make_mock_parent


class TestProfileToolsetGrant(unittest.TestCase):
    """Only profile-sourced toolsets may use the allowlist."""

    def _child_toolsets(self, grantable, *, from_profile=True,
                        parent_toolsets=("clarify", "delegation", "skills", "todo")):
        from tools.delegate_tool import _build_child_agent

        captured = {}

        def _fake_aiagent(**kwargs):
            captured.update(kwargs)
            agent = MagicMock()
            agent.session_id = "sid"
            agent.tools = []
            agent._memory_manager = None
            return agent

        parent = _make_mock_parent()
        parent.enabled_toolsets = list(parent_toolsets)

        with patch("run_agent.AIAgent", side_effect=_fake_aiagent), \
             patch("tools.delegate_tool._get_profile_grantable_toolsets",
                   return_value=set(grantable)):
            _build_child_agent(
                task_index=0, goal="g", context="c", toolsets=["web", "file"],
                model="m", max_iterations=3, task_count=1, parent_agent=parent,
                toolsets_from_profile=from_profile,
            )
        return captured.get("enabled_toolsets")

    def test_allowlisted_toolsets_reach_the_child(self):
        got = self._child_toolsets({"web", "file"})
        self.assertIn("web", got)
        self.assertIn("file", got)

    def test_without_allowlist_profile_toolsets_are_intersected_away(self):
        """Today's behaviour, preserved when the operator has not opted in."""
        got = self._child_toolsets(set())
        self.assertNotIn("web", got or [])
        self.assertNotIn("file", got or [])

    def test_partial_allowlist_grants_only_named_toolsets(self):
        got = self._child_toolsets({"file"})
        self.assertIn("file", got)
        self.assertNotIn("web", got)

    def test_non_profile_toolsets_never_use_the_allowlist(self):
        """The escalation guard still applies to any non-profile caller."""
        got = self._child_toolsets({"web", "file"}, from_profile=False)
        self.assertNotIn("web", got or [])
        self.assertNotIn("file", got or [])
