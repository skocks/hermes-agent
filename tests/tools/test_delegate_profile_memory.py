"""Profile-driven memory-provider access for delegate_task children.

Children are built with skip_memory=True, so agent_init never attaches the
external memory provider (hindsight). A profile opts back in by naming an
allow-list of provider tools. These verify the allow-list is honoured
exactly -- notably that a write tool the profile did not name stays out,
which is the whole point of an allow-list over a boolean.
"""

from tools.delegate_tool import _inject_allowed_memory_tools


class _FakeMemoryManager:
    def __init__(self, schemas):
        self._schemas = schemas

    def get_all_tool_schemas(self):
        return list(self._schemas)


class _FakeAgent:
    def __init__(self, mm=None, tools=None):
        self._memory_manager = mm
        self.tools = tools if tools is not None else []
        self.valid_tool_names = set()


_SCHEMAS = [
    {"name": "hindsight_recall", "description": "read", "parameters": {}},
    {"name": "hindsight_retain", "description": "write", "parameters": {}},
    {"name": "hindsight_reflect", "description": "write", "parameters": {}},
]


def _names(agent):
    return {t["function"]["name"] for t in agent.tools}


class TestInjectAllowedMemoryTools:
    def test_injects_only_allowlisted_tool(self):
        agent = _FakeAgent(_FakeMemoryManager(_SCHEMAS))
        added = _inject_allowed_memory_tools(agent, ["hindsight_recall"])
        assert added == 1
        assert _names(agent) == {"hindsight_recall"}
        assert "hindsight_retain" not in agent.valid_tool_names

    def test_write_tools_injected_when_named(self):
        agent = _FakeAgent(_FakeMemoryManager(_SCHEMAS))
        added = _inject_allowed_memory_tools(
            agent, ["hindsight_recall", "hindsight_retain"]
        )
        assert added == 2
        assert _names(agent) == {"hindsight_recall", "hindsight_retain"}

    def test_no_allowlist_injects_nothing(self):
        agent = _FakeAgent(_FakeMemoryManager(_SCHEMAS))
        assert _inject_allowed_memory_tools(agent, None) == 0
        assert agent.tools == []

    def test_no_memory_manager_is_a_noop(self):
        agent = _FakeAgent(None)
        assert _inject_allowed_memory_tools(agent, ["hindsight_recall"]) == 0
        assert agent.tools == []

    def test_unknown_name_in_allowlist_is_ignored(self):
        """A profile naming a tool the provider doesn't expose must not crash."""
        agent = _FakeAgent(_FakeMemoryManager(_SCHEMAS))
        added = _inject_allowed_memory_tools(agent, ["nope_not_a_tool"])
        assert added == 0
        assert agent.tools == []

    def test_does_not_duplicate_existing_tool(self):
        agent = _FakeAgent(
            _FakeMemoryManager(_SCHEMAS),
            tools=[{"type": "function", "function": {"name": "hindsight_recall"}}],
        )
        assert _inject_allowed_memory_tools(agent, ["hindsight_recall"]) == 0
        assert len(agent.tools) == 1

    def test_registers_valid_tool_name(self):
        agent = _FakeAgent(_FakeMemoryManager(_SCHEMAS))
        _inject_allowed_memory_tools(agent, ["hindsight_recall"])
        assert "hindsight_recall" in agent.valid_tool_names


# --- plumbing: profile.memory_tools must reach the child build -------------

import unittest
from unittest.mock import MagicMock, patch

from tools.agent_profiles import AgentProfileSpec
from tools.delegate_tool import delegate_task
from tests.tools.test_delegate_agent_profile import (  # reuse, don't duplicate
    _CREDS,
    _completed,
    _make_mock_parent,
)

_MEM_SPEC = AgentProfileSpec(
    skill_name="memory-researcher",
    enabled_toolsets=["web", "file"],
    memory_tools=["hindsight_recall"],
)

_PLAIN_SPEC = AgentProfileSpec(
    skill_name="researcher",
    enabled_toolsets=["web", "file"],
)


class TestProfileMemoryPlumbing(unittest.TestCase):
    def _capture(self, spec):
        captured = {}

        def _fake_build(**kwargs):
            captured.update(kwargs)
            child = MagicMock()
            child._delegate_output_schema = None
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value=_CREDS,
            ),
            patch(
                "tools.delegate_tool.agent_profile_for_installed",
                return_value=spec,
            ),
            patch(
                "tools.delegate_tool._build_child_preserving_parent_tools",
                side_effect=_fake_build,
            ),
            patch("tools.delegate_tool._run_single_child", return_value=_completed(0)),
        ):
            delegate_task(
                goal="Find prior findings on migration ROI",
                agent="memory-researcher",
                parent_agent=_make_mock_parent(),
            )
        return captured

    def test_memory_tools_reach_child_build(self):
        captured = self._capture(_MEM_SPEC)
        self.assertEqual(captured.get("profile_memory_tools"), ["hindsight_recall"])

    def test_profile_without_memory_tools_passes_none(self):
        """Default stays "no provider" -- the opt-in must be explicit."""
        captured = self._capture(_PLAIN_SPEC)
        self.assertIsNone(captured.get("profile_memory_tools"))


class TestSkipMemoryFlip(unittest.TestCase):
    """The load-bearing claim: memory_tools flips skip_memory off for the child.

    Without this, agent_init's ``if not skip_memory:`` gate never builds the
    provider, ``child._memory_manager`` stays None, and the injection helper
    is a no-op no matter what the profile allow-lists.
    """

    def _captured_skip_memory(self, memory_tools):
        from tools.delegate_tool import _build_child_agent

        captured = {}

        def _fake_aiagent(**kwargs):
            captured.update(kwargs)
            agent = MagicMock()
            agent.session_id = "sid"
            agent.tools = []
            agent._memory_manager = None
            return agent

        with patch("run_agent.AIAgent", side_effect=_fake_aiagent):
            _build_child_agent(
                task_index=0,
                goal="g",
                context="c",
                toolsets=None,
                model="m",
                max_iterations=3,
                task_count=1,
                parent_agent=_make_mock_parent(),
                profile_memory_tools=memory_tools,
            )
        return captured.get("skip_memory")

    def test_skip_memory_false_when_profile_allows_memory_tools(self):
        self.assertFalse(self._captured_skip_memory(["hindsight_recall"]))

    def test_skip_memory_true_by_default(self):
        self.assertTrue(self._captured_skip_memory(None))
