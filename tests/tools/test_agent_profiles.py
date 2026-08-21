"""Tests for tools/agent_profiles.py (round 24 of the RLM plugin audit).

An agent profile is a skill with a metadata.hermes.agent block. These verify
parsing and the installed-skill lookup, mirroring test_blueprints.py's
pattern for the sibling blueprints layer.
"""

from unittest.mock import patch

import pytest

from tools.agent_profiles import (
    AgentProfileError,
    AgentProfileSpec,
    agent_profile_for_installed,
    parse_agent_profile,
)


PROFILE_SKILL = """---
name: researcher
description: Research subagent profile.
metadata:
  hermes:
    agent:
      enabled_toolsets: [web, file]
      system_prompt_fragment: "You are a research subagent."
---

# Researcher
"""

PLAIN_SKILL = """---
name: not-a-profile
description: Just a regular skill.
metadata:
  hermes:
    tags: [misc]
---

# Not a profile
"""

MALFORMED_MISSING_TOOLSETS = """---
name: broken
description: Profile with no enabled_toolsets.
metadata:
  hermes:
    agent:
      system_prompt_fragment: "no toolsets here"
---

# Broken
"""

MALFORMED_NOT_A_LIST = """---
name: broken2
description: enabled_toolsets is a string, not a list.
metadata:
  hermes:
    agent:
      enabled_toolsets: web
---

# Broken 2
"""

MALFORMED_BAD_DISABLED = """---
name: broken3
description: disabled_toolsets is not a list.
metadata:
  hermes:
    agent:
      enabled_toolsets: [web]
      disabled_toolsets: terminal
---

# Broken 3
"""

MALFORMED_BAD_SCHEMA = """---
name: broken4
description: output_schema is not an object.
metadata:
  hermes:
    agent:
      enabled_toolsets: [web]
      output_schema: "not-a-dict"
---

# Broken 4
"""

FULL_SKILL = """---
name: file-auditor
description: File-auditor subagent profile.
metadata:
  hermes:
    agent:
      enabled_toolsets: [file, terminal]
      disabled_toolsets: [web]
      system_prompt_fragment: "No web access."
      output_schema:
        type: object
        properties:
          summary: {type: string}
---

# File auditor
"""


class TestParseAgentProfile:
    def test_parses_minimal_profile(self):
        spec = parse_agent_profile(PROFILE_SKILL)
        assert spec is not None
        assert spec.skill_name == "researcher"
        assert spec.enabled_toolsets == ["web", "file"]
        assert spec.system_prompt_fragment == "You are a research subagent."
        assert spec.disabled_toolsets is None
        assert spec.output_schema is None

    def test_parses_full_profile(self):
        spec = parse_agent_profile(FULL_SKILL)
        assert spec is not None
        assert spec.skill_name == "file-auditor"
        assert spec.enabled_toolsets == ["file", "terminal"]
        assert spec.disabled_toolsets == ["web"]
        assert spec.system_prompt_fragment == "No web access."
        assert spec.output_schema == {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }

    def test_plain_skill_returns_none(self):
        assert parse_agent_profile(PLAIN_SKILL) is None

    def test_missing_enabled_toolsets_raises(self):
        with pytest.raises(AgentProfileError):
            parse_agent_profile(MALFORMED_MISSING_TOOLSETS)

    def test_enabled_toolsets_not_a_list_raises(self):
        with pytest.raises(AgentProfileError):
            parse_agent_profile(MALFORMED_NOT_A_LIST)

    def test_disabled_toolsets_not_a_list_raises(self):
        with pytest.raises(AgentProfileError):
            parse_agent_profile(MALFORMED_BAD_DISABLED)

    def test_output_schema_not_a_dict_raises(self):
        with pytest.raises(AgentProfileError):
            parse_agent_profile(MALFORMED_BAD_SCHEMA)

    def test_no_frontmatter_returns_none(self):
        assert parse_agent_profile("# Just a heading, no frontmatter") is None

    def test_not_a_string_returns_none(self):
        assert parse_agent_profile(None) is None  # type: ignore[arg-type]


class TestAgentProfileForInstalled:
    def test_finds_and_parses_installed_profile(self, tmp_path):
        skills_dir = tmp_path / "skills"
        rec_dir = skills_dir / "agent-profiles" / "researcher"
        rec_dir.mkdir(parents=True)
        (rec_dir / "SKILL.md").write_text(PROFILE_SKILL, encoding="utf-8")

        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            spec = agent_profile_for_installed("researcher")
        assert spec is not None
        assert spec.enabled_toolsets == ["web", "file"]

    def test_plain_skill_returns_none(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "misc" / "not-a-profile"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(PLAIN_SKILL, encoding="utf-8")
        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            assert agent_profile_for_installed("not-a-profile") is None

    def test_missing_skill_returns_none(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            assert agent_profile_for_installed("does-not-exist") is None

    def test_malformed_installed_profile_raises(self, tmp_path):
        skills_dir = tmp_path / "skills"
        rec_dir = skills_dir / "agent-profiles" / "broken"
        rec_dir.mkdir(parents=True)
        (rec_dir / "SKILL.md").write_text(MALFORMED_MISSING_TOOLSETS, encoding="utf-8")
        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            with pytest.raises(AgentProfileError):
                agent_profile_for_installed("broken")



MEMORY_PROFILE_SKILL = """---
name: memory-researcher
description: Research profile that may recall shared knowledge.
metadata:
  hermes:
    agent:
      enabled_toolsets: [web, file]
      memory_tools: [hindsight_recall]
---

# Memory researcher
"""


class TestAgentProfileMemoryTools:
    """metadata.hermes.agent.memory_tools -- opt-in, allow-listed provider tools.

    Children are built with skip_memory=True, which gates the external memory
    provider (agent_init.py). A profile opts back in by naming exactly which
    provider tools its children may call. Absent means "no provider", so the
    default stays the current behaviour.
    """

    def test_absent_memory_tools_is_none(self):
        spec = parse_agent_profile(PROFILE_SKILL)
        assert spec is not None
        assert spec.memory_tools is None

    def test_parses_memory_tools_allowlist(self):
        spec = parse_agent_profile(MEMORY_PROFILE_SKILL)
        assert spec is not None
        assert spec.memory_tools == ["hindsight_recall"]

    def test_memory_tools_not_a_list_raises(self):
        bad = MEMORY_PROFILE_SKILL.replace(
            "memory_tools: [hindsight_recall]", "memory_tools: hindsight_recall"
        )
        with pytest.raises(AgentProfileError):
            parse_agent_profile(bad)

    def test_empty_memory_tools_raises(self):
        """An empty list is a typo, not "no memory" -- absent means that."""
        bad = MEMORY_PROFILE_SKILL.replace(
            "memory_tools: [hindsight_recall]", "memory_tools: []"
        )
        with pytest.raises(AgentProfileError):
            parse_agent_profile(bad)

    def test_memory_tools_coerced_to_strings(self):
        odd = MEMORY_PROFILE_SKILL.replace(
            "memory_tools: [hindsight_recall]", "memory_tools: [hindsight_recall, 7]"
        )
        spec = parse_agent_profile(odd)
        assert spec is not None
        assert spec.memory_tools == ["hindsight_recall", "7"]
