"""Agent profiles: shaped delegate_task children, layered on skills.

Round 24 (RLM audit): delegate_task already spawns real isolated
subagents — parallel, background, role leaf/orchestrator, spawn-depth
limits, output_schema validation, tasks[] fan-out, live action=list/
steer/stop. What it never had was a way to say WHICH SHAPE of agent to
spawn: every child inherited the parent's full tool surface, because
_build_child_agent's `toolsets` argument was hardcoded to None at the
delegate_task call site — "the model cannot choose or narrow them (no
model-facing toolsets arg)", a deliberate prior decision against letting
the model specify an arbitrary raw toolset list per call.

An "agent profile" is NOT a new object type, following tools/blueprints.py's
exact design argument for the same problem one level up (automations):
it's an ordinary skill (a SKILL.md the agent loads) that additionally
declares a tool shape in its frontmatter:

    metadata:
      hermes:
        agent:
          enabled_toolsets: [web, file]   # required, non-empty
          disabled_toolsets: [terminal]    # optional
          system_prompt_fragment: "..."    # optional, appended to the child's prompt
          output_schema: {...}             # optional default when the caller omits one

Because a profile is just a skill, it flows through the entire existing
skills-hub pipeline for free — search, inspect, quarantine, security scan,
install, lock-file provenance, audit log, taps, the centralized index, and
`hermes skills publish` for sharing. No new store, no new transport. This
extends the existing model-facing surface in a controlled, curated way
(a small set of profiles an operator/skill-author vetted) rather than
reopening the raw-toolsets-list door the original design closed — the
model picks a NAME, not an arbitrary tool list.

  * ``parse_agent_profile(skill_md_text)``   -> AgentProfileSpec | None
  * ``agent_profile_for_installed(name)``    -> AgentProfileSpec | None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "AgentProfileSpec",
    "parse_agent_profile",
    "agent_profile_for_installed",
    "AgentProfileError",
]


class AgentProfileError(ValueError):
    """Raised when an agent profile block is present but malformed."""


@dataclass
class AgentProfileSpec:
    """Parsed ``metadata.hermes.agent`` tool-shape spec for a skill."""

    skill_name: str
    enabled_toolsets: List[str]
    disabled_toolsets: Optional[List[str]] = None
    system_prompt_fragment: Optional[str] = None
    output_schema: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    memory_tools: Optional[List[str]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _split_frontmatter(text: str) -> Optional[Dict[str, Any]]:
    """Return the parsed YAML frontmatter mapping, or None if absent/invalid.

    Deliberately duplicated from tools/blueprints.py's identical helper
    rather than imported: both are small, standalone, and importing across
    these two sibling modules for four lines isn't worth the coupling.
    """
    if not isinstance(text, str):
        return None
    stripped = text.lstrip("﻿").lstrip()  # BOM is not whitespace; strip explicitly
    if not stripped.startswith("---"):
        return None
    after_open = stripped[3:]
    end = after_open.find("\n---")
    if end == -1:
        return None
    fm_text = after_open[:end]
    try:
        import yaml

        data = yaml.safe_load(fm_text)
    except Exception as e:  # pragma: no cover - malformed YAML
        logger.debug("agent_profiles: frontmatter YAML parse failed: %s", e)
        return None
    return data if isinstance(data, dict) else None


def parse_agent_profile(skill_md_text: str) -> Optional[AgentProfileSpec]:
    """Extract an AgentProfileSpec from a SKILL.md string, or None if not one.

    A skill is an agent profile iff ``metadata.hermes.agent`` is a mapping
    containing a non-empty ``enabled_toolsets`` list. Raises
    AgentProfileError if the block exists but is structurally invalid, so
    a typo surfaces instead of silently no-op'ing (same posture as
    parse_blueprint).
    """
    fm = _split_frontmatter(skill_md_text)
    if not fm:
        return None

    name = str(fm.get("name", "")).strip()

    meta = fm.get("metadata")
    hermes = meta.get("hermes") if isinstance(meta, dict) else None
    agent_block = hermes.get("agent") if isinstance(hermes, dict) else None
    if agent_block is None:
        return None
    if not isinstance(agent_block, dict):
        raise AgentProfileError("metadata.hermes.agent must be a mapping")

    toolsets = agent_block.get("enabled_toolsets")
    if not isinstance(toolsets, list) or not toolsets:
        raise AgentProfileError(
            "agent.enabled_toolsets is required and must be a non-empty list"
        )

    disabled = agent_block.get("disabled_toolsets")
    if disabled is not None and not isinstance(disabled, list):
        raise AgentProfileError("agent.disabled_toolsets must be a list when present")

    fragment = agent_block.get("system_prompt_fragment")
    if fragment is not None:
        fragment = str(fragment)

    schema = agent_block.get("output_schema")
    if schema is not None and not isinstance(schema, dict):
        raise AgentProfileError("agent.output_schema must be an object when present")

    # Children are built with skip_memory=True, which gates the external
    # memory provider off entirely (agent_init.py). A profile opts back in by
    # naming exactly which provider tools its children may call -- an explicit
    # allow-list rather than an on/off flag, because the provider exposes both
    # read (hindsight_recall) and write (hindsight_retain) tools and those are
    # very different grants. Absent means "no provider", so the default is
    # unchanged. An empty list is a typo, not a way to say "none".
    memory_tools = agent_block.get("memory_tools")
    if memory_tools is not None:
        if not isinstance(memory_tools, list) or not memory_tools:
            raise AgentProfileError(
                "agent.memory_tools must be a non-empty list when present; "
                "omit the key entirely to give children no memory provider"
            )

    description = fm.get("description")
    description = str(description).strip() if description else None

    return AgentProfileSpec(
        skill_name=name,
        enabled_toolsets=[str(t) for t in toolsets],
        disabled_toolsets=[str(t) for t in disabled] if disabled else None,
        system_prompt_fragment=fragment,
        output_schema=schema,
        description=description,
        memory_tools=[str(t) for t in memory_tools] if memory_tools else None,
        raw=agent_block,
    )


def agent_profile_for_installed(skill_name: str) -> Optional[AgentProfileSpec]:
    """Locate an installed skill's SKILL.md and parse its agent-profile block.

    Searches the standard skills tree for ``<skill_name>/SKILL.md`` (same
    glob shape as blueprint_spec_for_installed). Returns None if the skill
    isn't found or isn't an agent profile -- callers distinguish "no such
    skill" from "not a profile" only if they need to; delegate_task treats
    both as "profile not found" for a clear tool-error either way.
    """
    try:
        from tools.skills_hub import SKILLS_DIR
    except Exception:  # pragma: no cover - import guard
        return None

    base = Path(SKILLS_DIR)
    candidates = list(base.glob(f"**/{skill_name}/SKILL.md"))
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        spec = parse_agent_profile(text)
        if spec is not None:
            if not spec.skill_name:
                spec.skill_name = skill_name
            return spec
    return None
