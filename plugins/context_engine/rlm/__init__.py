"""RLM context engine plugin — registration entry point.

Loaded by plugins/context_engine/__init__.py's discover/load machinery
when `context.engine: rlm` is set in config.yaml. See engine.py for the
actual mechanism and plugin.yaml for the description.
"""

from __future__ import annotations

from .engine import RLMContextEngine


def register(ctx) -> None:
    ctx.register_context_engine(RLMContextEngine())
