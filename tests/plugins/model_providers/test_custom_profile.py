"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at — local Ollama, vLLM, llama.cpp, and hosted reasoning APIs like
GLM-5.2 on Volcengine ARK. Before #57601's salvage, ``CustomProfile`` emitted
nothing when reasoning was *enabled*, so a configured ``reasoning_effort``
was silently dropped for every custom endpoint.

These tests pin the wire-shape contract:
  - disabled            → extra_body.think = False (non-Ollama endpoints);
                          reasoning_effort "none" when the Ollama server
                          confirmed the model thinks (its /v1 disable switch)
  - enabled + effort    → top-level reasoning_effort (native OpenAI-compat
                          format GLM/ARK expect); OpenAI-only levels
                          (xhigh/minimal) map to the nearest accepted level
  - enabled + no effort → nothing emitted (endpoint's server default applies)
  - num_ctx/keep_alive  → never emitted (Ollama /v1 silently drops them; the
                          window is reconciled post-load from /api/ps)
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """Resolve the registered custom profile via the global registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    ``custom`` profile. Going through ``get_provider_profile`` keeps the test
    honest — if the registered class is ever downgraded to a plain
    ``ProviderProfile``, the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomReasoningWireShape:
    """``build_api_kwargs_extras`` produces the correct wire format."""

    def test_no_reasoning_config_emits_nothing(self, custom_profile):
        """Unset reasoning → omit everything so the endpoint's default applies."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_disabled_sends_think_false(self, custom_profile):
        """enabled=False → extra_body.think = False (Ollama thinking-off flag)."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → think=False, no effort."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {}

    @pytest.mark.parametrize(
        "effort", ["low", "medium", "high", "max"]
    )
    def test_enabled_effort_goes_top_level(self, custom_profile, effort):
        """enabled + effort → TOP-LEVEL reasoning_effort, passed through verbatim.

        GLM-5.2/ARK and OpenAI-compatible reasoning APIs read reasoning_effort
        as a top-level string, not nested in extra_body. ``max`` is GLM's
        native deep-reasoning level and must survive.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": effort}
        assert "reasoning_effort" not in eb
        assert "think" not in eb

    @pytest.mark.parametrize(
        ("effort", "expected"), [("xhigh", "max"), ("minimal", "low")]
    )
    def test_openai_only_efforts_map_to_endpoint_levels(
        self, custom_profile, effort, expected
    ):
        """Efforts that only exist on the OpenAI/Codex scale map to the
        nearest level these endpoints accept.

        Ollama validates reasoning_effort against high/medium/low/max/none
        and rejects "xhigh"/"minimal" with HTTP 400; GLM documents "high" and
        "max". Carrying a session's effort dial across a provider switch to a
        local model must not brick the chat.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": expected}
        assert "think" not in eb

    def test_enabled_without_effort_emits_nothing(self, custom_profile):
        """enabled but no effort → omit; do NOT force a level the user didn't pick."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_does_not_force_think_true_on_enable(self, custom_profile):
        """We must never send think=True on enable — it's Ollama-only and
        would 400 on GLM/vLLM endpoints that don't recognize it."""
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2"
        )
        assert eb.get("think") is not True


class TestCustomOllamaThinkingDisable:
    """Confirmed-thinking Ollama models disable via reasoning_effort 'none'."""

    def test_disabled_with_thinking_support_sends_effort_none(self, custom_profile):
        """ollama_supports_thinking=True → reasoning_effort 'none' top-level.

        Ollama's /v1 handler only parses reasoning_effort ('none' maps to
        think=false internally); an extra_body think flag is an unknown field
        Go silently drops. Verified live: think=False had no effect, effort
        'none' suppressed reasoning.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            ollama_supports_thinking=True,
            model="qwen3",
        )
        assert tl == {"reasoning_effort": "none"}
        assert eb == {}

    def test_effort_none_with_thinking_support_sends_effort_none(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
            ollama_supports_thinking=True,
            model="qwen3",
        )
        assert tl == {"reasoning_effort": "none"}
        assert eb == {}

    def test_no_thinking_support_emits_nothing(self, custom_profile):
        """ollama_supports_thinking=False → emit no reasoning fields at all.

        Ollama 400s on reasoning_effort (any value) for non-thinking models;
        a session's effort dial carried over from a thinking model must not
        brick the chat.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            ollama_supports_thinking=False,
            model="hermes3:8b",
        )
        assert eb == {}
        assert tl == {}


class TestCustomOllamaNoOptionsPassthrough:
    """Ollama /v1 silently drops options/keep_alive — never emit them."""

    def test_request_body_has_no_ollama_options(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="qwen3",
        )
        assert "options" not in eb
        assert "keep_alive" not in eb
        assert tl == {"reasoning_effort": "high"}
