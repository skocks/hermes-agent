"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled, Ollama endpoint → top-level
    reasoning_effort="none" (Ollama /v1/chat/completions ignores
    think=False — ollama#14820) + extra_body.think = False for /api/chat
    and proxies
  - reasoning_config disabled, non-Ollama endpoint → generic
    extra_body.reasoning={"enabled": False}. Some chat templates (e.g.
    Qwen3.8 served by vLLM/TabbyAPI) validate reasoning_effort against a
    fixed allow-list with no "off" sentinel and 400 on the literal
    "none" that Ollama needs (#68210) — the disable signal has to be the
    provider-neutral object, not the Ollama-only string.
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK/Qwen3.8 expect; unset
    omits it so the endpoint's server default applies)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _is_ollama_endpoint(base_url: str | None, ollama_num_ctx: int | None) -> bool:
    """Detect an actual Ollama backend among the many providers sharing
    provider="custom" (vLLM, llama.cpp, GLM/ARK, TabbyAPI, generic local
    OpenAI-compatible servers, …).

    Mirrors the detection ``run_agent.py::_is_ollama_glm_backend`` already
    uses for a different Ollama-only quirk: port 11434 (Ollama's default)
    or "ollama" in the base URL. Also treats an explicitly configured
    ``ollama_num_ctx`` as a strong signal — that knob only exists for
    Ollama. Deliberately narrow so generic local/private endpoints
    (vLLM, TabbyAPI, llama.cpp, Tailscale boxes) never match.
    """
    if ollama_num_ctx:
        return True
    base_url_lower = (base_url or "").lower()
    return "ollama" in base_url_lower or ":11434" in base_url_lower


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        base_url: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled  → Ollama: extra_body.think = False (its thinking-off
        #     flag) + top-level reasoning_effort="none" (see module
        #     docstring). Everyone else: generic extra_body.reasoning =
        #     {"enabled": False} — the same provider-neutral shape used for
        #     OpenRouter/Nous, which non-Ollama chat templates already
        #     resolve to their real enable_thinking toggle.
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                if _is_ollama_endpoint(base_url, ollama_num_ctx):
                    # Ollama's /v1/chat/completions silently ignores
                    # extra_body.think (only /api/chat honours it —
                    # ollama#14820) but respects the top-level
                    # reasoning_effort field, so both are needed to
                    # actually stop a thinking-capable model from
                    # reasoning (#25758).
                    top_level["reasoning_effort"] = "none"
                    extra_body["think"] = False
                else:
                    extra_body["reasoning"] = {"enabled": False}
            elif _effort:
                top_level["reasoning_effort"] = _effort

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
