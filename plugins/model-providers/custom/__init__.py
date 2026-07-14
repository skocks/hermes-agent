"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", plus the first-class
"ollama" provider (routed here by alias), and OpenAI-compatible reasoning
endpoints (GLM-5.2 on Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - reasoning_config disabled → reasoning_effort "none" on Ollama thinking
    models (its /v1 disable switch), extra_body.think = False elsewhere
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)

Ollama's OpenAI-compatible endpoint has no options passthrough: num_ctx
and keep_alive in the request body are silently dropped, so this profile
does not emit them. The context window is server-controlled (reconciled
post-load from /api/ps); keep_alive is refreshed via the native API.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_supports_thinking: bool | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled + Ollama thinking model → TOP-LEVEL reasoning_effort
        #     "none". Ollama's /v1 handler parses only reasoning_effort and
        #     maps "none" to think=false internally; an extra_body ``think``
        #     is an unknown field Go silently drops (verified live: think
        #     had no effect, effort "none" suppressed reasoning).
        #   - disabled elsewhere → extra_body.think = False (legacy shape for
        #     non-Ollama endpoints that do parse it, e.g. ARK).
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # Effort levels that only exist on the OpenAI/Codex scale are mapped
        # to the nearest level these endpoints accept — Ollama validates
        # against high/medium/low/max/none and 400s on "xhigh"/"minimal".
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if ollama_supports_thinking is False:
                # The Ollama server reports this model cannot think: any
                # reasoning_effort 400s ('"hermes3:8b" does not support
                # thinking') and a think flag is at best a no-op. Emit
                # nothing — a session's effort dial carried over from a
                # thinking model must not brick the chat. None means
                # unknown/non-Ollama and changes nothing.
                pass
            elif _effort == "none" or _enabled is False:
                if ollama_supports_thinking is True:
                    top_level["reasoning_effort"] = "none"
                else:
                    extra_body["think"] = False
            elif _effort:
                _aliases = {"xhigh": "max", "minimal": "low"}
                top_level["reasoning_effort"] = _aliases.get(_effort, _effort)

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
