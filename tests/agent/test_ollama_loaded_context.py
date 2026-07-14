"""Ollama loaded-context reconciliation.

Ollama sizes a model's real context window by free VRAM at load time —
often far below the GGUF trained max that /api/show reports — and the
OpenAI-compatible /v1 endpoint cannot change it (its request struct has no
options field, so per-request num_ctx is silently dropped). ``/api/ps`` is
the only source that reports the window in effect, and only after load.

Contract under test:
  - query_ollama_loaded_context() reads /api/ps for the model's effective
    window; None when unreachable / not loaded / malformed.
  - Selection-time resolution (get_model_context_length) keeps the trained
    max — the loaded window is transient state, not model metadata.
  - sync_ollama_loaded_context() resizes the compressor post-response and
    refreshes keep_alive via the native API (both impossible over /v1).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agent.model_metadata as mm
from agent.agent_runtime_helpers import sync_ollama_loaded_context
from agent.model_metadata import query_ollama_loaded_context


BASE_URL = "http://127.0.0.1:11434/v1"


def _ps_response(models):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"models": models}
    return resp


def _client_returning(resp):
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get.return_value = resp
    return client


@pytest.fixture(autouse=True)
def _clear_ps_cache():
    mm._OLLAMA_PS_CACHE.clear()
    yield
    mm._OLLAMA_PS_CACHE.clear()


class TestQueryOllamaLoadedContext:
    def test_returns_loaded_window_for_exact_tag(self):
        resp = _ps_response(
            [{"name": "gemma4:31b", "context_length": 32768, "size_vram": 20282741882}]
        )
        with patch("httpx.Client", return_value=_client_returning(resp)):
            assert query_ollama_loaded_context("gemma4:31b", BASE_URL) == 32768

    def test_tagless_config_matches_any_loaded_tag(self):
        resp = _ps_response([{"name": "gemma4:latest", "context_length": 16384}])
        with patch("httpx.Client", return_value=_client_returning(resp)):
            assert query_ollama_loaded_context("gemma4", BASE_URL) == 16384

    def test_different_tag_does_not_match(self):
        resp = _ps_response([{"name": "gemma4:31b", "context_length": 32768}])
        with patch("httpx.Client", return_value=_client_returning(resp)):
            assert query_ollama_loaded_context("gemma4:9b", BASE_URL) is None

    def test_not_loaded_returns_none(self):
        with patch("httpx.Client", return_value=_client_returning(_ps_response([]))):
            assert query_ollama_loaded_context("gemma4:31b", BASE_URL) is None

    def test_unreachable_returns_none(self):
        with patch("httpx.Client", side_effect=ConnectionError("refused")):
            assert query_ollama_loaded_context("gemma4:31b", BASE_URL) is None

    def test_non_200_returns_none(self):
        resp = MagicMock()
        resp.status_code = 404
        with patch("httpx.Client", return_value=_client_returning(resp)):
            assert query_ollama_loaded_context("gemma4:31b", BASE_URL) is None

    def test_cached_result_skips_second_probe(self):
        resp = _ps_response([{"name": "gemma4:31b", "context_length": 32768}])
        client = _client_returning(resp)
        with patch("httpx.Client", return_value=client) as factory:
            assert query_ollama_loaded_context("gemma4:31b", BASE_URL) == 32768
            assert query_ollama_loaded_context("gemma4:31b", BASE_URL) == 32768
            assert factory.call_count == 1

    def test_use_cache_false_reprobes(self):
        resp = _ps_response([{"name": "gemma4:31b", "context_length": 32768}])
        client = _client_returning(resp)
        with patch("httpx.Client", return_value=client) as factory:
            query_ollama_loaded_context("gemma4:31b", BASE_URL)
            query_ollama_loaded_context("gemma4:31b", BASE_URL, use_cache=False)
            assert factory.call_count == 2


class TestSelectionResolutionUnaffected:
    def test_get_model_context_length_ignores_loaded_window(self):
        """Selection surfaces show the trained max, not the transient loaded
        window — feeding /api/ps into resolution would also fail agent init's
        minimum-context check for a model resident with a small window."""
        with (
            patch.object(mm, "query_ollama_loaded_context") as ps_probe,
            patch.object(mm, "_query_ollama_api_show", return_value=262144),
            patch.object(mm, "_skip_persistent_context_cache", return_value=True),
        ):
            ctx = mm.get_model_context_length(
                "gemma4:31b", base_url=BASE_URL, provider="ollama"
            )
        assert ctx == 262144
        ps_probe.assert_not_called()


def _ollama_agent(context_length=262144, config_ctx=None, keep_alive=None):
    agent = MagicMock()
    agent.provider = "ollama"
    agent.model = "gemma4:31b"
    agent.base_url = BASE_URL
    agent.api_key = "dummy-test-key"
    agent.api_mode = "chat_completions"
    agent._config_context_length = config_ctx
    agent._ollama_keep_alive = keep_alive
    agent.context_compressor.context_length = context_length
    return agent


class TestSyncOllamaLoadedContext:
    def test_resizes_compressor_to_loaded_window(self):
        agent = _ollama_agent()
        with patch(
            "agent.model_metadata.query_ollama_loaded_context", return_value=32768
        ):
            sync_ollama_loaded_context(agent)
        agent.context_compressor.update_model.assert_called_once()
        assert (
            agent.context_compressor.update_model.call_args.kwargs["context_length"]
            == 32768
        )

    def test_noop_when_window_already_matches(self):
        agent = _ollama_agent(context_length=32768)
        with patch(
            "agent.model_metadata.query_ollama_loaded_context", return_value=32768
        ):
            sync_ollama_loaded_context(agent)
        agent.context_compressor.update_model.assert_not_called()

    def test_noop_for_non_ollama_provider(self):
        agent = _ollama_agent()
        agent.provider = "anthropic"
        with patch(
            "agent.model_metadata.query_ollama_loaded_context", return_value=32768
        ) as probe:
            sync_ollama_loaded_context(agent)
        probe.assert_not_called()
        agent.context_compressor.update_model.assert_not_called()

    def test_explicit_config_context_length_wins(self):
        agent = _ollama_agent(config_ctx=65536)
        with patch(
            "agent.model_metadata.query_ollama_loaded_context", return_value=32768
        ) as probe:
            sync_ollama_loaded_context(agent)
        probe.assert_not_called()
        agent.context_compressor.update_model.assert_not_called()

    def test_probe_failure_leaves_compressor_alone(self):
        agent = _ollama_agent()
        with patch(
            "agent.model_metadata.query_ollama_loaded_context", return_value=None
        ):
            sync_ollama_loaded_context(agent)
        agent.context_compressor.update_model.assert_not_called()

    def test_keep_alive_refreshed_via_native_api(self):
        agent = _ollama_agent(keep_alive="30m")
        with (
            patch(
                "agent.model_metadata.query_ollama_loaded_context", return_value=None
            ),
            patch(
                "agent.agent_runtime_helpers._refresh_ollama_keep_alive"
            ) as refresh,
        ):
            sync_ollama_loaded_context(agent)
        refresh.assert_called_once_with(
            "gemma4:31b", BASE_URL, "dummy-test-key", "30m"
        )

    def test_no_keep_alive_config_no_refresh(self):
        agent = _ollama_agent()
        with (
            patch(
                "agent.model_metadata.query_ollama_loaded_context", return_value=None
            ),
            patch(
                "agent.agent_runtime_helpers._refresh_ollama_keep_alive"
            ) as refresh,
        ):
            sync_ollama_loaded_context(agent)
        refresh.assert_not_called()
