"""Tests du backend LLM : sélection automatique du modèle et diagnostic."""

from __future__ import annotations

import httpx
import pytest

from app.ai.llm import (
    OllamaProvider,
    _parameter_billions,
    get_llm_provider,
    reset_llm_provider,
    score_model,
)
from app.core.config import reload_settings
from app.core.exceptions import ConfigurationError, LLMUnavailableError, ModelNotFoundError


def installed(*entries: tuple[str, str]) -> list[dict]:
    return [
        {"name": name, "details": {"parameter_size": size}} for name, size in entries
    ]


# ----------------------------------------------------------- taille lisible
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("7B", 7.0), ("7.6B", 7.6), ("500M", 0.5), ("", 0.0), ("weird", 0.0)],
)
def test_parameter_size_parsing(raw: str, expected: float) -> None:
    assert _parameter_billions({"parameter_size": raw}) == expected


# ------------------------------------------------------------- notation
@pytest.mark.parametrize(
    "name",
    ["nomic-embed-text", "bge-reranker", "codellama:7b", "llava:7b", "llama-guard3"],
)
def test_non_conversational_models_are_excluded(name: str) -> None:
    assert score_model(name, {"parameter_size": "7B"}, budget_gb=16) < 0


def test_instruct_models_are_preferred() -> None:
    plain = score_model("qwen2.5:7b", {"parameter_size": "7B"}, budget_gb=24)
    instruct = score_model("qwen2.5:7b-instruct", {"parameter_size": "7B"}, budget_gb=24)
    assert instruct > plain


def test_a_model_too_large_for_the_machine_is_penalised() -> None:
    """Un 70B sur 8 Go tournerait, mais si lentement que c'est inutilisable."""
    small = score_model("qwen2.5:3b-instruct", {"parameter_size": "3B"}, budget_gb=8)
    huge = score_model("qwen2.5:70b-instruct", {"parameter_size": "70B"}, budget_gb=8)
    assert small > huge


def test_bigger_is_preferred_when_the_machine_allows_it() -> None:
    small = score_model("qwen2.5:1.5b-instruct", {"parameter_size": "1.5B"}, budget_gb=64)
    large = score_model("qwen2.5:7b-instruct", {"parameter_size": "7B"}, budget_gb=64)
    assert large > small


# ------------------------------------------------- sélection automatique
def _provider_with(monkeypatch, models: list[dict], configured: str = "") -> OllamaProvider:
    monkeypatch.setenv("LLM_MODEL", configured)
    provider = OllamaProvider(reload_settings())
    monkeypatch.setattr(provider, "_installed", lambda: models)
    return provider


def test_configured_model_is_never_overridden(monkeypatch) -> None:
    provider = _provider_with(
        monkeypatch, installed(("qwen2.5:7b-instruct", "7B")), configured="my-model"
    )
    assert provider.resolve_model() == "my-model"
    assert provider.auto_selected is False


def test_a_single_installed_model_is_picked_automatically(monkeypatch) -> None:
    provider = _provider_with(monkeypatch, installed(("qwen2.5:3b-instruct", "3B")))
    assert provider.resolve_model() == "qwen2.5:3b-instruct"
    assert provider.auto_selected is True


def test_selection_skips_embedding_models(monkeypatch) -> None:
    provider = _provider_with(
        monkeypatch,
        installed(("nomic-embed-text", "137M"), ("llama3.2:3b-instruct", "3B")),
    )
    assert provider.resolve_model() == "llama3.2:3b-instruct"


def test_selection_prefers_an_instruct_model(monkeypatch) -> None:
    provider = _provider_with(
        monkeypatch, installed(("mistral:7b", "7B"), ("mistral:7b-instruct", "7B"))
    )
    assert provider.resolve_model() == "mistral:7b-instruct"


def test_nothing_installed_means_no_model(monkeypatch) -> None:
    provider = _provider_with(monkeypatch, [])
    assert provider.resolve_model() == ""
    assert provider.auto_selected is False


def test_only_unusable_models_means_no_model(monkeypatch) -> None:
    provider = _provider_with(monkeypatch, installed(("nomic-embed-text", "137M")))
    assert provider.resolve_model() == ""


def test_missing_model_raises_an_actionable_error(monkeypatch) -> None:
    provider = _provider_with(monkeypatch, [])
    with pytest.raises(ConfigurationError) as excinfo:
        provider.generate([{"role": "user", "content": "hi"}])
    assert "ollama pull" in excinfo.value.user_message


def test_status_says_when_the_model_was_auto_selected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "")
    provider = OllamaProvider(reload_settings())

    def _fake_get(self, url, **kwargs):  # noqa: ANN001, ARG001
        return httpx.Response(
            200,
            json={"models": installed(("qwen2.5:3b-instruct", "3B"))},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", _fake_get)
    monkeypatch.setattr(
        OllamaProvider, "_installed", lambda self: installed(("qwen2.5:3b-instruct", "3B"))
    )
    status = provider.status()
    assert status.available is True
    assert status.model == "qwen2.5:3b-instruct"
    assert status.detail == "auto-selected"


def test_status_reports_an_unreachable_ollama(monkeypatch) -> None:
    def _refuse(self, url, **kwargs):  # noqa: ANN001, ARG001
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.Client, "get", _refuse)
    status = OllamaProvider(reload_settings()).status()
    assert status.available is False
    assert "ollama" in status.detail.lower()


# ------------------------------------------------------------- transport
def test_http_404_becomes_a_model_not_found_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "ghost")
    provider = OllamaProvider(reload_settings())

    def _not_found(self, url, **kwargs):  # noqa: ANN001, ARG001
        return httpx.Response(404, text="model not found", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", _not_found)
    with pytest.raises(ModelNotFoundError):
        provider.generate([{"role": "user", "content": "hi"}])


def test_connection_failure_becomes_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "some-model")
    provider = OllamaProvider(reload_settings())

    def _refuse(self, url, **kwargs):  # noqa: ANN001, ARG001
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.Client, "post", _refuse)
    with pytest.raises(LLMUnavailableError):
        provider.generate([{"role": "user", "content": "hi"}])


def test_provider_instance_is_reused() -> None:
    reset_llm_provider()
    assert get_llm_provider() is get_llm_provider()


def test_unknown_provider_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "telepathy")
    reset_llm_provider()
    with pytest.raises(ConfigurationError, match="unknown LLM provider"):
        get_llm_provider(reload_settings())
