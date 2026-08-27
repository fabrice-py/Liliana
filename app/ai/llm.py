"""Abstraction du modèle de langage.

Le reste de l'application ne connaît que :class:`LLMProvider`. Changer de
backend (Ollama, llama.cpp, autre) se fait ici et dans la configuration, sans
toucher au code métier (cf. cahier des charges §21).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConfigurationError,
    LLMError,
    LLMUnavailableError,
    ModelNotFoundError,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


#: Familles de modèles qui tiennent correctement une conversation multilingue et
#: respectent un format JSON. Score plus élevé = préféré à l'auto-sélection.
_FAMILY_PREFERENCE: tuple[tuple[str, int], ...] = (
    ("qwen", 40), ("llama", 35), ("mistral", 30), ("gemma", 28),
    ("phi", 22), ("granite", 20), ("command-r", 18), ("aya", 25),
)

#: Modèles à ne jamais choisir automatiquement : ils ne servent pas à discuter.
_EXCLUDED_MODEL_MARKERS: tuple[str, ...] = (
    "embed", "rerank", "code", "vision", "moondream", "llava", "guard", "math",
)


def _parameter_billions(details: dict[str, Any]) -> float:
    """Taille du modèle en milliards de paramètres, 0.0 si inconnue."""
    raw = str(details.get("parameter_size") or "").strip().upper()
    if not raw:
        return 0.0
    try:
        if raw.endswith("B"):
            return float(raw[:-1])
        if raw.endswith("M"):
            return float(raw[:-1]) / 1000
    except ValueError:
        pass
    return 0.0


def score_model(name: str, details: dict[str, Any], budget_gb: float) -> float:
    """Note un modèle installé pour l'auto-sélection. Négatif = à écarter."""
    lowered = name.lower()
    if any(marker in lowered for marker in _EXCLUDED_MODEL_MARKERS):
        return -1.0

    score = 0.0
    for family, bonus in _FAMILY_PREFERENCE:
        if family in lowered:
            score += bonus
            break
    if "instruct" in lowered or "chat" in lowered or "-it" in lowered:
        score += 25

    # Le plus gros modèle qui tient confortablement en mémoire. Au-delà d'un
    # tiers du budget, la latence devient pénible sur une machine sans GPU.
    billions = _parameter_billions(details)
    if billions:
        comfortable = max(1.0, budget_gb / 3.0)
        score += 20 * min(billions / comfortable, 1.0)
        if billions > comfortable * 2:
            score -= 15  # trop gros : ça tournera, mais très lentement
    return score


@dataclass(slots=True)
class LLMStatus:
    """État du backend LLM, affiché dans l'interface."""

    available: bool
    provider: str
    model: str
    detail: str = ""
    installed_models: tuple[str, ...] = ()


class LLMProvider(ABC):
    """Interface commune à tous les backends de génération."""

    name = "abstract"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Génère une réponse complète à partir d'un historique de messages."""

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Iterator[str]:
        """Génère la réponse par fragments (pour réduire la latence perçue)."""

    @abstractmethod
    def status(self) -> LLMStatus:
        """Vérifie que le backend et le modèle sont disponibles."""


class OllamaProvider(LLMProvider):
    """Backend Ollama (http://localhost:11434), 100 % local."""

    name = "ollama"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.llm_base_url.rstrip("/")
        self.model = self.settings.llm_model.strip()
        self.timeout = self.settings.llm_timeout
        #: Vrai quand le modèle a été choisi automatiquement faute de LLM_MODEL.
        self.auto_selected = False

    # -------------------------------------------------- sélection du modèle
    def _installed(self) -> list[dict[str, Any]]:
        """Modèles présents dans Ollama. Liste vide s'il est injoignable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError):
            return []
        return [model for model in models if isinstance(model, dict) and model.get("name")]

    def resolve_model(self) -> str:
        """Modèle à utiliser, choisi automatiquement si la configuration est vide.

        Devoir éditer ``.env`` avant la première phrase est le principal point de
        friction à l'installation. Si un seul modèle convenable est installé,
        autant s'en servir — le choix est journalisé et affiché dans l'interface,
        jamais silencieux.
        """
        if self.model:
            return self.model

        installed = self._installed()
        if not installed:
            return ""

        from app.core.hardware import detect_hardware

        hardware = detect_hardware()
        budget = hardware.vram_gb if (hardware.has_cuda and hardware.vram_gb) else hardware.total_ram_gb

        ranked = sorted(
            (
                (score_model(str(model["name"]), model.get("details") or {}, budget), str(model["name"]))
                for model in installed
            ),
            reverse=True,
        )
        best = next((name for score, name in ranked if score >= 0), "")
        if best:
            self.model = best
            self.auto_selected = True
            logger.info(
                "LLM_MODEL non renseigné : sélection automatique de '%s' "
                "parmi %d modèle(s) installé(s)",
                best,
                len(installed),
            )
        return best

    # ------------------------------------------------------------- interne
    def _require_model(self) -> str:
        if not self.resolve_model():
            raise ConfigurationError(
                "LLM_MODEL is empty",
                user_message=(
                    "No language model is configured. Install one with "
                    "`ollama pull <model>` and set LLM_MODEL in your .env file "
                    "(run `python scripts/check_env.py` for suggestions)."
                ),
            )
        return self.model

    def _payload(
        self, messages: list[dict[str, str]], temperature: float | None, json_mode: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._require_model(),
            "messages": messages,
            "options": {
                "temperature": (
                    self.settings.llm_temperature if temperature is None else temperature
                )
            },
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    @staticmethod
    def _translate_http_error(exc: httpx.HTTPStatusError) -> LLMError:
        body = exc.response.text.lower()
        if exc.response.status_code == 404 or "not found" in body:
            return ModelNotFoundError(f"model not found: {exc.response.text[:200]}")
        return LLMError(f"Ollama returned HTTP {exc.response.status_code}")

    # -------------------------------------------------------------- public
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        payload = self._payload(messages, temperature, json_mode)
        payload["stream"] = False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise self._translate_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"invalid JSON envelope from Ollama: {exc}") from exc

        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise LLMError("Ollama returned an empty answer")
        return content

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Iterator[str]:
        payload = self._payload(messages, temperature, json_mode)
        payload["stream"] = True
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = (chunk.get("message") or {}).get("content", "")
                        if piece:
                            yield piece
                        if chunk.get("done"):
                            break
        except httpx.HTTPStatusError as exc:
            raise self._translate_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc

    def status(self) -> LLMStatus:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                tags = response.json().get("models", [])
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
            logger.warning("Ollama injoignable sur %s (%s)", self.base_url, exc)
            return LLMStatus(
                available=False,
                provider=self.name,
                model=self.model,
                detail=LLMUnavailableError.user_message,
            )

        installed = tuple(str(tag.get("name", "")) for tag in tags)
        if not self.resolve_model():
            return LLMStatus(
                available=False,
                provider=self.name,
                model="",
                detail=(
                    "Ollama is running but has no usable conversational model. "
                    "Install one, for example `ollama pull qwen2.5:3b-instruct`."
                ),
                installed_models=installed,
            )
        # Ollama tolère `llama3` pour `llama3:latest` : on compare sans le tag.
        base_names = {name.split(":", 1)[0] for name in installed}
        if self.model not in installed and self.model.split(":", 1)[0] not in base_names:
            return LLMStatus(
                available=False,
                provider=self.name,
                model=self.model,
                detail=f"Model '{self.model}' is not installed. Run `ollama pull {self.model}`.",
                installed_models=installed,
            )
        return LLMStatus(
            available=True,
            provider=self.name,
            model=self.model,
            detail="auto-selected" if self.auto_selected else "ready",
            installed_models=installed,
        )


_PROVIDERS: dict[str, type[LLMProvider]] = {"ollama": OllamaProvider}

_cached_provider: LLMProvider | None = None
_cached_key: tuple[str, str, str] | None = None


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Retourne le backend LLM configuré (instance réutilisée)."""
    global _cached_provider, _cached_key

    settings = settings or get_settings()
    key = (settings.llm_provider, settings.llm_model, settings.llm_base_url)
    if _cached_provider is not None and _cached_key == key:
        return _cached_provider

    provider_class = _PROVIDERS.get(settings.llm_provider.lower())
    if provider_class is None:
        raise ConfigurationError(
            f"unknown LLM provider: {settings.llm_provider}",
            user_message=(
                f"Unknown LLM provider '{settings.llm_provider}'. "
                f"Supported: {', '.join(sorted(_PROVIDERS))}."
            ),
        )

    _cached_provider = provider_class(settings)
    _cached_key = key
    return _cached_provider


def reset_llm_provider() -> None:
    """Vide le cache du provider (config modifiée, tests)."""
    global _cached_provider, _cached_key
    _cached_provider = None
    _cached_key = None
