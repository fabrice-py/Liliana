"""Reconnaissance vocale (Speech-to-Text).

Backend par défaut : faster-whisper, 100 % local. Le modèle est chargé
paresseusement au premier usage et gardé en mémoire (le chargement coûte
plusieurs secondes, la transcription elle-même est rapide).
"""

from __future__ import annotations

import io
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConfigurationError,
    EmptyTranscriptionError,
    STTError,
    STTUnavailableError,
)
from app.core.hardware import resolve_stt_device
from app.core.logger import get_logger
from app.language.languages import LANGUAGES, whisper_code

logger = get_logger(__name__)

#: Codes ISO des langues que Liliana accepte en entrée (cf. §5).
SUPPORTED_WHISPER_CODES: tuple[str, ...] = tuple(
    sorted({language.whisper_code for language in LANGUAGES.values()})
)


@dataclass(slots=True)
class Transcription:
    text: str
    language: str          # code ISO renvoyé par le modèle ("en", "de", "fr")
    language_probability: float
    duration: float        # durée de l'audio, en secondes
    elapsed: float         # temps de transcription, en secondes
    segments: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "duration": round(self.duration, 2),
            "elapsed": round(self.elapsed, 2),
        }


class STTProvider(ABC):
    """Interface de reconnaissance vocale."""

    name = "abstract"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @abstractmethod
    def transcribe(self, audio: bytes, language: str | None = None) -> Transcription:
        """Transcrit un fichier audio (n'importe quel format lisible par ffmpeg)."""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(disponible, détail) — sans charger le modèle."""

    def warmup(self) -> None:  # pragma: no cover - dépend du modèle installé
        """Précharge le modèle pour éviter la latence du premier tour."""


class FasterWhisperSTT(STTProvider):
    """faster-whisper (CTranslate2). Local, rapide, CPU ou CUDA."""

    name = "faster-whisper"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self._model: Any | None = None
        self._lock = threading.Lock()
        self.device, self.compute_type = resolve_stt_device(
            self.settings.stt_device, self.settings.stt_compute_type
        )

    # ------------------------------------------------------------- interne
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise STTUnavailableError(f"faster-whisper is not installed: {exc}") from exc

            logger.info(
                "Chargement du modèle Whisper '%s' (device=%s, compute=%s)…",
                self.settings.stt_model,
                self.device,
                self.compute_type,
            )
            started = time.perf_counter()
            try:
                self._model = WhisperModel(
                    self.settings.stt_model,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(self.settings.models_dir / "whisper"),
                )
            except Exception as exc:  # noqa: BLE001 - erreurs très hétérogènes
                raise STTError(
                    f"cannot load Whisper model '{self.settings.stt_model}': {exc}",
                    user_message=(
                        f"Liliana could not load the speech model "
                        f"'{self.settings.stt_model}'. If this is the first run, "
                        "it needs to be downloaded once — check your Internet "
                        "connection, or pick a smaller STT_MODEL (tiny, base)."
                    ),
                ) from exc
            logger.info("Modèle Whisper chargé en %.1f s", time.perf_counter() - started)
            return self._model

    # -------------------------------------------------------------- public
    def transcribe(self, audio: bytes, language: str | None = None) -> Transcription:
        if not audio:
            raise EmptyTranscriptionError("empty audio payload")

        model = self._load()
        # En mode apprentissage, on privilégie la langue cible (cf. §5) ; sinon
        # Whisper détecte lui-même parmi les langues supportées.
        forced_code = whisper_code(language) if language else None

        started = time.perf_counter()
        try:
            segments, info = model.transcribe(
                io.BytesIO(audio),
                language=forced_code,
                beam_size=self.settings.stt_beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
            )
            collected = [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in segments
            ]
        except EmptyTranscriptionError:
            raise
        except Exception as exc:  # noqa: BLE001 - PyAV/CTranslate2 lèvent large
            raise STTError(
                f"transcription failed: {exc}",
                user_message=(
                    "Liliana could not decode your recording. Try again, and "
                    "check that your microphone is working."
                ),
            ) from exc

        text = " ".join(segment["text"].strip() for segment in collected).strip()
        if not text:
            raise EmptyTranscriptionError("no speech detected in audio")

        return Transcription(
            text=text,
            language=getattr(info, "language", forced_code or "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            elapsed=time.perf_counter() - started,
            segments=collected,
        )

    def is_available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, STTUnavailableError.user_message
        loaded = " (model loaded)" if self._model is not None else ""
        return True, f"faster-whisper on {self.device}/{self.compute_type}{loaded}"

    def warmup(self) -> None:  # pragma: no cover - dépend du modèle installé
        try:
            self._load()
        except (STTError, STTUnavailableError) as exc:
            logger.warning("Préchargement STT impossible : %s", exc)


_PROVIDERS: dict[str, type[STTProvider]] = {"faster-whisper": FasterWhisperSTT}

_cached_provider: STTProvider | None = None
_cached_key: tuple[str, str] | None = None


def get_stt_provider(settings: Settings | None = None) -> STTProvider:
    """Retourne le moteur STT configuré (instance réutilisée, modèle en cache)."""
    global _cached_provider, _cached_key

    settings = settings or get_settings()
    key = (settings.stt_provider, settings.stt_model)
    if _cached_provider is not None and _cached_key == key:
        return _cached_provider

    provider_class = _PROVIDERS.get(settings.stt_provider.lower())
    if provider_class is None:
        raise ConfigurationError(
            f"unknown STT provider: {settings.stt_provider}",
            user_message=(
                f"Unknown speech-to-text provider '{settings.stt_provider}'. "
                f"Supported: {', '.join(sorted(_PROVIDERS))}."
            ),
        )

    _cached_provider = provider_class(settings)
    _cached_key = key
    return _cached_provider


def reset_stt_provider() -> None:
    global _cached_provider, _cached_key
    _cached_provider = None
    _cached_key = None
