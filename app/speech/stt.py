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
from collections.abc import Iterator
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
    #: Mots horodatés avec leur probabilité acoustique. Rempli uniquement quand
    #: la transcription est demandée avec ``word_timestamps=True`` (analyse de
    #: prononciation) : le calcul coûte du temps, inutile en conversation.
    words: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "duration": round(self.duration, 2),
            "elapsed": round(self.elapsed, 2),
        }


@dataclass(slots=True)
class TranscriptionEvent:
    """Étape d'une transcription en cours."""

    text: str                              # texte cumulé jusqu'ici
    is_final: bool = False
    transcription: Transcription | None = None  # rempli uniquement sur l'évènement final


class STTProvider(ABC):
    """Interface de reconnaissance vocale."""

    name = "abstract"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @abstractmethod
    def transcribe(
        self, audio: bytes, language: str | None = None, word_timestamps: bool = False
    ) -> Transcription:
        """Transcrit un fichier audio (n'importe quel format lisible par ffmpeg).

        ``word_timestamps`` demande le détail mot à mot avec la probabilité
        acoustique de chacun — la matière première de l'analyse de prononciation.
        """

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(disponible, détail) — sans charger le modèle."""

    def transcribe_stream(
        self, audio: bytes, language: str | None = None
    ) -> Iterator[TranscriptionEvent]:
        """Transcrit en émettant les segments au fur et à mesure.

        Implémentation par défaut : un seul évènement, final. Les backends
        capables de décoder par segments surchargent cette méthode.
        """
        transcription = self.transcribe(audio, language)
        yield TranscriptionEvent(
            text=transcription.text, is_final=True, transcription=transcription
        )

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
    def _decode(
        self, audio: bytes, language: str | None, word_timestamps: bool = False
    ) -> tuple[Any, Any]:
        """Lance le décodage. Retourne (générateur de segments, informations)."""
        if not audio:
            raise EmptyTranscriptionError("empty audio payload")

        model = self._load()
        # En mode apprentissage, on privilégie la langue cible (cf. §5) ; sinon
        # Whisper détecte lui-même parmi les langues supportées.
        forced_code = whisper_code(language) if language else None
        try:
            return model.transcribe(
                io.BytesIO(audio),
                language=forced_code,
                beam_size=self.settings.stt_beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
                word_timestamps=word_timestamps,
            )
        except Exception as exc:  # noqa: BLE001 - PyAV/CTranslate2 lèvent large
            raise STTError(
                f"transcription failed: {exc}",
                user_message=(
                    "Liliana could not decode your recording. Try again, and "
                    "check that your microphone is working."
                ),
            ) from exc

    def transcribe_stream(
        self,
        audio: bytes,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> Iterator[TranscriptionEvent]:
        """Émet le texte au fil du décodage, puis la transcription complète.

        Le générateur de segments de faster-whisper est paresseux : consommer
        segment par segment permet d'afficher les premiers mots pendant que la
        fin de la phrase est encore en cours de décodage.
        """
        started = time.perf_counter()
        forced_code = whisper_code(language) if language else None
        segments, info = self._decode(audio, language, word_timestamps)

        collected: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []
        try:
            for segment in segments:
                collected.append(
                    {"start": segment.start, "end": segment.end, "text": segment.text}
                )
                for word in getattr(segment, "words", None) or ():
                    words.append(
                        {
                            "word": word.word.strip(),
                            "start": float(word.start),
                            "end": float(word.end),
                            # Confiance acoustique du modèle sur ce mot précis :
                            # basse = « j'ai entendu quelque chose d'approchant ».
                            "probability": float(getattr(word, "probability", 0.0) or 0.0),
                        }
                    )
                partial = " ".join(item["text"].strip() for item in collected).strip()
                if partial:
                    yield TranscriptionEvent(text=partial)
        except Exception as exc:  # noqa: BLE001 - le décodage est paresseux : il peut échouer ici
            raise STTError(
                f"transcription failed while decoding: {exc}",
                user_message=(
                    "Liliana could not decode your recording. Try again, and "
                    "check that your microphone is working."
                ),
            ) from exc

        text = " ".join(item["text"].strip() for item in collected).strip()
        if not text:
            raise EmptyTranscriptionError("no speech detected in audio")

        yield TranscriptionEvent(
            text=text,
            is_final=True,
            transcription=Transcription(
                text=text,
                language=getattr(info, "language", forced_code or "") or "",
                language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
                duration=float(getattr(info, "duration", 0.0) or 0.0),
                elapsed=time.perf_counter() - started,
                segments=collected,
                words=words,
            ),
        )

    def transcribe(
        self, audio: bytes, language: str | None = None, word_timestamps: bool = False
    ) -> Transcription:
        """Transcription complète, en consommant le flux jusqu'au bout."""
        final: Transcription | None = None
        for event in self.transcribe_stream(audio, language, word_timestamps):
            if event.is_final:
                final = event.transcription
        if final is None:  # pragma: no cover - transcribe_stream lève avant
            raise EmptyTranscriptionError("no speech detected in audio")
        return final

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
