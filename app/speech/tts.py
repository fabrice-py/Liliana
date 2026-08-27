"""Synthèse vocale (Text-to-Speech).

Backend par défaut : Piper, local et rapide. Deux chemins d'exécution sont
supportés, dans cet ordre :

1. le module Python ``piper`` (``pip install piper-tts``) — pas de sous-processus ;
2. l'exécutable ``piper`` présent dans le PATH ou désigné par ``TTS_BINARY``.

Les voix (``.onnx`` + ``.onnx.json``) sont cherchées dans ``models/piper/``.
Voir ``scripts/download_voices.py`` pour les télécharger.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, TTSError, TTSUnavailableError
from app.core.logger import get_logger

logger = get_logger(__name__)


#: Ponctuation qui termine une phrase, suivie d'une espace ou de la fin du texte.
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"')\]]*(?=\s|$)|[\n\r]+")

#: En deçà, on ne coupe pas : « Mr. », « 3.5 » ou « etc. » ne sont pas des phrases.
_MIN_SENTENCE_CHARS = 15


class SentenceBuffer:
    """Découpe un texte qui arrive par fragments en phrases prononçables.

    Permet de synthétiser la première phrase pendant que le modèle écrit encore
    la suite : c'est ce qui fait tomber la latence perçue (cf. §30).
    """

    def __init__(self, min_chars: int = _MIN_SENTENCE_CHARS) -> None:
        self.min_chars = min_chars
        self._pending = ""

    def feed(self, chunk: str) -> list[str]:
        """Ajoute un fragment. Retourne les phrases devenues complètes."""
        if not chunk:
            return []
        self._pending += chunk

        sentences: list[str] = []
        while True:
            match = self._next_boundary()
            if match is None:
                break
            sentence = self._pending[: match.end()].strip()
            self._pending = self._pending[match.end():].lstrip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def _next_boundary(self) -> re.Match[str] | None:
        """Première coupure acceptable dans le tampon courant."""
        for match in _SENTENCE_END_RE.finditer(self._pending):
            if match.end() >= self.min_chars:
                return match
        return None

    def flush(self) -> str:
        """Retourne le reste, même s'il ne se termine pas par une ponctuation."""
        remainder, self._pending = self._pending.strip(), ""
        return remainder

    @property
    def pending(self) -> str:
        return self._pending


@dataclass(slots=True)
class Speech:
    """Audio synthétisé, prêt à être renvoyé au navigateur."""

    audio: bytes
    mime_type: str = "audio/wav"
    voice: str = ""
    elapsed: float = 0.0


class TTSProvider(ABC):
    """Interface de synthèse vocale (cf. cahier des charges §7)."""

    name = "abstract"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @abstractmethod
    def synthesize(self, text: str, language: str, speed: float | None = None) -> Speech:
        """Synthétise ``text`` dans la voix associée à ``language``."""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(disponible, détail) sans rien synthétiser."""

    @abstractmethod
    def available_voices(self) -> dict[str, bool]:
        """Voix configurées par langue et présence du fichier de modèle."""


class PiperTTS(TTSProvider):
    """Synthèse Piper locale."""

    name = "piper"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.voices_dir = self.settings.models_dir / "piper"
        self._loaded: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------- résolution
    def voice_path(self, language: str) -> Path | None:
        """Chemin du fichier ``.onnx`` de la voix configurée, s'il existe."""
        voice = self.settings.tts_voice_for(language)
        if not voice:
            return None
        candidate = Path(voice)
        if candidate.suffix == ".onnx" and candidate.is_file():
            return candidate  # chemin absolu fourni dans la configuration
        for path in (
            self.voices_dir / f"{voice}.onnx",
            self.voices_dir / voice / f"{voice}.onnx",
        ):
            if path.is_file():
                return path
        return None

    def _binary(self) -> str | None:
        binary = self.settings.tts_binary
        return binary if (Path(binary).is_file() or shutil.which(binary)) else None

    def _python_voice(self, path: Path) -> Any | None:
        """Charge la voix via le module Python ``piper``, si installé."""
        key = str(path)
        if key in self._loaded:
            return self._loaded[key]
        try:
            from piper import PiperVoice
        except ImportError:
            return None
        with self._lock:
            if key not in self._loaded:
                logger.info("Chargement de la voix Piper %s", path.name)
                self._loaded[key] = PiperVoice.load(str(path))
        return self._loaded[key]

    # ------------------------------------------------------------ synthèse
    def _synthesize_python(self, voice: Any, text: str, length_scale: float) -> bytes | None:
        """Utilise l'API Python. Retourne ``None`` si l'API ne correspond pas."""
        buffer = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
        try:
            with wave.open(buffer, "wb") as wav_file:
                try:
                    from piper import SynthesisConfig

                    voice.synthesize_wav(
                        text, wav_file, syn_config=SynthesisConfig(length_scale=length_scale)
                    )
                except ImportError:
                    # piper-tts < 1.3 : signature différente, sans SynthesisConfig.
                    voice.synthesize(text, wav_file, length_scale=length_scale)
            buffer.seek(0)
            return buffer.read()
        except (AttributeError, TypeError) as exc:
            logger.debug("API Python Piper inattendue (%s), passage au binaire", exc)
            return None
        finally:
            buffer.close()

    def _synthesize_binary(
        self, binary: str, model: Path, text: str, length_scale: float
    ) -> bytes:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "speech.wav"
            command = [
                binary,
                "--model", str(model),
                "--output_file", str(output),
                "--length_scale", f"{length_scale:.2f}",
            ]
            try:
                result = subprocess.run(
                    command,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise TTSError(f"piper failed to run: {exc}") from exc

            if result.returncode != 0 or not output.is_file():
                detail = result.stderr.decode("utf-8", "replace").strip()[:300]
                raise TTSError(f"piper exited with {result.returncode}: {detail}")
            return output.read_bytes()

    def synthesize(self, text: str, language: str, speed: float | None = None) -> Speech:
        text = (text or "").strip()
        if not text:
            raise TTSError("nothing to synthesize", user_message="There was nothing to say.")

        model = self.voice_path(language)
        if model is None:
            voice_name = self.settings.tts_voice_for(language)
            raise TTSUnavailableError(
                f"voice file not found for {language} ({voice_name})",
                user_message=(
                    f"The Piper voice '{voice_name}' is missing. Download it with "
                    f"`python scripts/download_voices.py` — Liliana keeps answering "
                    "in text meanwhile."
                ),
            )

        # length_scale : > 1 ralentit la parole. `speed` est un multiplicateur de
        # vitesse (0.8 = 20 % plus lent), plus intuitif côté interface.
        length_scale = self.settings.tts_length_scale / max(0.3, min(speed or 1.0, 2.0))

        started = time.perf_counter()
        audio: bytes | None = None

        voice = self._python_voice(model)
        if voice is not None:
            audio = self._synthesize_python(voice, text, length_scale)

        if audio is None:
            binary = self._binary()
            if binary is None:
                raise TTSUnavailableError(
                    "neither the piper python module nor the piper binary is available"
                )
            audio = self._synthesize_binary(binary, model, text, length_scale)

        if not audio:
            raise TTSError("piper produced no audio")

        return Speech(
            audio=audio,
            mime_type="audio/wav",
            voice=model.stem,
            elapsed=time.perf_counter() - started,
        )

    # -------------------------------------------------------- disponibilité
    def is_available(self) -> tuple[bool, str]:
        try:
            import piper  # noqa: F401

            has_python = True
        except ImportError:
            has_python = False
        has_binary = self._binary() is not None

        if not has_python and not has_binary:
            return False, TTSUnavailableError.user_message

        voices = self.available_voices()
        missing = [name for name, present in voices.items() if not present]
        engine = "python module" if has_python else "binary"
        if not any(voices.values()):
            return False, (
                f"Piper ({engine}) is installed but no voice was found in "
                f"{self.voices_dir}. Run `python scripts/download_voices.py`."
            )
        detail = f"piper ({engine})"
        if missing:
            detail += f"; missing voices: {', '.join(missing)}"
        return True, detail

    def available_voices(self) -> dict[str, bool]:
        return {
            language: self.voice_path(language) is not None
            for language in ("english", "german", "french")
        }


class NullTTS(TTSProvider):
    """Repli silencieux : Liliana répond en texte uniquement.

    Utilisé quand ``TTS_PROVIDER=none``. L'application reste utilisable (cf. §36).
    """

    name = "none"

    def synthesize(self, text: str, language: str, speed: float | None = None) -> Speech:
        raise TTSUnavailableError(
            "TTS is disabled",
            user_message="Voice output is turned off. Liliana answers in text only.",
        )

    def is_available(self) -> tuple[bool, str]:
        return False, "Text-to-speech is disabled (TTS_PROVIDER=none)."

    def available_voices(self) -> dict[str, bool]:
        return {}


_PROVIDERS: dict[str, type[TTSProvider]] = {"piper": PiperTTS, "none": NullTTS}

_cached_provider: TTSProvider | None = None
_cached_key: str | None = None


def get_tts_provider(settings: Settings | None = None) -> TTSProvider:
    """Retourne le moteur TTS configuré (instance réutilisée)."""
    global _cached_provider, _cached_key

    settings = settings or get_settings()
    key = settings.tts_provider.lower()
    if _cached_provider is not None and _cached_key == key:
        return _cached_provider

    provider_class = _PROVIDERS.get(key)
    if provider_class is None:
        raise ConfigurationError(
            f"unknown TTS provider: {settings.tts_provider}",
            user_message=(
                f"Unknown text-to-speech provider '{settings.tts_provider}'. "
                f"Supported: {', '.join(sorted(_PROVIDERS))}."
            ),
        )

    _cached_provider = provider_class(settings)
    _cached_key = key
    return _cached_provider


def reset_tts_provider() -> None:
    global _cached_provider, _cached_key
    _cached_provider = None
    _cached_key = None
