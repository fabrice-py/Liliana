"""Utilitaires audio côté serveur.

Liliana n'enregistre jamais l'audio brut par défaut (cf. §34) : la sauvegarde
n'a lieu que si ``SAVE_AUDIO=true`` dans la configuration.
"""

from __future__ import annotations

import io
import wave
from datetime import datetime
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.exceptions import AudioError
from app.core.logger import get_logger

logger = get_logger(__name__)

#: Taille maximale acceptée pour un enregistrement (~5 minutes d'Opus).
MAX_AUDIO_BYTES = 25 * 1024 * 1024

#: Types MIME que le navigateur peut produire et que PyAV sait décoder.
ACCEPTED_MIME_PREFIXES = ("audio/", "video/webm", "application/octet-stream")


def validate_upload(data: bytes, content_type: str | None = None) -> None:
    """Vérifie qu'un enregistrement reçu est exploitable."""
    if not data:
        raise AudioError(
            "empty upload",
            user_message="Liliana received an empty recording. Please try again.",
        )
    if len(data) > MAX_AUDIO_BYTES:
        raise AudioError(
            f"upload too large: {len(data)} bytes",
            user_message="That recording is too long. Please keep turns under a few minutes.",
        )
    if content_type and not content_type.startswith(ACCEPTED_MIME_PREFIXES):
        raise AudioError(
            f"unsupported content type: {content_type}",
            user_message=f"Liliana cannot read audio of type '{content_type}'.",
        )


def maybe_save(data: bytes, suffix: str = ".webm", settings: Settings | None = None) -> Path | None:
    """Enregistre l'audio sur disque **uniquement** si ``SAVE_AUDIO=true``."""
    settings = settings or get_settings()
    if not settings.save_audio:
        return None

    directory = settings.log_dir / "audio"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}{suffix}"
    try:
        path.write_bytes(data)
    except OSError as exc:  # pragma: no cover - dépend du disque
        logger.warning("Impossible d'enregistrer l'audio : %s", exc)
        return None
    logger.debug("Audio enregistré (SAVE_AUDIO=true) : %s", path)
    return path


def wav_duration(data: bytes) -> float:
    """Durée d'un WAV en secondes, 0.0 si illisible."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            return wav_file.getnframes() / frame_rate if frame_rate else 0.0
    except (wave.Error, EOFError, OSError):
        return 0.0


def wav_to_pcm16(data: bytes) -> bytes:
    """Extrait les échantillons PCM 16 bits d'un WAV mono. b'' si illisible."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                return b""
            return wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError, OSError):
        return b""
