"""Détection d'activité vocale (Voice Activity Detection).

Le VAD temps réel s'exécute **dans le navigateur** (Web Audio API) : c'est là
que se trouve le micro, et détecter le silence côté client évite un aller-retour
réseau par fragment audio. Ce module :

* expose les paramètres du VAD, réglables dans ``.env`` (cf. §6) ;
* fournit un VAD d'appoint côté serveur, utilisé pour écarter très tôt un
  enregistrement qui ne contient que du bruit.
"""

from __future__ import annotations

import array
import math
from dataclasses import asdict, dataclass

from app.core.config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class VadSettings:
    """Paramètres transmis au VAD du navigateur."""

    silence_threshold: float   # secondes de silence marquant la fin d'une phrase
    energy_threshold: float    # RMS normalisé au-delà duquel on considère qu'on parle
    min_speech_duration: float # durée minimale d'une prise de parole utile

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def get_vad_settings(settings: Settings | None = None) -> VadSettings:
    settings = settings or get_settings()
    return VadSettings(
        silence_threshold=settings.vad_silence_threshold,
        energy_threshold=settings.vad_energy_threshold,
        min_speech_duration=settings.vad_min_speech_duration,
    )


def rms_of_pcm16(pcm: bytes) -> float:
    """Énergie RMS d'un buffer PCM 16 bits signé, normalisée dans [0, 1]."""
    if len(pcm) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    total = math.fsum(float(sample) * float(sample) for sample in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def contains_speech(pcm: bytes, settings: Settings | None = None) -> bool:
    """Heuristique simple : l'enregistrement contient-il autre chose que du bruit ?

    Volontairement permissive — le vrai filtrage est fait par le VAD de Whisper.
    """
    threshold = get_vad_settings(settings).energy_threshold
    return rms_of_pcm16(pcm) >= threshold * 0.5
