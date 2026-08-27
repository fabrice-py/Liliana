"""Configuration centralisée de Liliana.

Toute la configuration passe par ce module. Aucun paramètre réglable ne doit
être écrit en dur ailleurs dans le code (cf. README section "Configuration").

Les valeurs sont lues, par ordre de priorité :
1. variables d'environnement du process ;
2. fichier ``.env`` à la racine du projet ;
3. valeurs par défaut définies ci-dessous.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet (…/liliana), calculée depuis ce fichier.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

CorrectionMode = Literal["off", "minimal", "normal", "strict"]
LanguageCode = Literal["english", "german", "french"]


class Settings(BaseSettings):
    """Paramètres applicatifs."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Liliana"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # ------------------------------------------------------------------ LLM
    llm_provider: str = "ollama"
    llm_model: str = ""  # jamais de modèle en dur : voir scripts/check_env.py
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_temperature: float = 0.7
    llm_timeout: float = 120.0
    llm_max_history_turns: int = 12

    # ------------------------------------------------------------------ STT
    stt_provider: str = "faster-whisper"
    stt_model: str = "base"
    stt_device: str = "auto"          # auto | cpu | cuda
    stt_compute_type: str = "auto"    # auto | int8 | float16 | float32
    stt_beam_size: int = 1            # 1 = greedy, le plus rapide

    # ------------------------------------------------------------------ TTS
    tts_provider: str = "piper"
    tts_binary: str = "piper"         # chemin de l'exécutable piper
    tts_voice_english: str = "en_US-lessac-medium"
    tts_voice_german: str = "de_DE-thorsten-medium"
    tts_voice_french: str = "fr_FR-siwis-medium"
    tts_length_scale: float = 1.0     # > 1.0 = parle plus lentement

    # --------------------------------------------------------------- pédago
    default_language: LanguageCode = "english"
    correction_mode: CorrectionMode = "normal"

    # ------------------------------------------------------------------ VAD
    # Utilisés par le VAD navigateur, exposés via /api/config.
    vad_silence_threshold: float = 0.8   # secondes de silence = fin de phrase
    vad_energy_threshold: float = 0.015  # RMS normalisé au-dessus duquel on parle
    vad_min_speech_duration: float = 0.3 # ignore les bruits très courts

    # ------------------------------------------------------------- stockage
    database_path: Path = Path("data/liliana.db")
    models_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    save_audio: bool = False

    # ------------------------------------------------------------ validation
    @field_validator("database_path", "models_dir", "log_dir", mode="after")
    @classmethod
    def _absolutize(cls, value: Path) -> Path:
        """Rend les chemins relatifs absolus par rapport à la racine projet."""
        return value if value.is_absolute() else (BASE_DIR / value)

    @field_validator("correction_mode", mode="before")
    @classmethod
    def _lower_mode(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("default_language", mode="before")
    @classmethod
    def _lower_language(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    # ------------------------------------------------------------- helpers
    def tts_voice_for(self, language: str) -> str:
        """Nom de voix Piper configuré pour une langue donnée."""
        return {
            "english": self.tts_voice_english,
            "german": self.tts_voice_german,
            "french": self.tts_voice_french,
        }.get(language, self.tts_voice_english)

    def ensure_directories(self) -> None:
        """Crée les répertoires nécessaires au démarrage."""
        for directory in (
            self.database_path.parent,
            self.models_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instance unique de configuration (mise en cache)."""
    return Settings()


def reload_settings() -> Settings:
    """Vide le cache et relit la configuration (utile en tests)."""
    get_settings.cache_clear()
    return get_settings()


settings_field_names = tuple(Settings.model_fields)  # noqa: F401  (introspection)
