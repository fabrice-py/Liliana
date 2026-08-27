"""Fixtures de test.

Chaque test tourne sur une base SQLite temporaire et des moteurs factices : la
suite ne nécessite ni Ollama, ni modèle Whisper, ni voix Piper.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.llm import LLMProvider  # noqa: E402
from app.speech.stt import STTProvider  # noqa: E402
from app.speech.tts import TTSProvider  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Isole configuration, base de données et fichiers pour chaque test."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("SAVE_AUDIO", "false")
    # Les variables d'environnement priment sur le .env du développeur :
    # la suite est donc déterministe quelle que soit la machine.
    for name in ("CORRECTION_MODE", "DEFAULT_LANGUAGE", "TTS_PROVIDER", "STT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORRECTION_MODE", "normal")
    monkeypatch.setenv("DEFAULT_LANGUAGE", "english")

    from app.core.config import reload_settings
    from app.database import database as database_module

    database_module.reset_connection()
    database_module.forget_initialised()
    settings = reload_settings()
    settings.ensure_directories()

    from app.ai import llm as llm_module
    from app.speech import stt as stt_module, tts as tts_module

    llm_module.reset_llm_provider()
    stt_module.reset_stt_provider()
    tts_module.reset_tts_provider()

    yield settings

    database_module.reset_connection()
    database_module.forget_initialised()
    reload_settings()


class FakeLLM(LLMProvider):
    """Provider LLM déterministe.

    ``responses`` est une file de chaînes brutes renvoyées dans l'ordre ; la
    dernière est réutilisée une fois la file épuisée.
    """

    name = "fake"

    def __init__(self, responses: list[str] | None = None, chunk_size: int = 7) -> None:
        self.responses = responses or ['{"response": "Hello!", "errors": []}']
        self.chunk_size = chunk_size
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages, *, temperature=None, json_mode=False):  # noqa: ANN001
        self.calls.append(messages)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    def stream(self, messages, *, temperature=None, json_mode=False):  # noqa: ANN001
        """Découpe la réponse en petits fragments, comme le ferait un vrai modèle."""
        text = self.generate(messages, temperature=temperature, json_mode=json_mode)
        size = max(1, self.chunk_size)
        for start in range(0, len(text), size):
            yield text[start : start + size]

    def status(self):
        from app.ai.llm import LLMStatus

        return LLMStatus(available=True, provider="fake", model="fake-model", detail="ready")


class FakeSTT(STTProvider):
    """Provider STT déterministe.

    Hérite de STTProvider afin d'exercer le vrai contrat — y compris le repli
    ``transcribe_stream`` fourni par la classe de base.
    """

    name = "fake"

    def __init__(self, text: str = "Yesterday I go to the cinema.") -> None:
        super().__init__()
        self.text = text
        self.calls = 0

    def transcribe(self, audio: bytes, language: str | None = None):  # noqa: ANN001
        from app.core.exceptions import EmptyTranscriptionError
        from app.speech.stt import Transcription

        self.calls += 1
        if not audio:
            raise EmptyTranscriptionError("empty")
        return Transcription(
            text=self.text,
            language=(language or "english")[:2],
            language_probability=0.99,
            duration=2.0,
            elapsed=0.05,
        )

    def is_available(self):
        return True, "fake stt"

    def warmup(self):
        return None


class FakeTTS(TTSProvider):
    """Provider TTS déterministe (renvoie un petit WAV valide)."""

    name = "fake"
    WAV = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x40\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )

    def synthesize(self, text: str, language: str, speed: float | None = None):  # noqa: ANN001
        from app.speech.tts import Speech

        return Speech(audio=self.WAV, mime_type="audio/wav", voice=f"fake-{language}", elapsed=0.01)

    def is_available(self):
        return True, "fake tts"

    def available_voices(self):
        return {"english": True, "german": True, "french": True}


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def user_id() -> int:
    from app.database.repositories import users

    return int(users.get_or_create_default()["id"])


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Client HTTP de test, avec les trois moteurs simulés."""
    from fastapi.testclient import TestClient

    from app.ai import llm as llm_module, tutor as tutor_module
    from app.language import (
        assessment as assessment_module,
        correction as correction_module,
        grammar as grammar_module,
        vocabulary as vocabulary_module,
    )
    from app.main import create_app
    from app.speech import stt as stt_module, tts as tts_module

    fake_llm = FakeLLM()
    fake_stt = FakeSTT()
    fake_tts = FakeTTS()

    monkeypatch.setattr(llm_module, "get_llm_provider", lambda settings=None: fake_llm)
    monkeypatch.setattr(stt_module, "get_stt_provider", lambda settings=None: fake_stt)
    monkeypatch.setattr(tts_module, "get_tts_provider", lambda settings=None: fake_tts)

    # Les routes et les services importent les fabriques directement.
    import app.api.routes as routes_module

    monkeypatch.setattr(routes_module, "get_llm_provider", lambda settings=None: fake_llm)
    monkeypatch.setattr(routes_module, "get_stt_provider", lambda settings=None: fake_stt)
    monkeypatch.setattr(routes_module, "get_tts_provider", lambda settings=None: fake_tts)

    for module in (
        tutor_module,
        correction_module,
        grammar_module,
        vocabulary_module,
        assessment_module,
    ):
        monkeypatch.setattr(module, "get_llm_provider", lambda settings=None: fake_llm)

    with TestClient(create_app()) as test_client:
        test_client.fake_llm = fake_llm
        test_client.fake_stt = fake_stt
        test_client.fake_tts = fake_tts
        yield test_client
