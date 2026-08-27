"""Tests des couches parole : sélection des providers, VAD, garde-fous audio."""

from __future__ import annotations

import pytest

from app.core.config import reload_settings
from app.core.exceptions import (
    AudioError,
    ConfigurationError,
    LilianaError,
    TTSUnavailableError,
)
from app.speech.audio import MAX_AUDIO_BYTES, maybe_save, validate_upload, wav_duration
from app.speech.stt import FasterWhisperSTT, get_stt_provider, reset_stt_provider
from app.speech.tts import NullTTS, PiperTTS, get_tts_provider, reset_tts_provider
from app.speech.vad import contains_speech, get_vad_settings, rms_of_pcm16


# ------------------------------------------------------------------ upload
def test_valid_upload_passes() -> None:
    validate_upload(b"x" * 2000, "audio/webm")


def test_empty_upload_is_rejected() -> None:
    with pytest.raises(AudioError, match="empty"):
        validate_upload(b"", "audio/webm")


def test_oversized_upload_is_rejected() -> None:
    with pytest.raises(AudioError):
        validate_upload(b"x" * (MAX_AUDIO_BYTES + 1), "audio/webm")


def test_unsupported_mime_type_is_rejected() -> None:
    with pytest.raises(AudioError, match="unsupported"):
        validate_upload(b"x" * 100, "text/html")


@pytest.mark.parametrize("mime", ["audio/webm", "audio/ogg", "video/webm", "application/octet-stream"])
def test_browser_mime_types_are_accepted(mime: str) -> None:
    validate_upload(b"x" * 100, mime)


# ------------------------------------------------------------ sauvegarde
def test_audio_is_not_saved_by_default(isolated_settings) -> None:
    assert maybe_save(b"audio") is None


def test_audio_is_saved_only_when_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SAVE_AUDIO", "true")
    settings = reload_settings()
    path = maybe_save(b"audio-bytes", settings=settings)
    assert path is not None
    assert path.read_bytes() == b"audio-bytes"


# ---------------------------------------------------------------------- VAD
def test_vad_settings_come_from_configuration(monkeypatch) -> None:
    monkeypatch.setenv("VAD_SILENCE_THRESHOLD", "1.5")
    settings = reload_settings()
    assert get_vad_settings(settings).silence_threshold == 1.5


def test_rms_of_silence_is_zero() -> None:
    assert rms_of_pcm16(b"\x00\x00" * 500) == 0.0


def test_rms_grows_with_amplitude() -> None:
    quiet = rms_of_pcm16((500).to_bytes(2, "little", signed=True) * 500)
    loud = rms_of_pcm16((20000).to_bytes(2, "little", signed=True) * 500)
    assert 0 < quiet < loud <= 1.0


def test_contains_speech_discriminates_silence() -> None:
    assert contains_speech((12000).to_bytes(2, "little", signed=True) * 500) is True
    assert contains_speech(b"\x00\x00" * 500) is False


def test_rms_handles_truncated_buffers() -> None:
    assert rms_of_pcm16(b"\x01") == 0.0
    assert rms_of_pcm16(b"") == 0.0


# -------------------------------------------------------------- providers
def test_stt_provider_is_cached() -> None:
    reset_stt_provider()
    assert get_stt_provider() is get_stt_provider()


def test_unknown_stt_provider_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("STT_PROVIDER", "magic-ears")
    reset_stt_provider()
    with pytest.raises(ConfigurationError, match="unknown STT provider"):
        get_stt_provider(reload_settings())


def test_unknown_tts_provider_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("TTS_PROVIDER", "singing-robot")
    reset_tts_provider()
    with pytest.raises(ConfigurationError, match="unknown TTS provider"):
        get_tts_provider(reload_settings())


def test_stt_device_resolution_never_assumes_a_gpu() -> None:
    provider = FasterWhisperSTT()
    assert provider.device in ("cpu", "cuda")
    assert provider.compute_type in ("int8", "float16", "float32")


def test_piper_reports_missing_voices(isolated_settings) -> None:
    provider = PiperTTS(isolated_settings)
    assert provider.voice_path("english") is None
    available, detail = provider.is_available()
    assert available is False
    assert detail


def _install_fake_voice(settings, name: str, with_config: bool = True) -> None:
    voices_dir = settings.models_dir / "piper"
    voices_dir.mkdir(parents=True, exist_ok=True)
    (voices_dir / f"{name}.onnx").write_bytes(b"fake-model")
    if with_config:
        (voices_dir / f"{name}.onnx.json").write_text('{"audio": {"sample_rate": 22050}}')


def test_piper_finds_an_installed_voice(isolated_settings) -> None:
    _install_fake_voice(isolated_settings, isolated_settings.tts_voice_english)

    provider = PiperTTS(isolated_settings)
    assert provider.voice_path("english") is not None
    assert provider.available_voices()["english"] is True
    assert provider.available_voices()["german"] is False


def test_a_half_downloaded_voice_is_not_considered_installed(isolated_settings) -> None:
    """Un .onnx sans son .onnx.json = téléchargement interrompu, pas une voix."""
    _install_fake_voice(isolated_settings, isolated_settings.tts_voice_english, with_config=False)

    provider = PiperTTS(isolated_settings)
    assert provider.voice_path("english") is None
    assert provider.incomplete_voices() == [isolated_settings.tts_voice_english]

    available, detail = provider.is_available()
    assert available is False
    assert "incomplete" in detail.lower()


def test_a_half_downloaded_voice_gives_an_actionable_error(isolated_settings) -> None:
    _install_fake_voice(isolated_settings, isolated_settings.tts_voice_english, with_config=False)
    with pytest.raises(TTSUnavailableError) as excinfo:
        PiperTTS(isolated_settings).synthesize("Hello", "english")
    assert "download_voices" in excinfo.value.user_message


def test_a_corrupt_voice_model_degrades_instead_of_crashing(isolated_settings) -> None:
    """Un .onnx illisible ne doit jamais remonter une erreur brute (§36)."""
    pytest.importorskip("piper")
    _install_fake_voice(isolated_settings, isolated_settings.tts_voice_english)

    provider = PiperTTS(isolated_settings)
    assert provider.voice_path("english") is not None
    with pytest.raises(LilianaError):
        provider.synthesize("Hello", "english")


def test_null_tts_degrades_gracefully() -> None:
    provider = NullTTS()
    available, detail = provider.is_available()
    assert available is False
    assert "disabled" in detail
    with pytest.raises(TTSUnavailableError):
        provider.synthesize("hello", "english")


def test_voice_names_are_configured_per_language(isolated_settings) -> None:
    assert isolated_settings.tts_voice_for("english").startswith("en_")
    assert isolated_settings.tts_voice_for("german").startswith("de_")
    assert isolated_settings.tts_voice_for("french").startswith("fr_")


# ------------------------------------------------------------------- WAV
def test_wav_duration_of_invalid_data_is_zero() -> None:
    assert wav_duration(b"not a wav file") == 0.0


def test_tts_provider_none_is_selectable(monkeypatch) -> None:
    """`TTS_PROVIDER=none` doit produire un NullTTS utilisable, pas une erreur."""
    monkeypatch.setenv("TTS_PROVIDER", "none")
    reset_tts_provider()
    provider = get_tts_provider(reload_settings())
    assert isinstance(provider, NullTTS)
    assert provider.is_available()[0] is False
    assert provider.available_voices() == {}


def test_every_provider_accepts_settings(isolated_settings) -> None:
    """Toutes les fabriques passent `settings` : chaque provider doit l'accepter."""
    assert NullTTS(isolated_settings).settings is isolated_settings
    assert PiperTTS(isolated_settings).settings is isolated_settings
    assert FasterWhisperSTT(isolated_settings).settings is isolated_settings
