#!/usr/bin/env python3
"""Vérifie l'environnement de Liliana et dit quoi faire s'il manque quelque chose.

    python scripts/check_env.py
    python run.py --check
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.hardware import detect_hardware  # noqa: E402

OK = "  [ok]"
KO = "  [--]"
INFO = "       "


def _title(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def _check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 10)
    print(f"{OK if ok else KO} Python {version.major}.{version.minor}.{version.micro}")
    if not ok:
        print(f"{INFO} Liliana needs Python 3.10 or newer.")
    return ok


def _check_packages() -> bool:
    required = {
        "fastapi": "the local web server",
        "uvicorn": "the ASGI server",
        "pydantic_settings": "configuration loading",
        "httpx": "talking to Ollama",
        "multipart": "receiving audio from the browser",
    }
    optional = {
        "faster_whisper": "speech-to-text (required to talk to Liliana)",
        "piper": "text-to-speech via the Python module (the `piper` binary also works)",
    }

    all_present = True
    for module, purpose in required.items():
        present = importlib.util.find_spec(module) is not None
        print(f"{OK if present else KO} {module} — {purpose}")
        all_present &= present
    for module, purpose in optional.items():
        present = importlib.util.find_spec(module) is not None
        print(f"{OK if present else KO} {module} (optional) — {purpose}")

    if not all_present:
        print(f"{INFO} Install everything with: pip install -r requirements.txt")
    return all_present


def _check_llm() -> bool:
    from app.ai.llm import get_llm_provider

    settings = get_settings()
    status = get_llm_provider().status()

    if status.available:
        origin = " (chosen automatically)" if status.detail == "auto-selected" else ""
        print(f"{OK} Ollama is running with model '{status.model}'{origin}")
        return True

    print(f"{KO} {status.detail}")
    if status.installed_models:
        print(f"{INFO} Models already installed: {', '.join(status.installed_models)}")
        if not settings.llm_model:
            print(f"{INFO} Pick one and set LLM_MODEL=<name> in your .env file.")
    else:
        print(f"{INFO} 1. Install Ollama: https://ollama.com/download")
        print(f"{INFO} 2. Start it:       ollama serve")
        print(f"{INFO} 3. Pull a model:   see the recommendation below")
    return False


def _check_stt() -> bool:
    from app.speech.stt import get_stt_provider

    settings = get_settings()
    available, detail = get_stt_provider().is_available()
    print(f"{OK if available else KO} Speech-to-text: {detail}")

    cache = settings.models_dir / "whisper"
    if available and not any(cache.glob("**/*.bin")) and not any(cache.glob("**/*.safetensors")):
        print(
            f"{INFO} The '{settings.stt_model}' model is not downloaded yet. "
            "It downloads automatically on the first recording (needs Internet once)."
        )
    return available


def _check_tts() -> bool:
    from app.speech.tts import get_tts_provider

    provider = get_tts_provider()
    available, detail = provider.is_available()
    print(f"{OK if available else KO} Text-to-speech: {detail}")

    voices = provider.available_voices()
    for language, present in voices.items():
        marker = OK if present else KO
        print(f"{marker} voice for {language}: {get_settings().tts_voice_for(language)}")
    if not all(voices.values()):
        print(f"{INFO} Download the missing voices: python scripts/download_voices.py")
    if not shutil.which(get_settings().tts_binary) and importlib.util.find_spec("piper") is None:
        print(f"{INFO} Piper itself is missing: pip install piper-tts")
        print(f"{INFO} (Liliana still works without it — she answers in text only.)")
    return available


def _check_phonetics() -> bool:
    """L'analyse phonétique de la prononciation dépend d'espeak-ng (piper-tts)."""
    from app.language import phonemes

    available = phonemes.is_available()
    if available:
        print(f"{OK} Phonetic analysis: espeak-ng ready (sound-by-sound feedback)")
    else:
        print(f"{KO} Phonetic analysis unavailable: pip install piper-tts")
        print(f"{INFO} Pronunciation practice falls back to comparing words.")
    return available


def _check_storage() -> bool:
    settings = get_settings()
    try:
        settings.ensure_directories()
        from app.database.database import init_database

        init_database()
    except Exception as exc:  # noqa: BLE001
        print(f"{KO} Local storage: {exc}")
        return False
    print(f"{OK} Database ready: {settings.database_path}")
    print(f"{OK} Audio recordings saved to disk: {'yes' if settings.save_audio else 'no'}")
    return True


def main() -> int:
    settings = get_settings()
    print(f"\n{settings.app_name} — environment check")

    _title("Runtime")
    python_ok = _check_python()

    _title("Python packages")
    packages_ok = _check_packages()

    _title("Hardware")
    info = detect_hardware()
    print(f"{INFO} {info.os_name} {info.os_version} ({info.architecture})")
    print(f"{INFO} {info.cpu_count} CPU cores, {info.total_ram_gb} GB RAM")
    print(f"{INFO} GPU: {info.gpu_name or 'none detected'}"
          + (f" ({info.vram_gb} GB VRAM)" if info.vram_gb else ""))
    print(f"\n{INFO} Recommended for this machine:")
    for key, value in info.recommendations.items():
        print(f"{INFO}   {key}: {value}")

    _title("Engines")
    llm_ok = _check_llm()
    stt_ok = _check_stt()
    _check_tts()  # jamais bloquant : Liliana peut répondre en texte
    _check_phonetics()  # jamais bloquant : l'analyse retombe sur l'orthographe

    _title("Local storage")
    storage_ok = _check_storage()

    ready = python_ok and packages_ok and llm_ok and stt_ok and storage_ok
    _title("Result")
    if ready:
        print("  Liliana is ready. Start her with:  python run.py")
    else:
        print("  Liliana is not ready yet — fix the [--] lines above, then run this again.")
    print()
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
