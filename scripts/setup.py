#!/usr/bin/env python3
"""Prépare Liliana en une commande.

    python scripts/setup.py

Crée la configuration, télécharge les modèles nécessaires et dit précisément ce
qui manque encore. Peut être relancé sans risque : tout ce qui est déjà en place
est laissé tel quel.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import BASE_DIR, get_settings, reload_settings  # noqa: E402
from app.core.hardware import detect_hardware  # noqa: E402

OK, KO, INFO = "  [ok]", "  [--]", "       "


def _title(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


# ------------------------------------------------------------- configuration
def ensure_env_file() -> bool:
    """Crée ``.env`` depuis ``.env.example`` s'il n'existe pas."""
    env, example = BASE_DIR / ".env", BASE_DIR / ".env.example"
    if env.is_file():
        print(f"{OK} .env already exists (left untouched)")
        return True
    if not example.is_file():
        print(f"{KO} .env.example is missing — cannot create the configuration")
        return False
    shutil.copyfile(example, env)
    reload_settings()
    print(f"{OK} .env created from .env.example")
    return True


# ------------------------------------------------------------------ Ollama
def check_ollama(pull: bool) -> bool:
    """Vérifie Ollama et, si demandé, installe un modèle adapté à la machine."""
    from app.ai.llm import get_llm_provider, reset_llm_provider

    reset_llm_provider()
    status = get_llm_provider().status()

    if status.available:
        chosen = "auto-selected" if status.detail == "auto-selected" else "configured"
        print(f"{OK} Ollama is running — using '{status.model}' ({chosen})")
        return True

    if not status.installed_models and "cannot reach" in status.detail.lower():
        print(f"{KO} {status.detail}")
        print(f"{INFO} Install it from https://ollama.com/download, then run `ollama serve`.")
        return False

    recommendation = detect_hardware().recommendations["llm_model"]
    suggested = _suggested_pull(recommendation)

    if status.installed_models:
        print(f"{KO} {status.detail}")
        print(f"{INFO} Installed: {', '.join(status.installed_models)}")
    else:
        print(f"{KO} Ollama is running but has no model installed.")

    if not pull:
        print(f"{INFO} Recommended for this machine: {recommendation}")
        print(f"{INFO} Run `ollama pull {suggested}`, or re-run with --pull-model.")
        return False

    if not shutil.which("ollama"):
        print(f"{KO} The `ollama` command is not on your PATH — pull the model manually.")
        return False

    print(f"{INFO} Downloading '{suggested}' — this can take several minutes…")
    try:
        completed = subprocess.run(["ollama", "pull", suggested], check=False)
    except OSError as exc:
        print(f"{KO} Could not run ollama: {exc}")
        return False
    if completed.returncode != 0:
        print(f"{KO} `ollama pull {suggested}` failed. Try it by hand.")
        return False

    reset_llm_provider()
    print(f"{OK} Model '{suggested}' installed")
    return True


def _suggested_pull(recommendation: str) -> str:
    """Extrait le nom de modèle de la recommandation matérielle."""
    marker = "ollama pull "
    if marker in recommendation:
        return recommendation.split(marker, 1)[1].strip(" `)").split()[0]
    return "qwen2.5:3b-instruct"


# ------------------------------------------------------------------ Whisper
def preload_whisper() -> bool:
    """Télécharge le modèle de transcription pour que le 1er tour ne traîne pas."""
    from app.core.exceptions import LilianaError
    from app.speech.stt import get_stt_provider, reset_stt_provider

    settings = get_settings()
    reset_stt_provider()
    provider = get_stt_provider()

    available, detail = provider.is_available()
    if not available:
        print(f"{KO} {detail}")
        return False

    print(f"{INFO} Preparing the '{settings.stt_model}' speech model (first run downloads it)…")
    started = time.perf_counter()
    try:
        provider.warmup()
        # warmup() avale ses erreurs : on vérifie que le modèle est bien chargé.
        provider._load()  # noqa: SLF001 - vérification volontaire à l'installation
    except LilianaError as exc:
        print(f"{KO} {exc.user_message}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"{KO} Could not prepare the speech model: {exc}")
        return False

    print(f"{OK} Speech model ready in {time.perf_counter() - started:.1f}s")
    return True


# -------------------------------------------------------------------- Piper
def install_voices() -> bool:
    from scripts.download_voices import main as download_main

    print(f"{INFO} Downloading the Piper voices named in your .env…")
    return download_main() == 0


def check_phonetics() -> None:
    """L'analyse phonétique dépend d'espeak-ng, fourni par piper-tts."""
    from app.language import phonemes

    if phonemes.is_available():
        print(f"{OK} Phonetic analysis enabled (espeak-ng via piper-tts)")
    else:
        print(f"{KO} Phonetic analysis unavailable — run `pip install piper-tts`")
        print(f"{INFO} Pronunciation practice still works, comparing words instead of sounds.")


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Set Liliana up in one command.")
    parser.add_argument(
        "--pull-model", action="store_true",
        help="download a language model with `ollama pull` if none is installed",
    )
    parser.add_argument("--skip-voices", action="store_true", help="do not download Piper voices")
    parser.add_argument("--skip-whisper", action="store_true", help="do not preload the speech model")
    arguments = parser.parse_args()

    settings = get_settings()
    print(f"\n{settings.app_name} — setup")

    _title("Configuration")
    ensure_env_file()
    settings = reload_settings()
    settings.ensure_directories()

    _title("Hardware")
    info = detect_hardware()
    print(f"{INFO} {info.os_name} {info.architecture}, {info.cpu_count} cores, "
          f"{info.total_ram_gb} GB RAM, GPU: {info.gpu_name or 'none'}")
    print(f"{INFO} {info.recommendations['note']}")

    _title("Language model")
    llm_ready = check_ollama(pull=arguments.pull_model)

    _title("Speech recognition")
    stt_ready = True if arguments.skip_whisper else preload_whisper()

    _title("Voice and phonetics")
    if not arguments.skip_voices:
        install_voices()
    check_phonetics()

    _title("Result")
    if llm_ready and stt_ready:
        print("  Liliana is ready. Start her with:  python run.py")
        print("  Then open http://127.0.0.1:8000\n")
        return 0
    print("  Almost there — fix the [--] lines above, then run this again.")
    print("  `python scripts/check_env.py` re-checks without downloading anything.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
