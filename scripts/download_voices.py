#!/usr/bin/env python3
"""Télécharge les voix Piper configurées dans ``models/piper/``.

    python scripts/download_voices.py                  # les voix du .env
    python scripts/download_voices.py --list           # voix courantes disponibles
    python scripts/download_voices.py en_US-amy-low    # une voix précise

Les voix proviennent du dépôt public `rhasspy/piper-voices` (Hugging Face).
C'est le seul moment où Liliana a besoin d'Internet : une fois les voix
téléchargées, tout fonctionne hors ligne.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

#: Quelques voix éprouvées, pour dépanner sans aller lire le catalogue complet.
SUGGESTED_VOICES: dict[str, tuple[str, ...]] = {
    "English": ("en_US-lessac-medium", "en_US-amy-medium", "en_GB-alba-medium", "en_US-ryan-high"),
    "German": ("de_DE-thorsten-medium", "de_DE-eva_k-x_low", "de_DE-karlsson-low"),
    "French": ("fr_FR-siwis-medium", "fr_FR-upmc-medium", "fr_FR-gilles-low"),
}


def voice_urls(voice: str) -> tuple[str, str]:
    """Construit les URLs du modèle et de sa configuration.

    ``en_US-lessac-medium`` -> ``en/en_US/lessac/medium/en_US-lessac-medium.onnx``
    """
    try:
        locale, name, quality = voice.split("-", 2)
        family = locale.split("_")[0]
    except ValueError as exc:
        raise ValueError(
            f"'{voice}' is not a valid Piper voice name "
            "(expected something like en_US-lessac-medium)"
        ) from exc

    stem = f"{BASE_URL}/{family}/{locale}/{name}/{quality}/{voice}"
    return f"{stem}.onnx", f"{stem}.onnx.json"


def download(url: str, destination: Path) -> bool:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"  already there: {destination.name}")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"  downloading {destination.name} …", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while chunk := response.read(256 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  downloading {destination.name} … {100 * downloaded // total}%",
                          end="", flush=True)
        temporary.replace(destination)
        print(f"\r  downloaded  {destination.name}      ")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        print(f"\r  FAILED      {destination.name}: {exc}")
        return False


def install_voice(voice: str, target_dir: Path) -> bool:
    print(f"\n{voice}")
    try:
        model_url, config_url = voice_urls(voice)
    except ValueError as exc:
        print(f"  {exc}")
        return False
    ok = download(model_url, target_dir / f"{voice}.onnx")
    ok &= download(config_url, target_dir / f"{voice}.onnx.json")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Piper voices for Liliana.")
    parser.add_argument("voices", nargs="*", help="voice names (default: the ones from your .env)")
    parser.add_argument("--list", action="store_true", help="list a few well-tested voices")
    arguments = parser.parse_args()

    if arguments.list:
        print("\nWell-tested Piper voices:\n")
        for language, voices in SUGGESTED_VOICES.items():
            print(f"  {language}")
            for voice in voices:
                print(f"    {voice}")
        print(
            "\nFull catalogue: https://huggingface.co/rhasspy/piper-voices/tree/main"
            "\nSet the one you want in .env (TTS_VOICE_ENGLISH, TTS_VOICE_GERMAN…).\n"
        )
        return 0

    settings = get_settings()
    target_dir = settings.models_dir / "piper"
    voices = arguments.voices or [
        settings.tts_voice_english, settings.tts_voice_german, settings.tts_voice_french
    ]
    voices = [voice for voice in dict.fromkeys(voices) if voice]

    print(f"Installing {len(voices)} voice(s) into {target_dir}")
    failures = [voice for voice in voices if not install_voice(voice, target_dir)]

    if failures:
        print(
            f"\n{len(failures)} voice(s) could not be downloaded: {', '.join(failures)}"
            "\nCheck your Internet connection, or run with --list to pick another name."
            "\nLiliana still runs without them — she will answer in text only.\n"
        )
        return 1

    print("\nAll voices installed. Liliana can now speak out loud.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
