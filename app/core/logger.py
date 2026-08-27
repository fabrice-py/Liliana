"""Configuration du logging.

Écrit à la fois sur la console et dans ``logs/liliana.log`` (rotation à 2 Mo).
Les conversations audio brutes ne sont jamais journalisées.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

from app.core.config import get_settings

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging() -> None:
    """Installe les handlers. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_dir / "liliana.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Ces bibliothèques sont très bavardes en DEBUG.
    for noisy in ("httpx", "httpcore", "faster_whisper", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé, en s'assurant que le logging est configuré."""
    setup_logging()
    return logging.getLogger(name)
