#!/usr/bin/env python3
"""Lance Liliana.

    python run.py
    python run.py --port 8080 --reload

Ouvre ensuite http://127.0.0.1:8000 dans un navigateur.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

# Permet `python run.py` depuis n'importe quel répertoire.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from app.core.config import get_settings

    settings = get_settings()

    parser = argparse.ArgumentParser(description="Start Liliana, your local AI language tutor.")
    parser.add_argument("--host", default=settings.host, help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=settings.port, help="port (default: %(default)s)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument(
        "--check", action="store_true", help="check the environment and exit"
    )
    arguments = parser.parse_args()

    if arguments.check:
        from scripts.check_env import main as check_main

        return check_main()

    try:
        import uvicorn
    except ImportError:
        print(
            "Missing dependencies. Install them first:\n\n"
            "    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return 1

    url = f"http://{arguments.host}:{arguments.port}"
    print(f"\n  {settings.app_name} — local AI language tutor")
    print(f"  Open {url} in your browser.\n")

    if not arguments.no_browser and not arguments.reload:
        # `open_new_tab` est non bloquant ; le serveur démarre juste après.
        try:
            webbrowser.open_new_tab(url)
        except Exception:  # noqa: BLE001 - pas de navigateur disponible
            pass

    uvicorn.run(
        "app.main:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
