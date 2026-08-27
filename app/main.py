"""Point d'entrée de l'application Liliana (FastAPI).

Sert l'API sous ``/api`` et l'interface statique sous ``/``. L'application
n'écoute que sur ``127.0.0.1`` par défaut : elle n'est pas exposée au réseau.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import BASE_DIR, get_settings
from app.core.exceptions import LilianaError
from app.core.logger import get_logger, setup_logging
from app.database.database import init_database

logger = get_logger(__name__)

FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prépare la base et journalise l'état des moteurs au démarrage."""
    setup_logging()
    settings = get_settings()
    settings.ensure_directories()
    init_database()

    logger.info("%s démarre sur http://%s:%s", settings.app_name, settings.host, settings.port)
    logger.info(
        "LLM=%s/%s | STT=%s/%s | TTS=%s",
        settings.llm_provider,
        settings.llm_model or "(non configuré)",
        settings.stt_provider,
        settings.stt_model,
        settings.tts_provider,
    )
    yield
    logger.info("%s s'arrête.", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        description="Local AI language tutor — English & German.",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs",
    )

    @application.exception_handler(LilianaError)
    async def _liliana_error_handler(_: Request, exc: LilianaError) -> JSONResponse:
        """Toute erreur métier non rattrapée devient un message compréhensible."""
        logger.warning("Erreur non rattrapée : %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": type(exc).__name__, "message": exc.user_message},
        )

    application.include_router(router)

    if FRONTEND_DIR.is_dir():
        application.mount(
            "/static", StaticFiles(directory=FRONTEND_DIR), name="static"
        )

        @application.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

        @application.get("/favicon.ico", include_in_schema=False, response_model=None)
        async def favicon() -> FileResponse | JSONResponse:
            icon = FRONTEND_DIR / "favicon.svg"
            if icon.is_file():
                return FileResponse(icon)
            return JSONResponse(status_code=404, content={})
    else:  # pragma: no cover - installation incomplète
        logger.warning("Répertoire frontend introuvable : %s", FRONTEND_DIR)

    return application


app = create_app()
