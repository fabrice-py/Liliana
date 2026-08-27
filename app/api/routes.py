"""Routes HTTP de Liliana.

L'API est locale (127.0.0.1 par défaut) et sans authentification : elle n'est pas
exposée sur le réseau. Aucune requête SQL ici — tout passe par les dépôts.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ai.tutor import tutor
from app.core.config import get_settings
from app.core.exceptions import EmptyTranscriptionError, LilianaError, TTSError
from app.core.hardware import detect_hardware
from app.core.logger import get_logger
from app.database.repositories import (
    app_settings,
    language_profiles,
    messages as message_repo,
    pronunciation as pronunciation_repo,
    sessions as session_repo,
    users as user_repo,
    vocabulary as vocabulary_repo,
)
from app.language import pronunciation as pronunciation_module
from app.language.assessment import assessment_service, build_test
from app.language.commands import detect_command
from app.language.correction import correction_service
from app.language.grammar import EXERCISE_TYPES, exercise_service
from app.language.languages import LANGUAGES, TARGET_LANGUAGES, is_supported
from app.language.vocabulary import vocabulary_service
from app.learning.curriculum import build_lesson
from app.learning.progress import progress_tracker
from app.ai.llm import get_llm_provider
from app.ai.prompts import CONVERSATION_MODES, DEFAULT_MODE
from app.speech.audio import maybe_save, validate_upload
from app.speech.stt import get_stt_provider
from app.speech.tts import get_tts_provider
from app.speech.vad import get_vad_settings

logger = get_logger(__name__)
router = APIRouter(prefix="/api")

CorrectionModeName = Literal["off", "minimal", "normal", "strict"]


# ------------------------------------------------------------------ helpers
def _current_user_id() -> int:
    return int(user_repo.get_or_create_default()["id"])


def _resolve_language(language: str | None) -> str:
    settings = get_settings()
    candidate = (language or app_settings.get("language") or settings.default_language).lower()
    return candidate if is_supported(candidate) else settings.default_language


def _resolve_mode(mode: str | None) -> str:
    candidate = (mode or app_settings.get("mode") or DEFAULT_MODE).lower()
    return candidate if candidate in CONVERSATION_MODES else DEFAULT_MODE


def _fail(exc: LilianaError, status: int = 503) -> HTTPException:
    """Transforme une erreur métier en réponse HTTP lisible par l'utilisateur."""
    logger.warning("%s: %s", type(exc).__name__, exc)
    return HTTPException(
        status_code=status,
        detail={"error": type(exc).__name__, "message": exc.user_message},
    )


def _speak(text: str, language: str, speed: float | None = None) -> dict[str, Any] | None:
    """Synthétise la réponse. ``None`` si le TTS est indisponible — l'utilisateur
    garde le texte, l'application ne plante pas (cf. §36)."""
    try:
        speech = get_tts_provider().synthesize(text, language, speed=speed)
    except (TTSError, LilianaError) as exc:
        logger.info("TTS indisponible : %s", exc)
        return None
    return {
        "audio_base64": base64.b64encode(speech.audio).decode("ascii"),
        "mime_type": speech.mime_type,
        "voice": speech.voice,
        "elapsed": round(speech.elapsed, 2),
    }


# -------------------------------------------------------------------- SSE
#: En-têtes qui empêchent tout intermédiaire de tamponner le flux : sans eux,
#: le streaming n'apporte rien (l'utilisateur reçoit tout d'un bloc à la fin).
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: str, data: dict[str, Any]) -> str:
    """Formate un évènement Server-Sent Events."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_error(exc: LilianaError) -> str:
    logger.warning("Flux interrompu — %s: %s", type(exc).__name__, exc)
    return _sse("error", {"error": type(exc).__name__, "message": exc.user_message})


def _stream_turn(
    *,
    user_id: int,
    text: str,
    language: str,
    mode: str,
    correction_mode: str | None,
    is_voice: bool,
    speak: bool,
    speed: float,
    duration_seconds: int = 0,
) -> Iterator[str]:
    """Diffuse un tour de conversation : texte au fil de l'eau, voix par phrase.

    C'est ici que se joue la latence perçue : la première phrase est synthétisée
    et envoyée pendant que le modèle écrit encore la suite (cf. §30).
    """
    command = detect_command(text)
    applied_command: dict[str, Any] | None = None
    if command:
        applied_command = {"action": command.action, **command.payload}
        language, mode, correction_mode, speed = _apply_command(
            command, language, mode, correction_mode, speed
        )
        yield _sse("command", applied_command)

    session = session_repo.get_or_create_open(user_id, language, mode)
    started = time.perf_counter()
    spoken_chunks = 0

    for event in tutor.respond_stream(
        user_id=user_id,
        session_id=int(session["id"]),
        text=text,
        language=language,
        mode=mode,
        correction_mode=correction_mode,
        is_voice=is_voice,
        duration_seconds=duration_seconds,
    ):
        if event.kind == "delta":
            yield _sse("delta", {"text": event.text})
        elif event.kind == "sentence" and speak:
            speech = _speak(event.text, language, speed)
            if speech is not None:
                yield _sse("audio", {**speech, "index": spoken_chunks, "text": event.text})
                spoken_chunks += 1
        elif event.kind == "done" and event.result is not None:
            payload = event.result.as_dict()
            payload["command"] = applied_command
            payload["language"] = language
            payload["mode"] = mode
            payload["llm_elapsed"] = round(time.perf_counter() - started, 2)
            payload["spoken_chunks"] = spoken_chunks
            yield _sse("done", payload)


def _apply_command(
    command: Any, language: str, mode: str, correction_mode: str | None, speed: float
) -> tuple[str, str, str | None, float]:
    """Applique une commande vocale et retourne l'état mis à jour."""
    if command.action == "switch_language":
        language = _resolve_language(command.payload["language"])
        app_settings.set("language", language)
    elif command.action == "set_mode":
        mode = _resolve_mode(command.payload["mode"])
        app_settings.set("mode", mode)
    elif command.action == "set_correction_mode":
        correction_mode = command.payload["correction_mode"]
        app_settings.set("correction_mode", correction_mode)
    elif command.action in ("speak_slower", "speak_faster"):
        speed = float(command.payload["speed"])
    return language, mode, correction_mode, speed


# ---------------------------------------------------------------- schémas
class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str | None = None
    mode: str | None = None
    correction_mode: CorrectionModeName | None = None
    speak: bool = True
    speed: float = Field(default=1.0, ge=0.3, le=2.0)


class CorrectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str | None = None
    correction_mode: CorrectionModeName | None = None


class ExplainRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    language: str | None = None


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str | None = None
    speed: float = Field(default=1.0, ge=0.3, le=2.0)


class SettingsRequest(BaseModel):
    language: str | None = None
    mode: str | None = None
    correction_mode: CorrectionModeName | None = None


class ExerciseRequest(BaseModel):
    language: str | None = None
    topic: str | None = None
    exercise_type: str | None = None


class ExerciseAnswer(BaseModel):
    exercise_id: int
    answer: str = Field(default="", max_length=2000)


class VocabularyReview(BaseModel):
    language: str | None = None
    word: str = Field(min_length=1, max_length=120)
    remembered: bool


class VocabularyTeach(BaseModel):
    language: str | None = None
    theme: str = Field(default="everyday life", max_length=200)
    count: int = Field(default=5, ge=1, le=12)


class AssessmentSubmission(BaseModel):
    language: str
    answers: dict[str, str] = Field(default_factory=dict)
    productions: dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------------ status
@router.get("/health")
def health() -> dict[str, Any]:
    """État des trois moteurs. Alimente l'indicateur ● LOCAL / OFFLINE (§33)."""
    settings = get_settings()

    llm_status = get_llm_provider().status()
    stt_available, stt_detail = get_stt_provider().is_available()
    tts_provider = get_tts_provider()
    tts_available, tts_detail = tts_provider.is_available()

    return {
        "app": settings.app_name,
        "ready": llm_status.available and stt_available,
        "llm": {
            "provider": llm_status.provider,
            "model": llm_status.model,
            "available": llm_status.available,
            "detail": llm_status.detail,
            "installed_models": list(llm_status.installed_models),
        },
        "stt": {
            "provider": settings.stt_provider,
            "model": settings.stt_model,
            "available": stt_available,
            "detail": stt_detail,
        },
        "tts": {
            "provider": settings.tts_provider,
            "available": tts_available,
            "detail": tts_detail,
            "voices": tts_provider.available_voices(),
        },
        # Tout tourne en local : aucune donnée ne quitte la machine.
        "offline_capable": True,
    }


@router.get("/config")
def configuration() -> dict[str, Any]:
    """Tout ce dont l'interface a besoin pour se construire."""
    settings = get_settings()
    user_id = _current_user_id()
    user = user_repo.get(user_id) or {}

    return {
        "app_name": settings.app_name,
        "languages": [
            {
                "code": code,
                "name": LANGUAGES[code].english_name,
                "native_name": LANGUAGES[code].native_name,
                "flag": LANGUAGES[code].flag,
            }
            for code in TARGET_LANGUAGES
        ],
        "modes": [
            {"key": mode.key, "label": mode.label, "description": mode.description}
            for mode in CONVERSATION_MODES.values()
        ],
        "correction_modes": ["off", "minimal", "normal", "strict"],
        "exercise_types": list(EXERCISE_TYPES),
        "vad": get_vad_settings().as_dict(),
        "current": {
            "language": _resolve_language(None),
            "mode": _resolve_mode(None),
            "correction_mode": app_settings.get("correction_mode", settings.correction_mode),
        },
        "onboarded": bool(user.get("onboarded")),
        "profiles": language_profiles.list_for_user(user_id),
    }


@router.get("/hardware")
def hardware() -> dict[str, Any]:
    """Détection matérielle et modèles recommandés (§31)."""
    return detect_hardware().as_dict()


@router.post("/settings")
def update_settings(payload: SettingsRequest) -> dict[str, Any]:
    if payload.language:
        if not is_supported(payload.language):
            raise HTTPException(422, detail={"message": f"Unknown language '{payload.language}'."})
        app_settings.set("language", payload.language.lower())
    if payload.mode:
        if payload.mode not in CONVERSATION_MODES:
            raise HTTPException(422, detail={"message": f"Unknown mode '{payload.mode}'."})
        app_settings.set("mode", payload.mode)
    if payload.correction_mode:
        app_settings.set("correction_mode", payload.correction_mode)
    return configuration()["current"]


# ---------------------------------------------------------------- sessions
@router.post("/session/new")
def new_session(language: str | None = None, mode: str | None = None) -> dict[str, Any]:
    """Ferme la session courante et en ouvre une nouvelle."""
    user_id = _current_user_id()
    resolved_language, resolved_mode = _resolve_language(language), _resolve_mode(mode)
    current = session_repo.get_open(user_id, resolved_language, resolved_mode)
    if current:
        session_repo.close(int(current["id"]))
    return session_repo.create(user_id, resolved_language, resolved_mode)


@router.get("/session/current")
def current_session(language: str | None = None, mode: str | None = None) -> dict[str, Any]:
    user_id = _current_user_id()
    session = session_repo.get_or_create_open(
        user_id, _resolve_language(language), _resolve_mode(mode)
    )
    return {
        **session,
        "messages": message_repo.history(int(session["id"]), limit=100),
    }


@router.post("/session/{session_id}/close")
def close_session(session_id: int) -> dict[str, Any]:
    session = session_repo.close(session_id)
    if session is None:
        raise HTTPException(404, detail={"message": "Unknown session."})
    return session


# --------------------------------------------------------- tour de parole
def _handle_turn(
    *,
    user_id: int,
    text: str,
    language: str,
    mode: str,
    correction_mode: str | None,
    is_voice: bool,
    speak: bool,
    speed: float,
    duration_seconds: int = 0,
) -> dict[str, Any]:
    """Chemin commun aux tours vocaux et écrits."""
    command = detect_command(text)
    applied_command: dict[str, Any] | None = None
    if command:
        applied_command = {"action": command.action, **command.payload}
        language, mode, correction_mode, speed = _apply_command(
            command, language, mode, correction_mode, speed
        )

    session = session_repo.get_or_create_open(user_id, language, mode)
    session_id = int(session["id"])

    started = time.perf_counter()
    try:
        result = tutor.respond(
            user_id=user_id,
            session_id=session_id,
            text=text,
            language=language,
            mode=mode,
            correction_mode=correction_mode,
            is_voice=is_voice,
            duration_seconds=duration_seconds,
        )
    except LilianaError as exc:
        raise _fail(exc) from exc

    payload = result.as_dict()
    payload["command"] = applied_command
    payload["language"] = language
    payload["mode"] = mode
    payload["llm_elapsed"] = round(time.perf_counter() - started, 2)
    payload["speech"] = _speak(result.response, language, speed) if speak else None
    return payload


@router.post("/chat/turn")
def chat_turn(payload: TurnRequest) -> dict[str, Any]:
    """Tour de conversation écrit."""
    return _handle_turn(
        user_id=_current_user_id(),
        text=payload.text,
        language=_resolve_language(payload.language),
        mode=_resolve_mode(payload.mode),
        correction_mode=payload.correction_mode,
        is_voice=False,
        speak=payload.speak,
        speed=payload.speed,
    )


@router.post("/voice/turn")
async def voice_turn(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    mode: str | None = Form(default=None),
    correction_mode: str | None = Form(default=None),
    speak: bool = Form(default=True),
    speed: float = Form(default=1.0),
) -> dict[str, Any]:
    """Tour de conversation vocal : audio -> transcription -> réponse -> voix."""
    data = await audio.read()
    try:
        validate_upload(data, audio.content_type)
    except LilianaError as exc:
        raise _fail(exc, status=400) from exc

    maybe_save(data)  # no-op tant que SAVE_AUDIO=false

    resolved_language = _resolve_language(language)
    try:
        transcription = get_stt_provider().transcribe(data, language=resolved_language)
    except EmptyTranscriptionError as exc:
        raise _fail(exc, status=422) from exc
    except LilianaError as exc:
        raise _fail(exc) from exc

    payload = _handle_turn(
        user_id=_current_user_id(),
        text=transcription.text,
        language=resolved_language,
        mode=_resolve_mode(mode),
        correction_mode=correction_mode,
        is_voice=True,
        speak=speak,
        speed=max(0.3, min(speed, 2.0)),
        duration_seconds=int(transcription.duration),
    )
    payload["transcription"] = transcription.as_dict()
    return payload


@router.post("/chat/turn/stream")
def chat_turn_stream(payload: TurnRequest) -> StreamingResponse:
    """Tour écrit, diffusé en Server-Sent Events."""
    user_id = _current_user_id()
    language = _resolve_language(payload.language)
    mode = _resolve_mode(payload.mode)

    def events() -> Iterator[str]:
        try:
            yield from _stream_turn(
                user_id=user_id,
                text=payload.text,
                language=language,
                mode=mode,
                correction_mode=payload.correction_mode,
                is_voice=False,
                speak=payload.speak,
                speed=payload.speed,
            )
        except LilianaError as exc:
            yield _sse_error(exc)

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/voice/turn/stream")
async def voice_turn_stream(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    mode: str | None = Form(default=None),
    correction_mode: str | None = Form(default=None),
    speak: bool = Form(default=True),
    speed: float = Form(default=1.0),
) -> StreamingResponse:
    """Tour vocal diffusé de bout en bout.

    L'utilisateur voit sa transcription se former, puis la réponse s'écrire, et
    entend la première phrase pendant que la suite est encore en génération.
    """
    data = await audio.read()
    try:
        validate_upload(data, audio.content_type)
    except LilianaError as exc:
        raise _fail(exc, status=400) from exc

    maybe_save(data)  # no-op tant que SAVE_AUDIO=false

    user_id = _current_user_id()
    resolved_language = _resolve_language(language)
    resolved_mode = _resolve_mode(mode)
    resolved_speed = max(0.3, min(speed, 2.0))

    def events() -> Iterator[str]:
        try:
            transcription = None
            for step in get_stt_provider().transcribe_stream(data, language=resolved_language):
                if step.is_final:
                    transcription = step.transcription
                else:
                    yield _sse("transcription", {"text": step.text, "partial": True})

            if transcription is None:  # pragma: no cover - le flux lève avant
                raise EmptyTranscriptionError("no speech detected")
            yield _sse("transcription", {**transcription.as_dict(), "partial": False})

            yield from _stream_turn(
                user_id=user_id,
                text=transcription.text,
                language=resolved_language,
                mode=resolved_mode,
                correction_mode=correction_mode,
                is_voice=True,
                speak=speak,
                speed=resolved_speed,
                duration_seconds=int(transcription.duration),
            )
        except LilianaError as exc:
            yield _sse_error(exc)

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/speak")
def speak(payload: SpeakRequest) -> dict[str, Any]:
    """Synthèse à la demande (commande « repeat that », « speak more slowly »)."""
    speech = _speak(payload.text, _resolve_language(payload.language), payload.speed)
    if speech is None:
        available, detail = get_tts_provider().is_available()
        raise HTTPException(503, detail={"message": detail})
    return speech


# -------------------------------------------------------------- correction
@router.post("/correct")
def correct(payload: CorrectRequest) -> dict[str, Any]:
    user_id = _current_user_id()
    language = _resolve_language(payload.language)
    profile = language_profiles.get_or_create(user_id, language)
    settings = get_settings()
    mode = (
        payload.correction_mode
        or app_settings.get("correction_mode")
        or settings.correction_mode
    )
    try:
        result = correction_service.correct(
            payload.text, language, str(profile.get("level") or "A1"), mode
        )
    except LilianaError as exc:
        raise _fail(exc) from exc
    return result.as_dict()


@router.post("/explain")
def explain(payload: ExplainRequest) -> dict[str, Any]:
    user_id = _current_user_id()
    language = _resolve_language(payload.language)
    profile = language_profiles.get_or_create(user_id, language)
    try:
        text = correction_service.explain(
            payload.topic, language, str(profile.get("level") or "A1")
        )
    except LilianaError as exc:
        raise _fail(exc) from exc
    return {"topic": payload.topic, "explanation": text}


# --------------------------------------------------------------- exercices
@router.post("/exercise/generate")
def generate_exercise(payload: ExerciseRequest) -> dict[str, Any]:
    user_id = _current_user_id()
    language = _resolve_language(payload.language)
    profile = language_profiles.get_or_create(user_id, language)
    try:
        return exercise_service.generate(
            user_id,
            language,
            str(profile.get("level") or "A1"),
            topic=payload.topic,
            exercise_type=payload.exercise_type,
        )
    except LilianaError as exc:
        raise _fail(exc) from exc


@router.post("/exercise/check")
def check_exercise(payload: ExerciseAnswer) -> dict[str, Any]:
    user_id = _current_user_id()
    try:
        return exercise_service.check(user_id, payload.exercise_id, payload.answer)
    except LilianaError as exc:
        raise _fail(exc) from exc


# -------------------------------------------------------------- vocabulaire
@router.get("/vocabulary/due")
def vocabulary_due(language: str | None = None, limit: int = 10) -> dict[str, Any]:
    user_id = _current_user_id()
    resolved = _resolve_language(language)
    return {
        "language": resolved,
        "due": vocabulary_service.due_words(user_id, resolved, limit=max(1, min(limit, 50))),
        "known_words": vocabulary_repo.count(user_id, resolved),
    }


@router.post("/vocabulary/review")
def review_vocabulary(payload: VocabularyReview) -> dict[str, Any]:
    user_id = _current_user_id()
    return vocabulary_service.review_word(
        user_id, _resolve_language(payload.language), payload.word, payload.remembered
    )


@router.post("/vocabulary/teach")
def teach_vocabulary(payload: VocabularyTeach) -> dict[str, Any]:
    user_id = _current_user_id()
    language = _resolve_language(payload.language)
    profile = language_profiles.get_or_create(user_id, language)
    try:
        words = vocabulary_service.teach(
            user_id,
            language,
            str(profile.get("level") or "A1"),
            theme=payload.theme,
            count=payload.count,
        )
    except LilianaError as exc:
        raise _fail(exc) from exc
    return {"language": language, "theme": payload.theme, "words": words}


# ------------------------------------------------------------ prononciation
@router.post("/pronunciation/check")
async def check_pronunciation(
    audio: UploadFile = File(...),
    expected: str = Form(...),
    language: str | None = Form(default=None),
) -> dict[str, Any]:
    """Compare la phrase attendue à ce que le moteur STT entend réellement."""
    data = await audio.read()
    try:
        validate_upload(data, audio.content_type)
    except LilianaError as exc:
        raise _fail(exc, status=400) from exc

    resolved = _resolve_language(language)
    try:
        transcription = get_stt_provider().transcribe(data, language=resolved)
    except EmptyTranscriptionError as exc:
        raise _fail(exc, status=422) from exc
    except LilianaError as exc:
        raise _fail(exc) from exc

    result = pronunciation_module.analyse(expected, transcription.text, resolved)
    pronunciation_repo.add(
        _current_user_id(),
        resolved,
        expected,
        transcription.text,
        result.score,
        result.problem_sounds,
    )
    payload = result.as_dict()
    payload["transcription"] = transcription.as_dict()
    return payload


# ------------------------------------------------------------- évaluation
@router.get("/assessment/{language}")
def assessment_test(language: str) -> dict[str, Any]:
    if not is_supported(language):
        raise HTTPException(404, detail={"message": f"Unknown language '{language}'."})
    return build_test(language.lower())


@router.post("/assessment")
def submit_assessment(payload: AssessmentSubmission) -> dict[str, Any]:
    if not is_supported(payload.language):
        raise HTTPException(404, detail={"message": f"Unknown language '{payload.language}'."})
    return assessment_service.evaluate(
        _current_user_id(), payload.language.lower(), payload.answers, payload.productions
    )


# -------------------------------------------------------------- dashboard
@router.get("/dashboard")
def dashboard() -> dict[str, Any]:
    """Tableau de bord toutes langues confondues (§25)."""
    user_id = _current_user_id()
    return {
        "languages": [
            progress_tracker.dashboard(user_id, language) for language in TARGET_LANGUAGES
        ],
        "sessions": session_repo.list_recent(user_id, limit=10),
    }


@router.get("/lesson")
def lesson(language: str | None = None, minutes: int = 30) -> dict[str, Any]:
    """Plan d'une séance guidée (bouton START LESSON, §26)."""
    user_id = _current_user_id()
    resolved = _resolve_language(language)
    profile = language_profiles.get_or_create(user_id, resolved)
    return build_lesson(user_id, resolved, str(profile.get("level") or "A1"), minutes=minutes)
