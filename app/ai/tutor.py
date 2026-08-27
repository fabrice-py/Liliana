"""Cœur pédagogique de Liliana.

Orchestre un tour de conversation complet :

    texte utilisateur -> contexte -> LLM -> JSON structuré -> mémoire -> réponse

C'est le seul module qui décide de ce qui est enregistré en base à l'issue d'un
tour de parole.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app.ai.llm import LLMProvider, get_llm_provider
from app.ai.prompts import (
    TURN_RESPONSE_SCHEMA,
    TutorContext,
    build_turn_prompt,
    get_mode,
)
from app.ai.structured import (
    ResponseStreamParser,
    extract_json,
    is_meaningful_response,
    normalise_turn,
)
from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logger import get_logger
from app.database.repositories import (
    errors as error_repo,
    language_profiles,
    messages as message_repo,
    sessions as session_repo,
    users as user_repo,
    vocabulary as vocabulary_repo,
)
from app.language.languages import get_language, is_supported
from app.learning.progress import progress_tracker
from app.learning.spaced_repetition import spaced_repetition
from app.speech.tts import SentenceBuffer

logger = get_logger(__name__)


@dataclass(slots=True)
class TurnResult:
    """Résultat complet d'un tour de conversation."""

    response: str
    correction: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    vocabulary: list[dict[str, Any]] = field(default_factory=list)
    detected_language: str = ""
    difficulty: str = ""
    session_id: int | None = None
    structured: bool = True
    level: str = "A1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "correction": self.correction,
            "errors": self.errors,
            "vocabulary": self.vocabulary,
            "detected_language": self.detected_language,
            "difficulty": self.difficulty,
            "session_id": self.session_id,
            "structured": self.structured,
            "level": self.level,
        }


@dataclass(slots=True)
class TurnEvent:
    """Étape d'un tour de conversation diffusé en continu.

    ``kind`` vaut :

    * ``delta``    — un fragment de texte à afficher immédiatement ;
    * ``sentence`` — une phrase complète, prête à être synthétisée ;
    * ``done``     — le tour est terminé, ``result`` porte le tour complet.
    """

    kind: str
    text: str = ""
    result: "TurnResult | None" = None


class Tutor:
    """Assemble contexte, LLM et mémoire pour produire un tour de conversation."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        return self._llm or get_llm_provider()

    # ------------------------------------------------------------- contexte
    def build_context(
        self, user_id: int, language: str, mode: str, correction_mode: str | None = None
    ) -> TutorContext:
        """Rassemble tout ce que Liliana sait de l'utilisateur pour cette langue."""
        settings = get_settings()
        user = user_repo.get(user_id) or {}
        profile = language_profiles.get_or_create(user_id, language)
        conversation_mode = get_mode(mode)

        effective_correction = (
            correction_mode
            or self._stored_correction_mode()
            or conversation_mode.default_correction_mode
            or settings.correction_mode
        )

        return TutorContext(
            language=language,
            level=str(profile.get("level") or "A1"),
            mode=conversation_mode.key,
            correction_mode=effective_correction,
            native_language=str(user.get("native_language") or "french"),
            weaknesses=error_repo.top_weaknesses(user_id, language, limit=5),
            recent_errors=error_repo.recent(user_id, language, limit=8),
            recent_vocabulary=[
                item["word"] for item in vocabulary_repo.list_for(user_id, language, limit=20)
            ],
            review_items=spaced_repetition.due_keys(user_id, language, limit=10),
        )

    @staticmethod
    def _stored_correction_mode() -> str | None:
        from app.database.repositories import app_settings

        return app_settings.get("correction_mode")

    # ------------------------------------------------------------ historique
    def _history(self, session_id: int) -> list[dict[str, str]]:
        """Historique envoyé au modèle : uniquement des échanges complets.

        La base conserve tout, y compris les tours ratés — c'est la trace de ce
        qui s'est passé. Mais le modèle, lui, ne doit relire qu'une conversation
        bien formée, question/réponse en alternance stricte. Deux écueils sinon :

        * une réponse vide (« {} ») relue dans l'historique se fait imiter au
          tour suivant, et la session ne produit plus que du vide ;
        * une question restée sans réponse, quand le tour a échoué, laisse le
          modèle face à plusieurs questions d'affilée — une forme qu'il n'a
          jamais vue à l'entraînement, et à laquelle il répond mal.

        Une session déjà abîmée se remet donc d'elle-même au tour suivant.
        """
        limit = get_settings().llm_max_history_turns * 2
        turns: list[dict[str, str]] = []

        for message in message_repo.history(session_id, limit=limit):
            role, content = message["role"], message["content"]
            if role not in ("user", "assistant") or not content:
                continue

            if role == "user":
                if turns and turns[-1]["role"] == "user":
                    turns.pop()  # la précédente est restée sans réponse
                turns.append({"role": "user", "content": content})
            elif turns and turns[-1]["role"] == "user":
                if is_meaningful_response(content):
                    turns.append({"role": "assistant", "content": content})
                else:
                    turns.pop()  # tour raté : on retire la question avec lui

        if turns and turns[-1]["role"] == "user":
            turns.pop()  # le tour en cours est ajouté séparément par _prepare
        return turns

    # ----------------------------------------------------------------- tour
    def _prepare(
        self,
        *,
        user_id: int,
        session_id: int,
        text: str,
        language: str,
        mode: str,
        correction_mode: str | None,
        is_voice: bool,
    ) -> tuple[str, list[dict[str, str]]]:
        """Valide l'entrée, enregistre le tour utilisateur, assemble le prompt.

        Retourne (langue effective, conversation prête pour le modèle).
        """
        text = (text or "").strip()
        if not text:
            raise LLMError(
                "empty user turn",
                user_message="Liliana did not receive anything to answer.",
            )
        if not is_supported(language):
            language = get_settings().default_language

        context = self.build_context(user_id, language, mode, correction_mode)
        history = self._history(session_id)

        # Le message utilisateur est enregistré avant l'appel LLM : si le modèle
        # échoue, ce que l'utilisateur a dit n'est pas perdu.
        message_repo.add(session_id, "user", text, language, is_voice)

        return language, [
            {"role": "system", "content": build_turn_prompt(context)},
            *history,
            {"role": "user", "content": text},
        ]

    def _finalise(
        self,
        *,
        user_id: int,
        session_id: int,
        language: str,
        turn: dict[str, Any],
        duration_seconds: int,
        is_voice: bool,
    ) -> TurnResult:
        """Contrôle la réponse, l'enregistre et met à jour la mémoire."""
        if not turn["response"]:
            raise LLMError(
                "model returned nothing usable",
                user_message="Liliana could not produce an answer. Please try again.",
            )

        message_repo.add(session_id, "assistant", turn["response"], language)
        self._persist_learning(user_id, session_id, language, turn, duration_seconds, is_voice)

        profile = language_profiles.get_or_create(user_id, language)
        return TurnResult(
            response=turn["response"],
            correction=turn["correction"],
            errors=turn["errors"],
            vocabulary=turn["vocabulary"],
            detected_language=turn["detected_language"],
            difficulty=turn["difficulty"],
            session_id=session_id,
            structured=turn["structured"],
            level=str(profile.get("level") or "A1"),
        )

    def respond(
        self,
        *,
        user_id: int,
        session_id: int,
        text: str,
        language: str,
        mode: str,
        correction_mode: str | None = None,
        is_voice: bool = False,
        duration_seconds: int = 0,
    ) -> TurnResult:
        """Traite un tour de parole et met à jour la mémoire de Liliana."""
        language, conversation = self._prepare(
            user_id=user_id, session_id=session_id, text=text, language=language,
            mode=mode, correction_mode=correction_mode, is_voice=is_voice,
        )
        raw = self.llm.generate(
            conversation,
            json_mode=True,
            schema=TURN_RESPONSE_SCHEMA,
            model=get_settings().llm_model_for(language),
        )
        turn = normalise_turn(extract_json(raw), fallback_text=raw)

        return self._finalise(
            user_id=user_id, session_id=session_id, language=language, turn=turn,
            duration_seconds=duration_seconds, is_voice=is_voice,
        )

    def respond_stream(
        self,
        *,
        user_id: int,
        session_id: int,
        text: str,
        language: str,
        mode: str,
        correction_mode: str | None = None,
        is_voice: bool = False,
        duration_seconds: int = 0,
    ) -> Iterator[TurnEvent]:
        """Même tour, diffusé au fil de la génération.

        Émet le texte dès qu'il arrive puis, phrase par phrase, de quoi
        synthétiser la voix sans attendre la fin de la réponse. Le JSON complet
        (correction, erreurs, vocabulaire) n'est exploité qu'à la fin, mais la
        clé ``response`` vient en tête du contrat de sortie : elle est donc
        disponible bien avant.
        """
        language, conversation = self._prepare(
            user_id=user_id, session_id=session_id, text=text, language=language,
            mode=mode, correction_mode=correction_mode, is_voice=is_voice,
        )

        parser = ResponseStreamParser()
        sentences = SentenceBuffer()

        def emit(delta: str) -> Iterator[TurnEvent]:
            if not delta:
                return
            yield TurnEvent(kind="delta", text=delta)
            for sentence in sentences.feed(delta):
                yield TurnEvent(kind="sentence", text=sentence)

        chunks = self.llm.stream(
            conversation,
            json_mode=True,
            schema=TURN_RESPONSE_SCHEMA,
            model=get_settings().llm_model_for(language),
        )
        for chunk in chunks:
            yield from emit(parser.feed(chunk))

        yield from emit(parser.flush())
        if remainder := sentences.flush():
            yield TurnEvent(kind="sentence", text=remainder)

        turn = parser.finish()
        yield TurnEvent(
            kind="done",
            result=self._finalise(
                user_id=user_id, session_id=session_id, language=language, turn=turn,
                duration_seconds=duration_seconds, is_voice=is_voice,
            ),
        )

    # ----------------------------------------------------------- mémoire
    def _persist_learning(
        self,
        user_id: int,
        session_id: int,
        language: str,
        turn: dict[str, Any],
        duration_seconds: int,
        is_voice: bool = False,
    ) -> None:
        """Enregistre erreurs, vocabulaire, planning de révision et statistiques."""
        allowed = set(get_language(language).error_types)
        recorded_errors = [
            error for error in turn["errors"] if error.get("topic") or error.get("type")
        ]
        for error in recorded_errors:
            # Un modèle local invente parfois des types : on retombe sur "grammar"
            # plutôt que de polluer les statistiques.
            if error["type"] not in allowed:
                error["type"] = "grammar"

        error_repo.add_many(user_id, session_id, language, recorded_errors, is_voice=is_voice)
        added_words = vocabulary_repo.add_many(user_id, language, turn["vocabulary"])

        # Les points faibles et les mots neufs entrent dans la répétition espacée.
        spaced_repetition.register_many(
            user_id,
            language,
            "grammar",
            [error["topic"] for error in recorded_errors if error.get("topic")],
        )
        spaced_repetition.register_many(user_id, language, "vocabulary", added_words)

        session_repo.touch(session_id, messages=1, errors=len(recorded_errors))
        progress_tracker.record_turn(
            user_id,
            language,
            seconds=duration_seconds,
            errors_found=len(recorded_errors),
            words_learned=len(added_words),
        )


tutor = Tutor()
