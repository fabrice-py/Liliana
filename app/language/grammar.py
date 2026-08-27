"""Génération et correction d'exercices de grammaire (cf. §8 mode 5).

Les exercices sont générés par le LLM local, ciblés sur les points faibles réels
de l'utilisateur, puis stockés pour suivre les résultats.
"""

from __future__ import annotations

from typing import Any

from app.ai.llm import LLMProvider, get_llm_provider
from app.ai.prompts import ANSWER_CHECK_SCHEMA_PROMPT, EXERCISE_SCHEMA_PROMPT
from app.ai.structured import extract_json, normalise_answer_check, normalise_exercise
from app.core.exceptions import LLMError
from app.core.logger import get_logger
from app.database.repositories import errors as error_repo, exercises as exercise_repo
from app.language.languages import get_language
from app.learning.progress import progress_tracker
from app.learning.spaced_repetition import spaced_repetition

logger = get_logger(__name__)

EXERCISE_TYPES: tuple[str, ...] = (
    "multiple_choice",
    "fill_in_the_blank",
    "conjugation",
    "sentence_correction",
    "transformation",
    "translation",
    "sentence_building",
)


class ExerciseService:
    """Fabrique des exercices adaptés et évalue les réponses."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        return self._llm or get_llm_provider()

    # ----------------------------------------------------------- ciblage
    def pick_topic(self, user_id: int, language: str) -> str:
        """Choisit le point à travailler : révision due, sinon point faible."""
        due = spaced_repetition.due(user_id, language, limit=1, item_type="grammar")
        if due:
            return str(due[0]["item_key"])
        weaknesses = error_repo.top_weaknesses(user_id, language, limit=1)
        if weaknesses:
            return str(weaknesses[0]["topic"])
        return "general review"

    # -------------------------------------------------------- génération
    def generate(
        self,
        user_id: int,
        language: str,
        level: str,
        topic: str | None = None,
        exercise_type: str | None = None,
    ) -> dict[str, Any]:
        """Génère un exercice et l'enregistre."""
        topic = topic or self.pick_topic(user_id, language)
        language_name = get_language(language).english_name

        constraints = [
            f"Target language: {language_name}.",
            f"Learner CEFR level: {level}.",
            f"Grammar or vocabulary topic to practise: {topic}.",
        ]
        if exercise_type in EXERCISE_TYPES:
            constraints.append(f"Exercise type: {exercise_type}.")
        else:
            constraints.append(f"Choose the most suitable type among: {', '.join(EXERCISE_TYPES)}.")
        constraints.append(
            "The exercise must be solvable in one short answer and written in the "
            "target language (instructions may use the target language too)."
        )

        raw = self.llm.generate(
            [
                {"role": "system", "content": EXERCISE_SCHEMA_PROMPT},
                {"role": "user", "content": "\n".join(constraints)},
            ],
            temperature=0.6,
            json_mode=True,
        )
        exercise = normalise_exercise(extract_json(raw))
        if exercise is None:
            raise LLMError(
                "model did not produce a usable exercise",
                user_message="Liliana could not build an exercise. Please try again.",
            )
        exercise.setdefault("topic", topic)
        if not exercise["topic"]:
            exercise["topic"] = topic

        stored = exercise_repo.create(user_id, language, exercise)
        stored["options"] = exercise["options"]
        # On ne renvoie jamais la réponse attendue au client avant qu'il ait répondu.
        stored.pop("answer", None)
        stored.pop("explanation", None)
        return stored

    # ---------------------------------------------------------- correction
    def check(
        self,
        user_id: int,
        exercise_id: int,
        user_answer: str,
        session_id: int | None = None,
    ) -> dict[str, Any]:
        """Évalue une réponse, enregistre le résultat et replanifie la révision."""
        exercise = exercise_repo.get(exercise_id)
        if not exercise:
            raise LLMError(
                f"unknown exercise {exercise_id}",
                user_message="Liliana lost track of that exercise. Ask for a new one.",
            )

        user_answer = (user_answer or "").strip()
        expected = str(exercise.get("answer") or "").strip()

        # Cas trivial : réponse strictement identique, on n'appelle pas le LLM.
        if expected and user_answer.lower().rstrip(".!?") == expected.lower().rstrip(".!?"):
            result = {
                "is_correct": True,
                "feedback": "Exactly right!",
                "corrected": expected,
                "errors": [],
            }
        else:
            question = (
                f"Language: {get_language(exercise['language']).english_name}\n"
                f"Topic: {exercise.get('topic')}\n"
                f"Exercise: {exercise.get('prompt')}\n"
                f"Expected answer: {expected or '(not provided)'}\n"
                f"Learner answer: {user_answer or '(empty)'}\n\n"
                "Decide whether the learner answer is acceptable. Accept equivalent "
                "correct answers even if they differ from the expected one."
            )
            raw = self.llm.generate(
                [
                    {"role": "system", "content": ANSWER_CHECK_SCHEMA_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
                json_mode=True,
            )
            result = normalise_answer_check(extract_json(raw))
            if not result["corrected"]:
                result["corrected"] = expected

        language = str(exercise["language"])
        exercise_repo.record_result(
            exercise_id,
            user_id,
            session_id,
            user_answer,
            result["is_correct"],
            result["feedback"],
        )
        progress_tracker.record_exercise(user_id, language, result["is_correct"])

        topic = str(exercise.get("topic") or "").strip()
        if topic:
            # Qualité SM-2 : 5 si réussi, 2 si raté.
            spaced_repetition.review(
                user_id, language, "grammar", topic, quality=5 if result["is_correct"] else 2
            )
        if not result["is_correct"]:
            error_repo.add_many(
                user_id,
                session_id,
                language,
                [
                    {
                        "type": error.get("type", "grammar"),
                        "topic": error.get("topic") or topic,
                        "original": user_answer,
                        "corrected": result["corrected"],
                        "explanation": result["feedback"],
                        "severity": error.get("severity", "minor"),
                    }
                    for error in (result["errors"] or [{"type": "grammar", "topic": topic}])
                ],
            )

        result["explanation"] = str(exercise.get("explanation") or "")
        result["exercise_id"] = exercise_id
        result["topic"] = topic
        return result


exercise_service = ExerciseService()
