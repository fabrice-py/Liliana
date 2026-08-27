"""Évaluation initiale du niveau (cf. §15).

Deux parties :

1. **Objective** — un questionnaire à choix multiple gradué A1 -> C1, corrigé
   localement, sans LLM. Rapide, déterministe, fonctionne même si Ollama est
   éteint.
2. **Production** — l'utilisateur répond librement (à l'oral ou à l'écrit) à
   deux consignes ; le LLM évalue et produit le profil final.

Le résultat alimente le profil linguistique (niveau + scores par compétence).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.ai.llm import LLMProvider, get_llm_provider
from app.ai.prompts import ASSESSMENT_SCHEMA_PROMPT
from app.ai.structured import extract_json, normalise_assessment
from app.core.logger import get_logger
from app.database.repositories import language_profiles, users as user_repo
from app.language.languages import (
    CEFR_LEVELS,
    SKILLS,
    clamp_level,
    get_language,
    level_index,
    score_to_level,
)
from app.learning.progress import progress_tracker

logger = get_logger(__name__)


@dataclass(slots=True)
class AssessmentItem:
    id: str
    level: str
    skill: str
    question: str
    options: list[str]
    answer: str

    def public(self) -> dict[str, Any]:
        """Version envoyée au client : sans la bonne réponse."""
        payload = asdict(self)
        payload.pop("answer")
        return payload


@dataclass(slots=True)
class ProductionTask:
    id: str
    level: str
    skill: str
    prompt: str

    def public(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------- banque QCM
_ENGLISH_ITEMS: tuple[AssessmentItem, ...] = (
    AssessmentItem("en-a1-1", "A1", "grammar", "She ___ a teacher.",
                   ["is", "are", "am", "be"], "is"),
    AssessmentItem("en-a1-2", "A1", "vocabulary", "You eat breakfast in the ___.",
                   ["morning", "evening", "night", "week"], "morning"),
    AssessmentItem("en-a2-1", "A2", "grammar", "Yesterday I ___ to the cinema.",
                   ["went", "go", "goed", "gone"], "went"),
    AssessmentItem("en-a2-2", "A2", "grammar", "There ___ any milk left.",
                   ["isn't", "aren't", "don't", "hasn't"], "isn't"),
    AssessmentItem("en-b1-1", "B1", "grammar", "If it ___ tomorrow, we'll stay home.",
                   ["rains", "will rain", "rained", "would rain"], "rains"),
    AssessmentItem("en-b1-2", "B1", "vocabulary", "I'm looking ___ to the weekend.",
                   ["forward", "ahead", "after", "up"], "forward"),
    AssessmentItem("en-b2-1", "B2", "grammar", "He said he ___ the report by then.",
                   ["had finished", "has finished", "finishes", "would finishing"],
                   "had finished"),
    AssessmentItem("en-b2-2", "B2", "reading",
                   "\"The proposal was met with lukewarm enthusiasm.\" This means people were:",
                   ["not very enthusiastic", "extremely excited", "openly hostile", "confused"],
                   "not very enthusiastic"),
    AssessmentItem("en-c1-1", "C1", "grammar",
                   "___ had I sat down when the phone rang.",
                   ["Hardly", "Rarely if", "Almost", "Barely than"], "Hardly"),
    AssessmentItem("en-c1-2", "C1", "vocabulary",
                   "Her argument was ___: nobody could find a flaw in it.",
                   ["watertight", "waterlogged", "watered down", "under water"], "watertight"),
)

_GERMAN_ITEMS: tuple[AssessmentItem, ...] = (
    AssessmentItem("de-a1-1", "A1", "grammar", "Ich ___ aus Frankreich.",
                   ["komme", "kommst", "kommt", "kommen"], "komme"),
    AssessmentItem("de-a1-2", "A1", "vocabulary", "___ Tisch steht im Zimmer.",
                   ["Der", "Die", "Das", "Den"], "Der"),
    AssessmentItem("de-a2-1", "A2", "grammar", "Ich gebe ___ Mann das Buch.",
                   ["dem", "den", "der", "des"], "dem"),
    AssessmentItem("de-a2-2", "A2", "grammar", "Gestern ___ ich ins Kino gegangen.",
                   ["bin", "habe", "war", "hatte"], "bin"),
    AssessmentItem("de-b1-1", "B1", "grammar", "Ich weiß, dass er morgen ___.",
                   ["kommt", "kommt an morgen", "ankommt morgen", "an kommt"], "kommt"),
    AssessmentItem("de-b1-2", "B1", "grammar", "Wir treffen uns ___ dem Bahnhof.",
                   ["vor", "für", "über", "durch"], "vor"),
    AssessmentItem("de-b2-1", "B2", "grammar", "Das ist der Mann, ___ Auto gestohlen wurde.",
                   ["dessen", "deren", "den", "dem"], "dessen"),
    AssessmentItem("de-b2-2", "B2", "grammar", "Wenn ich mehr Zeit ___, würde ich reisen.",
                   ["hätte", "habe", "hatte", "haben würde"], "hätte"),
    AssessmentItem("de-c1-1", "C1", "grammar",
                   "Der Vertrag ist ___ der neuen Regelung ungültig.",
                   ["aufgrund", "trotzdem", "obwohl", "während dem"], "aufgrund"),
    AssessmentItem("de-c1-2", "C1", "vocabulary",
                   "Seine Erklärung war an den Haaren ___.",
                   ["herbeigezogen", "hergestellt", "hingelegt", "abgeholt"], "herbeigezogen"),
)

_ITEM_BANK: dict[str, tuple[AssessmentItem, ...]] = {
    "english": _ENGLISH_ITEMS,
    "german": _GERMAN_ITEMS,
}

_PRODUCTION_TASKS: dict[str, tuple[ProductionTask, ...]] = {
    "english": (
        ProductionTask("en-prod-1", "A2", "speaking",
                       "Introduce yourself and describe what you did last weekend."),
        ProductionTask("en-prod-2", "B2", "writing",
                       "Do you think working from home is a good thing? Explain why."),
    ),
    "german": (
        ProductionTask("de-prod-1", "A2", "speaking",
                       "Stellen Sie sich vor und erzählen Sie, was Sie letztes Wochenende "
                       "gemacht haben."),
        ProductionTask("de-prod-2", "B2", "writing",
                       "Finden Sie Homeoffice gut? Begründen Sie Ihre Meinung."),
    ),
}


def get_items(language: str) -> tuple[AssessmentItem, ...]:
    return _ITEM_BANK.get(language, _ENGLISH_ITEMS)


def get_production_tasks(language: str) -> tuple[ProductionTask, ...]:
    return _PRODUCTION_TASKS.get(language, _PRODUCTION_TASKS["english"])


def build_test(language: str) -> dict[str, Any]:
    """Questionnaire complet envoyé au client (sans les réponses)."""
    return {
        "language": language,
        "language_name": get_language(language).english_name,
        "items": [item.public() for item in get_items(language)],
        "production": [task.public() for task in get_production_tasks(language)],
    }


# ------------------------------------------------------------- correction
@dataclass(slots=True)
class ObjectiveScore:
    correct: int
    total: int
    per_level: dict[str, tuple[int, int]] = field(default_factory=dict)
    estimated_level: str = "A1"

    @property
    def percent(self) -> float:
        return round(100.0 * self.correct / self.total, 1) if self.total else 0.0


def score_objective(language: str, answers: dict[str, str]) -> ObjectiveScore:
    """Corrige le QCM localement.

    Le niveau estimé est le plus haut palier où l'utilisateur répond
    majoritairement juste — un palier raté fait s'arrêter la progression.
    """
    items = get_items(language)
    per_level: dict[str, list[int]] = {level: [0, 0] for level in CEFR_LEVELS}
    correct = 0

    for item in items:
        given = str(answers.get(item.id, "")).strip()
        per_level[item.level][1] += 1
        if given and given.lower() == item.answer.lower():
            correct += 1
            per_level[item.level][0] += 1

    estimated = "A1"
    for level in CEFR_LEVELS:
        got, total = per_level[level]
        if total == 0:
            continue
        if got / total >= 0.5:
            estimated = level
        else:
            break

    # Réussir tous les paliers testés justifie un cran de plus.
    top_level = max((item.level for item in items), key=level_index, default="A1")
    if estimated == top_level and correct == len(items):
        estimated = clamp_level(level_index(top_level) + 1)

    return ObjectiveScore(
        correct=correct,
        total=len(items),
        per_level={level: tuple(counts) for level, counts in per_level.items() if counts[1]},
        estimated_level=estimated,
    )


class AssessmentService:
    """Évalue la production libre et enregistre le profil final."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        return self._llm or get_llm_provider()

    def evaluate(
        self,
        user_id: int,
        language: str,
        answers: dict[str, str],
        productions: dict[str, str],
    ) -> dict[str, Any]:
        """Corrige le test, évalue la production et enregistre le profil."""
        objective = score_objective(language, answers)
        tasks = {task.id: task for task in get_production_tasks(language)}

        written = "\n\n".join(
            f"Task ({tasks[task_id].skill}, target {tasks[task_id].level}): "
            f"{tasks[task_id].prompt}\nLearner answer: {text.strip()}"
            for task_id, text in productions.items()
            if task_id in tasks and text and text.strip()
        )

        llm_result: dict[str, Any] | None = None
        if written:
            language_name = get_language(language).english_name
            system = (
                f"You are a {language_name} examiner using the CEFR scale. "
                f"A multiple-choice test already placed this learner around "
                f"{objective.estimated_level} "
                f"({objective.correct}/{objective.total} correct). "
                "Judge the free production below and produce the final profile. "
                "Be honest but not harsh; scores are percentages within the CEFR scale.\n\n"
                + ASSESSMENT_SCHEMA_PROMPT
            )
            try:
                raw = self.llm.generate(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": written},
                    ],
                    temperature=0.2,
                    json_mode=True,
                )
                llm_result = normalise_assessment(extract_json(raw))
            except Exception as exc:  # noqa: BLE001 - l'évaluation ne doit jamais bloquer
                logger.warning("Évaluation LLM indisponible, repli sur le QCM : %s", exc)

        scores, level, summary = self._merge(objective, llm_result)

        language_profiles.update_scores(user_id, language, scores, level=level, assessed=True)
        progress_tracker.refresh(user_id, language)
        user_repo.mark_onboarded(user_id)

        return {
            "language": language,
            "level": level,
            "scores": scores,
            "objective": {
                "correct": objective.correct,
                "total": objective.total,
                "percent": objective.percent,
                "estimated_level": objective.estimated_level,
                "per_level": {
                    key: {"correct": value[0], "total": value[1]}
                    for key, value in objective.per_level.items()
                },
            },
            "strengths": (llm_result or {}).get("strengths", []),
            "weaknesses": (llm_result or {}).get("weaknesses", []),
            "summary": summary,
            "llm_used": llm_result is not None,
        }

    # ------------------------------------------------------------- interne
    @staticmethod
    def _merge(
        objective: ObjectiveScore, llm_result: dict[str, Any] | None
    ) -> tuple[dict[str, float], str, str]:
        """Combine QCM (objectif) et jugement du LLM (production)."""
        baseline = 100.0 * (level_index(objective.estimated_level) + 0.5) / len(CEFR_LEVELS)
        scores: dict[str, float] = {skill: round(baseline, 1) for skill in SKILLS}
        # Le QCM mesure directement grammaire et vocabulaire.
        scores["grammar"] = round(0.5 * baseline + 0.5 * objective.percent, 1)
        scores["vocabulary"] = round(0.6 * baseline + 0.4 * objective.percent, 1)

        if llm_result:
            for skill, value in llm_result.get("scores", {}).items():
                if skill in SKILLS:
                    scores[skill] = round(0.4 * scores[skill] + 0.6 * float(value), 1)
            level = llm_result.get("level") or objective.estimated_level
            summary = llm_result.get("summary") or ""
        else:
            level = objective.estimated_level
            summary = (
                f"Based on the placement test you answered "
                f"{objective.correct}/{objective.total} correctly. "
                "Liliana will refine this estimate as you talk to her."
            )

        # Cohérence finale : le niveau doit rester proche des scores obtenus.
        derived = score_to_level(sum(scores.values()) / len(scores))
        if abs(level_index(derived) - level_index(level)) > 1:
            level = clamp_level((level_index(derived) + level_index(level)) // 2)

        return scores, level, summary


assessment_service = AssessmentService()
