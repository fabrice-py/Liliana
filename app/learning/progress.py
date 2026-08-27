"""Calcul de la progression et du profil linguistique.

Les scores par compétence (§14) ne sont jamais codés en dur : ils sont recalculés
à partir des données réellement produites par l'utilisateur (messages, erreurs,
exercices, vocabulaire, prononciation), puis lissés pour éviter les à-coups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.logger import get_logger
from app.database.repositories import (
    errors as error_repo,
    exercises as exercise_repo,
    language_profiles,
    messages as message_repo,
    progress as progress_repo,
    pronunciation as pronunciation_repo,
    reviews as review_repo,
    vocabulary as vocabulary_repo,
)
from app.language.languages import SKILLS, level_index, score_to_level

logger = get_logger(__name__)

#: Poids du nouveau score dans la moyenne glissante. Faible = progression douce.
SMOOTHING = 0.25

#: En dessous de ce nombre de messages, les scores restent des estimations.
MIN_MESSAGES_FOR_CONFIDENCE = 10


@dataclass(slots=True)
class SkillSnapshot:
    scores: dict[str, float]
    level: str
    overall: float
    is_estimate: bool


def _pct(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _accuracy_score(total: int, mistakes: int, *, baseline: float = 45.0) -> float:
    """Score dérivé d'un taux d'erreur.

    Peu de données -> on reste proche de la valeur de base pour ne pas conclure
    trop vite. Beaucoup de données -> le taux d'erreur domine.
    """
    if total <= 0:
        return baseline
    error_rate = mistakes / total
    measured = 100.0 * max(0.0, 1.0 - error_rate)
    confidence = min(total / MIN_MESSAGES_FOR_CONFIDENCE, 1.0)
    return baseline * (1 - confidence) + measured * confidence


class ProgressTracker:
    """Recalcule le profil d'une langue et alimente le tableau de bord."""

    # ------------------------------------------------------------- calcul
    def compute_scores(self, user_id: int, language: str) -> SkillSnapshot:
        profile = language_profiles.get_or_create(user_id, language)

        total_messages = message_repo.count_for_user(user_id, language)
        exercise_stats = exercise_repo.stats(user_id, language)
        never_assessed = not profile.get("assessed_at")

        # Aucune donnée : on n'invente pas un niveau. Le profil reste tel quel
        # (A1 par défaut) jusqu'à ce que l'évaluation initiale ou une vraie
        # conversation apporte de la matière.
        if total_messages == 0 and exercise_stats["done"] == 0 and never_assessed:
            stored = {skill: float(profile.get(skill, 0.0)) for skill in SKILLS}
            return SkillSnapshot(
                scores=stored,
                level=str(profile.get("level") or "A1"),
                overall=round(sum(stored.values()) / len(stored), 1),
                is_estimate=True,
            )

        total_errors = error_repo.count(user_id, language)
        recent_errors = error_repo.recent(user_id, language, limit=200)

        by_kind: dict[str, int] = {}
        for error in recent_errors:
            key = str(error.get("error_type") or "grammar")
            by_kind[key] = by_kind.get(key, 0) + 1

        grammar_like = sum(
            count
            for kind, count in by_kind.items()
            if kind
            not in ("vocabulary", "false_friends", "spelling", "pronunciation", "register")
        )
        vocabulary_like = sum(
            by_kind.get(kind, 0) for kind in ("vocabulary", "false_friends", "spelling")
        )

        known_words = vocabulary_repo.count(user_id, language)
        review_items = review_repo.all_for(user_id, language)
        mean_confidence = (
            sum(float(item.get("confidence", 0.0)) for item in review_items) / len(review_items)
            if review_items
            else 0.0
        )

        # --- grammaire : justesse en conversation + réussite aux exercices
        grammar = _accuracy_score(total_messages, grammar_like)
        if exercise_stats["done"]:
            drill = 100.0 * exercise_stats["correct"] / exercise_stats["done"]
            weight = min(exercise_stats["done"] / 20.0, 0.5)
            grammar = grammar * (1 - weight) + drill * weight

        # --- vocabulaire : étendue (mots connus) + solidité (confiance SRS)
        breadth = 100.0 * (1.0 - 0.5 ** (known_words / 150.0)) if known_words else 0.0
        vocabulary = 0.6 * breadth + 40.0 * mean_confidence
        vocabulary = max(vocabulary, _accuracy_score(total_messages, vocabulary_like) * 0.6)

        # --- oral / écrit : justesse mesurée séparément sur chaque canal.
        # Sans données sur un canal, on n'invente rien : la baseline s'applique
        # (_accuracy_score retourne la baseline quand le total vaut 0).
        spoken_turns = message_repo.count_for_user(user_id, language, is_voice=True)
        written_turns = message_repo.count_for_user(user_id, language, is_voice=False)
        spoken_errors = error_repo.count(user_id, language, is_voice=True)
        written_errors = error_repo.count(user_id, language, is_voice=False)

        speaking = _accuracy_score(spoken_turns, spoken_errors, baseline=40.0)
        writing = _accuracy_score(written_turns, written_errors, baseline=45.0)

        # --- compréhension : proxy = volume d'échange réellement soutenu
        listening = _pct(35.0 + min(total_messages, 200) * 0.2)

        # --- prononciation : moyenne des tentatives, sinon estimation prudente
        pronunciation_avg = pronunciation_repo.average_score(user_id, language)
        pronunciation = (
            _pct(pronunciation_avg)
            if pronunciation_avg is not None
            else _pct(speaking * 0.85)
        )

        computed = {
            "grammar": _pct(grammar),
            "vocabulary": _pct(vocabulary),
            "speaking": _pct(speaking),
            "listening": listening,
            "writing": _pct(writing),
            "pronunciation": pronunciation,
        }

        # Lissage exponentiel avec les scores déjà enregistrés.
        smoothed = {
            skill: _pct(
                float(profile.get(skill, 0.0)) * (1 - SMOOTHING) + computed[skill] * SMOOTHING
                if float(profile.get(skill, 0.0)) > 0
                else computed[skill]
            )
            for skill in SKILLS
        }

        overall = round(sum(smoothed.values()) / len(smoothed), 1)
        level = score_to_level(overall)

        # Le niveau ne redescend jamais de plus d'un cran d'un coup : une mauvaise
        # session ne doit pas effacer des semaines de travail.
        previous = str(profile.get("level") or "A1")
        if level_index(level) < level_index(previous) - 1:
            from app.language.languages import clamp_level

            level = clamp_level(level_index(previous) - 1)

        return SkillSnapshot(
            scores=smoothed,
            level=level,
            overall=overall,
            is_estimate=total_messages < MIN_MESSAGES_FOR_CONFIDENCE,
        )

    # ---------------------------------------------------------- persistance
    def refresh(self, user_id: int, language: str) -> dict[str, Any]:
        """Recalcule, enregistre et retourne le profil de la langue."""
        snapshot = self.compute_scores(user_id, language)
        profile = language_profiles.update_scores(
            user_id, language, snapshot.scores, level=snapshot.level
        )
        progress_repo.snapshot_level(user_id, language, snapshot.level, snapshot.overall)
        profile["overall_score"] = snapshot.overall
        profile["is_estimate"] = snapshot.is_estimate
        return profile

    def record_turn(
        self,
        user_id: int,
        language: str,
        *,
        seconds: int = 0,
        errors_found: int = 0,
        words_learned: int = 0,
    ) -> None:
        """Comptabilise un tour de conversation dans les statistiques du jour."""
        progress_repo.add(
            user_id,
            language,
            seconds=seconds,
            messages=1,
            errors=errors_found,
            words=words_learned,
        )

    def record_exercise(self, user_id: int, language: str, is_correct: bool) -> None:
        progress_repo.add(
            user_id, language, exercises=1, exercises_ok=1 if is_correct else 0
        )

    # ------------------------------------------------------------ dashboard
    def dashboard(self, user_id: int, language: str) -> dict[str, Any]:
        """Toutes les données du tableau de bord pour une langue (§25)."""
        profile = self.refresh(user_id, language)
        totals = progress_repo.totals(user_id, language)
        exercise_stats = exercise_repo.stats(user_id, language)
        history = progress_repo.history(user_id, language, days=30)
        weaknesses = error_repo.top_weaknesses(user_id, language, limit=5)

        success_rate = (
            round(100.0 * exercise_stats["correct"] / exercise_stats["done"], 1)
            if exercise_stats["done"]
            else None
        )

        return {
            "language": language,
            "level": profile.get("level", "A1"),
            "overall_score": profile.get("overall_score", 0.0),
            "is_estimate": profile.get("is_estimate", True),
            "skills": {skill: float(profile.get(skill, 0.0)) for skill in SKILLS},
            "totals": {
                "seconds_studied": totals.get("seconds", 0),
                "messages": message_repo.count_for_user(user_id, language),
                "words_learned": vocabulary_repo.count(user_id, language),
                "errors_corrected": error_repo.count(user_id, language),
                "exercises_done": exercise_stats["done"],
                "exercises_correct": exercise_stats["correct"],
                "success_rate": success_rate,
            },
            "weaknesses": weaknesses,
            "history": history,
            "reviews_due": len(review_repo.due(user_id, language, limit=100)),
        }


progress_tracker = ProgressTracker()
