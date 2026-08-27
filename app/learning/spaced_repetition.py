"""Répétition espacée (algorithme SM-2 simplifié).

S'applique indifféremment au vocabulaire et aux points de grammaire : la table
``review_schedule`` stocke ``item_type`` + ``item_key`` (cf. §18).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.database.repositories import reviews

#: Intervalles (en jours) des deux premières répétitions réussies.
FIRST_INTERVALS: tuple[float, float] = (1.0, 3.0)

MIN_EASE = 1.3
MAX_EASE = 2.8
DEFAULT_EASE = 2.5

#: Après un échec, on revoit l'élément très vite.
FAILURE_INTERVAL_DAYS = 10 / (60 * 24)  # 10 minutes


@dataclass(slots=True)
class ReviewOutcome:
    interval_days: float
    ease: float
    repetitions: int
    confidence: float
    next_review: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _confidence(repetitions: int, success: int, failure: int) -> float:
    """Confiance dans [0, 1] : taux de réussite pondéré par le nombre de révisions."""
    attempts = success + failure
    if attempts == 0:
        return 0.0
    accuracy = success / attempts
    maturity = min(repetitions / 5.0, 1.0)
    return round(0.35 * accuracy + 0.65 * accuracy * maturity, 3)


def compute_next_review(
    *,
    quality: int,
    repetitions: int,
    interval_days: float,
    ease: float,
    success_count: int,
    failure_count: int,
    now: datetime | None = None,
) -> ReviewOutcome:
    """Calcule la prochaine échéance.

    ``quality`` suit la convention SM-2 : 0-2 = échec, 3-5 = réussite.
    Fonction pure, donc directement testable.
    """
    now = now or _now()
    quality = max(0, min(int(quality), 5))
    ease = max(MIN_EASE, min(float(ease or DEFAULT_EASE), MAX_EASE))

    if quality < 3:
        failure_count += 1
        repetitions = 0
        interval_days = FAILURE_INTERVAL_DAYS
        ease = max(MIN_EASE, ease - 0.2)
    else:
        success_count += 1
        if repetitions == 0:
            interval_days = FIRST_INTERVALS[0]
        elif repetitions == 1:
            interval_days = FIRST_INTERVALS[1]
        else:
            interval_days = round(max(interval_days, 1.0) * ease, 2)
        repetitions += 1
        # Formule SM-2 d'ajustement de la facilité.
        ease = max(
            MIN_EASE,
            min(ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)), MAX_EASE),
        )

    return ReviewOutcome(
        interval_days=round(interval_days, 4),
        ease=round(ease, 3),
        repetitions=repetitions,
        confidence=_confidence(repetitions, success_count, failure_count),
        next_review=now + timedelta(days=interval_days),
    )


class SpacedRepetition:
    """Façade métier au-dessus du dépôt ``review_schedule``."""

    def __init__(self, repository: Any = reviews) -> None:
        self.repository = repository

    def register(self, user_id: int, language: str, item_type: str, item_key: str) -> dict[str, Any]:
        """Ajoute un élément au planning s'il n'y est pas déjà (dû immédiatement)."""
        return self.repository.upsert(user_id, language, item_type, item_key)

    def register_many(
        self, user_id: int, language: str, item_type: str, keys: list[str]
    ) -> int:
        added = 0
        for key in keys:
            if key and not self.repository.get(user_id, language, item_type, key):
                self.repository.upsert(user_id, language, item_type, key)
                added += 1
        return added

    def review(
        self, user_id: int, language: str, item_type: str, item_key: str, quality: int
    ) -> dict[str, Any]:
        """Enregistre le résultat d'une révision et replanifie l'élément."""
        current = self.repository.get(user_id, language, item_type, item_key) or {}
        success = int(current.get("success_count", 0))
        failure = int(current.get("failure_count", 0))

        outcome = compute_next_review(
            quality=quality,
            repetitions=int(current.get("repetitions", 0)),
            interval_days=float(current.get("interval_days", 0.0)),
            ease=float(current.get("difficulty", DEFAULT_EASE)),
            success_count=success,
            failure_count=failure,
        )
        if quality < 3:
            failure += 1
        else:
            success += 1

        return self.repository.upsert(
            user_id,
            language,
            item_type,
            item_key,
            difficulty=outcome.ease,
            interval_days=outcome.interval_days,
            repetitions=outcome.repetitions,
            success_count=success,
            failure_count=failure,
            confidence=outcome.confidence,
            last_review=outcome.next_review.strftime("%Y-%m-%d %H:%M:%S"),
            next_review=outcome.next_review.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def due(self, user_id: int, language: str, limit: int = 20, item_type: str | None = None):
        return self.repository.due(user_id, language, limit=limit, item_type=item_type)

    def due_keys(self, user_id: int, language: str, limit: int = 10) -> list[str]:
        return [item["item_key"] for item in self.due(user_id, language, limit=limit)]


spaced_repetition = SpacedRepetition()
