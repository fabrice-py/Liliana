"""Construction d'une séance d'apprentissage (bouton START LESSON, §26).

Le plan est adapté au niveau, au temps disponible et aux points faibles observés.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.database.repositories import errors as error_repo, reviews as review_repo
from app.language.languages import level_index


@dataclass(slots=True)
class LessonBlock:
    mode: str
    label: str
    minutes: int
    focus: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Durée plancher d'un bloc, en minutes.
MIN_BLOCK_MINUTES = 3

#: Répartition de référence, en proportion du temps total.
_BASE_PLAN: tuple[tuple[str, str, float], ...] = (
    ("free_conversation", "Warm-up conversation", 0.15),
    ("grammar_training", "Grammar", 0.25),
    ("vocabulary_training", "Vocabulary", 0.25),
    ("free_conversation", "Speaking", 0.25),
    ("vocabulary_training", "Review", 0.10),
)


def _allocate_minutes(total: int, weights: list[float]) -> list[int]:
    """Répartit ``total`` minutes selon ``weights``, en entiers dont la somme est
    exactement ``total`` (méthode des plus forts restes)."""
    if not weights:
        return []
    weight_sum = sum(weights) or 1.0
    exact = [total * weight / weight_sum for weight in weights]
    allocation = [max(MIN_BLOCK_MINUTES, int(value)) for value in exact]

    # Corrige l'écart en ajoutant/retirant une minute là où c'est le plus juste.
    remainders = sorted(range(len(exact)), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    drift = total - sum(allocation)
    index = 0
    while drift > 0:
        allocation[remainders[index % len(remainders)]] += 1
        drift -= 1
        index += 1
    while drift < 0:
        candidates = [i for i in reversed(remainders) if allocation[i] > MIN_BLOCK_MINUTES]
        if not candidates:
            break
        allocation[candidates[0]] -= 1
        drift += 1
    return allocation


def build_lesson(
    user_id: int, language: str, level: str, minutes: int = 30
) -> dict[str, Any]:
    """Construit un plan de séance de ``minutes`` minutes.

    - débutants (A1/A2) : plus de vocabulaire, moins de conversation libre ;
    - avancés (B2+) : plus de conversation, moins de drill ;
    - le bloc "Review" est supprimé s'il n'y a rien à réviser.
    """
    minutes = max(10, min(int(minutes), 90))
    weights = {label: weight for _, label, weight in _BASE_PLAN}

    if level_index(level) <= 1:  # A1 / A2
        weights["Vocabulary"] += 0.10
        weights["Speaking"] -= 0.10
    elif level_index(level) >= 3:  # B2 / C1 / C2
        weights["Speaking"] += 0.10
        weights["Grammar"] -= 0.10

    due_count = len(review_repo.due(user_id, language, limit=50))
    if due_count == 0:
        redistributed = weights.pop("Review")
        weights["Speaking"] += redistributed

    weaknesses = error_repo.top_weaknesses(user_id, language, limit=3)
    grammar_focus = weaknesses[0]["topic"] if weaknesses else "general review"

    ordered = [(mode, label) for mode, label, _ in _BASE_PLAN if label in weights]
    # Une séance courte ne peut pas contenir tous les blocs : on garde les plus
    # lourds, à raison de MIN_BLOCK_MINUTES minutes chacun au minimum.
    while len(ordered) * MIN_BLOCK_MINUTES > minutes and len(ordered) > 1:
        lightest = min(ordered, key=lambda item: weights[item[1]])
        ordered.remove(lightest)
        weights.pop(lightest[1])

    allocations = _allocate_minutes(minutes, [weights[label] for _, label in ordered])

    blocks: list[LessonBlock] = []
    for (mode, label), block_minutes in zip(ordered, allocations, strict=True):
        focus = ""
        if label == "Grammar":
            focus = grammar_focus
        elif label == "Review":
            focus = f"{due_count} item(s) due"
        blocks.append(LessonBlock(mode=mode, label=label, minutes=block_minutes, focus=focus))

    return {
        "language": language,
        "level": level,
        "total_minutes": sum(block.minutes for block in blocks),
        "blocks": [block.as_dict() for block in blocks],
        "weaknesses": [item["topic"] for item in weaknesses],
        "reviews_due": due_count,
    }
