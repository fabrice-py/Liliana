"""Extraction et validation robustes du JSON produit par le LLM (cf. §23).

Un modèle local se trompe régulièrement de format : texte avant/après, bloc
markdown, virgule finale, guillemets typographiques, JSON tronqué. Liliana ne
doit jamais planter à cause de cela : au pire, on retombe sur la réponse en
texte brut.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.logger import get_logger
from app.language.languages import CEFR_LEVELS

logger = get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})


def _balanced_slice(text: str) -> str | None:
    """Extrait le premier objet JSON équilibré, en ignorant les accolades
    présentes à l'intérieur des chaînes."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    # JSON tronqué : on referme la chaîne puis les accolades manquantes.
    if depth > 0:
        repaired = text[start:].rstrip().rstrip(",")
        if in_string:
            repaired = repaired.rstrip("\\") + '"'
        return repaired + "}" * depth
    return None


def extract_json(raw: str) -> dict[str, Any] | None:
    """Tente d'extraire un objet JSON d'une réponse LLM. ``None`` si impossible."""
    if not raw or not raw.strip():
        return None

    candidates: list[str] = []
    fence = _FENCE_RE.search(raw)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(raw)

    for candidate in candidates:
        candidate = candidate.strip().translate(_SMART_QUOTES)
        for attempt in (candidate, _balanced_slice(candidate) or ""):
            if not attempt:
                continue
            for text in (attempt, _TRAILING_COMMA_RE.sub(r"\1", attempt)):
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    return parsed
    logger.debug("Impossible d'extraire du JSON de la réponse LLM (%d caractères)", len(raw))
    return None


# ------------------------------------------------------------ normalisation
def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_level(value: Any, default: str = "") -> str:
    level = _as_str(value).upper()
    return level if level in CEFR_LEVELS else default


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str) and item.strip():
            # Certains modèles renvoient une simple liste de chaînes.
            result.append({"type": item.strip(), "topic": item.strip()})
    return result


def normalise_turn(parsed: dict[str, Any] | None, fallback_text: str) -> dict[str, Any]:
    """Normalise la réponse d'un tour de conversation.

    Garantit la présence et le type de chaque champ, même si le modèle a produit
    une structure incomplète. ``fallback_text`` est utilisé comme réponse parlée
    lorsque le JSON est inexploitable.
    """
    parsed = parsed if isinstance(parsed, dict) else {}

    response = _as_str(parsed.get("response"))
    if not response:
        # Le modèle a répondu en texte libre : on parle ce texte, sans le JSON.
        response = _as_str(fallback_text)

    correction = parsed.get("correction")
    if isinstance(correction, dict):
        original = _as_str(correction.get("original"))
        corrected = _as_str(correction.get("corrected"))
        explanation = _as_str(correction.get("explanation"))
        # Une "correction" identique à l'original n'en est pas une.
        if not corrected or corrected == original:
            correction = None
        else:
            correction = {
                "original": original,
                "corrected": corrected,
                "explanation": explanation,
            }
    else:
        correction = None

    errors: list[dict[str, Any]] = []
    for error in _as_list_of_dicts(parsed.get("errors")):
        error_type = _as_str(error.get("type")) or _as_str(error.get("error_type")) or "grammar"
        severity = _as_str(error.get("severity"), "minor").lower()
        errors.append(
            {
                "type": error_type,
                "topic": _as_str(error.get("topic")) or error_type,
                "original": _as_str(error.get("original")),
                "corrected": _as_str(error.get("corrected")),
                "explanation": _as_str(error.get("explanation")),
                "severity": severity if severity in ("minor", "major") else "minor",
            }
        )

    vocabulary: list[dict[str, Any]] = []
    for entry in _as_list_of_dicts(parsed.get("vocabulary")):
        word = _as_str(entry.get("word")) or _as_str(entry.get("type"))
        if not word:
            continue
        vocabulary.append(
            {
                "word": word,
                "translation": _as_str(entry.get("translation")),
                "example": _as_str(entry.get("example")),
                "part_of_speech": _as_str(entry.get("part_of_speech")),
                "difficulty": _as_level(entry.get("difficulty"), "A1"),
            }
        )

    return {
        "response": response,
        "correction": correction,
        "errors": errors,
        "vocabulary": vocabulary,
        "detected_language": _as_str(parsed.get("detected_language")).lower(),
        "difficulty": _as_level(parsed.get("difficulty")),
        "suggested_level": _as_level(parsed.get("suggested_level")),
        "structured": bool(parsed),
    }


def normalise_exercise(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalise un exercice généré. ``None`` si inexploitable."""
    if not isinstance(parsed, dict):
        return None
    prompt = _as_str(parsed.get("prompt")) or _as_str(parsed.get("question"))
    if not prompt:
        return None

    options = parsed.get("options")
    options = [_as_str(option) for option in options if _as_str(option)] if isinstance(options, list) else []

    return {
        "exercise_type": _as_str(parsed.get("exercise_type"), "open") or "open",
        "topic": _as_str(parsed.get("topic")),
        "level": _as_level(parsed.get("level"), "A1"),
        "prompt": prompt,
        "options": options,
        "answer": _as_str(parsed.get("answer")),
        "explanation": _as_str(parsed.get("explanation")),
    }


def normalise_answer_check(parsed: dict[str, Any] | None, fallback: str = "") -> dict[str, Any]:
    """Normalise l'évaluation d'une réponse à un exercice."""
    parsed = parsed if isinstance(parsed, dict) else {}
    raw_correct = parsed.get("is_correct")
    if isinstance(raw_correct, str):
        is_correct = raw_correct.strip().lower() in ("true", "yes", "1", "correct")
    else:
        is_correct = bool(raw_correct)
    return {
        "is_correct": is_correct,
        "feedback": _as_str(parsed.get("feedback")) or _as_str(fallback),
        "corrected": _as_str(parsed.get("corrected")),
        "errors": [
            {
                "type": _as_str(error.get("type"), "grammar") or "grammar",
                "topic": _as_str(error.get("topic")),
                "severity": _as_str(error.get("severity"), "minor").lower(),
            }
            for error in _as_list_of_dicts(parsed.get("errors"))
        ],
    }


def normalise_assessment(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalise le résultat d'une évaluation de niveau."""
    if not isinstance(parsed, dict):
        return None
    scores_raw = parsed.get("scores")
    scores: dict[str, float] = {}
    if isinstance(scores_raw, dict):
        for skill, value in scores_raw.items():
            try:
                scores[str(skill).lower()] = max(0.0, min(100.0, float(value)))
            except (TypeError, ValueError):
                continue
    level = _as_level(parsed.get("level"))
    if not level and not scores:
        return None
    return {
        "level": level,
        "scores": scores,
        "strengths": [_as_str(item) for item in parsed.get("strengths", []) if _as_str(item)]
        if isinstance(parsed.get("strengths"), list)
        else [],
        "weaknesses": [_as_str(item) for item in parsed.get("weaknesses", []) if _as_str(item)]
        if isinstance(parsed.get("weaknesses"), list)
        else [],
        "summary": _as_str(parsed.get("summary")),
    }
