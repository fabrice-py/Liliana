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


#: Enveloppe JSON vide : « {} », « [] », « null », ou de la ponctuation seule.
_EMPTY_ENVELOPE_RE = re.compile(r"^(?:[\s{}\[\]\"',:;.]*|null|none|n/a)$", re.IGNORECASE)


def is_meaningful_response(text: str) -> bool:
    """Vrai si ``text`` porte quelque chose qu'on puisse réellement dire.

    Un petit modèle renvoie parfois une enveloppe vide (« {} ») au lieu du tour
    attendu. La traiter comme une réponse en texte libre a deux conséquences :
    la synthèse vocale n'a aucun phonème à produire, et surtout « {} » est
    enregistré comme message de l'assistant — au tour suivant le modèle relit sa
    propre sortie vide dans l'historique et l'imite, indéfiniment. Mieux vaut
    une erreur franche, que l'appelant sait présenter.
    """
    stripped = (text or "").strip()
    return bool(stripped) and not _EMPTY_ENVELOPE_RE.match(stripped)


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
        # Sauf s'il n'a rien produit d'exploitable : voir is_meaningful_response.
        fallback = _as_str(fallback_text)
        response = fallback if is_meaningful_response(fallback) else ""

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


# ------------------------------------------------- lecture JSON incrémentale
class ResponseStreamParser:
    """Extrait le champ ``response`` d'un JSON **pendant** sa réception.

    Le contrat de sortie (cf. ``RESPONSE_SCHEMA_PROMPT``) place ``response`` en
    premier : on peut donc commencer à parler la réponse alors que la correction
    et les erreurs sont encore en train d'arriver. C'est tout l'intérêt du
    streaming — sans cela il faudrait attendre le JSON complet.

    Deux modes, choisis automatiquement :

    * **JSON** — le flux commence par ``{`` : on cherche la clé ``response`` et
      on décode sa valeur au fur et à mesure.
    * **texte brut** — le modèle a ignoré la consigne : tout ce qui arrive est
      considéré comme la réponse parlée (même repli que ``normalise_turn``).

    Utilisation ::

        parser = ResponseStreamParser()
        for chunk in llm.stream(messages):
            if delta := parser.feed(chunk):
                ...  # texte nouvellement disponible
        raw = parser.raw
    """

    _KEY_RE = re.compile(r'"response"\s*:\s*"')
    #: Nombre de caractères reçus sans voir la clé ``response`` au-delà duquel on
    #: abandonne l'hypothèse JSON. Assez large pour laisser passer un préambule
    #: bavard ("Sure! Here you go: {…"), assez court pour que le repli en texte
    #: brut reste fluide.
    _PLAIN_TEXT_AFTER = 80

    def __init__(self) -> None:
        self.raw = ""            # tout ce qui a été reçu, pour le parsing final
        self._emitted = ""       # texte déjà rendu à l'appelant
        self._value_start = -1   # index, dans self.raw, du 1er caractère de la valeur
        self._plain = False      # mode texte brut
        self._closed = False     # guillemet fermant rencontré

    # ------------------------------------------------------------- interne
    @staticmethod
    def _safe_prefix(raw_value: str) -> str:
        """Plus long préfixe ne se terminant pas au milieu d'une séquence d'échappement.

        Un fragment réseau peut couper ``\\n`` en deux, ou ``\\u00e9`` n'importe où :
        décoder un tel préfixe échouerait.
        """
        # Antislashs finaux en nombre impair : le dernier ouvre un échappement.
        trailing = len(raw_value) - len(raw_value.rstrip("\\"))
        if trailing % 2:
            raw_value = raw_value[:-1]

        # Séquence \uXXXX incomplète en fin de tampon.
        marker = raw_value.rfind("\\u")
        if marker != -1 and len(raw_value) - marker < 6:
            # Vérifie que cet antislash est bien un échappement, pas un `\\` littéral.
            preceding = len(raw_value[:marker]) - len(raw_value[:marker].rstrip("\\"))
            if preceding % 2 == 0:
                raw_value = raw_value[:marker]
        return raw_value

    def _decode(self, raw_value: str) -> str:
        try:
            return json.loads(f'"{self._safe_prefix(raw_value)}"')
        except (json.JSONDecodeError, ValueError):
            return ""

    def _detect_mode(self) -> None:
        """Décide entre JSON et texte brut dès qu'il y a de quoi trancher."""
        if self._plain or self._value_start != -1:
            return

        stripped = self.raw.lstrip()
        if not stripped:
            return

        # La clé peut arriver après du bavardage ("Sure! Here you go: {…").
        match = self._KEY_RE.search(self.raw)
        if match:
            self._value_start = match.end()
            return

        # Toujours pas de clé après un volume significatif : le modèle a répondu
        # en texte libre. Une règle unique, car juger trop tôt sur l'absence
        # d'accolade ferait basculer à tort un préambule comme "Sure!\n".
        if len(stripped) >= self._PLAIN_TEXT_AFTER:
            self._plain = True

    # -------------------------------------------------------------- public
    def feed(self, chunk: str) -> str:
        """Ajoute un fragment. Retourne le texte **nouvellement** disponible."""
        if not chunk:
            return ""
        self.raw += chunk
        self._detect_mode()

        if self._plain:
            delta = self.raw[len(self._emitted):]
            self._emitted = self.raw
            return delta

        if self._value_start == -1 or self._closed:
            return ""

        # Cherche le guillemet fermant non échappé de la valeur.
        raw_value = self.raw[self._value_start:]
        end = self._find_closing_quote(raw_value)
        if end != -1:
            raw_value = raw_value[:end]
            self._closed = True

        decoded = self._decode(raw_value)
        if not decoded.startswith(self._emitted):
            # Le décodage d'un préfixe plus long peut réviser un caractère
            # partiellement décodé : on repart proprement de la version longue.
            self._emitted = decoded
            return decoded

        delta = decoded[len(self._emitted):]
        self._emitted = decoded
        return delta

    @staticmethod
    def _find_closing_quote(raw_value: str) -> int:
        escaped = False
        for index, char in enumerate(raw_value):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return index
        return -1

    def flush(self) -> str:
        """À appeler en fin de flux. Retourne ce qui restait à émettre.

        Indispensable pour une réponse en texte libre plus courte que
        ``_PLAIN_TEXT_AFTER`` : le seuil n'a jamais été atteint, donc ``feed``
        n'a rien émis.
        """
        if self._plain or self._value_start != -1:
            return ""
        self._plain = True
        delta = self.raw[len(self._emitted):]
        self._emitted = self.raw
        return delta

    @property
    def text(self) -> str:
        """Réponse parlée telle que reconstituée jusqu'ici."""
        return self._emitted

    @property
    def is_plain_text(self) -> bool:
        """Le modèle a-t-il ignoré le format JSON demandé ?"""
        return self._plain

    def finish(self) -> dict[str, Any]:
        """Parse le flux complet et retourne un tour normalisé."""
        return normalise_turn(extract_json(self.raw), fallback_text=self.raw)
