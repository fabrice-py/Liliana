"""Analyse de prononciation (cf. §8 mode 7 et §7 de la roadmap).

Module volontairement indépendant du reste : il ne dépend que d'une
transcription et de la phrase attendue.

Méthode utilisée dans cette version : **comparaison transcription / cible**.
On fait lire une phrase connue à l'utilisateur, on la transcrit avec Whisper,
puis on mesure l'écart. Si Whisper « entend » autre chose, c'est un indice fort
d'une prononciation qui ne passe pas — et la nature de la substitution permet
d'identifier le son en cause.

Ce n'est pas de l'alignement phonétique forcé (prévu en phase 7), mais cela
fonctionne réellement, sans modèle supplémentaire.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from app.language.languages import get_language

#: Substitutions typiques -> son à travailler. (attendu, entendu) -> étiquette.
_SUBSTITUTION_HINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "english": (
        ("th", "s"), ("th", "z"), ("th", "f"), ("th", "d"), ("th", "t"),
        ("r", "w"), ("v", "b"), ("w", "v"), ("h", ""), ("ing", "in"),
    ),
    "german": (
        ("ü", "u"), ("ö", "o"), ("ä", "e"), ("ch", "sh"), ("ch", "k"),
        ("r", "r"), ("z", "s"), ("ei", "i"), ("eu", "o"),
    ),
    "french": (),
}

#: Étiquette lisible associée à un son.
_LABELS: dict[str, str] = {
    "th": "the English TH sound",
    "r": "the R sound",
    "v": "the V/W distinction",
    "w": "the V/W distinction",
    "h": "the aspirated H",
    "ing": "the -ing ending",
    "ü": "the German Umlaut Ü",
    "ö": "the German Umlaut Ö",
    "ä": "the German Umlaut Ä",
    "ch": "the German CH (ich-Laut / ach-Laut)",
    "z": "the German Z (ts)",
    "ei": "the German diphthong EI",
    "eu": "the German diphthong EU",
}


@dataclass(slots=True)
class WordComparison:
    expected: str
    heard: str
    similarity: float
    ok: bool


@dataclass(slots=True)
class PronunciationResult:
    expected: str
    heard: str
    score: float                      # 0-100
    word_accuracy: float              # 0-100
    words: list[WordComparison] = field(default_factory=list)
    problem_sounds: list[str] = field(default_factory=list)
    feedback: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["words"] = [asdict(word) for word in self.words]
        return payload


def _normalise(text: str, keep_accents: bool = True) -> str:
    text = (text or "").lower().strip()
    if not keep_accents:
        text = "".join(
            char
            for char in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(char)
        )
    return re.sub(r"[^\w\s'-]", "", text, flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [token for token in _normalise(text).split() if token]


def _similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def _detect_sounds(expected: str, heard: str, language: str) -> list[str]:
    """Devine les sons fautifs à partir des substitutions observées."""
    hints = _SUBSTITUTION_HINTS.get(language, ())
    found: list[str] = []
    matched_targets: list[str] = []
    expected_norm, heard_norm = _normalise(expected), _normalise(heard)

    for target, substitute in hints:
        # Un digramme déjà signalé (« th ») explique la disparition de ses
        # lettres : inutile de signaler aussi « h » ou « t » séparément.
        if any(target in longer for longer in matched_targets):
            continue
        # Le son attendu était présent, il ne l'est plus, et le son de
        # substitution est apparu : signature typique d'une erreur.
        if target in expected_norm and target not in heard_norm:
            if not substitute or substitute in heard_norm:
                matched_targets.append(target)
                label = _LABELS.get(target, target)
                if label not in found:
                    found.append(label)

    # Sons spécifiques à la langue jamais reconnus dans la transcription.
    for phoneme in get_language(language).phoneme_focus:
        key = phoneme.split("_")[-1].lower()
        if key and key in expected_norm and key not in heard_norm:
            label = _LABELS.get(key, phoneme.replace("_", " "))
            if label not in found:
                found.append(label)
    return found


def _feedback(score: float, problems: list[str], language: str) -> str:
    if score >= 90:
        return "Excellent — that was clear and accurate."
    if score >= 75:
        base = "Good. Almost everything came through clearly."
    elif score >= 55:
        base = "Understandable, but several words were unclear."
    else:
        base = "That was hard to make out. Slow down and articulate each word."
    if problems:
        base += " Focus on " + ", ".join(problems[:3]) + "."
    else:
        language_name = get_language(language).english_name
        base += f" Try reading the sentence again in {language_name}, a little slower."
    return base


def analyse(expected: str, heard: str, language: str = "english") -> PronunciationResult:
    """Compare la phrase attendue à ce que le moteur STT a réellement entendu."""
    expected_tokens = _tokens(expected)
    heard_tokens = _tokens(heard)

    if not expected_tokens:
        return PronunciationResult(
            expected=expected, heard=heard, score=0.0, word_accuracy=0.0,
            feedback="No target sentence was provided.",
        )
    if not heard_tokens:
        return PronunciationResult(
            expected=expected, heard=heard, score=0.0, word_accuracy=0.0,
            feedback="Liliana did not hear anything. Please record again.",
        )

    # Alignement mot à mot par le plus long sous-ensemble commun.
    matcher = difflib.SequenceMatcher(None, expected_tokens, heard_tokens)
    comparisons: list[WordComparison] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                word = expected_tokens[i1 + offset]
                comparisons.append(WordComparison(word, word, 1.0, True))
        elif tag == "replace":
            for offset in range(max(i2 - i1, j2 - j1)):
                expected_word = expected_tokens[i1 + offset] if i1 + offset < i2 else ""
                heard_word = heard_tokens[j1 + offset] if j1 + offset < j2 else ""
                ratio = _similarity(expected_word, heard_word) if expected_word else 0.0
                comparisons.append(
                    WordComparison(expected_word, heard_word, round(ratio, 2), ratio >= 0.8)
                )
        elif tag == "delete":  # mot attendu, non entendu
            for offset in range(i1, i2):
                comparisons.append(WordComparison(expected_tokens[offset], "", 0.0, False))
        elif tag == "insert":  # mot entendu en trop
            for offset in range(j1, j2):
                comparisons.append(WordComparison("", heard_tokens[offset], 0.0, False))

    scored = [word for word in comparisons if word.expected]
    word_accuracy = 100.0 * sum(1 for word in scored if word.ok) / len(scored) if scored else 0.0
    # Le score global mêle exactitude mot à mot et ressemblance globale : une
    # phrase presque juste ne doit pas être notée comme une phrase fausse.
    char_similarity = 100.0 * _similarity(" ".join(expected_tokens), " ".join(heard_tokens))
    score = round(0.6 * word_accuracy + 0.4 * char_similarity, 1)

    # On analyse aussi les quasi-correspondances : « möchte » entendu « mochte »
    # est un mot juste mais un Umlaut manqué.
    problems: list[str] = []
    for word in comparisons:
        if word.expected and _normalise(word.expected) != _normalise(word.heard):
            problems.extend(_detect_sounds(word.expected, word.heard, language))
    unique_problems = list(dict.fromkeys(problems))

    return PronunciationResult(
        expected=expected,
        heard=heard,
        score=score,
        word_accuracy=round(word_accuracy, 1),
        words=comparisons,
        problem_sounds=unique_problems,
        feedback=_feedback(score, unique_problems, language),
    )
