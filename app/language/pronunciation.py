"""Analyse de prononciation (cf. §8 mode 7 et §7 de la roadmap).

Module volontairement indépendant du reste : il ne dépend que d'une
transcription et de la phrase attendue.

On fait lire une phrase connue à l'utilisateur, on la transcrit, puis on mesure
l'écart sur trois plans complémentaires :

1. **Phonèmes** — la phrase attendue et la phrase entendue sont converties en
   IPA par espeak-ng (embarqué dans ``piper-tts``), puis alignées. C'est ce qui
   permet de dire « votre Ö est prononcé O » là où une comparaison de lettres ne
   voyait qu'une faute d'accent. Voir :mod:`app.language.phonemes`.
2. **Mots** — quels mots la reconnaissance a-t-elle manqués ou confondus.
3. **Confiance acoustique** — la probabilité que Whisper attribue à chaque mot.
   Un mot correctement reconnu mais avec une confiance faible signale une
   articulation approximative, invisible autrement.

Si la phonémisation n'est pas disponible (``piper-tts`` absent), l'analyse
retombe automatiquement sur la comparaison orthographique : dégradée, mais
fonctionnelle.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from app.language import phonemes
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
    #: Confiance de Whisper sur ce mot (0-1). ``None`` si la transcription n'a
    #: pas été demandée avec les horodatages par mot.
    confidence: float | None = None


@dataclass(slots=True)
class PronunciationResult:
    expected: str
    heard: str
    score: float                      # 0-100, note globale
    word_accuracy: float              # 0-100, mots correctement reconnus
    words: list[WordComparison] = field(default_factory=list)
    problem_sounds: list[str] = field(default_factory=list)
    feedback: str = ""
    #: 0-100. ``None`` quand la phonémisation n'est pas disponible.
    phoneme_accuracy: float | None = None
    #: 0-100. ``None`` quand Whisper n'a pas fourni les probabilités par mot.
    acoustic_confidence: float | None = None
    #: Détail des écarts phonétiques, pour l'affichage.
    phoneme_diffs: list[dict[str, str]] = field(default_factory=list)
    #: Ce sur quoi la note s'appuie réellement, pour ne rien surinterpréter.
    method: str = "spelling"

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
        base = "Excellent — that was clear and accurate."
    elif score >= 75:
        base = "Good. Almost everything came through clearly."
    elif score >= 55:
        base = "Understandable, but several sounds were off."
    else:
        base = "That was hard to make out. Slow down and articulate each word."

    if problems:
        return f"{base} Focus on {', '.join(problems[:3])}."
    if score >= 90:
        return base
    language_name = get_language(language).english_name
    return f"{base} Try reading the sentence again in {language_name}, a little slower."


def _confidence_by_word(words: list[dict[str, Any]] | None) -> dict[str, float]:
    """Probabilité acoustique la plus basse observée pour chaque mot.

    On garde la plus basse : si un mot revient deux fois et n'a été bien
    articulé qu'une fois, c'est l'occurrence ratée qui doit ressortir.
    """
    if not words:
        return {}
    scores: dict[str, float] = {}
    for entry in words:
        token = _normalise(str(entry.get("word", "")))
        if not token:
            continue
        probability = float(entry.get("probability", 0.0) or 0.0)
        scores[token] = min(scores.get(token, 1.0), probability)
    return scores


def analyse(
    expected: str,
    heard: str,
    language: str = "english",
    words: list[dict[str, Any]] | None = None,
) -> PronunciationResult:
    """Compare la phrase attendue à ce que le moteur STT a réellement entendu.

    ``words`` est la liste horodatée renvoyée par le STT quand on lui demande
    ``word_timestamps=True`` : elle apporte la confiance acoustique par mot.
    """
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

    confidences = _confidence_by_word(words)

    # --- alignement mot à mot par la plus longue sous-séquence commune
    matcher = difflib.SequenceMatcher(None, expected_tokens, heard_tokens)
    comparisons: list[WordComparison] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                word = expected_tokens[i1 + offset]
                comparisons.append(
                    WordComparison(word, word, 1.0, True, confidences.get(word))
                )
        elif tag == "replace":
            for offset in range(max(i2 - i1, j2 - j1)):
                expected_word = expected_tokens[i1 + offset] if i1 + offset < i2 else ""
                heard_word = heard_tokens[j1 + offset] if j1 + offset < j2 else ""
                ratio = _similarity(expected_word, heard_word) if expected_word else 0.0
                comparisons.append(
                    WordComparison(
                        expected_word, heard_word, round(ratio, 2), ratio >= 0.8,
                        confidences.get(heard_word),
                    )
                )
        elif tag == "delete":  # mot attendu, non entendu
            for offset in range(i1, i2):
                comparisons.append(WordComparison(expected_tokens[offset], "", 0.0, False))
        elif tag == "insert":  # mot entendu en trop
            for offset in range(j1, j2):
                comparisons.append(
                    WordComparison("", heard_tokens[offset], 0.0, False,
                                   confidences.get(heard_tokens[offset]))
                )

    scored = [word for word in comparisons if word.expected]
    word_accuracy = 100.0 * sum(1 for word in scored if word.ok) / len(scored) if scored else 0.0

    # --- comparaison phonémique (le cœur de l'analyse quand elle est disponible)
    comparison = phonemes.compare(expected, heard, language)
    phoneme_accuracy = round(100.0 * comparison.accuracy, 1) if comparison else None

    # --- confiance acoustique sur les mots effectivement reconnus
    heard_confidences = [
        word.confidence for word in comparisons if word.ok and word.confidence is not None
    ]
    acoustic = (
        round(100.0 * sum(heard_confidences) / len(heard_confidences), 1)
        if heard_confidences
        else None
    )

    # --- note globale : moyenne pondérée des signaux réellement disponibles
    signals: list[tuple[float, float]] = [(word_accuracy, 0.30)]
    if phoneme_accuracy is not None:
        signals.append((phoneme_accuracy, 0.50))
        method = "phonemes+acoustics" if acoustic is not None else "phonemes"
    else:
        # Sans phonèmes, la ressemblance orthographique globale prend le relais.
        char_similarity = 100.0 * _similarity(
            " ".join(expected_tokens), " ".join(heard_tokens)
        )
        signals.append((char_similarity, 0.50))
        method = "spelling+acoustics" if acoustic is not None else "spelling"
    if acoustic is not None:
        signals.append((acoustic, 0.20))

    total_weight = sum(weight for _, weight in signals)
    score = round(sum(value * weight for value, weight in signals) / total_weight, 1)

    # --- sons à travailler
    if comparison is not None:
        problems = comparison.labels
        diffs = [diff.as_dict() for diff in comparison.diffs]
    else:
        raw: list[str] = []
        for word in comparisons:
            if word.expected and _normalise(word.expected) != _normalise(word.heard):
                raw.extend(_detect_sounds(word.expected, word.heard, language))
        problems = list(dict.fromkeys(raw))
        diffs = []

    return PronunciationResult(
        expected=expected,
        heard=heard,
        score=score,
        word_accuracy=round(word_accuracy, 1),
        words=comparisons,
        problem_sounds=problems,
        feedback=_feedback(score, problems, language),
        phoneme_accuracy=phoneme_accuracy,
        acoustic_confidence=acoustic,
        phoneme_diffs=diffs,
        method=method,
    )


# --------------------------------------------------- phrases d'entraînement
#: Phrases courtes, chacune saturée d'un son précis. Locale et instantanée :
#: l'entraînement à la prononciation fonctionne même sans modèle de langage.
PRACTICE_SENTENCES: dict[str, tuple[tuple[str, str], ...]] = {
    "english": (
        ("TH", "I think this thing is worth thirty pounds."),
        ("TH", "They gathered together on Thursday to breathe."),
        ("R", "Robert reported a rare red parrot in the garden."),
        ("W/V", "We were very well when we visited Vienna."),
        ("short/long I", "The ship on the sheep field will slip and sleep."),
        ("-ING", "She is singing, running and bringing everything."),
        ("H", "How happy he was to hear her helpful answer."),
        ("schwa", "The banana and the camera are in the cinema."),
        ("consonant clusters", "The sixth strong athlete stretched his strength."),
        ("silent letters", "The knight walked through the castle at midnight."),
    ),
    "german": (
        ("ich-Laut", "Ich möchte nicht schlecht sprechen, ich brauche Licht."),
        ("ach-Laut", "Nach acht Wochen kochte er noch Kuchen in der Nacht."),
        ("Ü", "Über fünf grüne Hügel führt die kühle Brücke."),
        ("Ö", "Der schöne König möchte zwölf rote Brötchen."),
        ("Ä", "Die Bäckerin erzählt täglich von ihren Käsekuchen."),
        ("R", "Der rote Rucksack rollte rasch durch die Straße."),
        ("Z (ts)", "Zwölf Katzen sitzen zusammen auf dem kurzen Platz."),
        ("PF", "Der Pfarrer pflückte Pflaumen und pfiff ein Pfundlied."),
        ("SCH", "Schnell schrieb der Schüler seine schöne Geschichte."),
        ("EU / EI", "Heute Abend feiern neun Leute bei mir zu Hause."),
    ),
}

#: Rapproche un son signalé par l'analyse d'une catégorie de la banque.
#:
#: Les motifs sont volontairement distinctifs et l'ordre compte : un motif trop
#: court attraperait n'importe quoi (« TH » se trouve dans « the »). Ils sont
#: comparés tels quels, sans passer en minuscules, car les Umlauts suffisent
#: à eux seuls à identifier une catégorie.
_SOUND_TO_CATEGORY: tuple[tuple[str, str], ...] = (
    # Allemand : les plus spécifiques d'abord.
    ("ich-Laut", "ich-Laut"),
    ("ach-Laut", "ach-Laut"),
    ("German CH", "ich-Laut"),
    ("Ü", "Ü"),
    ("Ö", "Ö"),
    ("Ä", "Ä"),
    ("German Z", "Z (ts)"),
    ("German PF", "PF"),
    ("SCH", "SCH"),
    ("German R", "R"),
    # Anglais.
    ("TH / S", "TH"),
    ("voiced TH", "TH"),
    ("unvoiced TH", "TH"),
    ("W / V", "W/V"),
    ("English R", "R"),
    ("aspirated H", "H"),
    ("-NG", "-ING"),
    ("schwa", "schwa"),
    ("long I", "short/long I"),
    ("long EE", "short/long I"),
    ("short I (", "short/long I"),
    ("V sound", "W/V"),
    ("W sound", "W/V"),
    # Repli générique, en dernier.
    ("R sound", "R"),
)


def categories_for(language: str) -> list[str]:
    """Catégories de sons disponibles à l'entraînement pour une langue."""
    return list(dict.fromkeys(category for category, _ in PRACTICE_SENTENCES.get(language, ())))


def category_for_sound(label: str) -> str | None:
    """Catégorie d'entraînement correspondant à un son signalé par l'analyse."""
    for needle, category in _SOUND_TO_CATEGORY:
        if needle in label:
            return category
    return None


def practice_sentences(language: str, category: str | None = None) -> list[tuple[str, str]]:
    """Phrases disponibles, éventuellement restreintes à une catégorie."""
    bank = PRACTICE_SENTENCES.get(language, PRACTICE_SENTENCES["english"])
    if not category:
        return list(bank)
    return [item for item in bank if item[0].lower() == category.lower()] or list(bank)


def pick_sentence(
    language: str,
    category: str | None = None,
    weak_sounds: list[str] | None = None,
    exclude: str | None = None,
    rotation: int = 0,
) -> tuple[str, str]:
    """Choisit une phrase à faire lire.

    Priorité aux sons que l'utilisateur rate réellement (``weak_sounds`` vient
    de ses tentatives passées) : l'entraînement suit ses difficultés au lieu de
    dérouler une liste. ``rotation`` fait tourner les phrases d'une même
    catégorie d'un appel à l'autre.
    """
    if category is None and weak_sounds:
        for label in weak_sounds:
            if matched := category_for_sound(label):
                category = matched
                break

    candidates = practice_sentences(language, category)
    if exclude:
        remaining = [item for item in candidates if item[1] != exclude]
        candidates = remaining or candidates
    return candidates[rotation % len(candidates)]
