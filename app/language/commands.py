"""Reconnaissance des commandes vocales naturelles (cf. §29).

Pas de wake word dans le MVP : le bouton micro suffit. On détecte en revanche
les instructions courantes prononcées au début d'un tour, en anglais, en
allemand et en français, afin d'agir immédiatement (changer de langue, changer
de mode…) sans passer par le LLM.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

#: Nom de l'assistante, optionnel en tête de commande.
_WAKE = r"(?:liliana[\s,]*)?"


@dataclass(frozen=True, slots=True)
class Command:
    action: str
    payload: dict[str, Any]
    matched_text: str


#: (action, payload, motifs). Le premier motif qui correspond gagne.
_PATTERNS: tuple[tuple[str, dict[str, Any], tuple[str, ...]], ...] = (
    (
        "switch_language",
        {"language": "english"},
        (
            rf"{_WAKE}(?:let'?s |we )?(?:switch|change|go|move) (?:to|in|into) english",
            rf"{_WAKE}(?:let'?s )?(?:practi[cs]e|speak|talk|do) (?:in )?english",
            rf"{_WAKE}(?:passe|passons|on passe) (?:a|à) l'?anglais",
        ),
    ),
    (
        "switch_language",
        {"language": "german"},
        (
            rf"{_WAKE}(?:let'?s |we )?(?:switch|change|go|move) (?:to|in|into) german",
            rf"{_WAKE}(?:let'?s )?(?:practi[cs]e|speak|talk|do) (?:in )?(?:german|deutsch)",
            rf"{_WAKE}(?:lass uns )?(?:deutsch|auf deutsch) (?:sprechen|reden|üben)",
            rf"{_WAKE}(?:passe|passons|on passe) (?:a|à) l'?allemand",
        ),
    ),
    (
        "set_mode",
        {"mode": "immersion"},
        (rf"{_WAKE}(?:switch to |start |go )?immersion(?: mode)?",),
    ),
    (
        "set_mode",
        {"mode": "grammar_training"},
        (
            rf"{_WAKE}(?:give me |let'?s do |start )(?:an? )?(?:grammar )?exercise",
            rf"{_WAKE}(?:let'?s )?(?:practi[cs]e|do) grammar",
        ),
    ),
    (
        "set_mode",
        {"mode": "vocabulary_training"},
        (rf"{_WAKE}(?:let'?s )?(?:practi[cs]e|do|review) vocabulary",),
    ),
    (
        "set_mode",
        {"mode": "pronunciation_training"},
        (rf"{_WAKE}(?:let'?s )?(?:practi[cs]e|work on|do) (?:my )?pronunciation",),
    ),
    (
        "set_mode",
        {"mode": "just_talk"},
        (
            rf"{_WAKE}(?:let'?s )?just talk",
            rf"{_WAKE}start a conversation",
        ),
    ),
    (
        "set_correction_mode",
        {"correction_mode": "strict"},
        (rf"{_WAKE}correct (?:me|everything)(?: please)?", rf"{_WAKE}correct all my mistakes"),
    ),
    (
        "set_correction_mode",
        {"correction_mode": "off"},
        (rf"{_WAKE}(?:stop|don'?t) correct(?:ing)?(?: me)?",),
    ),
    ("repeat", {}, (rf"{_WAKE}(?:can you |please )?repeat(?: that| it)?(?: please)?",)),
    (
        "speak_slower",
        {"speed": 0.75},
        (
            rf"{_WAKE}(?:speak|talk|say it) (?:more )?slow(?:er|ly)?",
            rf"{_WAKE}(?:parle|parlez) (?:plus )?lentement",
        ),
    ),
    (
        "speak_faster",
        {"speed": 1.25},
        (rf"{_WAKE}(?:speak|talk) (?:more )?(?:quick|fast)(?:er|ly)?",),
    ),
    (
        "explain",
        {},
        (
            rf"{_WAKE}explain (?:that|this|the) (?:grammar )?rule",
            rf"{_WAKE}(?:why|what) (?:was|is) (?:that|the) (?:mistake|correction)",
        ),
    ),
    (
        "translate",
        {},
        (rf"{_WAKE}translate (?:this|that|it)", rf"{_WAKE}what does (?:this|that) word mean"),
    ),
    ("start_lesson", {}, (rf"{_WAKE}start (?:a |the )?lesson",)),
)

_COMPILED: tuple[tuple[str, dict[str, Any], re.Pattern[str]], ...] = tuple(
    (action, payload, re.compile(pattern, re.IGNORECASE))
    for action, payload, patterns in _PATTERNS
    for pattern in patterns
)


def _normalise(text: str) -> str:
    """Minuscules, sans accents ni ponctuation superflue."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().strip()
    return re.sub(r"[.!?;:]+$", "", text).strip()


def detect_command(text: str, max_words: int = 12) -> Command | None:
    """Détecte une commande dans un tour de parole.

    Seuls les tours courts sont analysés : une phrase longue est une vraie prise
    de parole à corriger, pas une commande.
    """
    normalised = _normalise(text)
    if not normalised or len(normalised.split()) > max_words:
        return None

    for action, payload, pattern in _COMPILED:
        match = pattern.search(normalised)
        if match:
            return Command(action=action, payload=dict(payload), matched_text=match.group(0))
    return None
