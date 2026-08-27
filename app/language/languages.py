"""Langues supportées, niveaux CECRL et taxonomie des erreurs.

Ajouter une langue = ajouter une entrée dans ``LANGUAGES`` et une voix TTS dans
la configuration. Aucun autre module ne code de langue en dur (cf. §37).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- niveaux
CEFR_LEVELS: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")

#: Compétences suivies dans le profil linguistique (cf. §14).
SKILLS: tuple[str, ...] = (
    "grammar",
    "vocabulary",
    "speaking",
    "listening",
    "writing",
    "pronunciation",
)


def clamp_level(index: int) -> str:
    """Indice -> niveau CECRL, borné à [A1, C2]."""
    return CEFR_LEVELS[max(0, min(index, len(CEFR_LEVELS) - 1))]


def level_index(level: str) -> int:
    """Niveau CECRL -> indice. Retourne 0 (A1) si inconnu."""
    try:
        return CEFR_LEVELS.index(level.upper())
    except (ValueError, AttributeError):
        return 0


def score_to_level(score: float) -> str:
    """Convertit un score global 0-100 en niveau CECRL."""
    thresholds = ((20, "A1"), (35, "A2"), (50, "B1"), (75, "B2"), (90, "C1"))
    for limit, level in thresholds:
        if score < limit:
            return level
    return "C2"


# ---------------------------------------------------------------- langues
@dataclass(frozen=True, slots=True)
class Language:
    code: str            # identifiant interne, utilisé en base
    english_name: str    # nom affiché
    native_name: str
    whisper_code: str    # code ISO-639-1 attendu par Whisper
    espeak_voice: str    # voix espeak-ng, pour la phonémisation (analyse de prononciation)
    flag: str
    is_target: bool      # langue enseignée (vs langue d'appui)
    error_types: tuple[str, ...]
    phoneme_focus: tuple[str, ...]


#: Types d'erreurs communs à toutes les langues (cf. §10).
_COMMON_ERRORS: tuple[str, ...] = (
    "grammar",
    "conjugation",
    "verb_tense",
    "articles",
    "prepositions",
    "word_order",
    "plurals",
    "auxiliaries",
    "modals",
    "vocabulary",
    "false_friends",
    "spelling",
    "naturalness",
    "register",
    "pronunciation",
)

#: Spécificités allemandes qui s'ajoutent aux types communs.
_GERMAN_ERRORS: tuple[str, ...] = (
    "gender_der_die_das",
    "cases",
    "nominative",
    "accusative",
    "dative",
    "genitive",
    "declension",
    "verb_position",
    "separable_verbs",
    "adjective_endings",
    "umlaut",
)

LANGUAGES: dict[str, Language] = {
    "english": Language(
        code="english",
        english_name="English",
        native_name="English",
        whisper_code="en",
        espeak_voice="en-us",
        flag="🇬🇧",
        is_target=True,
        error_types=_COMMON_ERRORS,
        phoneme_focus=("TH", "english_R", "vowel_length", "schwa", "word_stress"),
    ),
    "german": Language(
        code="german",
        english_name="German",
        native_name="Deutsch",
        whisper_code="de",
        espeak_voice="de",
        flag="🇩🇪",
        is_target=True,
        error_types=_COMMON_ERRORS + _GERMAN_ERRORS,
        phoneme_focus=("umlaut_ü", "umlaut_ö", "ich_laut_CH", "ach_laut_CH", "german_R"),
    ),
    # Langue d'appui : Liliana la comprend mais ne l'enseigne pas dans le MVP.
    "french": Language(
        code="french",
        english_name="French",
        native_name="Français",
        whisper_code="fr",
        espeak_voice="fr",
        flag="🇫🇷",
        is_target=False,
        error_types=_COMMON_ERRORS,
        phoneme_focus=(),
    ),
}

#: Langues effectivement enseignées.
TARGET_LANGUAGES: tuple[str, ...] = tuple(
    code for code, language in LANGUAGES.items() if language.is_target
)


def get_language(code: str) -> Language:
    """Retourne la langue demandée, ou l'anglais par défaut."""
    return LANGUAGES.get((code or "").lower(), LANGUAGES["english"])


def is_supported(code: str) -> bool:
    return (code or "").lower() in LANGUAGES


def error_types_for(code: str) -> tuple[str, ...]:
    return get_language(code).error_types


def whisper_code(code: str) -> str:
    return get_language(code).whisper_code


def espeak_voice(code: str) -> str:
    return get_language(code).espeak_voice
