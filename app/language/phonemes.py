"""Phonémisation locale et comparaison de sons (cf. §8 mode 7, phase 7).

La comparaison lettre à lettre du module de prononciation était une
approximation : « think » entendu « sink » se voit, mais « möchte » entendu
« mochte » passe pour une simple faute d'accent. Ici on compare ce qui compte
réellement — les **phonèmes**.

La phonémisation utilise espeak-ng, **embarqué dans le paquet ``piper-tts``** :
aucune dépendance système supplémentaire, et cela fonctionne hors ligne. Si
``piper-tts`` n'est pas installé, :func:`is_available` retourne ``False`` et
l'appelant retombe sur l'heuristique orthographique.
"""

from __future__ import annotations

import difflib
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.logger import get_logger
from app.language.languages import espeak_voice

logger = get_logger(__name__)

#: Marques de proéminence : non pertinentes pour juger un son isolé.
_STRESS_MARKS = frozenset("ˈˌ")

#: Signes qui modifient le phonème précédent (longueur, nasalisation…) et
#: doivent rester collés à lui.
_MODIFIERS = frozenset("ːˑ̃ʰʷʲ˔˕")

_phonemizer: Any | None = None
_phonemizer_lock = threading.Lock()
_phonemizer_failed = False


# ------------------------------------------------------------- disponibilité
def _get_phonemizer() -> Any | None:
    """Instance partagée du phonémiseur espeak-ng, ou ``None`` s'il est absent."""
    global _phonemizer, _phonemizer_failed

    if _phonemizer is not None or _phonemizer_failed:
        return _phonemizer
    with _phonemizer_lock:
        if _phonemizer is not None or _phonemizer_failed:
            return _phonemizer
        try:
            from piper.phonemize_espeak import EspeakPhonemizer

            _phonemizer = EspeakPhonemizer()
            logger.debug("Phonémiseur espeak-ng chargé (fourni par piper-tts)")
        except Exception as exc:  # noqa: BLE001 - import, données, binaire natif
            logger.info(
                "Analyse phonétique indisponible (%s) — repli sur la comparaison "
                "orthographique. Installez piper-tts pour l'activer.",
                exc,
            )
            _phonemizer_failed = True
    return _phonemizer


def is_available() -> bool:
    """La phonémisation est-elle utilisable sur cette installation ?"""
    return _get_phonemizer() is not None


def reset() -> None:
    """Oublie l'instance chargée (tests)."""
    global _phonemizer, _phonemizer_failed
    _phonemizer = None
    _phonemizer_failed = False


# ------------------------------------------------------------ phonémisation
def _tokenise(chars: list[str]) -> list[list[str]]:
    """Transforme la sortie brute d'espeak en mots, chacun une liste de phonèmes.

    Les marques de proéminence sont écartées et les signes modificateurs
    (longueur, nasalisation) restent collés au phonème qu'ils qualifient.
    """
    words: list[list[str]] = []
    current: list[str] = []

    for char in chars:
        if char in _STRESS_MARKS:
            continue
        # espeak émet la ponctuation telle quelle : sans ce filtre, le point
        # final d'une phrase compterait comme un phonème manquant et pénaliserait
        # toute lecture correcte.
        if unicodedata.category(char).startswith("P"):
            continue
        if char.isspace():
            if current:
                words.append(current)
                current = []
            continue
        if (char in _MODIFIERS or unicodedata.combining(char)) and current:
            current[-1] += char
            continue
        current.append(char)

    if current:
        words.append(current)

    # espeak renvoie de l'Unicode décomposé (« ç » = c + cédille combinante) :
    # sans normalisation, aucun phonème accentué ne correspondrait aux tables
    # d'étiquettes ni aux familles de sons.
    return [[unicodedata.normalize("NFC", phoneme) for phoneme in word] for word in words]


@lru_cache(maxsize=512)
def _phonemize_cached(voice: str, text: str) -> tuple[tuple[str, ...], ...]:
    phonemizer = _get_phonemizer()
    if phonemizer is None or not text.strip():
        return ()
    try:
        sentences = phonemizer.phonemize(voice, text)
    except Exception as exc:  # noqa: BLE001 - espeak lève large (voix, encodage)
        logger.debug("Phonémisation impossible pour %r (%s) : %s", text, voice, exc)
        return ()

    words: list[list[str]] = []
    for sentence in sentences:
        words.extend(_tokenise(list(sentence)))
    return tuple(tuple(word) for word in words)


def phonemize(text: str, language: str) -> list[list[str]]:
    """Phonèmes IPA de ``text``, un sous-tableau par mot. ``[]`` si indisponible."""
    return [list(word) for word in _phonemize_cached(espeak_voice(language), text)]


def phonemize_flat(text: str, language: str) -> list[str]:
    """Suite de phonèmes, mots confondus."""
    return [phoneme for word in phonemize(text, language) for phoneme in word]


# ------------------------------------------------------- étiquettes lisibles
#: Phonème IPA -> description compréhensible par l'apprenant.
_PHONEME_LABELS: dict[str, str] = {
    # anglais
    "θ": "the unvoiced TH (as in “think”)",
    "ð": "the voiced TH (as in “this”)",
    "ɹ": "the English R",
    "w": "the W sound",
    "v": "the V sound",
    "h": "the aspirated H",
    "ŋ": "the -NG ending",
    "æ": "the flat A (as in “cat”)",
    "ə": "the unstressed schwa",
    "ɜ": "the long ER vowel",
    "ɪ": "the short I (as in “ship”)",
    "i": "the long EE (as in “sheep”)",
    "iː": "the long EE (as in “sheep”)",
    "ʊ": "the short OO (as in “book”)",
    "uː": "the long OO (as in “food”)",
    "ʌ": "the U in “cup”",
    "ɑː": "the long AH",
    "ɒ": "the short O",
    "ɛ": "the E in “bed”",
    "ɔ": "the open O",
    "ɔː": "the long AW",
    "eɪ": "the AY diphthong",
    "aɪ": "the I diphthong (as in “time”)",
    "aʊ": "the OW diphthong (as in “now”)",
    "oʊ": "the OH diphthong",
    "ɔɪ": "the OY diphthong",
    "ʒ": "the ZH sound (as in “measure”)",
    "dʒ": "the J sound",
    "tʃ": "the CH sound",
    # allemand
    "ç": "the German soft CH (ich-Laut)",
    "x": "the German hard CH (ach-Laut)",
    "øː": "the German Ö",
    "œ": "the German Ö (short)",
    "yː": "the German Ü",
    "ʏ": "the German Ü (short)",
    "ɛː": "the German Ä",
    "ʁ": "the German R",
    "ʀ": "the German R",
    "pf": "the German PF",
    "ts": "the German Z (ts)",
    "z": "the voiced S",
    "ʃ": "the SCH sound",
}

#: Familles de sons : sert à décrire une confusion quand le phonème exact
#: n'a pas d'étiquette dédiée.
_FAMILIES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset("θðsz"), "the TH / S distinction"),
    (frozenset("wv"), "the W / V distinction"),
    (frozenset("ɹʁʀr"), "the R sound"),
    (frozenset({"ç", "x", "ʃ", "k"}), "the German CH"),
    (frozenset({"øː", "œ", "oː", "ɔ"}), "the German Ö versus O"),
    (frozenset({"yː", "ʏ", "uː", "ʊ"}), "the German Ü versus U"),
    (frozenset({"ɛː", "ɛ", "eː", "e"}), "the German Ä versus E"),
    (frozenset({"ɪ", "i", "iː"}), "short versus long I (“ship” / “sheep”)"),
    (frozenset({"ʊ", "u", "uː"}), "short versus long OO"),
)


def describe(expected: str, heard: str = "") -> str:
    """Décrit en clair une substitution de phonème."""
    base_expected = expected.rstrip("ː")
    if heard:
        for members, label in _FAMILIES:
            if expected in members and heard in members:
                return label
            if base_expected in members and heard.rstrip("ː") in members:
                return label
    return (
        _PHONEME_LABELS.get(expected)
        or _PHONEME_LABELS.get(base_expected)
        or f"the “{expected}” sound"
    )


# -------------------------------------------------------------- comparaison
@dataclass(slots=True)
class PhonemeDiff:
    """Écart constaté entre la prononciation attendue et celle entendue."""

    kind: str          # "substitution" | "missing" | "extra"
    expected: str
    heard: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "expected": self.expected,
            "heard": self.heard,
            "label": self.label,
        }


@dataclass(slots=True)
class PhonemeComparison:
    accuracy: float                # 0-1, proportion de phonèmes corrects
    diffs: list[PhonemeDiff]
    expected: list[str]
    heard: list[str]

    @property
    def labels(self) -> list[str]:
        """Sons à travailler, sans doublon, du plus fréquent au moins fréquent."""
        counts: dict[str, int] = {}
        for diff in self.diffs:
            counts[diff.label] = counts.get(diff.label, 0) + 1
        return sorted(counts, key=lambda label: -counts[label])


def compare(expected_text: str, heard_text: str, language: str) -> PhonemeComparison | None:
    """Aligne les phonèmes attendus et entendus. ``None`` si indisponible."""
    expected = phonemize_flat(expected_text, language)
    heard = phonemize_flat(heard_text, language)
    if not expected:
        return None

    matcher = difflib.SequenceMatcher(None, expected, heard, autojunk=False)
    diffs: list[PhonemeDiff] = []
    matched = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
        elif tag == "replace":
            for offset in range(max(i2 - i1, j2 - j1)):
                want = expected[i1 + offset] if i1 + offset < i2 else ""
                got = heard[j1 + offset] if j1 + offset < j2 else ""
                if want:
                    diffs.append(
                        PhonemeDiff("substitution", want, got, describe(want, got))
                    )
        elif tag == "delete":
            for index in range(i1, i2):
                diffs.append(
                    PhonemeDiff("missing", expected[index], "", describe(expected[index]))
                )
        elif tag == "insert":
            for index in range(j1, j2):
                diffs.append(PhonemeDiff("extra", "", heard[index], describe(heard[index])))

    return PhonemeComparison(
        accuracy=matched / len(expected) if expected else 0.0,
        diffs=diffs,
        expected=expected,
        heard=heard,
    )
