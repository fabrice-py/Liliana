"""Mot d'éveil : décider si une phrase entendue s'adresse à Liliana.

En écoute permanente, le micro capte tout ce qui se dit dans la pièce. Whisper
transcrit chaque prise de parole — c'est bon marché — mais le modèle de langue,
lui, ne doit répondre que si on l'a appelée. Ce module est ce filtre.

Il ne compare pas des chaînes à l'identique : Whisper écrit rarement « Liliana »
deux fois de la même façon. Sur une voix francophone il entend « Lilliana »,
« Liliane », « Lily Anna », « Leliana ». Un test d'égalité stricte rendrait
l'éveil inutilisable ; on mesure donc une ressemblance, sur le premier fragment
de la phrase uniquement — le nom se dit en s'adressant à quelqu'un, en tête.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

#: Nom par défaut. Configurable via ``WAKE_WORD``.
DEFAULT_WAKE_WORD = "Liliana"

#: Mots de politesse ou d'appel qui précèdent souvent le nom, dans les trois
#: langues de Liliana. Ils sont ignorés avant de chercher le nom lui-même.
_GREETINGS = frozenset(
    {
        "hello", "hi", "hey", "ok", "okay", "yo",
        "hallo", "he", "guten", "tag", "morgen",
        "bonjour", "salut", "coucou", "eh", "dis",
    }
)

#: Nombre de mots examinés en tête de phrase. Au-delà, le nom prononcé au milieu
#: d'un récit (« … et j'ai dit à Liliana que… ») ne doit pas déclencher un tour.
_LOOKAHEAD_WORDS = 4

#: Un nom peut être entendu en deux morceaux (« Lily Anna ») : on teste donc
#: aussi les paires de mots consécutifs.
_MAX_TOKENS_PER_NAME = 2


@dataclass(frozen=True, slots=True)
class WakeWordMatch:
    """Résultat de l'écoute d'une phrase.

    ``heard``     — le nom a été reconnu, la phrase s'adresse à Liliana ;
    ``matched``   — ce que Whisper a réellement écrit à la place du nom ;
    ``remainder`` — la phrase débarrassée de l'appel, à traiter comme un tour ;
    ``score``     — ressemblance retenue, utile pour régler le seuil.
    """

    heard: bool
    matched: str = ""
    remainder: str = ""
    score: float = 0.0


def _normalise(text: str) -> str:
    """Minuscules, sans accents ni ponctuation. Même convention que commands.py."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower().strip()


def _similarity(heard: str, expected: str) -> float:
    """Ressemblance entre deux mots, dans [0, 1]."""
    return SequenceMatcher(None, heard, expected).ratio()


def parse_wake_words(configured: str) -> tuple[str, ...]:
    """Lit le réglage ``WAKE_WORD`` : un nom, ou plusieurs séparés par des virgules."""
    names = tuple(name.strip() for name in (configured or "").split(",") if name.strip())
    return names or (DEFAULT_WAKE_WORD,)


def detect(
    text: str,
    wake_words: tuple[str, ...] = (DEFAULT_WAKE_WORD,),
    *,
    threshold: float = 0.8,
) -> WakeWordMatch:
    """Cherche un appel à Liliana en tête de ``text``.

    Retourne le reste de la phrase quand le nom est reconnu. Ce reste peut être
    vide — « Liliana ? » tout court est un appel valide, auquel elle doit
    répondre en invitant à continuer.
    """
    words = re.findall(r"[\w']+", _normalise(text))
    if not words:
        return WakeWordMatch(heard=False)

    expected = [_normalise(name) for name in wake_words if name.strip()]
    if not expected:
        return WakeWordMatch(heard=False)

    # On saute les salutations : « Hello Liliana » appelle autant que « Liliana ».
    start = 0
    while start < len(words) and words[start] in _GREETINGS:
        start += 1

    best = WakeWordMatch(heard=False)
    limit = min(start + _LOOKAHEAD_WORDS, len(words))

    for index in range(start, limit):
        for span in range(1, _MAX_TOKENS_PER_NAME + 1):
            end = index + span
            if end > len(words):
                break
            candidate = "".join(words[index:end])  # « lily anna » -> « lilyanna »
            for name in expected:
                score = _similarity(candidate, name.replace(" ", ""))
                if score >= threshold and score > best.score:
                    best = WakeWordMatch(
                        heard=True,
                        matched=" ".join(words[index:end]),
                        remainder=_remainder(text, words, end),
                        score=round(score, 3),
                    )
    return best


def _remainder(original: str, words: list[str], consumed: int) -> str:
    """Ce qu'il reste à dire une fois l'appel retiré.

    On repart du texte d'origine — ponctuation et majuscules comprises — plutôt
    que des mots normalisés : c'est cette phrase-là qui sera corrigée, elle doit
    rester exactement telle que l'apprenant l'a prononcée.
    """
    if consumed >= len(words):
        return ""
    tail = words[consumed]
    # Retrouve le mot suivant dans le texte d'origine, sans tenir compte de la casse.
    match = re.search(rf"\b{re.escape(tail)}\b", original, flags=re.IGNORECASE)
    remainder = original[match.start():] if match else " ".join(words[consumed:])
    return remainder.strip(" ,;:!?.-—–").strip()
