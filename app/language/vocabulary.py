"""Entraînement au vocabulaire piloté par la répétition espacée (§8 mode 6).

Les mots proviennent de deux sources :
* ceux que Liliana a introduits pendant les conversations (table ``vocabulary``) ;
* ceux qu'elle génère à la demande pour un thème donné.
"""

from __future__ import annotations

from typing import Any

from app.ai.llm import LLMProvider, get_llm_provider
from app.ai.structured import extract_json
from app.core.logger import get_logger
from app.database.repositories import vocabulary as vocabulary_repo
from app.language.languages import get_language
from app.learning.spaced_repetition import spaced_repetition

logger = get_logger(__name__)

_GENERATION_PROMPT = """\
You are a {language} teacher. Propose {count} words or expressions to teach a
learner at CEFR level {level} about: {theme}.

Answer with a single JSON object and nothing else:

{{
  "words": [
    {{"word": "<the word in {language}>",
      "translation": "<translation into {native}>",
      "example": "<one short example sentence in {language}>",
      "part_of_speech": "<noun|verb|adjective|adverb|expression>",
      "difficulty": "<A1..C2>"}}
  ]
}}

Avoid words the learner already knows: {known}.\
"""


class VocabularyService:
    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        return self._llm or get_llm_provider()

    # ------------------------------------------------------------ révision
    def due_words(self, user_id: int, language: str, limit: int = 10) -> list[dict[str, Any]]:
        """Mots à réviser aujourd'hui, enrichis de leur fiche."""
        due = spaced_repetition.due(user_id, language, limit=limit, item_type="vocabulary")
        known = {
            item["word"]: item
            for item in vocabulary_repo.list_for(user_id, language, limit=500)
        }
        result: list[dict[str, Any]] = []
        for item in due:
            word = str(item["item_key"])
            entry = known.get(word, {"word": word, "translation": "", "example": ""})
            result.append(
                {
                    **entry,
                    "confidence": float(item.get("confidence", 0.0)),
                    "success_count": int(item.get("success_count", 0)),
                    "failure_count": int(item.get("failure_count", 0)),
                    "next_review": item.get("next_review"),
                }
            )
        return result

    def review_word(
        self, user_id: int, language: str, word: str, remembered: bool
    ) -> dict[str, Any]:
        """Enregistre le résultat d'une révision de mot."""
        return spaced_repetition.review(
            user_id, language, "vocabulary", word, quality=5 if remembered else 2
        )

    # ---------------------------------------------------------- génération
    def teach(
        self,
        user_id: int,
        language: str,
        level: str,
        theme: str = "everyday life",
        count: int = 5,
        native_language: str = "French",
    ) -> list[dict[str, Any]]:
        """Génère de nouveaux mots, les enregistre et les planifie en révision."""
        language_name = get_language(language).english_name
        known = [item["word"] for item in vocabulary_repo.list_for(user_id, language, limit=40)]

        raw = self.llm.generate(
            [
                {
                    "role": "system",
                    "content": _GENERATION_PROMPT.format(
                        language=language_name,
                        count=max(1, min(count, 12)),
                        level=level,
                        theme=theme,
                        native=native_language,
                        known=", ".join(known) if known else "none",
                    ),
                },
                {"role": "user", "content": f"Theme: {theme}"},
            ],
            temperature=0.7,
            json_mode=True,
        )

        parsed = extract_json(raw) or {}
        words = parsed.get("words")
        if not isinstance(words, list):
            logger.warning("Génération de vocabulaire inexploitable pour le thème %r", theme)
            return []

        entries = [
            {
                "word": str(item.get("word", "")).strip(),
                "translation": str(item.get("translation", "")).strip(),
                "example": str(item.get("example", "")).strip(),
                "part_of_speech": str(item.get("part_of_speech", "")).strip(),
                "difficulty": str(item.get("difficulty", level)).strip().upper(),
            }
            for item in words
            if isinstance(item, dict) and str(item.get("word", "")).strip()
        ]

        added = vocabulary_repo.add_many(user_id, language, entries)
        spaced_repetition.register_many(user_id, language, "vocabulary", added)
        return entries


vocabulary_service = VocabularyService()
