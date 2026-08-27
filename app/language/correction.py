"""Correction linguistique hors conversation.

Utilisé par l'endpoint ``/api/correct`` et par les commandes « explain that
rule » / « try again ». La correction faite pendant un tour de conversation
passe, elle, par :mod:`app.ai.tutor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ai.llm import LLMProvider, get_llm_provider
from app.ai.prompts import correction_instruction
from app.ai.structured import extract_json
from app.core.logger import get_logger
from app.language.languages import error_types_for, get_language

logger = get_logger(__name__)

#: Erreurs remontées à l'utilisateur selon le mode de correction (cf. §11).
_SEVERITY_FILTER: dict[str, tuple[str, ...]] = {
    "off": (),
    "minimal": ("major",),
    "normal": ("major", "minor"),
    "strict": ("major", "minor"),
}

_CORRECTION_PROMPT = """\
You are a {language} teacher correcting a learner whose CEFR level is {level}.

{correction_instruction}

Correct the sentence below. Answer with a single JSON object and nothing else:

{{
  "corrected": "<the corrected sentence, or the original if it is already correct>",
  "is_correct": true | false,
  "explanation": "<short explanation addressed to the learner, 1-2 sentences>",
  "errors": [
    {{"type": "<one of: {error_types}>", "topic": "<specific topic>",
      "original": "<wrong fragment>", "corrected": "<fixed fragment>",
      "severity": "minor" | "major"}}
  ],
  "natural_alternative": "<a more natural way to say it, or an empty string>"
}}\
"""


@dataclass(slots=True)
class CorrectionResult:
    original: str
    corrected: str
    is_correct: bool
    explanation: str = ""
    natural_alternative: str = ""
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "corrected": self.corrected,
            "is_correct": self.is_correct,
            "explanation": self.explanation,
            "natural_alternative": self.natural_alternative,
            "errors": self.errors,
        }


def filter_by_mode(errors: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Ne garde que les erreurs qu'il faut montrer dans ce mode de correction."""
    allowed = _SEVERITY_FILTER.get((mode or "normal").lower(), _SEVERITY_FILTER["normal"])
    if not allowed:
        return []
    return [
        error
        for error in errors
        if str(error.get("severity", "minor")).lower() in allowed
    ]


class CorrectionService:
    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        return self._llm or get_llm_provider()

    def correct(
        self,
        text: str,
        language: str,
        level: str = "A1",
        correction_mode: str = "normal",
    ) -> CorrectionResult:
        """Corrige une phrase isolée."""
        text = (text or "").strip()
        if not text:
            return CorrectionResult(original="", corrected="", is_correct=True)
        if correction_mode.lower() == "off":
            return CorrectionResult(original=text, corrected=text, is_correct=True)

        system = _CORRECTION_PROMPT.format(
            language=get_language(language).english_name,
            level=level,
            correction_instruction=correction_instruction(correction_mode),
            error_types=", ".join(error_types_for(language)),
        )
        raw = self.llm.generate(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            temperature=0.2,
            json_mode=True,
        )
        parsed = extract_json(raw) or {}

        corrected = str(parsed.get("corrected") or "").strip() or text
        errors = [
            error for error in parsed.get("errors", []) if isinstance(error, dict)
        ]
        is_correct = bool(parsed.get("is_correct", corrected == text)) and corrected == text

        return CorrectionResult(
            original=text,
            corrected=corrected,
            is_correct=is_correct,
            explanation=str(parsed.get("explanation") or "").strip(),
            natural_alternative=str(parsed.get("natural_alternative") or "").strip(),
            errors=filter_by_mode(errors, correction_mode),
        )

    def explain(self, topic: str, language: str, level: str = "A1") -> str:
        """Explique un point de grammaire, brièvement, avec des exemples."""
        system = (
            f"You are a {get_language(language).english_name} teacher. "
            f"The learner is level {level}. Explain the following point in at most "
            "5 short sentences, with 2 or 3 examples, then ask them to produce one "
            "sentence using it. Plain text only, no markdown."
        )
        return self.llm.generate(
            [{"role": "system", "content": system}, {"role": "user", "content": topic}],
            temperature=0.4,
        ).strip()


correction_service = CorrectionService()
