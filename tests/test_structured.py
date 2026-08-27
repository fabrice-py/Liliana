"""Tests du parsing JSON robuste des réponses du LLM (§23)."""

from __future__ import annotations

import pytest

from app.ai.structured import (
    extract_json,
    normalise_answer_check,
    normalise_assessment,
    normalise_exercise,
    normalise_turn,
)


@pytest.mark.parametrize(
    "raw",
    [
        '{"response": "Hi"}',
        '```json\n{"response": "Hi"}\n```',
        'Sure, here you go:\n{"response": "Hi"}\nHope this helps!',
        '{"response": "Hi",}',                       # virgule finale
        '{“response”: “Hi”}',                        # guillemets typographiques
        '{"response": "Hi"',                         # objet tronqué
        '{"response": "Hi',                          # chaîne tronquée
    ],
)
def test_extract_json_survives_malformed_output(raw: str) -> None:
    assert extract_json(raw) == {"response": "Hi"}


def test_extract_json_ignores_braces_inside_strings() -> None:
    parsed = extract_json('{"response": "He said {hi} to me"}')
    assert parsed == {"response": "He said {hi} to me"}


@pytest.mark.parametrize("raw", ["", "   ", "no json at all", "[1, 2, 3]"])
def test_extract_json_returns_none_when_hopeless(raw: str) -> None:
    assert extract_json(raw) is None


def test_normalise_turn_falls_back_to_plain_text() -> None:
    turn = normalise_turn(None, "Just a plain answer.")
    assert turn["response"] == "Just a plain answer."
    assert turn["errors"] == []
    assert turn["correction"] is None
    assert turn["structured"] is False


def test_normalise_turn_fills_missing_fields() -> None:
    turn = normalise_turn({"response": "Hi"}, "")
    assert set(turn) == {
        "response", "correction", "errors", "vocabulary",
        "detected_language", "difficulty", "suggested_level", "structured",
    }
    assert turn["vocabulary"] == []


def test_normalise_turn_drops_no_op_correction() -> None:
    turn = normalise_turn(
        {"response": "Hi", "correction": {"original": "same", "corrected": "same"}}, ""
    )
    assert turn["correction"] is None


def test_normalise_turn_accepts_errors_as_plain_strings() -> None:
    turn = normalise_turn({"response": "Hi", "errors": ["past_simple"]}, "")
    assert turn["errors"][0]["type"] == "past_simple"
    assert turn["errors"][0]["severity"] == "minor"


def test_normalise_turn_rejects_invalid_severity_and_level() -> None:
    turn = normalise_turn(
        {
            "response": "Hi",
            "errors": [{"type": "grammar", "severity": "catastrophic"}],
            "difficulty": "Z9",
        },
        "",
    )
    assert turn["errors"][0]["severity"] == "minor"
    assert turn["difficulty"] == ""


def test_normalise_exercise_requires_a_prompt() -> None:
    assert normalise_exercise({"exercise_type": "translation"}) is None
    exercise = normalise_exercise({"prompt": "Translate: le chat", "answer": "the cat"})
    assert exercise["exercise_type"] == "open"
    assert exercise["options"] == []


def test_normalise_answer_check_reads_string_booleans() -> None:
    assert normalise_answer_check({"is_correct": "true"})["is_correct"] is True
    assert normalise_answer_check({"is_correct": "no"})["is_correct"] is False


def test_normalise_assessment_clamps_scores() -> None:
    result = normalise_assessment(
        {"level": "B1", "scores": {"grammar": 150, "speaking": -20, "unknown": 50}}
    )
    assert result["scores"]["grammar"] == 100.0
    assert result["scores"]["speaking"] == 0.0
    assert result["level"] == "B1"


def test_normalise_assessment_returns_none_when_empty() -> None:
    assert normalise_assessment({"summary": "nothing useful"}) is None
    assert normalise_assessment(None) is None


@pytest.mark.parametrize("envelope", ["{}", "[]", "null", "  { }  ", "...", '""'])
def test_an_empty_json_envelope_is_not_an_answer(envelope: str) -> None:
    """« {} » n'est pas du texte libre : le parler ou l'enregistrer casse tout.

    Enregistré comme message de l'assistant, il revient dans l'historique du tour
    suivant et le modèle imite sa propre sortie vide — la session ne produit plus
    que « {} ». Une réponse vide fait lever une erreur claire à l'appelant.
    """
    turn = normalise_turn(None, fallback_text=envelope)
    assert turn["response"] == ""


def test_free_text_is_still_used_as_the_answer() -> None:
    """Le repli sur le texte brut reste la règle quand le modèle oublie le JSON."""
    turn = normalise_turn(None, fallback_text="Sorry, I forgot the JSON.")
    assert turn["response"] == "Sorry, I forgot the JSON."
