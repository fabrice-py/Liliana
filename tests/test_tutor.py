"""Tests du tuteur : contexte, mémoire et robustesse d'un tour de conversation."""

from __future__ import annotations

import json

import pytest

from app.ai.tutor import Tutor
from app.core.exceptions import LLMError
from app.database.repositories import (
    app_settings,
    errors,
    language_profiles,
    messages,
    reviews,
    sessions,
    vocabulary,
)
from tests.conftest import FakeLLM

RICH_TURN = json.dumps({
    "response": "Sounds fun! Who did you go with?",
    "correction": {
        "original": "Yesterday I go to Paris",
        "corrected": "Yesterday I went to Paris",
        "explanation": "Past tense with 'yesterday'.",
    },
    "errors": [{"type": "verb_tense", "topic": "past_simple", "severity": "major"}],
    "vocabulary": [{"word": "commute", "translation": "trajet", "example": "My commute is short."}],
    "detected_language": "english",
    "difficulty": "A2",
})


@pytest.fixture
def session_id(user_id: int) -> int:
    return int(sessions.create(user_id, "english", "free_conversation")["id"])


def test_turn_stores_messages_errors_and_vocabulary(user_id: int, session_id: int) -> None:
    tutor = Tutor(llm=FakeLLM([RICH_TURN]))
    result = tutor.respond(
        user_id=user_id, session_id=session_id, text="Yesterday I go to Paris",
        language="english", mode="free_conversation",
    )

    assert result.response == "Sounds fun! Who did you go with?"
    assert result.correction["corrected"] == "Yesterday I went to Paris"
    assert [message["role"] for message in messages.history(session_id)] == ["user", "assistant"]
    assert errors.count(user_id, "english") == 1
    assert vocabulary.count(user_id, "english") == 1


def test_turn_schedules_weak_topics_and_new_words(user_id: int, session_id: int) -> None:
    Tutor(llm=FakeLLM([RICH_TURN])).respond(
        user_id=user_id, session_id=session_id, text="Yesterday I go to Paris",
        language="english", mode="free_conversation",
    )
    scheduled = {
        (item["item_type"], item["item_key"])
        for item in reviews.all_for(user_id, "english")
    }
    assert ("grammar", "past_simple") in scheduled
    assert ("vocabulary", "commute") in scheduled


def test_invented_error_types_are_normalised(user_id: int, session_id: int) -> None:
    payload = json.dumps({
        "response": "Ok.",
        "errors": [{"type": "quantum_syntax", "topic": "past_simple", "severity": "minor"}],
    })
    Tutor(llm=FakeLLM([payload])).respond(
        user_id=user_id, session_id=session_id, text="Hi", language="english",
        mode="free_conversation",
    )
    stored = errors.recent(user_id, "english")[0]
    assert stored["error_type"] == "grammar"
    assert stored["topic"] == "past_simple"


def test_german_specific_error_types_are_kept(user_id: int) -> None:
    session = int(sessions.create(user_id, "german", "german_teacher")["id"])
    payload = json.dumps({
        "response": "Gut!",
        "errors": [{"type": "cases", "topic": "dative", "severity": "major"}],
    })
    Tutor(llm=FakeLLM([payload])).respond(
        user_id=user_id, session_id=session, text="Ich gebe der Mann das Buch",
        language="german", mode="german_teacher",
    )
    assert errors.recent(user_id, "german")[0]["error_type"] == "cases"


def test_unstructured_answer_is_still_spoken(user_id: int, session_id: int) -> None:
    result = Tutor(llm=FakeLLM(["Sorry, I forgot the JSON."])).respond(
        user_id=user_id, session_id=session_id, text="Hi", language="english",
        mode="free_conversation",
    )
    assert result.response == "Sorry, I forgot the JSON."
    assert result.structured is False
    assert result.errors == []


def test_empty_model_answer_raises_a_clear_error(user_id: int, session_id: int) -> None:
    with pytest.raises(LLMError) as excinfo:
        Tutor(llm=FakeLLM(["   "])).respond(
            user_id=user_id, session_id=session_id, text="Hi", language="english",
            mode="free_conversation",
        )
    assert "could not produce an answer" in excinfo.value.user_message


def test_empty_user_turn_is_rejected(user_id: int, session_id: int) -> None:
    with pytest.raises(LLMError):
        Tutor(llm=FakeLLM()).respond(
            user_id=user_id, session_id=session_id, text="   ", language="english",
            mode="free_conversation",
        )


def test_unsupported_language_falls_back_to_the_default(user_id: int, session_id: int) -> None:
    result = Tutor(llm=FakeLLM()).respond(
        user_id=user_id, session_id=session_id, text="Hello", language="klingon",
        mode="free_conversation",
    )
    assert result.session_id == session_id


def test_context_reflects_stored_history(user_id: int, session_id: int) -> None:
    errors.add_many(user_id, session_id, "english",
                    [{"type": "grammar", "topic": "articles"}] * 3)
    vocabulary.add_many(user_id, "english", [{"word": "sleeve"}])
    language_profiles.update_scores(user_id, "english", {"grammar": 55.0}, level="B1")

    context = Tutor(llm=FakeLLM()).build_context(user_id, "english", "free_conversation")
    assert context.level == "B1"
    assert context.weaknesses[0]["topic"] == "articles"
    assert "sleeve" in context.recent_vocabulary


def test_correction_mode_precedence(user_id: int) -> None:
    tutor = Tutor(llm=FakeLLM())
    # 1. l'appelant l'emporte sur tout
    assert tutor.build_context(user_id, "english", "just_talk", "strict").correction_mode == "strict"
    # 2. sinon le réglage enregistré
    app_settings.set("correction_mode", "minimal")
    assert tutor.build_context(user_id, "english", "free_conversation").correction_mode == "minimal"


def test_mode_default_correction_applies_without_a_stored_setting(user_id: int) -> None:
    # `just_talk` corrige au minimum par construction (cf. §27).
    context = Tutor(llm=FakeLLM()).build_context(user_id, "english", "just_talk")
    assert context.correction_mode == "minimal"


def test_history_is_capped(user_id: int, session_id: int, isolated_settings) -> None:
    for index in range(40):
        messages.add(session_id, "user", f"message {index}")
    fake = FakeLLM()
    Tutor(llm=fake).respond(
        user_id=user_id, session_id=session_id, text="Latest", language="english",
        mode="free_conversation",
    )
    sent = fake.calls[-1]
    assert len(sent) <= isolated_settings.llm_max_history_turns * 2 + 2


def test_voice_turns_are_recorded_as_spoken(user_id: int, session_id: int) -> None:
    Tutor(llm=FakeLLM([RICH_TURN])).respond(
        user_id=user_id, session_id=session_id, text="Yesterday I go to Paris",
        language="english", mode="free_conversation", is_voice=True,
    )
    assert errors.count(user_id, "english", is_voice=True) == 1
    assert errors.count(user_id, "english", is_voice=False) == 0


def test_text_turns_are_recorded_as_written(user_id: int, session_id: int) -> None:
    Tutor(llm=FakeLLM([RICH_TURN])).respond(
        user_id=user_id, session_id=session_id, text="Yesterday I go to Paris",
        language="english", mode="free_conversation", is_voice=False,
    )
    assert errors.count(user_id, "english", is_voice=False) == 1
