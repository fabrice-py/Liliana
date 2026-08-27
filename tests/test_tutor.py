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


def test_an_empty_envelope_does_not_poison_the_session(user_id: int, session_id: int) -> None:
    """Un « {} » du modèle ne doit jamais devenir un message de l'assistant.

    Sinon il revient dans l'historique et le modèle l'imite à chaque tour
    suivant : la conversation ne produit plus que des réponses vides.
    """
    with pytest.raises(LLMError):
        Tutor(llm=FakeLLM(["{}"])).respond(
            user_id=user_id, session_id=session_id, text="Hi", language="english",
            mode="free_conversation",
        )

    stored = messages.history(session_id, limit=10)
    assert [m["content"] for m in stored if m["role"] == "assistant"] == []

    # Le tour suivant repart proprement.
    result = Tutor(llm=FakeLLM([RICH_TURN])).respond(
        user_id=user_id, session_id=session_id, text="Hi again", language="english",
        mode="free_conversation",
    )
    assert result.response


def test_an_already_poisoned_history_is_not_replayed(user_id: int, session_id: int) -> None:
    """Une base déjà contaminée par une version antérieure doit se remettre seule."""
    messages.add(session_id, "user", "Tell me about Berlin", "english")
    messages.add(session_id, "assistant", "It is a lively city.", "english")
    messages.add(session_id, "user", "Hello", "english")
    messages.add(session_id, "assistant", "{}", "english")

    history = Tutor(llm=FakeLLM())._history(session_id)

    # Le tour raté disparaît entièrement, question comprise.
    assert [m["content"] for m in history] == ["Tell me about Berlin", "It is a lively city."]


def test_history_never_shows_two_questions_in_a_row(user_id: int, session_id: int) -> None:
    """Des tours en échec laissent des questions sans réponse en base.

    Les enchaîner donnerait au modèle une conversation qu'il n'a jamais vue —
    plusieurs questions d'affilée — et sa réponse s'en ressent.
    """
    messages.add(session_id, "user", "First question", "english")
    messages.add(session_id, "user", "Second question", "english")
    messages.add(session_id, "user", "Third question", "english")
    messages.add(session_id, "assistant", "An actual answer.", "english")

    history = Tutor(llm=FakeLLM())._history(session_id)

    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant"]
    assert history[0]["content"] == "Third question"


def test_a_complete_exchange_is_kept(user_id: int, session_id: int) -> None:
    messages.add(session_id, "user", "How are you?", "english")
    messages.add(session_id, "assistant", "Very well, thank you.", "english")

    history = Tutor(llm=FakeLLM())._history(session_id)

    assert history == [
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Very well, thank you."},
    ]


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


def test_the_turn_contract_is_sent_as_a_schema(user_id: int, session_id: int) -> None:
    """Le contrat de sortie doit contraindre le decodage, pas seulement le prompt.

    Sans cela le modele ecrit la correction dans `response` et laisse
    `correction` et `errors` vides : la carte Correction, les statistiques et la
    repetition espacee se retrouvent sans matiere.
    """
    from app.ai.prompts import TURN_RESPONSE_SCHEMA

    llm = FakeLLM([RICH_TURN])
    Tutor(llm=llm).respond(
        user_id=user_id, session_id=session_id, text="Hi", language="english",
        mode="free_conversation",
    )
    assert llm.schemas == [TURN_RESPONSE_SCHEMA]


def test_the_streamed_turn_uses_the_same_contract(user_id: int, session_id: int) -> None:
    from app.ai.prompts import TURN_RESPONSE_SCHEMA

    llm = FakeLLM([RICH_TURN])
    list(Tutor(llm=llm).respond_stream(
        user_id=user_id, session_id=session_id, text="Hi", language="english",
        mode="free_conversation",
    ))
    assert llm.schemas == [TURN_RESPONSE_SCHEMA]


def test_response_comes_first_so_the_voice_can_start_early() -> None:
    """L'ordre des champs du schema pilote l'ordre de generation.

    `response` doit rester en tete, sinon la premiere phrase n'est prononcable
    qu'apres la correction et tout le benefice du streaming disparait.
    """
    from app.ai.prompts import TURN_RESPONSE_SCHEMA

    assert list(TURN_RESPONSE_SCHEMA["properties"])[0] == "response"
    assert set(TURN_RESPONSE_SCHEMA["required"]) == set(TURN_RESPONSE_SCHEMA["properties"])


def test_the_model_follows_the_language_being_practised(user_id: int, monkeypatch) -> None:
    """Alterner anglais et allemand doit changer de modele, pas de qualite."""
    monkeypatch.setenv("LLM_MODEL_ENGLISH", "small-fast-model")
    monkeypatch.setenv("LLM_MODEL_GERMAN", "big-accurate-model")
    from app.core.config import reload_settings

    reload_settings()
    llm = FakeLLM([RICH_TURN])
    tutor = Tutor(llm=llm)

    for language in ("english", "german"):
        session = int(sessions.create(user_id, language, "free_conversation")["id"])
        tutor.respond(
            user_id=user_id, session_id=session, text="Hello", language=language,
            mode="free_conversation",
        )

    assert llm.models == ["small-fast-model", "big-accurate-model"]
