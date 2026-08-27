"""Tests de bout en bout de l'API, moteurs simulés."""

from __future__ import annotations

import base64
import json

from app.database.repositories import errors, messages, sessions, vocabulary

TURN_JSON = json.dumps(
    {
        "response": "Nice! What did you watch?",
        "correction": {
            "original": "Yesterday I go to the cinema.",
            "corrected": "Yesterday I went to the cinema.",
            "explanation": "Use the past tense with 'yesterday'.",
        },
        "errors": [
            {"type": "verb_tense", "topic": "past_simple", "original": "go",
             "corrected": "went", "severity": "major"}
        ],
        "vocabulary": [{"word": "screening", "translation": "projection",
                        "example": "The screening starts at 8."}],
        "detected_language": "english",
        "difficulty": "A2",
        "suggested_level": "B1",
    }
)


# --------------------------------------------------------------------- statut
def test_health_reports_every_engine(client) -> None:
    payload = client.get("/api/health").json()
    assert payload["ready"] is True
    assert payload["offline_capable"] is True
    assert set(payload) >= {"llm", "stt", "tts"}


def test_config_lists_languages_modes_and_vad(client) -> None:
    payload = client.get("/api/config").json()
    assert {language["code"] for language in payload["languages"]} == {"english", "german"}
    assert "free_conversation" in {mode["key"] for mode in payload["modes"]}
    assert payload["vad"]["silence_threshold"] > 0
    assert payload["current"]["language"] == "english"


def test_hardware_endpoint_returns_recommendations(client) -> None:
    payload = client.get("/api/hardware").json()
    assert payload["cpu_count"] >= 1
    assert "llm_model" in payload["recommendations"]


def test_index_page_is_served(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Liliana" in response.text


# ---------------------------------------------------------------- réglages
def test_settings_round_trip(client) -> None:
    updated = client.post(
        "/api/settings", json={"language": "german", "correction_mode": "strict"}
    ).json()
    assert updated["language"] == "german"
    assert updated["correction_mode"] == "strict"
    assert client.get("/api/config").json()["current"]["language"] == "german"


def test_settings_rejects_unknown_language(client) -> None:
    assert client.post("/api/settings", json={"language": "klingon"}).status_code == 422


def test_settings_rejects_unknown_mode(client) -> None:
    assert client.post("/api/settings", json={"mode": "hypnosis"}).status_code == 422


# ------------------------------------------------------------ tour écrit
def test_chat_turn_returns_answer_and_correction(client) -> None:
    client.fake_llm.responses = [TURN_JSON]
    payload = client.post(
        "/api/chat/turn", json={"text": "Yesterday I go to the cinema.", "language": "english"}
    ).json()

    assert payload["response"] == "Nice! What did you watch?"
    assert payload["correction"]["corrected"] == "Yesterday I went to the cinema."
    assert payload["errors"][0]["topic"] == "past_simple"
    assert payload["speech"]["mime_type"] == "audio/wav"
    assert base64.b64decode(payload["speech"]["audio_base64"]).startswith(b"RIFF")


def test_chat_turn_persists_everything(client, user_id: int) -> None:
    client.fake_llm.responses = [TURN_JSON]
    session_id = client.post(
        "/api/chat/turn", json={"text": "Yesterday I go to the cinema."}
    ).json()["session_id"]

    history = messages.history(session_id)
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert errors.count(user_id, "english") == 1
    assert vocabulary.count(user_id, "english") == 1


def test_chat_turn_survives_unstructured_answers(client) -> None:
    client.fake_llm.responses = ["I am not going to produce JSON today."]
    payload = client.post("/api/chat/turn", json={"text": "Hello"}).json()
    assert payload["response"] == "I am not going to produce JSON today."
    assert payload["structured"] is False
    assert payload["errors"] == []


def test_chat_turn_can_skip_speech(client) -> None:
    payload = client.post("/api/chat/turn", json={"text": "Hello", "speak": False}).json()
    assert payload["speech"] is None


def test_empty_text_is_rejected(client) -> None:
    assert client.post("/api/chat/turn", json={"text": ""}).status_code == 422


def test_conversation_context_is_sent_to_the_model(client) -> None:
    client.post("/api/chat/turn", json={"text": "First message"})
    client.post("/api/chat/turn", json={"text": "Second message"})
    last_call = client.fake_llm.calls[-1]
    assert last_call[0]["role"] == "system"
    assert any(message["content"] == "First message" for message in last_call)


def test_system_prompt_carries_level_and_mode(client) -> None:
    client.post("/api/chat/turn", json={"text": "Hello", "mode": "immersion"})
    system_prompt = client.fake_llm.calls[-1][0]["content"]
    assert "Immersion" in system_prompt
    assert "User CEFR level in this language: A1" in system_prompt


# ------------------------------------------------------------ tour vocal
def test_voice_turn_transcribes_then_answers(client) -> None:
    client.fake_llm.responses = [TURN_JSON]
    response = client.post(
        "/api/voice/turn",
        files={"audio": ("turn.webm", b"fake-audio-bytes" * 100, "audio/webm")},
        data={"language": "english"},
    )
    payload = response.json()
    assert payload["transcription"]["text"] == "Yesterday I go to the cinema."
    assert payload["response"] == "Nice! What did you watch?"
    assert payload["speech"] is not None


def test_voice_turn_rejects_empty_audio(client) -> None:
    response = client.post(
        "/api/voice/turn", files={"audio": ("turn.webm", b"", "audio/webm")}
    )
    assert response.status_code == 400
    assert "empty recording" in response.json()["detail"]["message"].lower()


def test_voice_turn_reports_silence_clearly(client, monkeypatch) -> None:
    from app.core.exceptions import EmptyTranscriptionError

    def _silent(audio, language=None):  # noqa: ANN001
        raise EmptyTranscriptionError("silence")

    monkeypatch.setattr(client.fake_stt, "transcribe", _silent)
    response = client.post(
        "/api/voice/turn", files={"audio": ("turn.webm", b"x" * 500, "audio/webm")}
    )
    assert response.status_code == 422
    assert "did not hear" in response.json()["detail"]["message"].lower()


# ------------------------------------------------------- commandes vocales
def test_voice_command_switches_language(client) -> None:
    client.fake_stt.text = "Liliana, switch to German."
    payload = client.post(
        "/api/voice/turn", files={"audio": ("t.webm", b"x" * 500, "audio/webm")}
    ).json()
    assert payload["command"] == {"action": "switch_language", "language": "german"}
    assert payload["language"] == "german"
    assert client.get("/api/config").json()["current"]["language"] == "german"


def test_voice_command_changes_correction_mode(client) -> None:
    payload = client.post("/api/chat/turn", json={"text": "Liliana, correct me."}).json()
    assert payload["command"]["correction_mode"] == "strict"
    assert client.get("/api/config").json()["current"]["correction_mode"] == "strict"


# ---------------------------------------------------------------- sessions
def test_session_endpoints(client, user_id: int) -> None:
    first = client.get("/api/session/current").json()
    client.post("/api/chat/turn", json={"text": "Hello"})

    reopened = client.get("/api/session/current").json()
    assert reopened["id"] == first["id"]
    assert len(reopened["messages"]) == 2

    fresh = client.post("/api/session/new").json()
    assert fresh["id"] != first["id"]

    closed = client.post(f"/api/session/{fresh['id']}/close").json()
    assert closed["ended_at"] is not None


def test_closing_an_unknown_session_returns_404(client) -> None:
    assert client.post("/api/session/999/close").status_code == 404


# --------------------------------------------------------------- correction
def test_correct_endpoint(client) -> None:
    client.fake_llm.responses = [
        json.dumps({
            "corrected": "Yesterday I went to Paris.",
            "is_correct": False,
            "explanation": "Past tense needed.",
            "errors": [{"type": "verb_tense", "topic": "past_simple", "severity": "major"}],
            "natural_alternative": "",
        })
    ]
    payload = client.post(
        "/api/correct", json={"text": "Yesterday I go to Paris.", "correction_mode": "normal"}
    ).json()
    assert payload["corrected"] == "Yesterday I went to Paris."
    assert payload["is_correct"] is False
    assert payload["errors"]


def test_correct_endpoint_is_a_no_op_when_correction_is_off(client) -> None:
    payload = client.post(
        "/api/correct", json={"text": "Yesterday I go.", "correction_mode": "off"}
    ).json()
    assert payload["corrected"] == "Yesterday I go."
    assert payload["is_correct"] is True


def test_explain_endpoint(client) -> None:
    client.fake_llm.responses = ["The past perfect describes an action before another past action."]
    payload = client.post("/api/explain", json={"topic": "past perfect"}).json()
    assert "past perfect" in payload["explanation"].lower()


# ---------------------------------------------------------------- exercices
EXERCISE_JSON = json.dumps({
    "exercise_type": "multiple_choice",
    "topic": "past_simple",
    "level": "A2",
    "prompt": "Yesterday I ___ to school.",
    "options": ["go", "went", "gone"],
    "answer": "went",
    "explanation": "'Yesterday' requires the past simple.",
})


def test_exercise_generation_hides_the_answer(client) -> None:
    client.fake_llm.responses = [EXERCISE_JSON]
    exercise = client.post("/api/exercise/generate", json={}).json()
    assert exercise["prompt"] == "Yesterday I ___ to school."
    assert exercise["options"] == ["go", "went", "gone"]
    assert "answer" not in exercise


def test_exercise_check_accepts_the_exact_answer(client, user_id: int) -> None:
    client.fake_llm.responses = [EXERCISE_JSON]
    exercise = client.post("/api/exercise/generate", json={}).json()

    result = client.post(
        "/api/exercise/check", json={"exercise_id": exercise["id"], "answer": "went"}
    ).json()
    assert result["is_correct"] is True
    assert result["explanation"] == "'Yesterday' requires the past simple."


def test_exercise_check_records_a_wrong_answer(client, user_id: int) -> None:
    client.fake_llm.responses = [
        EXERCISE_JSON,
        json.dumps({"is_correct": False, "feedback": "Not quite.", "corrected": "went",
                    "errors": [{"type": "verb_tense", "topic": "past_simple"}]}),
    ]
    exercise = client.post("/api/exercise/generate", json={}).json()
    result = client.post(
        "/api/exercise/check", json={"exercise_id": exercise["id"], "answer": "go"}
    ).json()

    assert result["is_correct"] is False
    assert errors.count(user_id, "english") == 1


def test_exercise_check_rejects_unknown_id(client) -> None:
    response = client.post("/api/exercise/check", json={"exercise_id": 4242, "answer": "x"})
    assert response.status_code == 503
    assert "lost track" in response.json()["detail"]["message"]


# -------------------------------------------------------------- vocabulaire
def test_vocabulary_due_and_review(client, user_id: int) -> None:
    client.fake_llm.responses = [TURN_JSON]
    client.post("/api/chat/turn", json={"text": "Yesterday I go to the cinema."})

    due = client.get("/api/vocabulary/due?language=english").json()
    assert [word["word"] for word in due["due"]] == ["screening"]

    client.post(
        "/api/vocabulary/review",
        json={"language": "english", "word": "screening", "remembered": True},
    )
    assert client.get("/api/vocabulary/due?language=english").json()["due"] == []


def test_vocabulary_teach_stores_words(client, user_id: int) -> None:
    client.fake_llm.responses = [json.dumps({"words": [
        {"word": "runway", "translation": "piste", "example": "The plane is on the runway."},
        {"word": "boarding pass", "translation": "carte d'embarquement", "example": "Show your boarding pass."},
    ]})]
    payload = client.post("/api/vocabulary/teach", json={"theme": "travel"}).json()
    assert len(payload["words"]) == 2
    assert vocabulary.count(user_id, "english") == 2


# ------------------------------------------------------------ prononciation
def test_pronunciation_check_scores_the_attempt(client, user_id: int) -> None:
    client.fake_stt.text = "I sink dis is fine"
    payload = client.post(
        "/api/pronunciation/check",
        files={"audio": ("t.webm", b"x" * 500, "audio/webm")},
        data={"expected": "I think this is fine", "language": "english"},
    ).json()
    assert payload["score"] < 100
    assert "the English TH sound" in payload["problem_sounds"]
    assert payload["transcription"]["text"] == "I sink dis is fine"


def test_pronunciation_perfect_reading_scores_full(client) -> None:
    client.fake_stt.text = "I think this is fine"
    payload = client.post(
        "/api/pronunciation/check",
        files={"audio": ("t.webm", b"x" * 500, "audio/webm")},
        data={"expected": "I think this is fine."},
    ).json()
    assert payload["score"] == 100.0
    assert payload["problem_sounds"] == []


# --------------------------------------------------------------- évaluation
def test_assessment_test_is_served_without_answers(client) -> None:
    payload = client.get("/api/assessment/german").json()
    assert len(payload["items"]) == 10
    assert all("answer" not in item for item in payload["items"])
    assert payload["production"]


def test_assessment_unknown_language_returns_404(client) -> None:
    assert client.get("/api/assessment/klingon").status_code == 404


def test_assessment_submission_sets_the_profile(client, user_id: int) -> None:
    from app.language.assessment import get_items

    client.fake_llm.responses = [json.dumps({
        "level": "B1",
        "scores": {"grammar": 60, "vocabulary": 58, "speaking": 55,
                   "listening": 57, "writing": 62, "pronunciation": 50},
        "strengths": ["clear structure"], "weaknesses": ["articles"],
        "summary": "Solid intermediate level.",
    })]

    answers = {item.id: item.answer for item in get_items("english") if item.level in ("A1", "A2", "B1")}
    result = client.post("/api/assessment", json={
        "language": "english", "answers": answers,
        "productions": {"en-prod-1": "I am a developer and I went to the cinema last weekend."},
    }).json()

    assert result["level"] in ("A2", "B1", "B2")
    assert result["llm_used"] is True
    assert client.get("/api/config").json()["onboarded"] is True


def test_assessment_works_without_the_llm(client, monkeypatch) -> None:
    from app.core.exceptions import LLMUnavailableError
    from app.language.assessment import get_items

    def _down(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise LLMUnavailableError("ollama is down")

    monkeypatch.setattr(client.fake_llm, "generate", _down)
    answers = {item.id: item.answer for item in get_items("english")}
    result = client.post("/api/assessment", json={
        "language": "english", "answers": answers, "productions": {"en-prod-1": "Some text."},
    }).json()

    assert result["llm_used"] is False
    assert result["objective"]["correct"] == 10


# ---------------------------------------------------------------- dashboard
def test_dashboard_covers_both_languages(client) -> None:
    payload = client.get("/api/dashboard").json()
    assert [entry["language"] for entry in payload["languages"]] == ["english", "german"]


def test_lesson_plan_endpoint(client) -> None:
    payload = client.get("/api/lesson?minutes=30").json()
    assert payload["total_minutes"] == 30
    assert payload["blocks"]


# ------------------------------------------------------------- robustesse
def test_llm_outage_is_reported_clearly(client, monkeypatch) -> None:
    from app.core.exceptions import LLMUnavailableError

    def _down(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise LLMUnavailableError("connection refused")

    monkeypatch.setattr(client.fake_llm, "generate", _down)
    response = client.post("/api/chat/turn", json={"text": "Hello"})
    assert response.status_code == 503
    assert "ollama" in response.json()["detail"]["message"].lower()


def test_tts_outage_still_returns_the_text(client, monkeypatch) -> None:
    from app.core.exceptions import TTSUnavailableError

    def _down(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise TTSUnavailableError("no voice installed")

    monkeypatch.setattr(client.fake_tts, "synthesize", _down)
    payload = client.post("/api/chat/turn", json={"text": "Hello"}).json()
    assert payload["response"]
    assert payload["speech"] is None


def test_user_message_is_kept_even_if_the_model_fails(client, user_id: int, monkeypatch) -> None:
    from app.core.exceptions import LLMError

    def _fail(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise LLMError("boom")

    monkeypatch.setattr(client.fake_llm, "generate", _fail)
    client.post("/api/chat/turn", json={"text": "Please remember this."})

    session = sessions.get_open(user_id, "english", "free_conversation")
    history = messages.history(int(session["id"]))
    assert [message["content"] for message in history] == ["Please remember this."]


def test_speak_endpoint_returns_audio(client) -> None:
    payload = client.post("/api/speak", json={"text": "Hello there", "speed": 0.8}).json()
    assert base64.b64decode(payload["audio_base64"]).startswith(b"RIFF")
