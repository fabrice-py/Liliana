"""Tests du streaming : parsing incrémental, découpage en phrases, endpoints SSE.

Le point que ces tests doivent réellement établir n'est pas seulement « le
contenu final est correct », mais « il arrive tôt » : c'est tout l'intérêt du
streaming.
"""

from __future__ import annotations

import json

import pytest

from app.ai.structured import ResponseStreamParser
from app.ai.tutor import Tutor
from app.database.repositories import errors, messages, sessions, vocabulary
from app.speech.tts import SentenceBuffer
from tests.conftest import FakeLLM

TURN_JSON = json.dumps(
    {
        "response": "That sounds great! What did you watch there? I love a good film.",
        "correction": {
            "original": "Yesterday I go to the cinema",
            "corrected": "Yesterday I went to the cinema",
            "explanation": "Past tense with 'yesterday'.",
        },
        "errors": [{"type": "verb_tense", "topic": "past_simple", "severity": "major"}],
        "vocabulary": [{"word": "screening", "translation": "projection"}],
        "detected_language": "english",
        "difficulty": "A2",
    }
)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Découpe un corps text/event-stream en (évènement, données)."""
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        name, payload = "", None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if name and payload is not None:
            events.append((name, payload))
    return events


def kinds(events: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in events]


# ------------------------------------------------- parsing JSON incrémental
@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (['{"resp', 'onse": "Hi ', 'there!"}'], "Hi there!"),
        (['{"response": "Line1\\', "nLine2\"}"], "Line1\nLine2"),
        (['{"response": "caf\\u0', '0e9"}'], "café"),
        (['```json\n{"response": "Fenced"}\n```'], "Fenced"),
        (['Sure!\n', '{"response": "After chatter"}'], "After chatter"),
        (['{"errors": [], "response": "not first"}'], "not first"),
        (['{"response": "truncated mid'], "truncated mid"),
    ],
)
def test_streaming_parser_recovers_the_response(chunks: list[str], expected: str) -> None:
    parser = ResponseStreamParser()
    emitted = "".join(parser.feed(chunk) for chunk in chunks) + parser.flush()
    assert emitted == expected
    assert parser.text == expected


def test_streaming_parser_falls_back_to_plain_text() -> None:
    parser = ResponseStreamParser()
    emitted = "".join(parser.feed(chunk) for chunk in ["I refuse ", "to produce JSON."])
    emitted += parser.flush()
    assert emitted == "I refuse to produce JSON."
    assert parser.is_plain_text is True


def test_streaming_parser_never_emits_the_same_text_twice() -> None:
    """Le texte est émis en deltas : leur concaténation ne doit rien dupliquer."""
    parser = ResponseStreamParser()
    deltas = [parser.feed(char) for char in '{"response": "abcdef"}']
    assert "".join(deltas) == "abcdef"


def test_streaming_parser_finishes_with_a_full_turn() -> None:
    parser = ResponseStreamParser()
    parser.feed(TURN_JSON)
    turn = parser.finish()
    assert turn["correction"]["corrected"] == "Yesterday I went to the cinema"
    assert turn["errors"][0]["topic"] == "past_simple"
    assert turn["structured"] is True


def test_streaming_parser_emits_before_the_json_is_closed() -> None:
    """La réponse doit être lisible AVANT la fin du JSON — sinon rien n'est gagné."""
    parser = ResponseStreamParser()
    early = parser.feed('{"response": "I can already be spoken')
    assert early == "I can already be spoken"


# --------------------------------------------------------- phrases pour TTS
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("That sounds great! What did you do?", ["That sounds great!", "What did you do?"]),
        ("Mr. Smith went to Paris. Then he left.", ["Mr. Smith went to Paris.", "Then he left."]),
        ("It costs 3.5 euros for one ticket. Nice!", ["It costs 3.5 euros for one ticket.", "Nice!"]),
        ('She said "hello there!" and smiled.', ['She said "hello there!"', "and smiled."]),
        ("No final punctuation here", ["No final punctuation here"]),
    ],
)
def test_sentence_buffer_splits_sensibly(text: str, expected: list[str]) -> None:
    buffer = SentenceBuffer()
    sentences = buffer.feed(text)
    if remainder := buffer.flush():
        sentences.append(remainder)
    assert sentences == expected


def test_sentence_buffer_works_character_by_character() -> None:
    buffer = SentenceBuffer()
    sentences: list[str] = []
    for char in "That sounds great! What did you do there?":
        sentences.extend(buffer.feed(char))
    if remainder := buffer.flush():
        sentences.append(remainder)
    assert sentences == ["That sounds great!", "What did you do there?"]


def test_sentence_buffer_loses_nothing() -> None:
    text = "One. Two! Three? And a tail without punctuation"
    buffer = SentenceBuffer()
    sentences = buffer.feed(text)
    sentences.append(buffer.flush())
    assert "".join(sentences).replace(" ", "") == text.replace(" ", "")


# ---------------------------------------------------- tuteur : incrémentalité
def test_first_sentence_is_ready_before_generation_ends(user_id: int) -> None:
    """Preuve du gain de latence : une phrase sort avant le dernier fragment."""
    session_id = int(sessions.create(user_id, "english", "free_conversation")["id"])
    llm = FakeLLM([TURN_JSON], chunk_size=5)

    consumed = 0
    original_stream = llm.stream

    def counting_stream(*args, **kwargs):
        nonlocal consumed
        for chunk in original_stream(*args, **kwargs):
            consumed += 1
            yield chunk

    llm.stream = counting_stream
    total_chunks = -(-len(TURN_JSON) // 5)

    first_sentence_at = None
    for event in Tutor(llm=llm).respond_stream(
        user_id=user_id, session_id=session_id, text="Yesterday I go to the cinema",
        language="english", mode="free_conversation",
    ):
        if event.kind == "sentence" and first_sentence_at is None:
            first_sentence_at = consumed

    assert first_sentence_at is not None
    assert first_sentence_at < total_chunks / 2, (
        f"première phrase après {first_sentence_at}/{total_chunks} fragments"
    )


def test_stream_and_non_stream_agree(user_id: int) -> None:
    """Les deux chemins doivent produire exactement le même tour."""
    streamed_session = int(sessions.create(user_id, "english", "free_conversation")["id"])
    direct_session = int(sessions.create(user_id, "german", "free_conversation")["id"])

    streamed = None
    for event in Tutor(llm=FakeLLM([TURN_JSON])).respond_stream(
        user_id=user_id, session_id=streamed_session, text="Yesterday I go",
        language="english", mode="free_conversation",
    ):
        if event.kind == "done":
            streamed = event.result

    direct = Tutor(llm=FakeLLM([TURN_JSON])).respond(
        user_id=user_id, session_id=direct_session, text="Yesterday I go",
        language="german", mode="free_conversation",
    )

    assert streamed is not None
    assert streamed.response == direct.response
    assert streamed.correction == direct.correction
    assert [e["topic"] for e in streamed.errors] == [e["topic"] for e in direct.errors]


def test_deltas_reconstruct_the_final_response(user_id: int) -> None:
    session_id = int(sessions.create(user_id, "english", "free_conversation")["id"])
    deltas, result = [], None
    for event in Tutor(llm=FakeLLM([TURN_JSON], chunk_size=3)).respond_stream(
        user_id=user_id, session_id=session_id, text="Hello",
        language="english", mode="free_conversation",
    ):
        if event.kind == "delta":
            deltas.append(event.text)
        elif event.kind == "done":
            result = event.result
    assert result is not None
    assert "".join(deltas) == result.response


def test_streaming_persists_the_same_way(user_id: int) -> None:
    session_id = int(sessions.create(user_id, "english", "free_conversation")["id"])
    for _ in Tutor(llm=FakeLLM([TURN_JSON])).respond_stream(
        user_id=user_id, session_id=session_id, text="Yesterday I go to the cinema",
        language="english", mode="free_conversation", is_voice=True,
    ):
        pass
    assert [m["role"] for m in messages.history(session_id)] == ["user", "assistant"]
    assert errors.count(user_id, "english", is_voice=True) == 1
    assert vocabulary.count(user_id, "english") == 1


# ------------------------------------------------------------ endpoints SSE
def test_chat_stream_event_sequence(client) -> None:
    client.fake_llm.responses = [TURN_JSON]
    response = client.post("/api/chat/turn/stream", json={"text": "Yesterday I go to the cinema"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_sse(response.text)
    names = kinds(events)
    assert "delta" in names
    assert "audio" in names
    assert names[-1] == "done"
    # La voix doit commencer avant la fin de la génération.
    assert names.index("audio") < len(names) - 1


def test_chat_stream_deltas_match_the_final_answer(client) -> None:
    client.fake_llm.responses = [TURN_JSON]
    events = parse_sse(
        client.post("/api/chat/turn/stream", json={"text": "Hello"}).text
    )
    deltas = "".join(data["text"] for name, data in events if name == "delta")
    done = next(data for name, data in events if name == "done")
    assert deltas == done["response"]
    assert done["correction"]["corrected"] == "Yesterday I went to the cinema"
    assert done["errors"][0]["topic"] == "past_simple"


def test_chat_stream_audio_chunks_carry_their_sentence(client) -> None:
    client.fake_llm.responses = [TURN_JSON]
    events = parse_sse(client.post("/api/chat/turn/stream", json={"text": "Hi"}).text)
    audio = [data for name, data in events if name == "audio"]

    assert len(audio) >= 2, "la réponse tient en plusieurs phrases : plusieurs chunks attendus"
    assert [chunk["index"] for chunk in audio] == list(range(len(audio)))
    assert all(chunk["text"] for chunk in audio)
    assert all(chunk["mime_type"] == "audio/wav" for chunk in audio)


def test_chat_stream_without_voice_emits_no_audio(client) -> None:
    client.fake_llm.responses = [TURN_JSON]
    events = parse_sse(
        client.post("/api/chat/turn/stream", json={"text": "Hi", "speak": False}).text
    )
    assert "audio" not in kinds(events)
    assert kinds(events)[-1] == "done"


def test_chat_stream_survives_plain_text_answers(client) -> None:
    client.fake_llm.responses = ["I am simply not going to produce any JSON today, sorry."]
    events = parse_sse(client.post("/api/chat/turn/stream", json={"text": "Hi"}).text)
    done = next(data for name, data in events if name == "done")
    assert done["response"].startswith("I am simply not going")
    assert done["structured"] is False


def test_chat_stream_reports_a_command(client) -> None:
    events = parse_sse(
        client.post("/api/chat/turn/stream", json={"text": "Liliana, switch to German."}).text
    )
    command = next(data for name, data in events if name == "command")
    assert command == {"action": "switch_language", "language": "german"}
    assert next(data for name, data in events if name == "done")["language"] == "german"


def test_chat_stream_reports_an_llm_outage(client, monkeypatch) -> None:
    from app.core.exceptions import LLMUnavailableError

    def _down(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise LLMUnavailableError("connection refused")
        yield  # pragma: no cover - fait de _down un générateur

    monkeypatch.setattr(client.fake_llm, "stream", _down)
    events = parse_sse(client.post("/api/chat/turn/stream", json={"text": "Hi"}).text)

    assert kinds(events) == ["error"]
    assert "ollama" in events[0][1]["message"].lower()


def test_chat_stream_reports_a_failure_after_partial_output(client, monkeypatch) -> None:
    """Une coupure en cours de route doit produire un `error`, pas un flux tronqué."""
    def _breaks(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        yield '{"response": "I started to ans'
        raise LLMError("connection dropped mid-stream")

    from app.core.exceptions import LLMError

    monkeypatch.setattr(client.fake_llm, "stream", _breaks)
    events = parse_sse(client.post("/api/chat/turn/stream", json={"text": "Hi"}).text)

    assert kinds(events)[0] == "delta"
    assert kinds(events)[-1] == "error"


def test_chat_stream_keeps_the_user_message_on_failure(client, user_id: int, monkeypatch) -> None:
    from app.core.exceptions import LLMError

    def _down(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise LLMError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr(client.fake_llm, "stream", _down)
    client.post("/api/chat/turn/stream", json={"text": "Please remember this."})

    session = sessions.get_open(user_id, "english", "free_conversation")
    assert [m["content"] for m in messages.history(int(session["id"]))] == [
        "Please remember this."
    ]


def test_tts_outage_does_not_break_the_stream(client, monkeypatch) -> None:
    from app.core.exceptions import TTSUnavailableError

    def _down(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise TTSUnavailableError("no voice installed")

    monkeypatch.setattr(client.fake_tts, "synthesize", _down)
    client.fake_llm.responses = [TURN_JSON]
    events = parse_sse(client.post("/api/chat/turn/stream", json={"text": "Hi"}).text)

    assert "audio" not in kinds(events)
    assert kinds(events)[-1] == "done"
    assert next(data for name, data in events if name == "done")["response"]


# ------------------------------------------------------------- voix en flux
def test_voice_stream_emits_transcription_then_answer(client) -> None:
    client.fake_llm.responses = [TURN_JSON]
    response = client.post(
        "/api/voice/turn/stream",
        files={"audio": ("turn.webm", b"fake-audio" * 100, "audio/webm")},
        data={"language": "english"},
    )
    events = parse_sse(response.text)
    names = kinds(events)

    assert names[0] == "transcription"
    assert names.index("transcription") < names.index("delta")
    assert names[-1] == "done"

    final = next(data for name, data in events if name == "transcription" and not data["partial"])
    assert final["text"] == "Yesterday I go to the cinema."


def test_voice_stream_reports_silence(client, monkeypatch) -> None:
    from app.core.exceptions import EmptyTranscriptionError

    def _silent(audio, language=None):  # noqa: ANN001
        raise EmptyTranscriptionError("silence")

    monkeypatch.setattr(client.fake_stt, "transcribe", _silent)
    events = parse_sse(
        client.post(
            "/api/voice/turn/stream",
            files={"audio": ("t.webm", b"x" * 500, "audio/webm")},
        ).text
    )
    assert kinds(events) == ["error"]
    assert "did not hear" in events[0][1]["message"].lower()


def test_voice_stream_rejects_an_empty_upload(client) -> None:
    response = client.post(
        "/api/voice/turn/stream", files={"audio": ("t.webm", b"", "audio/webm")}
    )
    assert response.status_code == 400


def test_voice_stream_emits_partial_transcriptions(client, monkeypatch) -> None:
    """Un backend capable de décoder par segments doit les faire remonter."""
    from app.speech.stt import Transcription, TranscriptionEvent

    def _staged(audio, language=None):  # noqa: ANN001
        yield TranscriptionEvent(text="Yesterday")
        yield TranscriptionEvent(text="Yesterday I go")
        yield TranscriptionEvent(
            text="Yesterday I go to the cinema.",
            is_final=True,
            transcription=Transcription(
                text="Yesterday I go to the cinema.", language="en",
                language_probability=0.99, duration=2.0, elapsed=0.1,
            ),
        )

    monkeypatch.setattr(client.fake_stt, "transcribe_stream", _staged)
    client.fake_llm.responses = [TURN_JSON]
    events = parse_sse(
        client.post(
            "/api/voice/turn/stream",
            files={"audio": ("t.webm", b"x" * 500, "audio/webm")},
        ).text
    )
    partials = [data["text"] for name, data in events if name == "transcription" and data["partial"]]
    assert partials == ["Yesterday", "Yesterday I go"]


def test_default_stt_provider_streams_a_single_final_event(client) -> None:
    """Le repli de STTProvider : un seul évènement, final, sans partiel."""
    events = list(client.fake_stt.transcribe_stream(b"audio", language="english"))
    assert len(events) == 1
    assert events[0].is_final is True
    assert events[0].transcription is not None
