"""Tests de la couche base de données et des dépôts."""

from __future__ import annotations

import pytest

from app.database.database import get_connection, init_database
from app.database.repositories import (
    errors,
    exercises,
    language_profiles,
    messages,
    progress,
    sessions,
    users,
    vocabulary,
)
from app.database.schema import SCHEMA_VERSION

EXPECTED_TABLES = {
    "users", "languages", "sessions", "messages", "vocabulary", "grammar_topics",
    "errors", "exercises", "exercise_results", "progress", "pronunciation_attempts",
    "review_schedule", "settings",
}


def test_schema_creates_every_table() -> None:
    init_database()
    rows = get_connection().execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    assert EXPECTED_TABLES <= {row["name"] for row in rows}


def test_schema_version_is_recorded() -> None:
    init_database()
    version = get_connection().execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION


def test_default_user_is_created_once() -> None:
    first = users.get_or_create_default()
    second = users.get_or_create_default()
    assert first["id"] == second["id"]


def test_language_profile_defaults_to_a1(user_id: int) -> None:
    profile = language_profiles.get_or_create(user_id, "german")
    assert profile["level"] == "A1"
    assert profile["grammar"] == 0


def test_update_scores_leaves_untouched_skills(user_id: int) -> None:
    language_profiles.update_scores(user_id, "english", {"grammar": 70.0, "writing": 60.0})
    language_profiles.update_scores(user_id, "english", {"grammar": 75.0}, level="B2")
    profile = language_profiles.get_or_create(user_id, "english")
    assert profile["grammar"] == 75.0
    assert profile["writing"] == 60.0
    assert profile["level"] == "B2"


def test_session_reuse_and_close(user_id: int) -> None:
    first = sessions.get_or_create_open(user_id, "english", "free_conversation")
    second = sessions.get_or_create_open(user_id, "english", "free_conversation")
    assert first["id"] == second["id"]

    sessions.close(int(first["id"]))
    third = sessions.get_or_create_open(user_id, "english", "free_conversation")
    assert third["id"] != first["id"]


def test_messages_are_returned_in_chronological_order(user_id: int) -> None:
    session = sessions.create(user_id, "english", "free_conversation")
    for index in range(5):
        messages.add(int(session["id"]), "user", f"message {index}")
    history = messages.history(int(session["id"]), limit=3)
    assert [item["content"] for item in history] == ["message 2", "message 3", "message 4"]


def test_top_weaknesses_ranks_by_frequency(user_id: int) -> None:
    session = sessions.create(user_id, "english", "free_conversation")
    errors.add_many(
        user_id,
        int(session["id"]),
        "english",
        [{"type": "grammar", "topic": "past_simple"}] * 3
        + [{"type": "grammar", "topic": "articles"}],
    )
    weaknesses = errors.top_weaknesses(user_id, "english")
    assert weaknesses[0]["topic"] == "past_simple"
    assert weaknesses[0]["occurrences"] == 3


def test_vocabulary_is_deduplicated(user_id: int) -> None:
    first = vocabulary.add_many(user_id, "english", [{"word": "sleeve"}, {"word": "cuff"}])
    second = vocabulary.add_many(user_id, "english", [{"word": "sleeve"}, {"word": "hem"}])
    assert first == ["sleeve", "cuff"]
    assert second == ["hem"]
    assert vocabulary.count(user_id, "english") == 3


def test_exercise_results_feed_statistics(user_id: int) -> None:
    exercise = exercises.create(
        user_id, "english", {"exercise_type": "multiple_choice", "prompt": "q", "answer": "a"}
    )
    exercises.record_result(int(exercise["id"]), user_id, None, "a", True)
    exercises.record_result(int(exercise["id"]), user_id, None, "b", False)
    assert exercises.stats(user_id, "english") == {"done": 2, "correct": 1}


def test_progress_accumulates_per_day(user_id: int) -> None:
    progress.add(user_id, "english", seconds=30, messages=1, errors=2)
    progress.add(user_id, "english", seconds=15, messages=1, words=3)
    totals = progress.totals(user_id, "english")
    assert totals["seconds"] == 45
    assert totals["messages"] == 2
    assert totals["errors"] == 2
    assert totals["words"] == 3


def test_foreign_keys_cascade(user_id: int) -> None:
    session = sessions.create(user_id, "english", "free_conversation")
    messages.add(int(session["id"]), "user", "hello")
    connection = get_connection()
    connection.execute("DELETE FROM sessions WHERE id = ?", (session["id"],))
    connection.commit()
    remaining = connection.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session["id"],)
    ).fetchone()["n"]
    assert remaining == 0


@pytest.mark.parametrize("language", ["english", "german"])
def test_profiles_are_independent_per_language(user_id: int, language: str) -> None:
    language_profiles.update_scores(user_id, language, {"grammar": 42.0}, level="B1")
    other = "german" if language == "english" else "english"
    assert language_profiles.get_or_create(user_id, other)["level"] == "A1"
