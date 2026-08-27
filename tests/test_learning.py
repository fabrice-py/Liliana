"""Tests de la répétition espacée, de la progression et du programme de séance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.database.repositories import errors, language_profiles, messages, reviews, sessions
from app.learning.curriculum import build_lesson
from app.learning.progress import progress_tracker
from app.learning.spaced_repetition import (
    DEFAULT_EASE,
    FIRST_INTERVALS,
    MIN_EASE,
    compute_next_review,
    spaced_repetition,
)


# ---------------------------------------------------------- répétition espacée
def test_first_success_uses_the_first_interval() -> None:
    outcome = compute_next_review(
        quality=5, repetitions=0, interval_days=0, ease=DEFAULT_EASE,
        success_count=0, failure_count=0,
    )
    assert outcome.interval_days == FIRST_INTERVALS[0]
    assert outcome.repetitions == 1


def test_intervals_grow_with_repeated_success() -> None:
    intervals = []
    repetitions, interval, ease = 0, 0.0, DEFAULT_EASE
    for _ in range(5):
        outcome = compute_next_review(
            quality=5, repetitions=repetitions, interval_days=interval, ease=ease,
            success_count=repetitions, failure_count=0,
        )
        repetitions, interval, ease = outcome.repetitions, outcome.interval_days, outcome.ease
        intervals.append(interval)
    assert intervals == sorted(intervals)
    assert intervals[-1] > intervals[0]


def test_failure_resets_repetitions_and_shortens_interval() -> None:
    outcome = compute_next_review(
        quality=1, repetitions=6, interval_days=40, ease=2.5,
        success_count=6, failure_count=0,
    )
    assert outcome.repetitions == 0
    assert outcome.interval_days < 1
    assert outcome.ease < 2.5


def test_ease_never_falls_below_the_floor() -> None:
    ease = DEFAULT_EASE
    for _ in range(20):
        ease = compute_next_review(
            quality=0, repetitions=0, interval_days=1, ease=ease,
            success_count=0, failure_count=1,
        ).ease
    assert ease >= MIN_EASE


def test_confidence_grows_with_successful_reviews() -> None:
    low = compute_next_review(
        quality=5, repetitions=1, interval_days=1, ease=2.5, success_count=1, failure_count=0
    ).confidence
    high = compute_next_review(
        quality=5, repetitions=8, interval_days=30, ease=2.5, success_count=8, failure_count=0
    ).confidence
    assert high > low


def test_review_persists_and_reschedules(user_id: int) -> None:
    spaced_repetition.register(user_id, "english", "vocabulary", "sleeve")
    assert spaced_repetition.due_keys(user_id, "english") == ["sleeve"]

    spaced_repetition.review(user_id, "english", "vocabulary", "sleeve", quality=5)
    assert spaced_repetition.due_keys(user_id, "english") == []

    item = reviews.get(user_id, "english", "vocabulary", "sleeve")
    assert item["success_count"] == 1
    assert item["repetitions"] == 1


def test_review_records_when_it_happened_not_when_it_is_due(user_id: int) -> None:
    """``last_review`` est la date de la révision, jamais celle de la suivante."""
    spaced_repetition.register(user_id, "english", "vocabulary", "sleeve")
    spaced_repetition.review(user_id, "english", "vocabulary", "sleeve", quality=5)

    item = reviews.get(user_id, "english", "vocabulary", "sleeve")
    assert item["last_review"] < item["next_review"]
    assert item["last_review"] <= datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def test_due_items_are_returned_when_the_date_has_passed(user_id: int) -> None:
    past = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    reviews.upsert(user_id, "german", "grammar", "dative", next_review=past)
    assert "dative" in spaced_repetition.due_keys(user_id, "german")


def test_register_many_is_idempotent(user_id: int) -> None:
    assert spaced_repetition.register_many(user_id, "english", "grammar", ["articles"]) == 1
    assert spaced_repetition.register_many(user_id, "english", "grammar", ["articles"]) == 0


# ------------------------------------------------------------------ progression
def test_scores_are_computed_not_hardcoded(user_id: int) -> None:
    session = sessions.create(user_id, "english", "free_conversation")
    for index in range(20):
        messages.add(int(session["id"]), "user", f"sentence {index}")

    clean = progress_tracker.compute_scores(user_id, "english")

    errors.add_many(
        user_id, int(session["id"]), "english",
        [{"type": "grammar", "topic": "past_simple"}] * 15,
    )
    noisy = progress_tracker.compute_scores(user_id, "english")

    assert noisy.scores["grammar"] < clean.scores["grammar"]


def test_new_profile_is_flagged_as_an_estimate(user_id: int) -> None:
    assert progress_tracker.compute_scores(user_id, "english").is_estimate is True


def test_level_never_drops_more_than_one_band(user_id: int) -> None:
    language_profiles.update_scores(user_id, "english", {skill: 95.0 for skill in
                                                         ("grammar", "vocabulary", "speaking",
                                                          "listening", "writing", "pronunciation")},
                                    level="C2")
    session = sessions.create(user_id, "english", "free_conversation")
    for index in range(30):
        messages.add(int(session["id"]), "user", f"sentence {index}")
    errors.add_many(
        user_id, int(session["id"]), "english",
        [{"type": "grammar", "topic": "articles"}] * 60,
    )
    assert progress_tracker.compute_scores(user_id, "english").level == "C1"


def test_refresh_persists_the_profile(user_id: int) -> None:
    profile = progress_tracker.refresh(user_id, "german")
    stored = language_profiles.get_or_create(user_id, "german")
    assert stored["level"] == profile["level"]
    assert "overall_score" in profile


def test_dashboard_exposes_every_section(user_id: int) -> None:
    data = progress_tracker.dashboard(user_id, "english")
    assert set(data) >= {
        "language", "level", "skills", "totals", "weaknesses", "history", "reviews_due"
    }
    assert set(data["skills"]) == {
        "grammar", "vocabulary", "speaking", "listening", "writing", "pronunciation"
    }


# -------------------------------------------------------------------- séance
@pytest.mark.parametrize("minutes", [10, 30, 60])
def test_lesson_respects_the_requested_duration(user_id: int, minutes: int) -> None:
    plan = build_lesson(user_id, "english", "B1", minutes=minutes)
    assert plan["total_minutes"] == minutes
    assert plan["blocks"]


def test_lesson_targets_the_actual_weaknesses(user_id: int) -> None:
    session = sessions.create(user_id, "english", "free_conversation")
    errors.add_many(
        user_id, int(session["id"]), "english",
        [{"type": "grammar", "topic": "prepositions"}] * 4,
    )
    plan = build_lesson(user_id, "english", "B1", minutes=30)
    grammar_block = next(block for block in plan["blocks"] if block["label"] == "Grammar")
    assert grammar_block["focus"] == "prepositions"


def test_lesson_drops_the_review_block_when_nothing_is_due(user_id: int) -> None:
    plan = build_lesson(user_id, "english", "B1", minutes=30)
    assert all(block["label"] != "Review" for block in plan["blocks"])

    spaced_repetition.register(user_id, "english", "vocabulary", "sleeve")
    plan = build_lesson(user_id, "english", "B1", minutes=30)
    assert any(block["label"] == "Review" for block in plan["blocks"])


def test_blank_profile_stays_at_a1(user_id: int) -> None:
    """Sans aucune donnée, Liliana n'invente pas un niveau (§14)."""
    snapshot = progress_tracker.compute_scores(user_id, "english")
    assert snapshot.level == "A1"
    assert snapshot.overall == 0.0
    assert snapshot.is_estimate is True


def test_assessed_profile_is_kept_even_without_conversation(user_id: int) -> None:
    language_profiles.update_scores(
        user_id, "german",
        {skill: 55.0 for skill in
         ("grammar", "vocabulary", "speaking", "listening", "writing", "pronunciation")},
        level="B1", assessed=True,
    )
    snapshot = progress_tracker.compute_scores(user_id, "german")
    assert snapshot.level in ("B1", "B2")


def test_speaking_and_writing_are_measured_per_channel(user_id: int) -> None:
    """Parler beaucoup et se tromper ne doit pas dégrader le score d'écrit."""
    session = int(sessions.create(user_id, "english", "free_conversation")["id"])
    for index in range(20):
        messages.add(session, "user", f"spoken {index}", "english", is_voice=True)
    for index in range(20):
        messages.add(session, "user", f"written {index}", "english", is_voice=False)

    errors.add_many(user_id, session, "english",
                    [{"type": "grammar", "topic": "articles"}] * 15, is_voice=True)

    scores = progress_tracker.compute_scores(user_id, "english").scores
    assert scores["writing"] > scores["speaking"]


def test_errors_are_attributed_to_the_right_channel(user_id: int) -> None:
    session = int(sessions.create(user_id, "english", "free_conversation")["id"])
    errors.add_many(user_id, session, "english",
                    [{"type": "grammar", "topic": "articles"}] * 3, is_voice=True)
    errors.add_many(user_id, session, "english",
                    [{"type": "grammar", "topic": "articles"}], is_voice=False)

    assert errors.count(user_id, "english", is_voice=True) == 3
    assert errors.count(user_id, "english", is_voice=False) == 1
    assert errors.count(user_id, "english") == 4
