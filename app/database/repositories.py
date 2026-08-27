"""Couche d'accès aux données.

Aucune requête SQL ne doit sortir de ce module : l'API et l'interface n'utilisent
que ces dépôts (cf. cahier des charges §19).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.database.database import get_connection, transaction
from app.language.languages import SKILLS

# ------------------------------------------------------------------ outils


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _today() -> str:
    return date.today().isoformat()


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def _row(cursor: sqlite3.Cursor) -> dict[str, Any] | None:
    row = cursor.fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------------- users
class UserRepository:
    """Un seul utilisateur local dans le MVP, mais le schéma en supporte N."""

    DEFAULT_NAME = "me"

    def get_or_create_default(self) -> dict[str, Any]:
        connection = get_connection()
        user = _row(
            connection.execute(
                "SELECT * FROM users WHERE name = ?", (self.DEFAULT_NAME,)
            )
        )
        if user:
            return user
        with transaction() as conn:
            conn.execute("INSERT INTO users (name) VALUES (?)", (self.DEFAULT_NAME,))
        return self.get_or_create_default()

    def get(self, user_id: int) -> dict[str, Any] | None:
        return _row(get_connection().execute("SELECT * FROM users WHERE id = ?", (user_id,)))

    def mark_onboarded(self, user_id: int) -> None:
        with transaction() as conn:
            conn.execute("UPDATE users SET onboarded = 1 WHERE id = ?", (user_id,))


# ------------------------------------------------------- profil par langue
class LanguageProfileRepository:
    """Niveau CECRL et scores par compétence, pour chaque langue apprise."""

    def get_or_create(self, user_id: int, language: str) -> dict[str, Any]:
        connection = get_connection()
        profile = _row(
            connection.execute(
                "SELECT * FROM languages WHERE user_id = ? AND language = ?",
                (user_id, language),
            )
        )
        if profile:
            return profile
        with transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO languages (user_id, language) VALUES (?, ?)",
                (user_id, language),
            )
        return self.get_or_create(user_id, language)

    def list_for_user(self, user_id: int) -> list[dict[str, Any]]:
        return _rows(
            get_connection().execute(
                "SELECT * FROM languages WHERE user_id = ? ORDER BY language",
                (user_id,),
            )
        )

    def update_scores(
        self,
        user_id: int,
        language: str,
        scores: dict[str, float],
        level: str | None = None,
        assessed: bool = False,
    ) -> dict[str, Any]:
        """Met à jour les compétences fournies. Les autres restent inchangées."""
        self.get_or_create(user_id, language)
        fields = {name: float(value) for name, value in scores.items() if name in SKILLS}
        assignments = [f"{name} = ?" for name in fields]
        values: list[Any] = list(fields.values())

        if level:
            assignments.append("level = ?")
            values.append(level)
        if assessed:
            assignments.append("assessed_at = ?")
            values.append(_now())
        assignments.append("updated_at = ?")
        values.append(_now())
        values.extend([user_id, language])

        with transaction() as conn:
            conn.execute(
                f"UPDATE languages SET {', '.join(assignments)} "
                "WHERE user_id = ? AND language = ?",
                values,
            )
        return self.get_or_create(user_id, language)


# ---------------------------------------------------------------- sessions
class SessionRepository:
    def create(self, user_id: int, language: str, mode: str) -> dict[str, Any]:
        with transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO sessions (user_id, language, mode) VALUES (?, ?, ?)",
                (user_id, language, mode),
            )
            session_id = int(cursor.lastrowid)
        return self.get(session_id) or {}

    def get(self, session_id: int) -> dict[str, Any] | None:
        return _row(
            get_connection().execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        )

    def get_open(self, user_id: int, language: str, mode: str) -> dict[str, Any] | None:
        return _row(
            get_connection().execute(
                "SELECT * FROM sessions "
                "WHERE user_id = ? AND language = ? AND mode = ? AND ended_at IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (user_id, language, mode),
            )
        )

    def get_or_create_open(self, user_id: int, language: str, mode: str) -> dict[str, Any]:
        return self.get_open(user_id, language, mode) or self.create(user_id, language, mode)

    def touch(self, session_id: int, *, messages: int = 0, errors: int = 0) -> None:
        with transaction() as conn:
            conn.execute(
                "UPDATE sessions SET message_count = message_count + ?, "
                "error_count = error_count + ? WHERE id = ?",
                (messages, errors, session_id),
            )

    def close(self, session_id: int) -> dict[str, Any] | None:
        session = self.get(session_id)
        if not session or session["ended_at"]:
            return session
        # SQLite écrit `datetime('now')` en UTC naïf : on compare à la même échelle.
        started = datetime.fromisoformat(session["started_at"])
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        duration = max(0, int((now - started).total_seconds()))
        with transaction() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, duration_sec = ? WHERE id = ?",
                (_now(), duration, session_id),
            )
        return self.get(session_id)

    def list_recent(self, user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        return _rows(
            get_connection().execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
        )


# ---------------------------------------------------------------- messages
class MessageRepository:
    def add(
        self,
        session_id: int,
        role: str,
        content: str,
        language: str | None = None,
        is_voice: bool = False,
    ) -> int:
        with transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, language, is_voice) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, language, int(is_voice)),
            )
            return int(cursor.lastrowid)

    def history(self, session_id: int, limit: int = 40) -> list[dict[str, Any]]:
        """Derniers messages de la session, dans l'ordre chronologique."""
        rows = _rows(
            get_connection().execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        )
        return list(reversed(rows))

    def count_for_user(
        self, user_id: int, language: str | None = None, is_voice: bool | None = None
    ) -> int:
        """Nombre de prises de parole de l'utilisateur, par canal si demandé."""
        sql = (
            "SELECT COUNT(*) AS n FROM messages m "
            "JOIN sessions s ON s.id = m.session_id "
            "WHERE s.user_id = ? AND m.role = 'user'"
        )
        params: list[Any] = [user_id]
        if language:
            sql += " AND s.language = ?"
            params.append(language)
        if is_voice is not None:
            sql += " AND m.is_voice = ?"
            params.append(int(is_voice))
        return int(_row(get_connection().execute(sql, params))["n"])


# ------------------------------------------------------------------ erreurs
class ErrorRepository:
    def add_many(
        self,
        user_id: int,
        session_id: int | None,
        language: str,
        errors: list[dict[str, Any]],
        is_voice: bool = False,
    ) -> int:
        if not errors:
            return 0
        payload = [
            (
                user_id,
                session_id,
                language,
                str(error.get("type") or "grammar"),
                str(error.get("topic") or ""),
                str(error.get("original") or ""),
                str(error.get("corrected") or ""),
                str(error.get("explanation") or ""),
                str(error.get("severity") or "minor"),
                int(is_voice),
            )
            for error in errors
        ]
        with transaction() as conn:
            conn.executemany(
                "INSERT INTO errors (user_id, session_id, language, error_type, topic, "
                "original, corrected, explanation, severity, is_voice) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                payload,
            )
        return len(payload)

    def recent(self, user_id: int, language: str, limit: int = 20) -> list[dict[str, Any]]:
        return _rows(
            get_connection().execute(
                "SELECT * FROM errors WHERE user_id = ? AND language = ? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, language, limit),
            )
        )

    def top_weaknesses(
        self, user_id: int, language: str, limit: int = 5, days: int = 60
    ) -> list[dict[str, Any]]:
        """Points faibles = types/thèmes d'erreurs les plus fréquents récemment."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        return _rows(
            get_connection().execute(
                "SELECT error_type, "
                "       CASE WHEN topic = '' THEN error_type ELSE topic END AS topic, "
                "       COUNT(*) AS occurrences, MAX(created_at) AS last_seen "
                "FROM errors "
                "WHERE user_id = ? AND language = ? AND created_at >= ? "
                "GROUP BY error_type, topic "
                "ORDER BY occurrences DESC, last_seen DESC LIMIT ?",
                (user_id, language, since, limit),
            )
        )

    def count(
        self, user_id: int, language: str | None = None, is_voice: bool | None = None
    ) -> int:
        """Nombre d'erreurs, éventuellement restreint à un canal (oral ou écrit)."""
        sql = "SELECT COUNT(*) AS n FROM errors WHERE user_id = ?"
        params: list[Any] = [user_id]
        if language:
            sql += " AND language = ?"
            params.append(language)
        if is_voice is not None:
            sql += " AND is_voice = ?"
            params.append(int(is_voice))
        return int(_row(get_connection().execute(sql, params))["n"])


# --------------------------------------------------------------- vocabulaire
class VocabularyRepository:
    def add_many(self, user_id: int, language: str, words: list[dict[str, Any]]) -> list[str]:
        """Insère les mots inconnus. Retourne la liste des mots réellement ajoutés."""
        added: list[str] = []
        if not words:
            return added
        with transaction() as conn:
            for entry in words:
                word = str(entry.get("word") or "").strip()
                if not word:
                    continue
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO vocabulary "
                    "(user_id, language, word, translation, example, part_of_speech, difficulty) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        language,
                        word,
                        str(entry.get("translation") or ""),
                        str(entry.get("example") or ""),
                        str(entry.get("part_of_speech") or ""),
                        str(entry.get("difficulty") or "A1"),
                    ),
                )
                if cursor.rowcount:
                    added.append(word)
        return added

    def list_for(self, user_id: int, language: str, limit: int = 100) -> list[dict[str, Any]]:
        return _rows(
            get_connection().execute(
                "SELECT * FROM vocabulary WHERE user_id = ? AND language = ? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, language, limit),
            )
        )

    def count(self, user_id: int, language: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM vocabulary WHERE user_id = ?"
        params: list[Any] = [user_id]
        if language:
            sql += " AND language = ?"
            params.append(language)
        return int(_row(get_connection().execute(sql, params))["n"])


# ------------------------------------------------------- répétition espacée
class ReviewRepository:
    def get(
        self, user_id: int, language: str, item_type: str, item_key: str
    ) -> dict[str, Any] | None:
        return _row(
            get_connection().execute(
                "SELECT * FROM review_schedule WHERE user_id = ? AND language = ? "
                "AND item_type = ? AND item_key = ?",
                (user_id, language, item_type, item_key),
            )
        )

    def upsert(
        self, user_id: int, language: str, item_type: str, item_key: str, **fields: Any
    ) -> dict[str, Any]:
        with transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO review_schedule "
                "(user_id, language, item_type, item_key) VALUES (?, ?, ?, ?)",
                (user_id, language, item_type, item_key),
            )
            if fields:
                assignments = ", ".join(f"{name} = ?" for name in fields)
                conn.execute(
                    f"UPDATE review_schedule SET {assignments} WHERE user_id = ? "
                    "AND language = ? AND item_type = ? AND item_key = ?",
                    [*fields.values(), user_id, language, item_type, item_key],
                )
        return self.get(user_id, language, item_type, item_key) or {}

    def due(
        self, user_id: int, language: str, limit: int = 20, item_type: str | None = None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM review_schedule WHERE user_id = ? AND language = ? "
            "AND next_review <= ?"
        )
        params: list[Any] = [user_id, language, _now()]
        if item_type:
            sql += " AND item_type = ?"
            params.append(item_type)
        sql += " ORDER BY next_review ASC, confidence ASC LIMIT ?"
        params.append(limit)
        return _rows(get_connection().execute(sql, params))

    def all_for(self, user_id: int, language: str) -> list[dict[str, Any]]:
        return _rows(
            get_connection().execute(
                "SELECT * FROM review_schedule WHERE user_id = ? AND language = ? "
                "ORDER BY next_review ASC",
                (user_id, language),
            )
        )


# --------------------------------------------------------------- exercices
class ExerciseRepository:
    def create(self, user_id: int, language: str, exercise: dict[str, Any]) -> dict[str, Any]:
        with transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO exercises (user_id, language, exercise_type, topic, level, "
                "prompt, options_json, answer, explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    language,
                    str(exercise.get("exercise_type") or "open"),
                    str(exercise.get("topic") or ""),
                    str(exercise.get("level") or "A1"),
                    str(exercise.get("prompt") or ""),
                    json.dumps(exercise.get("options") or [], ensure_ascii=False),
                    str(exercise.get("answer") or ""),
                    str(exercise.get("explanation") or ""),
                ),
            )
            exercise_id = int(cursor.lastrowid)
        return self.get(exercise_id) or {}

    def get(self, exercise_id: int) -> dict[str, Any] | None:
        row = _row(
            get_connection().execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,))
        )
        if row:
            row["options"] = json.loads(row.pop("options_json") or "[]")
        return row

    def record_result(
        self,
        exercise_id: int,
        user_id: int,
        session_id: int | None,
        user_answer: str,
        is_correct: bool,
        feedback: str = "",
    ) -> int:
        with transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO exercise_results "
                "(exercise_id, user_id, session_id, user_answer, is_correct, feedback) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (exercise_id, user_id, session_id, user_answer, int(is_correct), feedback),
            )
            return int(cursor.lastrowid)

    def stats(self, user_id: int, language: str | None = None) -> dict[str, int]:
        sql = (
            "SELECT COUNT(*) AS done, COALESCE(SUM(r.is_correct), 0) AS correct "
            "FROM exercise_results r JOIN exercises e ON e.id = r.exercise_id "
            "WHERE r.user_id = ?"
        )
        params: list[Any] = [user_id]
        if language:
            sql += " AND e.language = ?"
            params.append(language)
        row = _row(get_connection().execute(sql, params)) or {}
        return {"done": int(row.get("done", 0)), "correct": int(row.get("correct", 0))}


# --------------------------------------------------------------- progression
class ProgressRepository:
    def add(
        self,
        user_id: int,
        language: str,
        *,
        seconds: int = 0,
        messages: int = 0,
        errors: int = 0,
        words: int = 0,
        exercises: int = 0,
        exercises_ok: int = 0,
    ) -> None:
        """Incrémente l'instantané du jour."""
        with transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO progress (user_id, language, day) VALUES (?, ?, ?)",
                (user_id, language, _today()),
            )
            conn.execute(
                "UPDATE progress SET seconds_studied = seconds_studied + ?, "
                "messages = messages + ?, errors = errors + ?, "
                "words_learned = words_learned + ?, exercises_done = exercises_done + ?, "
                "exercises_ok = exercises_ok + ? "
                "WHERE user_id = ? AND language = ? AND day = ?",
                (
                    seconds, messages, errors, words, exercises, exercises_ok,
                    user_id, language, _today(),
                ),
            )

    def snapshot_level(self, user_id: int, language: str, level: str, score: float) -> None:
        with transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO progress (user_id, language, day) VALUES (?, ?, ?)",
                (user_id, language, _today()),
            )
            conn.execute(
                "UPDATE progress SET level = ?, overall_score = ? "
                "WHERE user_id = ? AND language = ? AND day = ?",
                (level, score, user_id, language, _today()),
            )

    def history(self, user_id: int, language: str, days: int = 30) -> list[dict[str, Any]]:
        since = (date.today() - timedelta(days=days)).isoformat()
        return _rows(
            get_connection().execute(
                "SELECT * FROM progress WHERE user_id = ? AND language = ? AND day >= ? "
                "ORDER BY day ASC",
                (user_id, language, since),
            )
        )

    def totals(self, user_id: int, language: str | None = None) -> dict[str, int]:
        sql = (
            "SELECT COALESCE(SUM(seconds_studied), 0) AS seconds, "
            "COALESCE(SUM(messages), 0) AS messages, "
            "COALESCE(SUM(errors), 0) AS errors, "
            "COALESCE(SUM(words_learned), 0) AS words, "
            "COALESCE(SUM(exercises_done), 0) AS exercises, "
            "COALESCE(SUM(exercises_ok), 0) AS exercises_ok "
            "FROM progress WHERE user_id = ?"
        )
        params: list[Any] = [user_id]
        if language:
            sql += " AND language = ?"
            params.append(language)
        return {key: int(value) for key, value in (_row(get_connection().execute(sql, params)) or {}).items()}


# ------------------------------------------------------------ prononciation
class PronunciationRepository:
    def add(
        self,
        user_id: int,
        language: str,
        expected: str,
        heard: str,
        score: float,
        phoneme_tags: list[str],
    ) -> int:
        with transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO pronunciation_attempts "
                "(user_id, language, expected, heard, score, phoneme_tags) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, language, expected, heard, score,
                 json.dumps(phoneme_tags, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def recent(self, user_id: int, language: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = _rows(
            get_connection().execute(
                "SELECT * FROM pronunciation_attempts WHERE user_id = ? AND language = ? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, language, limit),
            )
        )
        for row in rows:
            row["phoneme_tags"] = json.loads(row["phoneme_tags"] or "[]")
        return rows

    def average_score(self, user_id: int, language: str) -> float | None:
        row = _row(
            get_connection().execute(
                "SELECT AVG(score) AS avg_score FROM pronunciation_attempts "
                "WHERE user_id = ? AND language = ?",
                (user_id, language),
            )
        )
        value = (row or {}).get("avg_score")
        return float(value) if value is not None else None


# ---------------------------------------------------------------- réglages
class SettingsRepository:
    """Réglages modifiables à chaud depuis l'interface (mode de correction…)."""

    def get(self, key: str, default: str | None = None) -> str | None:
        row = _row(get_connection().execute("SELECT value FROM settings WHERE key = ?", (key,)))
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, value, _now()),
            )

    def all(self) -> dict[str, str]:
        return {
            row["key"]: row["value"]
            for row in _rows(get_connection().execute("SELECT key, value FROM settings"))
        }


# Instances partagées : les dépôts sont sans état.
users = UserRepository()
language_profiles = LanguageProfileRepository()
sessions = SessionRepository()
messages = MessageRepository()
errors = ErrorRepository()
vocabulary = VocabularyRepository()
reviews = ReviewRepository()
exercises = ExerciseRepository()
progress = ProgressRepository()
pronunciation = PronunciationRepository()
app_settings = SettingsRepository()
