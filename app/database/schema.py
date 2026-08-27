"""Schéma SQLite de Liliana.

Le schéma est appliqué de façon idempotente au démarrage (``CREATE TABLE IF NOT
EXISTS``). ``SCHEMA_VERSION`` est stocké dans ``PRAGMA user_version`` afin de
permettre des migrations futures.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- ----------------------------------------------------------------- users
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL DEFAULT 'me',
    native_language TEXT    NOT NULL DEFAULT 'french',
    onboarded       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ------------------------------------------------------------- languages
-- Une ligne par (utilisateur, langue apprise) : niveau et scores par compétence.
CREATE TABLE IF NOT EXISTS languages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    language       TEXT    NOT NULL,
    level          TEXT    NOT NULL DEFAULT 'A1',
    grammar        REAL    NOT NULL DEFAULT 0,
    vocabulary     REAL    NOT NULL DEFAULT 0,
    speaking       REAL    NOT NULL DEFAULT 0,
    listening      REAL    NOT NULL DEFAULT 0,
    writing        REAL    NOT NULL DEFAULT 0,
    pronunciation  REAL    NOT NULL DEFAULT 0,
    assessed_at    TEXT,
    updated_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, language)
);

-- -------------------------------------------------------------- sessions
CREATE TABLE IF NOT EXISTS sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    language       TEXT    NOT NULL,
    mode           TEXT    NOT NULL DEFAULT 'free_conversation',
    started_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at       TEXT,
    duration_sec   INTEGER NOT NULL DEFAULT 0,
    message_count  INTEGER NOT NULL DEFAULT 0,
    error_count    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, started_at DESC);

-- -------------------------------------------------------------- messages
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT    NOT NULL,
    language    TEXT,
    is_voice    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

-- ------------------------------------------------------------ vocabulary
CREATE TABLE IF NOT EXISTS vocabulary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    language      TEXT    NOT NULL,
    word          TEXT    NOT NULL,
    translation   TEXT    NOT NULL DEFAULT '',
    example       TEXT    NOT NULL DEFAULT '',
    part_of_speech TEXT   NOT NULL DEFAULT '',
    difficulty    TEXT    NOT NULL DEFAULT 'A1',
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, language, word)
);

-- -------------------------------------------------------- grammar_topics
CREATE TABLE IF NOT EXISTS grammar_topics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    language    TEXT NOT NULL,
    topic       TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    level       TEXT NOT NULL DEFAULT 'A1',
    UNIQUE (language, topic)
);

-- ---------------------------------------------------------------- errors
CREATE TABLE IF NOT EXISTS errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id  INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    language    TEXT    NOT NULL,
    error_type  TEXT    NOT NULL,
    topic       TEXT    NOT NULL DEFAULT '',
    original    TEXT    NOT NULL DEFAULT '',
    corrected   TEXT    NOT NULL DEFAULT '',
    explanation TEXT    NOT NULL DEFAULT '',
    severity    TEXT    NOT NULL DEFAULT 'minor',
    is_voice    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_errors_user ON errors(user_id, language, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_errors_topic ON errors(user_id, language, topic);

-- ------------------------------------------------------------- exercises
CREATE TABLE IF NOT EXISTS exercises (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    language       TEXT    NOT NULL,
    exercise_type  TEXT    NOT NULL,
    topic          TEXT    NOT NULL DEFAULT '',
    level          TEXT    NOT NULL DEFAULT 'A1',
    prompt         TEXT    NOT NULL,
    options_json   TEXT    NOT NULL DEFAULT '[]',
    answer         TEXT    NOT NULL DEFAULT '',
    explanation    TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ------------------------------------------------------- exercise_results
CREATE TABLE IF NOT EXISTS exercise_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    exercise_id   INTEGER NOT NULL REFERENCES exercises(id) ON DELETE CASCADE,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id    INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    user_answer   TEXT    NOT NULL DEFAULT '',
    is_correct    INTEGER NOT NULL DEFAULT 0,
    feedback      TEXT    NOT NULL DEFAULT '',
    answered_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_results_user ON exercise_results(user_id, answered_at DESC);

-- -------------------------------------------------------------- progress
-- Un instantané par jour et par langue, pour tracer les courbes du dashboard.
CREATE TABLE IF NOT EXISTS progress (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    language        TEXT    NOT NULL,
    day             TEXT    NOT NULL,
    level           TEXT    NOT NULL DEFAULT 'A1',
    overall_score   REAL    NOT NULL DEFAULT 0,
    seconds_studied INTEGER NOT NULL DEFAULT 0,
    messages        INTEGER NOT NULL DEFAULT 0,
    errors          INTEGER NOT NULL DEFAULT 0,
    words_learned   INTEGER NOT NULL DEFAULT 0,
    exercises_done  INTEGER NOT NULL DEFAULT 0,
    exercises_ok    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (user_id, language, day)
);

-- -------------------------------------------------- pronunciation_attempts
CREATE TABLE IF NOT EXISTS pronunciation_attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    language     TEXT    NOT NULL,
    expected     TEXT    NOT NULL,
    heard        TEXT    NOT NULL DEFAULT '',
    score        REAL    NOT NULL DEFAULT 0,
    phoneme_tags TEXT    NOT NULL DEFAULT '[]',
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ------------------------------------------------------- review_schedule
-- Répétition espacée : s'applique au vocabulaire comme aux points de grammaire.
CREATE TABLE IF NOT EXISTS review_schedule (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    language      TEXT    NOT NULL,
    item_type     TEXT    NOT NULL CHECK (item_type IN ('vocabulary', 'grammar')),
    item_key      TEXT    NOT NULL,
    difficulty    REAL    NOT NULL DEFAULT 2.5,
    interval_days REAL    NOT NULL DEFAULT 0,
    repetitions   INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    confidence    REAL    NOT NULL DEFAULT 0,
    last_review   TEXT,
    next_review   TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, language, item_type, item_key)
);
CREATE INDEX IF NOT EXISTS idx_review_due
    ON review_schedule(user_id, language, next_review);

-- -------------------------------------------------------------- settings
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""
