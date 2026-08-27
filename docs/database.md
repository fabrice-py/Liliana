# The database

SQLite, one file, `data/liliana.db`. Local, portable, and readable with any
SQLite browser if you want to inspect your own data.

## Access layer

Three modules, strictly separated:

| Module | Role |
|---|---|
| `app/database/schema.py` | The DDL, and `SCHEMA_VERSION` |
| `app/database/database.py` | Connections, pragmas, transactions |
| `app/database/repositories.py` | **The only place SQL is written** |

The API and the domain call repository methods. Nothing else touches the
database. That rule is what makes the storage swappable and the tests fast.

### Connections

One connection per thread (`threading.local`), because FastAPI serves requests
from a thread pool and a `sqlite3` connection is not shareable across threads.
Each connection sets:

```sql
PRAGMA foreign_keys = ON;     -- cascades actually cascade
PRAGMA journal_mode = WAL;    -- a read never blocks a write
PRAGMA synchronous = NORMAL;  -- durable enough for a local app, much faster
```

The schema is applied idempotently on first use (`CREATE TABLE IF NOT EXISTS`)
and the version recorded in `PRAGMA user_version`, leaving room for migrations.

## Tables

```
users ──┬── languages              level and six skill scores, per language
        ├── sessions ── messages   one conversation, its turns
        ├── errors                 every mistake, typed and dated
        ├── vocabulary             every word Liliana taught you
        ├── exercises ── exercise_results
        ├── progress               one row per day per language
        ├── pronunciation_attempts
        └── review_schedule        spaced repetition, vocabulary + grammar

grammar_topics                     grammar reference, per language
settings                           live preferences (language, mode, correction)
```

### `users`

One local user in practice (`name = 'me'`), but the schema supports several so
nothing has to change if it becomes a household application.

### `languages`

`UNIQUE (user_id, language)`. **English and German are tracked separately** —
being B2 in one says nothing about the other. Holds the CEFR level, the six
skill scores, and `assessed_at` (set by the placement test), which is how
`compute_scores()` distinguishes "no data yet" from "measured as low".

### `sessions` and `messages`

A session is one continuous conversation in one language and one mode. Reopened
automatically while it is still open, so closing the browser does not lose the
thread. Messages record `is_voice`, which lets the profile separate spoken from
written accuracy.

### `errors`

The heart of the tutoring. Each row stores `error_type` (from the language's
taxonomy), a specific `topic`, the original and corrected fragments, an
explanation, a severity, and `is_voice` — whether you spoke or typed that turn,
which is what lets the profile score speaking and writing independently.
Indexed on `(user_id, language, created_at)` and on `(user_id, language, topic)`
— the two ways it is read: recent history, and frequency ranking.

### `vocabulary`

`UNIQUE (user_id, language, word)`, so re-teaching a word is a no-op and the
insert reports which words were genuinely new — those are the ones scheduled for
review.

### `review_schedule`

SM-2 state, one row per item, with `item_type` in `('vocabulary', 'grammar')` so
one algorithm serves both. Indexed on `(user_id, language, next_review)`, which
is the only query that matters: what is due.

### `progress`

One row per day per language (`UNIQUE (user_id, language, day)`), accumulated
incrementally. Cheap to write, and it makes the dashboard's history a single
range scan instead of an aggregation over every message ever sent.

### `settings`

Key/value, for choices made in the interface — current language, mode,
correction level. Distinct from `.env`, which holds installation-level
configuration. This is why your choices survive a restart.

## Privacy

- The file never leaves your machine.
- **Audio is not stored**, unless you explicitly set `SAVE_AUDIO=true`.
- Transcriptions are stored, because that is what the tutoring is built on.
- Deleting `data/liliana.db` erases everything and starts you over.
- `data/` is git-ignored.

## Testing

Every test runs against a fresh temporary database created by the
`isolated_settings` fixture. Coverage includes schema creation, cascade
deletion, per-language independence, deduplication, chronological ordering and
daily accumulation.
