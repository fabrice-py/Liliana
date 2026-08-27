# The learning engine

What turns Liliana from a chatbot into a tutor: she remembers, she measures, and
she chooses what comes next.

## One turn, end to end

`app/ai/tutor.py` runs every turn the same way:

```
1. build_context()      level, mode, weaknesses, recent mistakes,
                        known vocabulary, items due for review
2. store the user's message         (before the model call — never lose it)
3. call the LLM                     with the assembled prompt + history
4. parse and normalise the JSON     app/ai/structured.py
5. store the assistant's message
6. persist the learning:
     · errors        → errors table
     · new words     → vocabulary table
     · error topics  → review_schedule (grammar)
     · new words     → review_schedule (vocabulary)
     · counters      → sessions, progress
```

Step 6 is what makes the next conversation different from this one.

## The linguistic profile

Six skills per language, on a 0–100 scale, plus a CEFR level. They are
**computed, never hardcoded** (`app/learning/progress.py`):

| Skill | Derived from |
|---|---|
| `grammar` | Grammar-type error rate per message, blended with exercise success |
| `vocabulary` | Breadth (words known) and solidity (mean spaced-repetition confidence) |
| `speaking` | Error rate on turns you spoke into the microphone |
| `writing` | Error rate on turns you typed |
| `listening` | Volume of sustained exchange (a proxy, honestly labelled) |
| `pronunciation` | Mean score of pronunciation attempts, or an estimate from speaking |

Three rules keep the numbers honest:

**Small samples do not produce confident numbers.** `_accuracy_score()` blends
the measured rate with a neutral baseline, weighted by how much data exists.
Below ten messages the profile is flagged `is_estimate` and the interface says
so.

**Each channel is measured on its own data.** Every mistake records whether
the turn was spoken or typed, so speaking and writing are computed from their
own turns. A skill with no turns yet sits at its neutral baseline rather than
borrowing a number from the other channel.

**A blank profile stays A1.** With no messages, no exercises and no placement
test, `compute_scores()` returns the stored profile untouched rather than
inventing a level from default constants.

**The level cannot collapse.** Scores are smoothed (`SMOOTHING = 0.25`) and the
level may drop at most one band at a time. One bad evening does not erase a
month.

## Placement test

`app/language/assessment.py`, run on first launch or on demand. Two parts:

1. **Ten multiple-choice questions**, two per band from A1 to C1, graded
   locally. Deterministic, instant, and **works with Ollama switched off**. The
   estimate is the highest band you pass at 50 % or better — a failed band stops
   the climb, so lucky guesses higher up do not inflate the result.
2. **Two free-production tasks**, graded by the model against the CEFR
   descriptors.

The two are merged, and the final level is reconciled against the resulting
scores so the badge and the bars cannot disagree by more than one band. If the
model is unavailable, part 1 alone produces the profile and `llm_used: false`
says so.

## Spaced repetition

`app/learning/spaced_repetition.py` implements SM-2, applied identically to
vocabulary and to grammar topics — `review_schedule` stores
`(item_type, item_key)`.

```
success (quality ≥ 3):   1 day → 3 days → interval × ease
failure (quality < 3):   back to 10 minutes, ease reduced, repetitions reset
ease:                    clamped to [1.3, 2.8]
```

`compute_next_review()` is a pure function — no clock, no database — which is
why it is exhaustively tested.

Items enter the schedule automatically: every mistake topic and every word
Liliana introduces is registered, due immediately. Exercises grade themselves
into it (5 for a correct answer, 2 for a wrong one), and the vocabulary panel
grades on *"I knew it"* / *"Forgot"*.

`confidence` combines accuracy with maturity, so a word answered right once does
not look as solid as one answered right five times.

## Weaknesses and adaptation

`ErrorRepository.top_weaknesses()` ranks error topics over the last 60 days.
That ranking is used in three places:

- **The prompt** — Liliana is told to weave practice for them into conversation.
- **Exercise generation** — `pick_topic()` prefers a due review item, then the
  top weakness.
- **The lesson planner** — the grammar block is labelled with the actual topic.

So making the same mistake repeatedly changes what Liliana talks about, what she
tests, and what the next session focuses on. That is the whole loop.

## Correction levels

Four levels (`app/ai/prompts.py`, filtered in `app/language/correction.py`):

| Level | Reported | In the spoken answer |
|---|---|---|
| `off` | nothing | nothing |
| `minimal` | major errors only | never mentioned |
| `normal` | major and minor, plus better phrasing | a light touch at most |
| `strict` | almost everything, including register | the main mistake, kindly |

Every mode has a sensible default — `just_talk` and `immersion` default to
`minimal`, drills to `strict` — and an explicit choice always wins.

Note the separation: what gets **recorded** and what gets **said** are different
decisions. In `just_talk` every mistake still lands in the database; the
conversation simply stays a conversation.

## Lesson planner

`app/learning/curriculum.py` builds a session of 10 to 90 minutes:

```
warm-up conversation · grammar · vocabulary · speaking · review
```

Proportions shift with level — beginners get more vocabulary, advanced learners
more speaking. The review block disappears when nothing is due. Minutes are
allocated by largest remainder so the blocks always sum exactly to the requested
duration, with a three-minute floor; a short session drops its lightest blocks
rather than slicing everything too thin.

## Pronunciation

You are given a sentence to read. Liliana transcribes what you actually said and
compares the two on three independent axes.

**Phonemes** (`app/language/phonemes.py`). Both sentences are converted to IPA
by espeak-ng — bundled inside `piper-tts`, so no system package and no network —
then aligned. This is the axis that carries the diagnosis:

```
expected  Ich möchte      →  ɪç mœçtə
heard     Ich mochte      →  ɪç mɔxtə
                              ─┬─ ─┬─
                               │   └── ç → x   the German CH
                               └────── œ → ɔ   the German Ö versus O
```

A spelling comparison sees one word off by an accent. The phoneme alignment sees
two distinct sounds missed, and can name both.

**Words.** A longest-common-subsequence alignment over the words, so a dropped
or swapped word is reported as such rather than smeared across the sentence.

**Acoustic confidence.** With `word_timestamps=True`, Whisper returns a
probability per word. A word recognised correctly but with low confidence was
mumbled — invisible to any text comparison. Requested only here, since the
extra decoding cost is not worth it on ordinary conversation turns.

The score is a weighted mean of whichever signals are actually available
(phonemes 0.50, words 0.30, clarity 0.20, renormalised), and `method` on the
result says which ones were used, so no number appears out of nowhere. Without
`piper-tts` the phoneme axis drops out and character similarity takes its place:
degraded, still useful, and honestly labelled in the interface.

Practice sentences come from a local bank, each saturated with one sound. The
choice is driven by the sounds you have actually missed before — the same
adaptive loop as the grammar drills — and rotates so you do not read the same
line twice.
