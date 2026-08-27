# Liliana

**A personal AI language tutor that runs entirely on your own computer.**

Liliana is not a chatbot that happens to speak English. She is a language
teacher: you talk to her out loud, she understands you, answers out loud,
notices the mistakes you actually make, remembers them, and builds the next
lessons around them.

She teaches **English 🇬🇧** and **German 🇩🇪**.

Everything runs locally: your voice, your conversations and your learning
profile never leave your machine.

---

## Table of contents

1. [What is Liliana?](#what-is-liliana)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Models](#models)
6. [Configuration](#configuration)
7. [Running Liliana](#running-liliana)
8. [Troubleshooting](#troubleshooting)
9. [Project structure](#project-structure)
10. [Development](#development)
11. [Testing](#testing)
12. [Future improvements](#future-improvements)

---

## What is Liliana?

You press the microphone button and speak. Liliana:

```
your microphone → silence detection → speech-to-text → linguistic analysis
      → local LLM → answer → pedagogical correction → speech synthesis → speaker
```

Each stage hands its output to the next as it is produced, so you hear the
first sentence while the rest is still being written.

She keeps a separate CEFR level (A1–C2) for each language, tracks six skills,
records every mistake by type, schedules what you got wrong for spaced
repetition, and adapts the difficulty as you improve.

A conversation looks like this:

> **You:** Yesterday I go to the cinema.
>
> **Liliana:** Oh nice! What did you watch?
>
> *(shown discreetly under the answer)*
> **Correction:** Yesterday I ~~go~~ **went** to the cinema.
> Use the past tense because the action happened yesterday. · `past_simple`

The conversation stays a conversation. The correction is recorded either way.

---

## Features

**Voice**
- Push-to-talk **or hands-free**: Liliana detects when you stop speaking
  (configurable silence threshold, default 0.8 s).
- Local speech-to-text with faster-whisper; the target language is preferred
  while you are practising it.
- Local speech synthesis with Piper, a different voice per language.
- **Streamed end to end**: your transcription appears as it decodes, Liliana's
  answer writes itself on screen, and she starts speaking after the first
  sentence instead of waiting for the whole turn. On a CPU-only laptop that
  takes time-to-first-word from ~2.9 s down to ~0.4 s.
- Adjustable speaking rate ("Liliana, speak more slowly").

**Teaching**
- Nine conversation modes: free conversation, just talk, English teacher,
  German teacher, immersion, grammar training, vocabulary training,
  pronunciation training, and "teach me" — where you name a grammar point and
  Liliana walks you through it: explanation, comprehension check, production,
  correction, exercise.
- Four correction levels: `off`, `minimal`, `normal`, `strict`.
- A full error taxonomy, including German-specific categories (cases,
  der/die/das, verb position, separable verbs, adjective endings, Umlaut…).
- Exercises generated on the topics you actually get wrong: multiple choice,
  fill in the blank, conjugation, sentence correction, transformation,
  translation, sentence building.
- **Pronunciation practice with real phonetic analysis**: Liliana gives you a
  sentence, converts both it and what she heard into IPA phonemes, aligns them,
  and names the exact sound to work on — the German Ö pronounced as an O, the
  TH turned into an S. She also uses the recogniser's per-word confidence, so
  a word that is technically right but mumbled still shows up.

**Memory and progress**
- A placement test on first launch, per language.
- A learning profile with six skills, recomputed from your real data — never
  hardcoded. Five of them are measured directly (grammar and vocabulary from
  your mistakes and drills, speaking and writing per channel, pronunciation from
  your recorded attempts); listening has no direct signal yet and is estimated
  from how much conversation you actually sustain.
- SM-2 spaced repetition for both vocabulary and grammar points.
- A dashboard: level, skills, study time, words learned, errors corrected,
  exercise success rate, weekly progression, what to work on next.
- Voice commands: *"Liliana, switch to German"*, *"correct me"*,
  *"give me an exercise"*, *"repeat that"*, *"speak more slowly"*…

**Privacy**
- No cloud account, no API key, no telemetry.
- Recordings are **never** written to disk unless you set `SAVE_AUDIO=true`.
- After the models are downloaded once, Liliana works with no Internet at all.

---

## Architecture

Liliana is a small local web application: a Python backend and a browser
front end talking to it over `http://127.0.0.1:8000`.

```
Browser (frontend/)                    Python backend (app/)
┌──────────────────────────┐           ┌──────────────────────────────┐
│ microphone capture       │  audio    │ /api/voice/turn              │
│ voice activity detection │ ────────► │   ├─ speech/stt.py  Whisper  │
│ conversation UI          │           │   ├─ ai/tutor.py    context  │
│ audio playback           │ ◄──────── │   ├─ ai/llm.py      Ollama   │
└──────────────────────────┘  answer   │   ├─ language/…     analysis │
                              + audio  │   └─ speech/tts.py  Piper    │
                                       └──────────────────────────────┘
                                                     │
                                              SQLite (data/liliana.db)
```

**Why the microphone lives in the browser.** The browser already owns the
microphone, handles the permission prompt, and gives us `MediaRecorder` and the
Web Audio API for free. Detecting silence on the client means no audio is sent
over the network until you have actually finished a sentence — lower latency,
and no native audio library to install (PyAudio and PortAudio are a recurring
source of pain on Windows).

Every engine sits behind an interface — `LLMProvider`, `STTProvider`,
`TTSProvider` — so swapping Ollama for something else, or Piper for another
synthesiser, is a change in one file plus configuration.

More detail: [`docs/architecture.md`](docs/architecture.md),
[`docs/voice_pipeline.md`](docs/voice_pipeline.md),
[`docs/learning_engine.md`](docs/learning_engine.md),
[`docs/llm.md`](docs/llm.md), [`docs/database.md`](docs/database.md).

---

## Installation

### 1. Python

Liliana needs **Python 3.10 or newer**.

- **Windows** — download it from [python.org](https://www.python.org/downloads/)
  and tick **"Add Python to PATH"** during installation.
- **macOS** — `brew install python@3.12`
- **Linux** — `sudo apt install python3 python3-venv python3-pip`

### 2. Liliana

**Windows** — double-click `install.bat`. It creates the virtual environment,
installs the dependencies, creates your `.env`, and downloads the voices.

**Linux / macOS**

```bash
git clone https://github.com/fabrice-py/Liliana.git
cd Liliana
./start_liliana --install
```

Or by hand, on any platform:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

### 3. The local language model (Ollama)

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Start it: `ollama serve` (on Windows it starts by itself after install).
3. Pull a model sized for your machine — see [Models](#models) below:

   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

**You do not need to edit `.env`.** Liliana picks the most suitable model among
the ones you have installed: it skips embedding, code and vision models, prefers
`instruct` variants, and avoids anything too large for your memory. The choice
is logged and shown in the interface, never silent. Set `LLM_MODEL` only if you
want to force a specific one.

### 4. Prepare everything in one command

```bash
python scripts/setup.py --pull-model
```

It creates your `.env`, downloads a language model if none is installed,
pre-downloads the speech model so your first recording is not slow, fetches the
Piper voices, and reports anything still missing. Safe to re-run.

To check without downloading anything:

```bash
python scripts/check_env.py     # or: run.bat --check  /  ./start_liliana --check
```

---

## Models

Liliana never hardcodes a model name. Run `python scripts/check_env.py` and it
recommends what fits your machine. As a rule of thumb:

| RAM (or VRAM with a GPU) | Language model | Whisper model |
|--------------------------|----------------|---------------|
| 16 GB or more            | a 7B–8B instruct model | `medium` |
| 8–16 GB                  | a 3B–4B instruct model | `small`  |
| 4–8 GB                   | a 1B–2B instruct model | `base`   |
| under 4 GB               | a sub-1B model, expect slow answers | `tiny` |

**Language model** — any instruct model served by Ollama works. Prefer one
that is good at multilingual output and at returning JSON:

```bash
ollama pull qwen2.5:3b-instruct     # good default on a laptop without a GPU
```

**Speech-to-text** — downloaded automatically on your first recording, into
`models/whisper/`. Set the size with `STT_MODEL` in `.env`.

**Speech synthesis and phonetics** — `piper-tts` (installed with the
requirements) provides two things: Liliana's voice, and the espeak-ng engine
that converts text to IPA phonemes for pronunciation analysis. The voices
themselves go in `models/piper/`:

```bash
python scripts/download_voices.py            # the voices named in your .env
python scripts/download_voices.py --list     # a few well-tested alternatives
```

The `piper` executable on your `PATH` works as an alternative to the Python
module. **Without either, Liliana still works** — she answers in text, and
pronunciation practice compares words instead of sounds.

No GPU is required anywhere. Everything runs on CPU, just more slowly.

**A model per language.** Liliana asks the model for a strict JSON contract —
answer, correction, error types — and constrains the decoding to it, so the model
no longer has to be clever about *format*, only about the *language*. How much
model that takes turns out to depend on which language: measured here, a
`qwen2.5:1.5b-instruct` corrects English exactly as well as a 3B and roughly three
times faster, but loses German cases and genders. If you alternate, say so:

```bash
LLM_MODEL_ENGLISH=qwen2.5:1.5b-instruct
LLM_MODEL_GERMAN=qwen2.5:3b-instruct
```

Ollama keeps both loaded, so switching language costs nothing. Leave them empty
and Liliana uses one model everywhere, as before.

---

## Configuration

Every setting lives in `.env` (start from `.env.example`); nothing is scattered
through the code. The ones you are most likely to touch:

| Variable | Default | What it does |
|---|---|---|
| `LLM_MODEL` | *(empty)* | Ollama model name. Leave empty to auto-select. |
| `LLM_MODEL_ENGLISH` / `_GERMAN` | *(empty)* | A model per language — see below. Empty falls back to `LLM_MODEL`. |
| `LLM_KEEP_ALIVE` | `30m` | How long Ollama keeps the model in memory between turns. |
| `STT_MODEL` | `base` | Whisper size: `tiny`…`large-v3` |
| `TTS_PROVIDER` | `piper` | `none` disables voice output entirely |
| `DEFAULT_LANGUAGE` | `english` | `english` or `german` |
| `CORRECTION_MODE` | `normal` | `off`, `minimal`, `normal`, `strict` |
| `VAD_SILENCE_THRESHOLD` | `0.8` | Seconds of silence that end your turn |
| `VAD_ENERGY_THRESHOLD` | `0.015` | Microphone sensitivity — lower it if you speak softly |
| `SAVE_AUDIO` | `false` | Keep recordings on disk. Off by default. |
| `PORT` | `8000` | Local port |

Language, mode and correction level can also be changed live in the interface —
that choice is stored in the database and survives a restart.

---

## Running Liliana

```bash
python run.py
```

Windows: double-click `run.bat`. Linux/macOS: `./start_liliana`.

Then open **<http://127.0.0.1:8000>** (`run.py` tries to open it for you).

Useful flags:

```bash
python run.py --port 8080     # another port
python run.py --no-browser    # do not open a browser
python run.py --reload        # auto-reload while developing
python run.py --check         # environment check, then exit
```

### First session

1. Pick your language in the top bar.
2. Optionally take the **placement test** — it takes two minutes and calibrates
   your level per language.
3. Press **Speak** (or the space bar) and talk.
4. With *Hands-free* on, just stop talking; Liliana takes it from there.

The tabs across the top hold the rest: **Exercises** (drills on the mistakes you
actually make), **Vocabulary** (spaced repetition), **Pronunciation** (read a
sentence aloud and see, sound by sound, what came out), and **Progress**.

The status chip in the corner shows `● LOCAL` when everything is running.
Click it for details on each engine.

---

## Troubleshooting

**"Liliana cannot access the microphone."**
Allow microphone access for the page in your browser, then check your operating
system's microphone privacy settings (on Windows: *Settings → Privacy & security
→ Microphone*). Browsers only allow microphone access on `localhost` or HTTPS —
use `http://127.0.0.1:8000`, not your machine's LAN address.

**"Liliana cannot reach Ollama."**
Ollama is not running. Start it with `ollama serve`, and check
<http://127.0.0.1:11434> answers. If it listens elsewhere, set `LLM_BASE_URL`.

**"The configured language model is not installed."**
Run `ollama list` to see what you have, `ollama pull <model>` to add one, then
set `LLM_MODEL` in `.env`.

**Liliana answers in text but never speaks.**
Piper or its voices are missing. Run `python scripts/download_voices.py`, and
install Piper with `pip install piper-tts`. This is not fatal — everything else
keeps working.

**She does not hear me / cuts me off too early.**
Lower `VAD_ENERGY_THRESHOLD` (try `0.008`) if she misses quiet speech, and raise
`VAD_SILENCE_THRESHOLD` (try `1.2`) if she cuts you off while you think.

**Answers are very slow.**
Use a smaller language model and a smaller `STT_MODEL`, and set
`STT_BEAM_SIZE=1`. On a CPU-only machine a 3B model is usually the sweet spot.

**The first recording takes ages.**
The Whisper model is being downloaded (once). Later recordings are fast.

**Anything else** — `python scripts/check_env.py` diagnoses most problems, and
`logs/liliana.log` has the details.

---

## Project structure

```
liliana/
├── run.py                    entry point — python run.py
├── install.bat / run.bat     Windows launchers
├── start_liliana             Linux & macOS launcher
│
├── app/
│   ├── main.py               FastAPI application
│   ├── core/                 config, logging, exceptions, hardware detection
│   ├── ai/                   LLM abstraction, prompts, tutor, JSON parsing
│   ├── speech/               speech-to-text, synthesis, VAD, audio helpers
│   ├── language/             correction, grammar, vocabulary, pronunciation,
│   │                         phonemes, placement test, voice commands,
│   │                         language data
│   ├── learning/             progress, spaced repetition, lesson planner
│   ├── database/             schema, connection, repositories
│   └── api/routes.py         HTTP endpoints
│
├── frontend/                 interface (HTML, CSS, one JS file)
│                             tabs: conversation, exercises, vocabulary,
│                             pronunciation, progress
├── models/                   Whisper and Piper models (not in Git)
├── data/                     SQLite database (not in Git)
├── logs/                     application log
├── scripts/                  setup.py, check_env.py, download_voices.py
├── tests/                    pytest suite
└── docs/                     architecture and design notes
```

---

## Development

```bash
python run.py --reload        # auto-reload on code changes
```

Interactive API documentation is served at <http://127.0.0.1:8000/docs>.

Ground rules for this codebase:

- No SQL outside `app/database/repositories.py`.
- No configuration values outside `app/core/config.py`.
- Every engine goes behind its provider interface.
- A missing model, a dead Ollama or invalid JSON must degrade gracefully and
  produce a message a human can act on — never a stack trace in the interface.

---

## Testing

```bash
pytest                 # the whole suite
pytest -v              # verbose
pytest tests/test_api.py
```

The suite runs against a temporary SQLite database and simulated engines, so
it needs neither Ollama, nor a Whisper model, nor a Piper voice. It covers the
database and repositories, malformed-JSON recovery, the correction pipeline,
progress computation, spaced repetition, voice commands, pronunciation scoring,
the placement test, session handling, exercises, the streaming pipeline, and the
failure paths (LLM down, TTS missing, silent recording, empty upload, a stream
cut off mid-answer).

Two of them are worth knowing about: one asserts that the first sentence is
ready for the synthesiser **before half the model output has arrived** — the
property the whole streaming design exists for — and another asserts that the
streamed and non-streamed paths produce an identical turn, so they cannot
silently drift apart.

---

## Future improvements

The architecture already accommodates these; they are deliberately not in the
current version:

- more languages (French, Spanish, Italian, Japanese) — add an entry to
  `app/language/languages.py` and a voice;
- a wake word, so you do not need the button at all;
- importing your own documents and books as learning material;
- job-interview practice, exam preparation, professional and travel modes.

---

## License

Personal project, provided as is.
