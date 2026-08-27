# Architecture

## Shape of the application

Liliana is a local web application. A Python process serves both an HTTP API and
a static front end on `127.0.0.1:8000`; a browser page talks to it.

```
┌────────────────────────────── your computer ───────────────────────────────┐
│                                                                            │
│  Browser                            Python (uvicorn + FastAPI)             │
│  ┌────────────────────┐             ┌──────────────────────────────────┐   │
│  │ frontend/app.js    │  HTTP/JSON  │ app/api/routes.py                │   │
│  │  · microphone      │ ──────────► │   │                              │   │
│  │  · VAD             │             │   ├─► app/speech/stt.py ──► Whisper  │
│  │  · UI              │             │   ├─► app/ai/tutor.py            │   │
│  │  · audio playback  │ ◄────────── │   │     └─► app/ai/llm.py ──► Ollama │
│  └────────────────────┘   answer    │   ├─► app/language/… analysis    │   │
│                          + audio    │   └─► app/speech/tts.py ──► Piper │   │
│                                     └──────────────────────────────────┘   │
│                                                    │                       │
│                                       SQLite  data/liliana.db              │
└────────────────────────────────────────────────────────────────────────────┘
```

Nothing leaves the machine. There is no cloud component, no account and no
telemetry.

## Why a browser front end

The obvious alternative was a desktop application capturing audio with
`sounddevice` or `PyAudio`. The browser won on three points:

1. **Installation.** PortAudio bindings are a recurring source of build failures
   on Windows. The browser needs nothing.
2. **Permissions.** The microphone prompt, device selection and the "which app
   is using your mic" indicator are handled by the operating system through the
   browser, with UI the user already recognises.
3. **Latency.** Silence detection runs where the audio already is. No audio
   crosses a socket until a full sentence is ready.

The trade-off — a browser must be open — is acceptable for an application whose
interface is a web page anyway.

## Layers

| Layer | Package | Responsibility |
|---|---|---|
| Transport | `app/api` | HTTP endpoints, request validation, error mapping |
| Orchestration | `app/ai/tutor.py` | One conversational turn, start to finish |
| Domain | `app/language`, `app/learning` | Correction, exercises, vocabulary, pronunciation, progress, spaced repetition |
| Engines | `app/ai/llm.py`, `app/speech/*` | LLM, speech-to-text, synthesis, behind interfaces |
| Persistence | `app/database` | Schema, connection, repositories |
| Cross-cutting | `app/core` | Configuration, logging, exceptions, hardware detection |

Two rules keep the layers honest:

- **No SQL outside `app/database/repositories.py`.** The API and the domain call
  repository methods, never `execute`.
- **No configuration outside `app/core/config.py`.** Anything tunable is a
  `Settings` field, read from the environment or `.env`.

## Swappable engines

Three abstract interfaces isolate the machine-learning parts:

```python
class LLMProvider:   # app/ai/llm.py
    def generate(self, messages, *, temperature=None, json_mode=False) -> str: ...
    def stream(self, messages, *, temperature=None) -> Iterator[str]: ...
    def status(self) -> LLMStatus: ...

class STTProvider:   # app/speech/stt.py
    def transcribe(self, audio: bytes, language: str | None) -> Transcription: ...
    def is_available(self) -> tuple[bool, str]: ...

class TTSProvider:   # app/speech/tts.py
    def synthesize(self, text: str, language: str, speed: float | None) -> Speech: ...
    def is_available(self) -> tuple[bool, str]: ...
```

Adding a backend means writing one class and registering it in the `_PROVIDERS`
dictionary of the module. Nothing else in the application changes.

## Failure model

Liliana is used by one person on one machine, where models get uninstalled,
Ollama gets stopped and microphones get unplugged. Every failure mode therefore
has a defined behaviour:

| Failure | Behaviour |
|---|---|
| Ollama not running | HTTP 503 with instructions to start it |
| Model not pulled | HTTP 503 naming the `ollama pull` command |
| Whisper model missing | Explains it downloads once, suggests a smaller size |
| Piper or voice missing | Answer is returned as **text**; the conversation continues |
| espeak-ng missing | Pronunciation scored on spelling instead of sounds, and says so |
| `LLM_MODEL` not set | The best installed Ollama model is chosen and reported |
| Model returns broken JSON | Raw text is used as the spoken answer, no correction |
| Stream breaks mid-answer | An `error` event closes it; the browser retries without streaming |
| Silent recording | HTTP 422, "Liliana did not hear anything" |
| Microphone denied | Explains browser *and* OS permissions |
| Empty or oversized upload | HTTP 400 before any model is loaded |

Every business exception in `app/core/exceptions.py` carries two messages: a
technical one for the log and a `user_message` written for the person using the
application. The API returns only the second.

## Dependencies

Each one earns its place:

| Package | Why | Could we drop it? |
|---|---|---|
| `fastapi` | HTTP routing and input validation | Not without rewriting both |
| `uvicorn` | ASGI server | Needed to serve FastAPI |
| `python-multipart` | Receives the audio upload | Required by FastAPI for file uploads |
| `pydantic-settings` | Typed `.env` loading | Yes, at the cost of hand-written parsing |
| `httpx` | HTTP client for Ollama, with streaming | `urllib` could do it, without streaming |
| `faster-whisper` | Local speech-to-text; bundles PyAV, so **no system ffmpeg** | It *is* the feature |
| `piper-tts` | Liliana's voice **and** the espeak-ng phonemizer behind the pronunciation analysis | Yes — voice off, pronunciation falls back to comparing words |
| `pytest` | Test suite | Development only |

The database uses the standard library's `sqlite3` rather than an ORM: the
schema is small, the queries are simple, and it removes a large dependency.

## Extending

- **A new language** — add a `Language` entry in `app/language/languages.py`
  (name, Whisper code, espeak voice, error taxonomy, sounds to watch), a voice
  in `.env`, and optionally a placement bank in `app/language/assessment.py`
  plus practice sentences in `app/language/pronunciation.py`.
- **A new conversation mode** — add a `ConversationMode` to
  `CONVERSATION_MODES` in `app/ai/prompts.py`. The UI picks it up from
  `/api/config` automatically.
- **A new exercise type** — add it to `EXERCISE_TYPES` in
  `app/language/grammar.py`; it becomes selectable in the interface.
