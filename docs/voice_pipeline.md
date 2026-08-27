# The voice pipeline

```
microphone
   ↓  browser: getUserMedia
voice activity detection
   ↓  browser: Web Audio API, RMS energy
MediaRecorder → WebM/Opus blob
   ↓  POST /api/voice/turn
speech-to-text
   ↓  faster-whisper (PyAV decodes the blob directly)
transcription
   ↓
linguistic analysis + local LLM
   ↓  app/ai/tutor.py → app/ai/llm.py → Ollama
structured answer (JSON)
   ↓
pedagogical correction, stored in SQLite
   ↓
text-to-speech
   ↓  Piper → WAV, base64 in the same HTTP response
speaker
```

## 1. Capture and voice activity detection (browser)

The microphone is opened once with `getUserMedia`, with echo cancellation,
noise suppression and automatic gain enabled. Two things then run in parallel:

- a `MediaRecorder`, producing the compressed audio;
- an `AnalyserNode`, sampled on every animation frame to compute the RMS energy
  of the signal.

The detection rule, driven by three settings from `.env` and served to the page
by `/api/config`:

| Setting | Default | Role |
|---|---|---|
| `VAD_ENERGY_THRESHOLD` | `0.015` | Above this normalised RMS, you are speaking |
| `VAD_MIN_SPEECH_DURATION` | `0.3` s | Below this, it was a noise, not a turn |
| `VAD_SILENCE_THRESHOLD` | `0.8` s | Silence this long after real speech ends the turn |

In hands-free mode the recorder stops by itself when the rule fires. A 60-second
ceiling protects against a microphone left open in a noisy room. The same RMS
value drives the level meter, so the sensitivity setting is visible rather than
mysterious.

Detecting silence in the browser is a deliberate choice: nothing is uploaded
until a complete sentence exists, so there is no per-chunk network round trip.

## 2. Format and decoding

`MediaRecorder` produces WebM/Opus in Chrome, Edge and Firefox, and MP4/AAC in
Safari. `frontend/app.js` picks the first format the browser supports and sends
the blob as-is.

faster-whisper decodes it through **PyAV**, which bundles its own ffmpeg
libraries. This is why Liliana needs **no ffmpeg installed on the system** — a
significant simplification on Windows. (Verified: a browser-shaped WebM/Opus
blob decodes to 16 kHz mono PCM through this path.)

The server refuses an empty upload, anything larger than 25 MB, and any content
type that is not audio — before loading a model.

## 3. Speech-to-text

`app/speech/stt.py` wraps faster-whisper. Notable choices:

- **The model is loaded lazily and kept in memory.** Loading costs seconds;
  transcribing costs a fraction of one. The first recording of a session is
  therefore slower than the rest.
- **Device and precision are resolved automatically.** `auto` becomes
  `cuda`/`float16` when an NVIDIA GPU is present, `cpu`/`int8` otherwise.
  Nothing assumes a GPU.
- **The target language is forced while you practise it.** Whisper can guess,
  but on a short sentence spoken by a learner it guesses badly; the mode you are
  in is better information than the acoustics.
- **Whisper's own VAD filter is enabled** as a second line of defence, which
  also strips leading and trailing silence.
- `STT_BEAM_SIZE=1` (greedy) by default — the accuracy difference on
  conversational speech does not pay for the latency.

An empty transcription raises `EmptyTranscriptionError`, which the API turns
into a 422 and the interface into "Liliana did not hear anything".

## 4. Language analysis and the model

The transcription first goes through `detect_command()`
(`app/language/commands.py`). Short utterances matching a known pattern —
*"switch to German"*, *"correct me"*, *"speak more slowly"* — are applied
immediately, without spending a model call. Anything over twelve words is
treated as real speech, never as a command.

Otherwise `app/ai/tutor.py` assembles the context (level, mode, weaknesses,
recent mistakes, known vocabulary, items due for review), calls the LLM, and
persists what comes back. See [`llm.md`](llm.md) and
[`learning_engine.md`](learning_engine.md).

## 5. Speech synthesis

`app/speech/tts.py` drives Piper by whichever route is available:

1. the Python module `piper`, if installed — no subprocess, and the voice stays
   loaded between turns;
2. otherwise the `piper` executable, given a temporary WAV file.

The WAV is base64-encoded into the same JSON response as the text, so one HTTP
round trip carries the whole turn.

Speaking rate maps to Piper's `length_scale` inverted, so that `speed=0.8` means
"20 % slower" — which is what *"Liliana, speak more slowly"* sets.

**If synthesis fails for any reason, the turn still succeeds**: `speech` is
`null` and the interface shows the text. Losing the voice must never lose the
lesson.

## 6. Latency

Where the time goes on a CPU-only laptop, for a five-second sentence:

| Stage | Typical | How it is kept down |
|---|---|---|
| VAD | 0 | Runs during your speech |
| Upload | < 50 ms | Loopback, Opus-compressed |
| Whisper (`base`, int8) | 0.5–1.5 s | Model kept warm, greedy decoding |
| LLM (3B, CPU) | 1–4 s | Bounded history, short answers requested |
| Piper | 0.2–0.6 s | Voice kept loaded |

The interface says which phase it is in — *Listening… / Transcribing… /
Thinking…* — and reports the real timings after each turn, so slowness is
attributable rather than mysterious.

Streaming (partial transcription, token-by-token generation, sentence-by-
sentence synthesis) is the next lever; `LLMProvider.stream()` already exists for
it.
