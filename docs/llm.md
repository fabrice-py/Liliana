# The language model

## Abstraction

Everything goes through one interface, in `app/ai/llm.py`:

```python
class LLMProvider(ABC):
    def generate(self, messages, *, temperature=None, json_mode=False,
                 schema=None) -> str: ...
    def stream(self, messages, *, temperature=None, json_mode=False,
               schema=None) -> Iterator[str]: ...
    def status(self) -> LLMStatus: ...
```

The only implementation today is `OllamaProvider`, talking to
`http://127.0.0.1:11434`. Adding another backend means writing the class and
registering it:

```python
_PROVIDERS: dict[str, type[LLMProvider]] = {"ollama": OllamaProvider}
```

The model name is **never** hardcoded, and it does not have to be configured
either. `LLM_MODEL` starts empty; when it is empty, `resolve_model()` asks
Ollama what is installed and scores the candidates:

- models that are not for conversation are excluded outright — anything matching
  `embed`, `rerank`, `code`, `vision`, `llava`, `guard`, `math`;
- known-good multilingual families score higher (qwen, llama, mistral, gemma…);
- `instruct` / `chat` variants beat their base counterparts;
- size is scored against the machine's memory: bigger is better up to about a
  third of the budget, then penalised — a 70B on 8 GB technically runs, and is
  unusable.

The chosen model is logged, and `/api/health` reports `auto-selected` so the
interface can show which one is in use. It is a default, not a lock: setting
`LLM_MODEL` always wins. With nothing usable installed, the error names the
`ollama pull` command to run.

## The system prompt

`app/ai/prompts.py` builds the prompt in layers.

**1. Persona** (`BASE_SYSTEM_PROMPT`) — Liliana is a teacher whose goal is that
you *produce* the language, not that she answers questions. It asks for short
spoken answers (they will be read aloud) that end by handing the turn back.

**2. Current context**, injected per turn:

```
Target language: German (Deutsch)
User CEFR level in this language: B1
User's native language: french
Session mode: Immersion
```

**3. Mode instructions** — one paragraph per mode, from `CONVERSATION_MODES`.

**4. Correction level** — `off`, `minimal`, `normal` or `strict`, spelled out as
behaviour rather than as a label.

**5. What Liliana knows about you** — the part that makes her a tutor rather
than a chatbot:

```
## Known weaknesses (weave practice for these into the conversation)
- dative (7 recent occurrences)
- verb_position (4 recent occurrences)

## Recent mistakes made by this user
- "mit der Mann" -> "mit dem Mann" [dative]

## Vocabulary already introduced (reuse it, do not re-teach it)
- Bahnhof

## Items due for review (prefer these when you introduce content)
- Umlaut
```

**6. The error taxonomy for that language** — so a German turn can report
`separable_verbs` or `gender_der_die_das`, which do not exist for English.

### The order of these layers is a performance decision

Ollama caches the longest prefix it has already evaluated; everything after the
first character that differs must be computed again. On a CPU-only machine that
costs real time — prompt evaluation runs at roughly 35-45 tokens per second,
against 0.2 s for a prefix that is entirely cached.

Layer 5 — what Liliana knows about you — changes on every single turn. Layers 1,
3, 4, 6 and the output contract never change. So the volatile layer is emitted
**last**, after the output contract, and only the tail is re-evaluated: about
7 s per turn instead of the whole prompt.

The conversation history follows it, which is why layer 5 is kept short: it sits
in front of the history, and whatever precedes a change is all that stays cached.

The order also happens to be sound pedagogy — what the model reads last is what
it weighs most.

## Structured output

Liliana asks for a single JSON object per turn:

```json
{
  "response": "That sounds great! What did you do there?",
  "correction": {
    "original": "Yesterday I go to Paris",
    "corrected": "Yesterday I went to Paris",
    "explanation": "Use the past tense because the action happened yesterday."
  },
  "errors": [{"type": "verb_tense", "topic": "past_simple", "severity": "major"}],
  "vocabulary": [],
  "detected_language": "english",
  "difficulty": "A2",
  "suggested_level": "B1"
}
```

Only `response` is ever spoken. The rest feeds the memory.

### Why the contract is a schema, not a paragraph

Describing the format in prose is not enough. Asked in prose, a 3B model
understands the task, finds the right correction — and writes it into
`response`, leaving `correction` and `errors` empty. The answer sounds fine and
the learning engine gets nothing: no correction card, no error statistics,
nothing to schedule for review. Measured on this machine, six sentences with an
obvious mistake produced **zero** structured corrections.

So `TURN_RESPONSE_SCHEMA` (`app/ai/prompts.py`) is passed to Ollama as `format`.
It is the same contract, expressed as JSON Schema, and it constrains decoding
itself: the fields can no longer be missing. The same six sentences then produced
**six** structured corrections.

Two properties of that schema are load-bearing:

- **every field is required** — that is the whole point;
- **`response` comes first** — fields are generated in schema order, so
  `ResponseStreamParser` can start speaking before the correction is written.

It costs almost nothing: 21.1 s with the schema against 19.9 s with plain
`format: "json"` for the same turn. A server too old to accept an object
`format` (Ollama < 0.5) is detected on the first refusal, logged once, and the
provider falls back to `format: "json"` for the rest of its life.

The parser still assumes nothing: a schema constrains shape, never sense.

### One model per language

Once the schema carries the structure, the model only has to be right about the
language — and how much model that takes depends on which language. Measured on
this contract, same prompt, same schema, mistakes repeated twice each:

| | English | German | median per turn |
|---|---|---|---|
| `qwen2.5:1.5b-instruct` | 16/16 | ~50% | **7.5 s** |
| `qwen2.5:3b-instruct` | 16/16 | ~70-75% | 21.5 s |

In English the small model is the equal of the larger one and nearly three times
faster. In German it collapses: it misses the `sein`/`haben` auxiliary entirely,
and on `Ich mag der Kaffee nicht` it deletes the article instead of fixing the
case — worse than silence for someone learning declensions. Cases and genders are
the first thing a model loses as it shrinks.

Hence `LLM_MODEL_ENGLISH` / `LLM_MODEL_GERMAN` / `LLM_MODEL_FRENCH`, resolved by
`Settings.llm_model_for()` and passed per call — the same shape as the per-language
Piper voices. Empty falls back to `LLM_MODEL`, then to auto-selection.

Two things worth knowing:

- **German accuracy varies between runs** (8, 9 and 11 out of 12 across three
  identical series). Treat a single measurement as an estimate, not a verdict.
- **Both models stay resident**, so alternating languages costs no reload —
  provided Ollama is allowed to keep more than one model loaded, which is its
  default. Inside the application, with history and a filled learner profile, an
  English turn lands around 13 s rather than the 7.5 s of the isolated bench.

## Surviving bad JSON

`app/ai/structured.py` recovers from what local models actually produce:

| What the model does | What happens |
|---|---|
| Wraps it in ```` ```json ```` | Fence is stripped |
| Adds "Sure, here you go:" before | The object is located inside the text |
| Leaves a trailing comma | Removed |
| Uses typographic quotes `“ ”` | Normalised |
| Emits `{` inside a string value | Brace matching ignores string contents |
| Gets cut off mid-object | String and braces are closed, then parsed |
| Returns `"errors": ["past_simple"]` | Strings are promoted to error objects |
| Invents `"severity": "catastrophic"` | Falls back to `minor` |
| Returns an empty envelope `{}` | Refused: it is not free text. Stored as the assistant's turn, it comes back in the next prompt and the model imitates it — the session then answers `{}` forever |
| Ignores JSON entirely | **The raw text becomes the spoken answer** |

That last row matters most: a model having a bad day costs you the correction
for one turn, never the conversation.

`normalise_turn()` then guarantees every field exists with the right type, so
nothing downstream needs defensive checks. A "correction" identical to the
original is discarded — models like to produce those.

## Guarding the taxonomy

Models invent error types. `app/ai/tutor.py` checks each one against the
language's taxonomy and falls back to `grammar`:

```python
if error["type"] not in allowed:
    error["type"] = "grammar"
```

The specific `topic` is kept either way, because that is what drives spaced
repetition and the weakness ranking.

## History

The last `LLM_MAX_HISTORY_TURNS` exchanges (default 12, so 24 messages) are
replayed each turn. Bounded, because context length is the main cost driver on a
CPU, and a language conversation rarely needs to reach far back.

The user's message is written to the database **before** the model is called.
If Ollama dies mid-turn, what you said is still there.

## Other calls

The same provider serves exercise generation, answer checking, standalone
correction, grammar explanations, vocabulary generation and the assessment
grading — each with its own schema prompt and a temperature suited to the task
(0.1–0.2 for grading, 0.6–0.7 for generation).

## Streaming

`LLMProvider.stream(messages, json_mode=True)` drives the streamed turn. The
output contract deliberately puts `response` first, so `ResponseStreamParser`
can hand that field to the speech synthesiser while the model is still writing
the correction and the error list.

`Tutor.respond_stream()` yields three kinds of event — `delta` (text to show),
`sentence` (text to speak), `done` (the complete, persisted turn) — and shares
`_prepare()` and `_finalise()` with the non-streaming `respond()`, so the two
paths cannot drift apart. A test asserts they produce the identical turn.

Details and measurements: [`voice_pipeline.md`](voice_pipeline.md), section
*Streaming*.
