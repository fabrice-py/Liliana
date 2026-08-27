# The language model

## Abstraction

Everything goes through one interface, in `app/ai/llm.py`:

```python
class LLMProvider(ABC):
    def generate(self, messages, *, temperature=None, json_mode=False) -> str: ...
    def stream(self, messages, *, temperature=None) -> Iterator[str]: ...
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

Ollama's `format: "json"` is requested, which constrains decoding — but small
local models still get it wrong regularly, so the parser assumes nothing.

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
