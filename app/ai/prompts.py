"""Prompts système de Liliana et construction dynamique du contexte.

Le prompt de base décrit la personnalité et la mission (cf. §22). Il est ensuite
enrichi à chaque tour avec : niveau, langue cible, mode, erreurs connues,
points faibles, vocabulaire récent et mode de correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.language.languages import CEFR_LEVELS, get_language

# --------------------------------------------------------------- persona
BASE_SYSTEM_PROMPT = """\
You are Liliana, a personal AI language teacher.

You teach English and German.

Your main goal is not simply to answer the user's questions.
Your goal is to actively improve the user's level in the target language.

You must:
- encourage the user;
- correct their mistakes;
- explain rules briefly;
- adapt the difficulty to their level;
- make them practise;
- reuse their past mistakes;
- introduce new vocabulary progressively;
- keep the conversation natural;
- detect gaps;
- avoid over-correcting when it hurts the conversation.

You clearly distinguish:
1. the conversational answer;
2. the correction;
3. the explanation;
4. the exercise;
5. the assessment.

When the user speaks the target language, answer in that language.
Do not translate automatically unless it is necessary or requested.
Do not give long grammar lectures.
Maximise the time during which the user actually produces the language.

Keep spoken answers short: 1 to 3 sentences, because they will be read aloud
by a speech synthesiser. Always end with a question or a prompt that invites
the user to speak again.\
"""


# ----------------------------------------------------------------- modes
@dataclass(frozen=True, slots=True)
class ConversationMode:
    key: str
    label: str
    description: str
    instructions: str
    default_correction_mode: str = "normal"
    forces_target_language: bool = True


CONVERSATION_MODES: dict[str, ConversationMode] = {
    "free_conversation": ConversationMode(
        key="free_conversation",
        label="Free conversation",
        description="Talk about anything. Liliana keeps the conversation going.",
        instructions=(
            "Have a natural, friendly conversation. Follow the user's topics, "
            "share small opinions and details about yourself so the exchange feels "
            "real, and always hand the turn back with a question."
        ),
    ),
    "just_talk": ConversationMode(
        key="just_talk",
        label="Just talk",
        description="No lesson at all — Liliana still records your mistakes silently.",
        instructions=(
            "No teaching, no exercises, no meta-comments about the language. "
            "Just be a warm, curious conversation partner. Corrections are still "
            "detected and stored, but never mentioned in your spoken answer."
        ),
        default_correction_mode="minimal",
    ),
    "english_teacher": ConversationMode(
        key="english_teacher",
        label="English teacher",
        description="Structured English practice with corrections and vocabulary.",
        instructions=(
            "Act as an English teacher. Make the user speak more than you do. "
            "Ask follow-up questions, offer one useful word or expression per turn, "
            "reformulate the user's sentence in a more natural way, and slowly "
            "raise the difficulty as they succeed."
        ),
    ),
    "german_teacher": ConversationMode(
        key="german_teacher",
        label="German teacher",
        description="Structured German practice with corrections and vocabulary.",
        instructions=(
            "Act as a German teacher. Make the user speak more than you do. "
            "Pay particular attention to gender (der/die/das), cases, verb position "
            "and adjective endings. Offer one useful word or expression per turn."
        ),
    ),
    "immersion": ConversationMode(
        key="immersion",
        label="Immersion",
        description="Target language only. No translation unless you ask.",
        instructions=(
            "Speak ONLY the target language, whatever language the user uses. "
            "Never translate unless the user explicitly asks. If the user does not "
            "understand, rephrase more simply in the target language."
        ),
        default_correction_mode="minimal",
    ),
    "grammar_training": ConversationMode(
        key="grammar_training",
        label="Grammar training",
        description="Grammar drills adapted to your weaknesses.",
        instructions=(
            "Drive a grammar drill. Give one short exercise per turn (multiple "
            "choice, fill in the blank, conjugation, sentence transformation, "
            "translation or sentence building), targeting the user's weak topics. "
            "Check their answer, explain briefly, then give the next item."
        ),
        default_correction_mode="strict",
    ),
    "vocabulary_training": ConversationMode(
        key="vocabulary_training",
        label="Vocabulary training",
        description="Spaced-repetition vocabulary with contextual examples.",
        instructions=(
            "Drive a vocabulary drill. Prefer the review items listed in the "
            "context. Always teach words inside a sentence or a mini-dialogue, "
            "never as a bare list, and ask the user to reuse the word."
        ),
    ),
    "pronunciation_training": ConversationMode(
        key="pronunciation_training",
        label="Pronunciation training",
        description="Repeat sentences; Liliana compares what she hears.",
        instructions=(
            "Give the user one short sentence to read aloud, chosen to exercise a "
            "specific sound. Comment on what you heard, then give the next sentence."
        ),
        default_correction_mode="strict",
    ),
    "teach_me": ConversationMode(
        key="teach_me",
        label="Teach me",
        description="Ask Liliana to teach a specific grammar point.",
        instructions=(
            "The user asked to be taught a specific point. Follow this order, one "
            "step per turn: (1) explain it briefly with 2-3 examples, (2) check "
            "understanding with a question, (3) make the user produce sentences, "
            "(4) correct them, (5) give one exercise, (6) confirm the result."
        ),
        default_correction_mode="strict",
    ),
}

DEFAULT_MODE = "free_conversation"


def get_mode(key: str) -> ConversationMode:
    return CONVERSATION_MODES.get((key or "").lower(), CONVERSATION_MODES[DEFAULT_MODE])


# ------------------------------------------------------- mode de correction
CORRECTION_INSTRUCTIONS: dict[str, str] = {
    "off": (
        "CORRECTION MODE: OFF. Do not correct anything. Leave `correction` null "
        "and `errors` empty."
    ),
    "minimal": (
        "CORRECTION MODE: MINIMAL. Only report mistakes that break comprehension "
        "or are clearly serious. Ignore small stylistic issues. Never mention the "
        "correction inside your spoken answer."
    ),
    "normal": (
        "CORRECTION MODE: NORMAL. Report real mistakes and, when useful, suggest a "
        "more natural phrasing. Keep your spoken answer conversational: at most a "
        "short, light touch on the mistake, never a grammar lecture."
    ),
    "strict": (
        "CORRECTION MODE: STRICT. Report almost every mistake, including word "
        "choice, register and naturalness. You may address the main mistake "
        "explicitly in your spoken answer, but stay encouraging and brief."
    ),
}


def correction_instruction(mode: str) -> str:
    return CORRECTION_INSTRUCTIONS.get((mode or "normal").lower(), CORRECTION_INSTRUCTIONS["normal"])


# -------------------------------------------------------- contexte pédago
@dataclass(slots=True)
class TutorContext:
    """Tout ce que Liliana sait de l'utilisateur au moment du tour de parole."""

    language: str = "english"
    level: str = "A1"
    mode: str = DEFAULT_MODE
    correction_mode: str = "normal"
    native_language: str = "french"
    weaknesses: list[dict[str, Any]] = field(default_factory=list)
    recent_errors: list[dict[str, Any]] = field(default_factory=list)
    recent_vocabulary: list[str] = field(default_factory=list)
    review_items: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    session_summary: str = ""


def _bullet_list(items: list[str], empty: str = "none yet") -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {empty}"


def build_system_prompt(context: TutorContext, output_format: str = "") -> str:
    """Assemble le prompt système complet pour un tour de conversation.

    L'ordre des sections n'est pas cosmétique. Ollama met en cache le plus long
    préfixe déjà évalué : tout ce qui suit le premier caractère qui change doit
    être recalculé. Or le profil de l'apprenant — points faibles, erreurs
    récentes, mots à réviser — change à chaque tour, tandis que la consigne de
    base et le contrat de sortie ne bougent jamais.

    Les parties stables viennent donc en tête, le profil en queue. Dans l'autre
    sens, le contrat de sortie (plusieurs centaines de tokens) était ré-évalué à
    chaque tour : une dizaine de secondes perdues sans GPU, à chaque phrase. Cet
    ordre a de surcroît un mérite pédagogique — ce que le modèle lit en dernier
    est ce dont il tient le plus compte.
    """
    language = get_language(context.language)
    mode = get_mode(context.mode)

    weaknesses = [
        f"{item.get('topic') or item.get('error_type')} "
        f"({item.get('occurrences', 0)} recent occurrences)"
        for item in context.weaknesses
    ]
    recent_errors = [
        f"\"{item.get('original', '')}\" -> \"{item.get('corrected', '')}\" "
        f"[{item.get('topic') or item.get('error_type')}]"
        for item in context.recent_errors[:6]
    ]

    # --- invariant d'un tour à l'autre : reste dans le cache du modèle --------
    sections = [
        BASE_SYSTEM_PROMPT,
        "",
        "## Current context",
        f"Target language: {language.english_name} ({language.native_name})",
        f"User's native language: {context.native_language}",
        f"Session mode: {mode.label}",
        "",
        "## Mode instructions",
        mode.instructions,
        "",
        "## " + correction_instruction(context.correction_mode).split(".")[0],
        correction_instruction(context.correction_mode),
        "",
        "## Error types you may use in the `errors` field",
        ", ".join(language.error_types),
    ]
    if output_format:
        sections += ["", "## Output format", output_format]

    # --- change à chaque tour : seule cette queue est ré-évaluée --------------
    sections += [
        "",
        "## This learner, right now",
        f"User CEFR level in this language: {context.level}",
        "",
        "## Known weaknesses (weave practice for these into the conversation)",
        _bullet_list(weaknesses, "no weakness identified yet"),
        "",
        "## Recent mistakes made by this user",
        _bullet_list(recent_errors, "no mistake recorded yet"),
        "",
        "## Vocabulary already introduced (reuse it, do not re-teach it)",
        _bullet_list(context.recent_vocabulary[:20], "none yet"),
        "",
        "## Items due for review (prefer these when you introduce content)",
        _bullet_list(context.review_items[:10], "none due"),
    ]

    if context.objectives:
        sections += ["", "## Learning objectives", _bullet_list(context.objectives)]
    if context.session_summary:
        sections += ["", "## What happened earlier in this session", context.session_summary]

    return "\n".join(sections)


# -------------------------------------------------- format de sortie JSON
RESPONSE_SCHEMA_PROMPT = """\
Answer with a single JSON object and nothing else. No markdown, no code fence.

{
  "response": "your spoken answer to the user, in the target language",
  "correction": {
    "original": "the user's sentence exactly as they said it",
    "corrected": "the corrected sentence",
    "explanation": "one or two short sentences explaining why"
  } | null,
  "errors": [
    {"type": "<error type>", "topic": "<specific grammar/vocabulary topic>",
     "original": "<the wrong fragment>", "corrected": "<the fixed fragment>",
     "severity": "minor" | "major"}
  ],
  "vocabulary": [
    {"word": "<word you introduced>", "translation": "<translation>",
     "example": "<short example sentence>", "part_of_speech": "<noun|verb|...>",
     "difficulty": "<A1..C2>"}
  ],
  "detected_language": "english" | "german" | "french" | "other",
  "difficulty": "<the CEFR level of the user's sentence, A1..C2>",
  "suggested_level": "<your best estimate of the user's level, A1..C2>"
}

Rules:
- `response` is the ONLY field that will be spoken aloud. Never put JSON,
  markdown or meta-commentary in it.
- If the user made no mistake, set `correction` to null and `errors` to [].
- `errors` only lists genuine mistakes, never stylistic preferences unless the
  correction mode is STRICT.
- `vocabulary` only lists words YOU deliberately introduce as new; leave it
  empty otherwise.
- Every field must be present.\
"""


def _object(**properties: Any) -> dict[str, Any]:
    """Objet JSON Schema dont tous les champs sont obligatoires.

    Les rendre obligatoires est le but même de la manœuvre : c'est ce qui empêche
    le modèle de « oublier » la correction et de la glisser dans sa réponse
    parlée, où plus rien ne peut l'exploiter.
    """
    return {"type": "object", "properties": properties, "required": list(properties)}


_STRING: dict[str, Any] = {"type": "string"}
_LEVEL: dict[str, Any] = {"type": "string", "enum": list(CEFR_LEVELS)}

#: Contrat de sortie d'un tour, en JSON Schema — le pendant exécutable de
#: ``RESPONSE_SCHEMA_PROMPT``.
#:
#: Décrire le format en prose ne suffit pas. Un modèle de 3B comprend la
#: consigne, trouve la bonne correction… et l'écrit dans ``response`` en laissant
#: ``correction`` et ``errors`` vides : la carte Correction, les statistiques
#: d'erreurs et la répétition espacée n'ont alors plus rien à exploiter. Transmis
#: à Ollama via ``format``, ce schéma contraint le décodage lui-même — les champs
#: ne peuvent plus manquer.
#:
#: ``response`` vient en tête, et cet ordre est structurant : les champs sont
#: générés dans l'ordre du schéma, ce qui laisse ``ResponseStreamParser``
#: commencer à parler avant que la correction ne soit écrite.
TURN_RESPONSE_SCHEMA: dict[str, Any] = _object(
    response=_STRING,
    correction=_object(original=_STRING, corrected=_STRING, explanation=_STRING),
    errors={
        "type": "array",
        "items": _object(
            type=_STRING,
            topic=_STRING,
            original=_STRING,
            corrected=_STRING,
            severity={"type": "string", "enum": ["minor", "major"]},
        ),
    },
    vocabulary={
        "type": "array",
        "items": _object(
            word=_STRING, translation=_STRING, example=_STRING, difficulty=_LEVEL
        ),
    },
    detected_language={
        "type": "string",
        "enum": ["english", "german", "french", "other"],
    },
    difficulty=_LEVEL,
    suggested_level=_LEVEL,
)


# ------------------------------------------------- tour en deux temps
#
# Un tour coûtait autrefois un seul appel produisant tout à la fois : la phrase
# parlée, la correction, les erreurs, le vocabulaire, le niveau. Mesuré sur une
# machine sans GPU, cet appel demandait 180 tokens et 52 secondes — alors que la
# phrase réellement prononcée en pesait 14, soit 3 secondes.
#
# L'analyse pédagogique est donc séparée de la conversation. Liliana répond
# d'abord, puis la correction arrive quelques secondes plus tard sous sa
# réponse. Rien n'est perdu : la correction est enregistrée dans tous les cas.

REPLY_INSTRUCTIONS = """React to WHAT THE LEARNER SAID — the content, not the grammar. Show interest,
add something of your own, then ask them a question that makes them speak again.

Never simply repeat their sentence back to them, corrected or otherwise. That
teaches nothing and ends the conversation. Their mistakes are handled elsewhere
and shown to them separately; your job here is to keep them talking.

    They say:  "Yesterday I go to the cinema with my friend."
    Bad:       "Yesterday I went to the cinema with my friend."
    Good:      "Oh nice! What did you watch? Was it any good?"

Steer towards what they need to practise: choose questions that make them
produce those forms again, without ever announcing that you are doing so.

One or two short sentences. Plain text — no JSON, no markdown, no quotation
marks around your answer, no meta-commentary. It will be read aloud."""


def build_reply_prompt(context: TutorContext) -> str:
    """Prompt du premier temps : la phrase que Liliana prononce.

    Elle garde ici ce qui fait d'elle un professeur — le niveau de l'apprenant
    et les points qu'il doit travailler — parce que c'est précisément ce qui doit
    orienter la conversation. Une réponse qui ignore les faiblesses de
    l'apprenant n'enseigne rien.

    Ce qu'elle laisse à l'analyse, en revanche, c'est le détail qui change à
    chaque phrase : la liste littérale des dernières erreurs et des derniers mots
    introduits. Cela ne rend pas la réponse meilleure, et cela suffisait à
    changer le prompt à chaque tour — donc à interdire à Ollama de le relire
    dans son cache. Les points faibles, eux, évoluent en jours, pas en phrases :
    ils restent, débarrassés de leurs compteurs.
    """
    language = get_language(context.language)
    mode = get_mode(context.mode)

    topics = [
        str(item.get("topic") or item.get("error_type") or "").strip()
        for item in context.weaknesses
    ]
    topics = [topic for topic in topics if topic][:5]

    sections = [
        BASE_SYSTEM_PROMPT,
        "",
        "## Current context",
        f"Target language: {language.english_name} ({language.native_name})",
        f"User's native language: {context.native_language}",
        f"Session mode: {mode.label}",
        "",
        "## Mode instructions",
        mode.instructions,
        "",
        "## " + correction_instruction(context.correction_mode).split(".")[0],
        correction_instruction(context.correction_mode),
        "",
        "## How to answer",
        REPLY_INSTRUCTIONS,
        "",
        "## This learner",
        f"CEFR level: {context.level}",
        "",
        "## What they need to practise (steer the conversation towards these)",
        _bullet_list(topics, "nothing identified yet — find out what they can do"),
    ]
    if context.review_items:
        sections += [
            "",
            "## Due for review (bring these back into the conversation)",
            _bullet_list(context.review_items[:8]),
        ]
    return "\n".join(sections)


ANALYSIS_INSTRUCTIONS = """You are analysing ONE sentence a learner produced. You are not conversing.

Report only genuine mistakes. If the sentence is correct, leave `correction`
empty and `errors` empty — do not invent problems to look useful.

`corrected` must keep the learner's meaning and their point of view: if they
said "I", the correction says "I", never "you".

`vocabulary` lists only words worth teaching them, drawn from your correction.
Leave it empty when there is nothing new."""


def build_analysis_prompt(context: TutorContext) -> str:
    """Prompt du second temps : l'analyse pédagogique, en arrière-plan.

    Il ne reprend ni la personnalité, ni le mode de conversation, ni l'historique
    — rien de tout cela ne sert à corriger une phrase. Le prompt tombe ainsi de
    ~900 à ~250 tokens, et ce qu'il produit n'est plus attendu par personne.
    """
    language = get_language(context.language)
    return "\n".join(
        [
            f"You are a {language.english_name} teacher correcting one sentence.",
            "",
            f"Learner's CEFR level: {context.level}",
            f"Learner's native language: {context.native_language}",
            "",
            "## " + correction_instruction(context.correction_mode).split(".")[0],
            correction_instruction(context.correction_mode),
            "",
            "## Error types you may use",
            ", ".join(language.error_types),
            "",
            "## How to analyse",
            ANALYSIS_INSTRUCTIONS,
        ]
    )


#: Contrat de l'analyse. Même logique que TURN_RESPONSE_SCHEMA : les champs sont
#: obligatoires, sans quoi un petit modèle les oublie. `response` en est absent —
#: la conversation a déjà eu lieu.
ANALYSIS_SCHEMA: dict[str, Any] = _object(
    correction=_object(original=_STRING, corrected=_STRING, explanation=_STRING),
    errors={
        "type": "array",
        "items": _object(
            type=_STRING,
            topic=_STRING,
            original=_STRING,
            corrected=_STRING,
            severity={"type": "string", "enum": ["minor", "major"]},
        ),
    },
    vocabulary={
        "type": "array",
        "items": _object(
            word=_STRING, translation=_STRING, example=_STRING, difficulty=_LEVEL
        ),
    },
    detected_language={
        "type": "string",
        "enum": ["english", "german", "french", "other"],
    },
    difficulty=_LEVEL,
)


def build_turn_prompt(context: TutorContext) -> str:
    """Prompt système + contrat de sortie JSON."""
    return build_system_prompt(context, output_format=RESPONSE_SCHEMA_PROMPT)


# --------------------------------------------------- prompts spécialisés
EXERCISE_SCHEMA_PROMPT = """\
Answer with a single JSON object and nothing else:

{
  "exercise_type": "multiple_choice" | "fill_in_the_blank" | "conjugation" |
                   "sentence_correction" | "transformation" | "translation" |
                   "sentence_building",
  "topic": "<grammar or vocabulary topic>",
  "level": "<A1..C2>",
  "prompt": "<the question shown to the learner>",
  "options": ["<option>", "..."],
  "answer": "<the expected answer>",
  "explanation": "<why this is the answer, in one or two sentences>"
}

`options` must be empty for open-ended exercise types.\
"""

ANSWER_CHECK_SCHEMA_PROMPT = """\
Answer with a single JSON object and nothing else:

{
  "is_correct": true | false,
  "feedback": "<short encouraging feedback, in the target language>",
  "corrected": "<the correct answer>",
  "errors": [{"type": "<error type>", "topic": "<topic>", "severity": "minor" | "major"}]
}\
"""

ASSESSMENT_SCHEMA_PROMPT = """\
Answer with a single JSON object and nothing else:

{
  "level": "<A1..C2>",
  "scores": {"grammar": 0-100, "vocabulary": 0-100, "speaking": 0-100,
             "listening": 0-100, "writing": 0-100, "pronunciation": 0-100},
  "strengths": ["..."],
  "weaknesses": ["..."],
  "summary": "<two or three sentences addressed to the learner>"
}\
"""
