"""Tests des commandes vocales, de la prononciation, des prompts et du placement."""

from __future__ import annotations

import pytest

from app.ai.prompts import CONVERSATION_MODES, TutorContext, build_system_prompt, get_mode
from app.language.assessment import get_items, score_objective
from app.language.commands import detect_command
from app.language.correction import filter_by_mode
from app.language.languages import (
    TARGET_LANGUAGES,
    clamp_level,
    error_types_for,
    get_language,
    is_supported,
    level_index,
    score_to_level,
)
from app.language.pronunciation import analyse


# ------------------------------------------------------------------ langues
def test_target_languages_are_english_and_german() -> None:
    assert TARGET_LANGUAGES == ("english", "german")


def test_german_has_its_own_error_types() -> None:
    german = set(error_types_for("german"))
    english = set(error_types_for("english"))
    assert {"cases", "gender_der_die_das", "separable_verbs", "umlaut"} <= german
    assert german > english


def test_unknown_language_falls_back_to_english() -> None:
    assert get_language("klingon").code == "english"
    assert is_supported("klingon") is False


@pytest.mark.parametrize(
    ("score", "level"),
    [(0, "A1"), (10, "A1"), (25, "A2"), (45, "B1"), (60, "B2"), (80, "C1"), (95, "C2")],
)
def test_score_maps_to_cefr_level(score: float, level: str) -> None:
    assert score_to_level(score) == level


def test_level_index_round_trip() -> None:
    for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
        assert clamp_level(level_index(level)) == level


def test_clamp_level_stays_within_bounds() -> None:
    assert clamp_level(-5) == "A1"
    assert clamp_level(99) == "C2"


# --------------------------------------------------------- commandes vocales
@pytest.mark.parametrize(
    ("phrase", "action", "payload"),
    [
        ("Liliana, let's practice English.", "switch_language", {"language": "english"}),
        ("Liliana, switch to German.", "switch_language", {"language": "german"}),
        ("Passe à l'allemand", "switch_language", {"language": "german"}),
        ("Liliana, correct me.", "set_correction_mode", {"correction_mode": "strict"}),
        ("Liliana, stop correcting me", "set_correction_mode", {"correction_mode": "off"}),
        ("Liliana, give me an exercise", "set_mode", {"mode": "grammar_training"}),
        ("Liliana, let's just talk", "set_mode", {"mode": "just_talk"}),
        ("immersion mode", "set_mode", {"mode": "immersion"}),
        ("Liliana, speak more slowly", "speak_slower", {"speed": 0.75}),
        ("Liliana, repeat that", "repeat", {}),
        ("Liliana, translate this", "translate", {}),
        ("start a lesson", "start_lesson", {}),
    ],
)
def test_voice_commands_are_recognised(phrase: str, action: str, payload: dict) -> None:
    command = detect_command(phrase)
    assert command is not None
    assert command.action == action
    assert command.payload == payload


def test_long_sentences_are_not_treated_as_commands() -> None:
    sentence = (
        "Yesterday I went to the cinema with my brother and we watched a very long film"
    )
    assert detect_command(sentence) is None


def test_ordinary_speech_is_not_a_command() -> None:
    assert detect_command("I like learning German because it is logical") is None
    assert detect_command("") is None


# ------------------------------------------------------------- prononciation
def test_perfect_reading_scores_full_marks() -> None:
    result = analyse("The weather is nice today", "the weather is nice today", "english")
    assert result.score == 100.0
    assert result.problem_sounds == []


def test_th_substitution_is_detected() -> None:
    result = analyse("I think this thing", "I sink dis sing", "english")
    assert result.score < 75
    assert any("TH" in sound for sound in result.problem_sounds)


def test_missing_umlaut_is_detected() -> None:
    result = analyse("Ich möchte ein Brötchen", "Ich mochte ein Brotchen", "german")
    assert any("Ö" in sound for sound in result.problem_sounds)


def test_empty_transcription_is_handled() -> None:
    result = analyse("Hello there", "", "english")
    assert result.score == 0.0
    assert "did not hear" in result.feedback


def test_missing_target_sentence_is_handled() -> None:
    assert analyse("", "hello", "english").score == 0.0


def test_word_comparisons_cover_every_expected_word() -> None:
    result = analyse("one two three", "one three", "english")
    assert [word.expected for word in result.words if word.expected] == ["one", "two", "three"]
    assert result.word_accuracy < 100


# -------------------------------------------------------- modes de correction
def test_correction_filter_respects_the_mode() -> None:
    found = [
        {"type": "grammar", "severity": "major"},
        {"type": "register", "severity": "minor"},
    ]
    assert filter_by_mode(found, "off") == []
    assert len(filter_by_mode(found, "minimal")) == 1
    assert len(filter_by_mode(found, "normal")) == 2
    assert len(filter_by_mode(found, "strict")) == 2


def test_unknown_correction_mode_falls_back_to_normal() -> None:
    found = [{"type": "grammar", "severity": "minor"}]
    assert filter_by_mode(found, "aggressive") == found


# --------------------------------------------------------------- prompts
def test_every_mode_has_instructions() -> None:
    for mode in CONVERSATION_MODES.values():
        assert mode.instructions
        assert mode.label
        assert mode.default_correction_mode in ("off", "minimal", "normal", "strict")


def test_unknown_mode_falls_back_to_free_conversation() -> None:
    assert get_mode("nonsense").key == "free_conversation"


def test_system_prompt_injects_the_learner_context() -> None:
    context = TutorContext(
        language="german",
        level="B1",
        mode="immersion",
        correction_mode="strict",
        weaknesses=[{"topic": "dative", "occurrences": 7}],
        recent_errors=[{"original": "mit der Mann", "corrected": "mit dem Mann", "topic": "dative"}],
        recent_vocabulary=["Bahnhof"],
        review_items=["Umlaut"],
    )
    prompt = build_system_prompt(context)
    assert "German" in prompt
    assert "B1" in prompt
    assert "dative (7 recent occurrences)" in prompt
    assert "mit dem Mann" in prompt
    assert "Bahnhof" in prompt
    assert "STRICT" in prompt
    # Le prompt doit annoncer les types d'erreurs propres à la langue.
    assert "separable_verbs" in prompt


def test_what_changes___turn_comes_last_in_the_prompt() -> None:
    """Le profil de l'apprenant doit etre en queue de prompt.

    Ollama ne garde en cache que le plus long prefixe deja evalue : tout ce qui
    suit le premier caractere qui change est recalcule. Le profil bouge a chaque
    tour ; le contrat de sortie, jamais. Place avant, il faisait re-evaluer
    plusieurs centaines de tokens a chaque phrase.
    """
    from app.ai.prompts import build_turn_prompt

    prompt = build_turn_prompt(
        TutorContext(weaknesses=[{"topic": "dative", "occurrences": 7}])
    )

    stable = prompt.index("## Output format")
    volatile = prompt.index("## This learner, right now")
    assert stable < volatile, "le contrat de sortie doit preceder le profil"

    # Tout ce qui bouge d'un tour a l'autre est apres le point de bascule.
    for section in ("Known weaknesses", "Recent mistakes", "Items due for review"):
        assert prompt.index(section) > stable


def test_system_prompt_handles_a_blank_profile() -> None:
    prompt = build_system_prompt(TutorContext())
    assert "no weakness identified yet" in prompt
    assert "no mistake recorded yet" in prompt


# ------------------------------------------------------------- placement
@pytest.mark.parametrize("language", ["english", "german"])
def test_placement_bank_covers_every_level(language: str) -> None:
    levels = {item.level for item in get_items(language)}
    assert levels == {"A1", "A2", "B1", "B2", "C1"}


@pytest.mark.parametrize("language", ["english", "german"])
def test_placement_answers_are_among_the_options(language: str) -> None:
    for item in get_items(language):
        assert item.answer in item.options


def test_placement_stops_at_the_first_failed_band() -> None:
    items = get_items("english")
    answers = {item.id: item.answer for item in items if item.level in ("A1", "A2")}
    assert score_objective("english", answers).estimated_level == "A2"


def test_placement_with_no_answers_returns_a1() -> None:
    assert score_objective("english", {}).estimated_level == "A1"


def test_a_perfect_placement_goes_one_band_above_the_bank() -> None:
    items = get_items("german")
    answers = {item.id: item.answer for item in items}
    result = score_objective("german", answers)
    assert result.estimated_level == "C2"
    assert result.percent == 100.0


def test_placement_answers_are_case_insensitive() -> None:
    items = get_items("english")
    answers = {item.id: item.answer.upper() for item in items if item.level == "A1"}
    assert score_objective("english", answers).correct == 2


# ------------------------------------------------- analyse phonétique (§7)
def test_punctuation_is_not_a_phoneme() -> None:
    """Un point final ne doit pas compter comme un son manqué."""
    from app.language import phonemes

    if not phonemes.is_available():
        pytest.skip("phonémisation indisponible sur cette installation")
    comparison = phonemes.compare("I think this is fine.", "I think this is fine", "english")
    assert comparison.accuracy == 1.0
    assert comparison.diffs == []


def test_phoneme_comparison_sees_what_spelling_cannot() -> None:
    """« möchte » entendu « mochte » : deux vrais sons manqués, pas un accent."""
    from app.language import phonemes

    if not phonemes.is_available():
        pytest.skip("phonémisation indisponible sur cette installation")
    comparison = phonemes.compare("Ich möchte", "Ich mochte", "german")
    assert comparison.accuracy < 1.0
    assert any("Ö" in label for label in comparison.labels)


def test_phoneme_labels_map_to_real_drill_categories() -> None:
    """Chaque son signalé doit renvoyer vers une catégorie qui existe vraiment."""
    from app.language.phonemes import _FAMILIES, _PHONEME_LABELS
    from app.language.pronunciation import categories_for, category_for_sound

    known = set(categories_for("english")) | set(categories_for("german"))
    for label in list(_PHONEME_LABELS.values()) + [name for _, name in _FAMILIES]:
        category = category_for_sound(label)
        assert category is None or category in known, f"{label!r} -> {category!r}"


def test_analysis_degrades_without_the_phonemizer(monkeypatch) -> None:
    """Sans piper-tts, l'analyse doit rester utilisable, pas planter."""
    from app.language import phonemes
    from app.language.pronunciation import analyse

    monkeypatch.setattr(phonemes, "compare", lambda *args, **kwargs: None)
    result = analyse("I think this thing", "I sink dis sing", "english")
    assert result.method == "spelling"
    assert result.phoneme_accuracy is None
    assert result.score < 100
    assert result.problem_sounds


@pytest.mark.parametrize("language", ["english", "german"])
def test_practice_sentences_exist_for_every_category(language: str) -> None:
    from app.language.pronunciation import categories_for, practice_sentences

    for category in categories_for(language):
        assert practice_sentences(language, category), category


def test_practice_sentence_targets_a_recorded_weakness() -> None:
    from app.language.pronunciation import pick_sentence

    category, sentence = pick_sentence("german", weak_sounds=["the German Ö versus O"])
    assert category == "Ö"
    assert "ö" in sentence.lower()


def test_practice_sentence_rotates() -> None:
    from app.language.pronunciation import pick_sentence

    first = pick_sentence("english", category="TH", rotation=0)[1]
    second = pick_sentence("english", category="TH", rotation=1)[1]
    assert first != second


# ------------------------------------------------------------- mot d'eveil
from app.language import wake_word  # noqa: E402


@pytest.mark.parametrize(
    ("phrase", "reste"),
    [
        ("Liliana, how are you today?", "how are you today"),
        ("Hello Liliana, I want to practise English.", "I want to practise English"),
        ("Hey Liliana what is a phrasal verb", "what is a phrasal verb"),
        ("Bonjour Liliana, on parle allemand ?", "on parle allemand"),
    ],
)
def test_the_call_is_recognised_and_stripped(phrase: str, reste: str) -> None:
    """Le nom sert a appeler ; ce qui suit est la vraie prise de parole."""
    heard = wake_word.detect(phrase)
    assert heard.heard
    assert heard.remainder == reste


@pytest.mark.parametrize("phrase", ["Lilliana can you help me", "Liliane, comment vas-tu ?",
                                    "Lily Anna are you there", "Lilyana I need help"])
def test_whisper_mishearings_still_wake_her(phrase: str) -> None:
    """Whisper n'ecrit presque jamais le nom deux fois de la meme facon.

    Une egalite stricte rendrait l'eveil inutilisable sur une voix accentuee.
    """
    assert wake_word.detect(phrase).heard


@pytest.mark.parametrize(
    "phrase",
    [
        "Yesterday I go to the cinema.",
        "What is the weather like today?",
        "I told my friend Liliana that the film was good.",
        "",
        "   ",
    ],
)
def test_speech_that_is_not_addressed_to_her_is_ignored(phrase: str) -> None:
    """Le nom prononce au milieu d'un recit ne doit pas declencher un tour."""
    assert wake_word.detect(phrase).heard is False


def test_the_name_alone_is_a_valid_call() -> None:
    """« Liliana ? » appelle sans rien demander : l'appelant attend une invite."""
    heard = wake_word.detect("Liliana?")
    assert heard.heard
    assert heard.remainder == ""


def test_a_stricter_threshold_rejects_approximations() -> None:
    assert wake_word.detect("Liliane hello", threshold=0.99).heard is False
    assert wake_word.detect("Liliana hello", threshold=0.99).heard is True


def test_several_names_can_be_configured() -> None:
    assert wake_word.parse_wake_words("Liliana, Lili") == ("Liliana", "Lili")
    assert wake_word.parse_wake_words("") == ("Liliana",)
    assert wake_word.detect("Lili, what does this mean?", ("Liliana", "Lili")).heard


def test_the_reply_prompt_knows_what_the_learner_must_practise() -> None:
    """Une reponse qui ignore les faiblesses de l'apprenant n'enseigne rien.

    C'est ce qui distingue une professeure d'un agent conversationnel : elle
    oriente la conversation vers ce qui coince.
    """
    from app.ai.prompts import build_reply_prompt

    prompt = build_reply_prompt(
        TutorContext(
            language="english",
            mode="english_teacher",
            level="B1",
            weaknesses=[{"topic": "past_simple", "occurrences": 12}],
            review_items=["irregular verbs"],
        )
    )
    assert "past_simple" in prompt
    assert "irregular verbs" in prompt
    assert "B1" in prompt


def test_the_reply_prompt_leaves_out_what_changes_every_single_turn() -> None:
    """Le detail litteral des dernieres erreurs appartient a l'analyse.

    Il n'ameliore pas la reponse parlee, et il suffisait a changer le prompt a
    chaque phrase — donc a interdire a Ollama de le relire dans son cache.
    Les points faibles, eux, evoluent en jours : ils restent.
    """
    from app.ai.prompts import build_reply_prompt

    def prompt_with(**extra):
        return build_reply_prompt(
            TutorContext(
                language="english", mode="english_teacher", level="B1",
                weaknesses=[{"topic": "past_simple", "occurrences": 12}], **extra,
            )
        )

    stable = prompt_with()
    assert prompt_with(recent_errors=[{"original": "I go", "corrected": "I went"}]) == stable
    assert prompt_with(recent_vocabulary=["commute", "sleeve"]) == stable
    # Le compteur d'occurrences change a chaque tour : lui non plus n'y figure pas.
    assert "12" not in stable


def test_the_analysis_prompt_does_depend_on_the_learner() -> None:
    """Corriger, en revanche, demande de savoir a qui on parle."""
    from app.ai.prompts import build_analysis_prompt

    a = build_analysis_prompt(TutorContext(language="english", level="A1"))
    b = build_analysis_prompt(TutorContext(language="english", level="C1"))
    assert a != b
