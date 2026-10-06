from atr.evalset.models import QuestionGroup
from atr.evalset.review import blind_flag, leak_tokens, render_group


def group(**overrides: object) -> QuestionGroup:
    fields: dict[str, object] = {
        "group_id": "A-01",
        "type": "paraphrase",
        "gold": ["L1.1"],
        "en": "May the round thing rotate twice?",
        "es": "¿Puede el objeto girar dos veces?",
        "cs": "¿Puede the round thing girar twice?",
        "answer": "Yes.",
        "note": "Invented.",
    }
    return QuestionGroup.model_validate({**fields, **overrides})


def test_leak_tokens_lists_shared_content_words_sorted() -> None:
    assert leak_tokens("Does the zorb spin twice?", ["A zorb spins once or twice."]) == [
        "spin",
        "twice",
        "zorb",
    ]


def test_leak_tokens_is_empty_for_a_clean_paraphrase() -> None:
    assert leak_tokens("May the round thing rotate?", ["A zorb spins once."]) == []


def test_blind_flag_passes_when_every_gold_id_was_picked() -> None:
    assert blind_flag(group(), ["L2.2", "L1.1"]) is None


def test_blind_flag_names_missing_gold_and_what_was_picked() -> None:
    assert blind_flag(group(), ["L2.2"]) == "blind check missed L1.1; it picked L2.2"


def test_blind_flag_for_out_of_scope_fires_when_anything_was_picked() -> None:
    oos = group(type="out_of_scope", gold=[])
    assert blind_flag(oos, []) is None
    assert blind_flag(oos, ["L3.1"]) == "blind check found an answer in L3.1"


def test_blind_flag_is_none_when_the_group_was_not_checked() -> None:
    assert blind_flag(group(), None) is None


def test_render_group_shows_texts_gold_label_excerpt_and_flags() -> None:
    en = {"L1.1": "A zorb is a round thing that may rotate."}
    es = {"L1.1": "Un zorb es un objeto que puede girar."}
    labels = {"L1.1": "The Zorb / Shape"}
    text = render_group(group(), en, es, labels, picked=["L9.9"], excerpt_words=4)
    assert "### A-01 (paraphrase)" in text
    assert "- EN: May the round thing rotate twice?" in text
    assert "`L1.1` The Zorb / Shape: A zorb is a [...]" in text
    assert "FLAG paraphrase shares words with its gold passage (en): rotate, round, thing" in text
    assert "FLAG paraphrase shares words with its gold passage (es): girar, objeto" in text
    assert "FLAG blind check missed L1.1; it picked L9.9" in text


def test_render_group_has_no_flag_lines_when_clean() -> None:
    en = {"L1.1": "A zorb spins."}
    es = {"L1.1": "Un zorb da vueltas."}
    text = render_group(group(), en, es, {"L1.1": "The Zorb"}, picked=["L1.1"], excerpt_words=40)
    assert "FLAG" not in text
