import pytest

from atr.corpus.models import Passage, count_words
from atr.finetune.synth import (
    QUESTIONS_PER_LANGUAGE,
    SynthExample,
    examples_from_reply,
    mixes_within_sentence,
    parse_synthetic,
    synthetic_hash,
    training_passages,
)
from atr.llm import Completion


def passage(passage_id: str, law: int | None, text: str = "The ball is round.") -> Passage:
    return Passage(
        passage_id=passage_id,
        language="en",
        law_number=law,
        law_title="Title",
        section_index=1 if law is not None else None,
        section_title="Section",
        subheading=None,
        source_slug="slug",
        text=text,
        word_count=count_words(text),
    )


def test_training_passages_drop_held_out_laws_and_keep_the_glossary() -> None:
    kept = training_passages(
        [passage("L1.1", 1), passage("L3.1", 3), passage("L5.2", 5), passage("G.x.ball", None)]
    )
    assert [p.passage_id for p in kept] == ["L1.1", "G.x.ball"]


def test_training_passages_take_only_english_passages() -> None:
    spanish = passage("L1.1", 1).model_copy(update={"language": "es"})
    assert training_passages([spanish]) == []


def test_parse_synthetic_reads_two_questions_per_language() -> None:
    reply = (
        'Sure:\n{"en": ["What is x?", "Why x?"], "es": ["¿Qué es x?", "¿Por qué x?"], '
        '"cs": ["¿Qué es the x?", "Why el x?"]}'
    )
    parsed = parse_synthetic(reply)
    assert parsed == {
        "en": ["What is x?", "Why x?"],
        "es": ["¿Qué es x?", "¿Por qué x?"],
        "cs": ["¿Qué es the x?", "Why el x?"],
    }


def test_parse_synthetic_rejects_missing_language_or_wrong_shape() -> None:
    assert parse_synthetic('{"en": ["a?"], "es": ["b?"]}') is None
    assert parse_synthetic('{"en": "a?", "es": ["b?"], "cs": ["c?"]}') is None
    assert parse_synthetic("no json here") is None
    assert parse_synthetic('{"en": [], "es": ["b?"], "cs": ["c?"]}') is None


def test_parse_synthetic_strips_blank_and_duplicate_questions() -> None:
    parsed = parse_synthetic('{"en": [" a? ", "a?", ""], "es": ["b?"], "cs": ["c?"]}')
    assert parsed is not None
    assert parsed["en"] == ["a?"]


def test_examples_from_reply_tags_each_question_with_its_passage_and_language() -> None:
    reply = Completion('{"en": ["a?"], "es": ["b?"], "cs": ["c?"]}', 12.0)
    examples = examples_from_reply("L1.1", reply)
    assert examples == [
        SynthExample(passage_id="L1.1", language="en", question="a?"),
        SynthExample(passage_id="L1.1", language="es", question="b?"),
        SynthExample(passage_id="L1.1", language="cs", question="c?"),
    ]


def test_examples_from_reply_is_empty_for_an_unusable_reply() -> None:
    assert examples_from_reply("L1.1", Completion("garbage", 1.0)) == []


def test_synthetic_hash_depends_on_content_not_order() -> None:
    a = SynthExample(passage_id="L1.1", language="en", question="a?")
    b = SynthExample(passage_id="L1.2", language="es", question="b?")
    assert synthetic_hash([a, b]) == synthetic_hash([b, a])
    assert synthetic_hash([a]) != synthetic_hash([a, b])


def test_questions_per_language_is_two() -> None:
    assert QUESTIONS_PER_LANGUAGE == 2


def test_example_rejects_blank_question() -> None:
    with pytest.raises(ValueError):
        SynthExample(passage_id="L1.1", language="en", question="  ")


def test_mixes_within_sentence_needs_both_languages_in_one_sentence() -> None:
    assert mixes_within_sentence("¿Es offside si the ball comes off a defender?")
    assert not mixes_within_sentence("Can the keeper hold the ball? ¿Cuánto tiempo puede tenerlo?")
    assert not mixes_within_sentence("What is the size of the goal?")
