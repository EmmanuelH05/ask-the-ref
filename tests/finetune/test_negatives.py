import pytest

from atr.finetune.negatives import (
    Example,
    build_examples,
    mine_hard_negatives,
    section_key,
)
from atr.finetune.synth import SynthExample


def test_section_key_groups_subsections_with_their_section() -> None:
    assert section_key("L12.1") == "L12.1"
    assert section_key("L12.1.3") == "L12.1"
    assert section_key("L12.0") == "L12.0"
    assert section_key("G.football-terms.offside") == "G.football-terms.offside"


def test_mine_hard_negatives_skips_the_gold_and_its_section_siblings() -> None:
    ranked = ["L12.1", "L12.1.2", "L4.3", "L12.1.9", "L7.2", "L1.1"]
    assert mine_hard_negatives(ranked, "L12.1.1", n=2) == ["L4.3", "L7.2"]


def test_mine_hard_negatives_returns_fewer_when_the_ranking_runs_out() -> None:
    assert mine_hard_negatives(["L1.1", "L2.1"], "L1.1", n=3) == ["L2.1"]


def test_mine_hard_negatives_rejects_bad_n() -> None:
    with pytest.raises(ValueError):
        mine_hard_negatives(["L1.1"], "L2.1", n=0)


def test_mine_hard_negatives_lets_other_glossary_entries_be_negatives() -> None:
    assert mine_hard_negatives(["G.a.x", "G.a.y", "L1.1"], "G.a.x", n=2) == ["G.a.y", "L1.1"]


def test_build_examples_prefixes_texts_and_keeps_every_negative_with_its_id() -> None:
    examples = [SynthExample(passage_id="L1.1", language="en", question="How big?")]
    texts = {"L1.1": "Size rules.", "L2.1": "Ball rules.", "L3.1": "Player rules."}
    rows = build_examples(
        examples,
        {"How big?": ["L1.1", "L2.1", "L3.1"]},
        texts,
        negatives_per_example=2,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    assert rows == [
        Example(
            "query: How big?",
            "passage: Size rules.",
            "L1.1",
            ("passage: Ball rules.", "passage: Player rules."),
            ("L2.1", "L3.1"),
        )
    ]


def test_build_examples_fails_loudly_on_a_question_without_a_ranking() -> None:
    examples = [SynthExample(passage_id="L1.1", language="en", question="How big?")]
    with pytest.raises(KeyError):
        build_examples(
            examples, {}, {"L1.1": "x"}, negatives_per_example=1, query_prefix="", passage_prefix=""
        )


def test_build_examples_skips_a_question_with_no_usable_negative() -> None:
    examples = [SynthExample(passage_id="L1.1", language="en", question="q?")]
    rows = build_examples(
        examples,
        {"q?": ["L1.1", "L1.1.2"]},
        {"L1.1": "x", "L1.1.2": "y"},
        negatives_per_example=1,
        query_prefix="",
        passage_prefix="",
    )
    assert rows == []
