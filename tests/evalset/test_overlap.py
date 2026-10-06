import pytest

from atr.evalset.models import QuestionGroup
from atr.evalset.overlap import STOPWORDS, GroupOverlap, content_tokens, group_overlap, overlap

# All text here is invented, about a made-up game called zorb.

EN = {"L1.1": "A zorb may leap only at the gate."}
ES = {"L1.1": "Un zorb puede saltar solo en la puerta."}


def _group(**changes: object) -> QuestionGroup:
    fields: dict[str, object] = {
        "group_id": "g1",
        "type": "plain",
        "gold": ["L1.1"],
        "en": "How many zorbs may leap the gate twice?",
        "es": "¿Cuántos zorbs pueden saltar la puerta dos veces?",
        "cs": "How many zorbs pueden saltar la puerta twice?",
        "answer": "invented",
        "note": "invented",
    }
    return QuestionGroup(**{**fields, **changes})  # type: ignore[arg-type]


def test_stopwords_size_and_normal_form() -> None:
    assert 150 <= len(STOPWORDS) <= 300
    assert all(word == word.lower() and word.isascii() and word.isalnum() for word in STOPWORDS)
    assert {"is", "was", "los", "las", "the", "el", "what", "que"} <= STOPWORDS
    assert not {"ball", "goal", "player", "referee", "balon", "gol", "jugador", "arbitro"} & (
        STOPWORDS
    )


def test_accents_folded() -> None:
    assert content_tokens("Zorbísta") == ["zorbista"]


def test_case_folded() -> None:
    assert content_tokens("ZORB Gate") == ["zorb", "gat"]


def test_punctuation_and_inverted_marks_dropped() -> None:
    assert content_tokens("¿Cuándo, zorb gate?!") == ["zorb", "gat"]


def test_english_stopwords_dropped() -> None:
    assert content_tokens("What is the gate") == ["gat"]
    assert content_tokens("This was the gate") == ["gat"]


def test_spanish_stopwords_dropped() -> None:
    assert content_tokens("¿Qué es el portal?") == ["portal"]
    assert content_tokens("Los zorbs y las puertas") == ["zorb", "puerta"]


def test_trailing_s_only_on_tokens_longer_than_three() -> None:
    assert content_tokens("bus gas zorbs gates") == ["bus", "gas", "zorb", "gat"]


def test_double_s_is_not_a_plural() -> None:
    assert content_tokens("zorbss") == ["zorbss"]


def test_order_and_duplicates_kept_and_digits_are_tokens() -> None:
    assert content_tokens("gate 12 zorb gate") == ["gat", "12", "zorb", "gat"]


def test_overlap_fraction() -> None:
    # question content: zorb, leap, gate, twice. passage content: zorb, leap, gate. 3/4.
    assert overlap("How many zorbs may leap the gate twice?", [EN["L1.1"]]) == 0.75


def test_overlap_full_when_all_present_across_passages() -> None:
    passages = [EN["L1.1"], "Twice per turn, a zorb must rest."]
    assert overlap("How many zorbs may leap the gate twice?", passages) == 1.0


def test_overlap_counts_distinct_question_tokens() -> None:
    # distinct question tokens: zorb, gate, leap. passage has only zorb. 1/3, not 2/5.
    assert overlap("zorb zorb gate gate leap", ["a zorb"]) == pytest.approx(1 / 3)


def test_overlap_zero_for_all_stopword_question() -> None:
    assert overlap("What is the?", [EN["L1.1"]]) == 0.0


def test_overlap_zero_for_empty_passage_list() -> None:
    assert overlap("zorb gate", []) == 0.0


def test_group_overlap_normal() -> None:
    result = group_overlap(_group(), EN, ES)
    assert result == GroupOverlap(
        group_id="g1",
        en_vs_en=0.75,  # {zorb leap gate twice} vs {zorb leap gate}
        es_vs_es=0.6,  # {zorb saltar puerta dos vece} vs {zorb saltar solo puerta}
        cs_vs_en=0.25,  # {zorb saltar puerta twice} vs {zorb leap gate}: zorb only
        cs_vs_es=0.75,  # {zorb saltar puerta twice} vs {zorb saltar solo puerta}
        es_vs_en=0.2,  # {zorb saltar puerta dos vece} vs {zorb leap gate}: zorb only
        en_vs_es=0.25,  # {zorb leap gate twice} vs {zorb saltar solo puerta}: zorb only
    )


def test_group_overlap_out_of_scope_is_all_none() -> None:
    result = group_overlap(_group(type="out_of_scope", gold=[]), EN, ES)
    assert result == GroupOverlap(
        group_id="g1",
        en_vs_en=None,
        es_vs_es=None,
        cs_vs_en=None,
        cs_vs_es=None,
        es_vs_en=None,
        en_vs_es=None,
    )


def test_group_overlap_gold_missing_from_spanish() -> None:
    result = group_overlap(_group(), EN, {})
    assert result.es_vs_es is None
    assert result.cs_vs_es is None
    assert result.en_vs_es is None
    assert result.en_vs_en == 0.75
    assert result.cs_vs_en == 0.25
    assert result.es_vs_en == 0.2


def test_group_overlap_none_when_any_gold_is_missing() -> None:
    group = _group(type="multi_rule", gold=["L1.1", "L1.2"])
    result = group_overlap(group, EN, ES)
    assert result.en_vs_en is None
    assert result.es_vs_es is None


def test_single_letter_tokens_are_dropped() -> None:
    # A possessive leaves a stray "s" ("keeper's" -> "keeper", "s") that would match any other.
    assert content_tokens("the zorb's wobble") == ["zorb", "wobbl"]


def test_cannot_antes_despues_and_durante_are_stopwords() -> None:
    assert content_tokens("cannot antes después durante zorb") == ["zorb"]


def test_contractions_and_possessives_do_not_leave_debris() -> None:
    assert content_tokens("the zorb's keeper doesn't wobble, they\u2019re late") == [
        "zorb",
        "keeper",
        "wobbl",
        "lat",
    ]


def test_plural_and_singular_meet_in_both_languages() -> None:
    assert content_tokens("jugadores jugador") == ["jugador", "jugador"]
    assert content_tokens("offences offence") == ["offenc", "offenc"]
    assert content_tokens("passes pass") == ["pass", "pass"]
    assert content_tokens("pases pase") == ["pas", "pas"]


def test_esa_is_a_stopword() -> None:
    assert content_tokens("esa zorba") == ["zorba"]
