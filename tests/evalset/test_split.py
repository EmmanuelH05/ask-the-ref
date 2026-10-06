import random
from collections import Counter

import pytest

from atr.evalset.models import QUESTION_TYPES, QuestionGroup, QuestionType
from atr.evalset.split import assign_splits

_GOLD_BY_TYPE: dict[str, list[str]] = {
    "plain": ["L11.1"],
    "paraphrase": ["L12.4"],
    "scenario": ["L12.4.3"],
    "multi_rule": ["L11.1", "L12.4"],
    "false_premise": ["L11.2"],
    "out_of_scope": [],
}


def make_group(
    group_id: str, type_: QuestionType = "plain", gold: list[str] | None = None
) -> QuestionGroup:
    return QuestionGroup(
        group_id=group_id,
        type=type_,
        gold=_GOLD_BY_TYPE[type_] if gold is None else gold,
        en=f"Invented question {group_id}?",
        es=f"Pregunta inventada {group_id}?",
        cs=f"Pregunta inventada {group_id}, okay?",
        answer="Invented answer.",
        note="Invented note.",
    )


def make_many(
    type_: QuestionType, count: int, gold: list[str] | None = None
) -> list[QuestionGroup]:
    return [make_group(f"{type_}-{i:03d}", type_, gold) for i in range(count)]


def test_any_held_out_gold_gives_heldout_whatever_the_type() -> None:
    groups = [
        make_group("a", "plain", ["L3.1"]),
        make_group("b", "paraphrase", ["L5.2"]),
        make_group("c", "scenario", ["L16.1.2"]),
        make_group("d", "multi_rule", ["L11.1", "L16.3"]),
        make_group("e", "false_premise", ["L5.1"]),
    ]
    result = assign_splits(groups, seed=0)
    assert result == dict.fromkeys("abcde", "heldout")


def test_out_of_scope_is_never_heldout() -> None:
    groups = make_many("out_of_scope", 10)
    result = assign_splits(groups, seed=0)
    assert Counter(result.values()) == Counter({"test": 5, "dev": 5})


def test_exact_counts_per_type_and_split() -> None:
    groups: list[QuestionGroup] = []
    for type_ in QUESTION_TYPES:
        groups += make_many(type_, 9)
    held = [make_group(f"held-{i}", "plain", ["L3.1"]) for i in range(3)]
    result = assign_splits(groups + held, seed=1)
    for type_ in QUESTION_TYPES:
        got = Counter(result[g.group_id] for g in groups if g.type == type_)
        # round(9 * 0.5) == 4 (banker's rounding)
        assert got == Counter({"test": 4, "dev": 5}), type_
    assert Counter(result[g.group_id] for g in held) == Counter({"heldout": 3})
    assert Counter(result.values()) == Counter({"test": 24, "dev": 30, "heldout": 3})


def test_test_share_follows_fraction() -> None:
    groups = make_many("scenario", 10)
    result = assign_splits(groups, seed=3, test_fraction=0.3)
    assert Counter(result.values()) == Counter({"test": 3, "dev": 7})
    result = assign_splits(groups, seed=3, test_fraction=0.25)
    assert Counter(result.values()) == Counter({"test": 2, "dev": 8})


def test_heldout_groups_do_not_count_toward_the_test_share() -> None:
    groups = make_many("plain", 10) + [
        make_group(f"held-{i}", "plain", ["L5.1"]) for i in range(10)
    ]
    result = assign_splits(groups, seed=2)
    assert Counter(result.values()) == Counter({"heldout": 10, "test": 5, "dev": 5})


def test_same_seed_same_result_and_input_order_does_not_matter() -> None:
    groups = make_many("plain", 20) + make_many("multi_rule", 12)
    first = assign_splits(groups, seed=7)
    assert assign_splits(groups, seed=7) == first
    shuffled = list(groups)
    random.Random(99).shuffle(shuffled)
    assert dict(assign_splits(shuffled, seed=7)) == dict(first)


def test_different_seed_gives_different_assignment() -> None:
    groups = make_many("plain", 60)
    assert assign_splits(groups, seed=1) != assign_splits(groups, seed=2)


def test_other_type_does_not_change_a_types_assignment() -> None:
    plain = make_many("plain", 20)
    alone = assign_splits(plain, seed=5)
    mixed = assign_splits(plain + make_many("scenario", 15), seed=5)
    assert {g.group_id: mixed[g.group_id] for g in plain} == alone


def test_result_key_order_equals_input_order() -> None:
    groups = make_many("plain", 10)
    random.Random(4).shuffle(groups)
    groups.append(make_group("zzz", "plain", ["L3.1"]))
    assert list(assign_splits(groups, seed=0)) == [g.group_id for g in groups]


def test_duplicate_group_ids_raise() -> None:
    with pytest.raises(ValueError, match="dup"):
        assign_splits([make_group("dup"), make_group("dup", "scenario")], seed=0)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_bad_test_fraction_raises(fraction: float) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        assign_splits(make_many("plain", 4), seed=0, test_fraction=fraction)


def test_single_group_type_does_not_crash() -> None:
    # round(1 * 0.5) == 0 in Python, so the lone group lands in dev.
    assert assign_splits([make_group("only")], seed=0) == {"only": "dev"}
    assert assign_splits([make_group("only")], seed=0, test_fraction=0.9) == {"only": "test"}


def test_empty_input_gives_empty_result() -> None:
    assert assign_splits([], seed=0) == {}
