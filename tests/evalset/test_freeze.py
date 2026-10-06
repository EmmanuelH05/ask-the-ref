import hashlib

import pytest
from pydantic import ValidationError

from atr.evalset.freeze import EvalManifest, build_manifest, canonical_bytes, evalset_hash
from atr.evalset.models import QuestionGroup, QuestionType, Split


def make_group(
    group_id: str = "g1",
    *,
    type: QuestionType = "plain",
    gold: list[str] | None = None,
    en: str = "May a goalkeeper hold the ball for ten seconds?",
) -> QuestionGroup:
    return QuestionGroup(
        group_id=group_id,
        type=type,
        gold=["L12.1"] if gold is None else gold,
        en=en,
        es="es text",
        cs="cs text",
        answer="an answer",
        note="a note",
    )


def two_groups() -> tuple[list[QuestionGroup], dict[str, Split]]:
    groups = [make_group("a"), make_group("b", en="Second invented question")]
    return groups, {"a": "dev", "b": "test"}


def test_exact_bytes_for_one_tiny_group() -> None:
    group = QuestionGroup(
        group_id="g1", type="plain", gold=["L1.1"], en="a", es="b", cs="c", answer="x", note="n"
    )

    result = canonical_bytes([group], {"g1": "dev"})

    assert result == (
        b'[{"answer":"x","cs":"c","en":"a","es":"b","gold":["L1.1"],"group_id":"g1",'
        b'"note":"n","split":"dev","type":"plain"}]'
    )


def test_order_of_groups_does_not_matter() -> None:
    groups, splits = two_groups()

    assert canonical_bytes(groups, splits) == canonical_bytes(groups[::-1], splits)


@pytest.mark.parametrize("language", ["en", "es", "cs"])
def test_changing_any_question_text_changes_bytes(language: str) -> None:
    groups, splits = two_groups()
    changed = [groups[0].model_copy(update={language: "different words"}), groups[1]]

    assert canonical_bytes(changed, splits) != canonical_bytes(groups, splits)


def test_changing_a_gold_id_changes_bytes() -> None:
    groups, splits = two_groups()
    changed = [groups[0].model_copy(update={"gold": ["L12.2"]}), groups[1]]

    assert canonical_bytes(changed, splits) != canonical_bytes(groups, splits)


def test_changing_type_changes_bytes() -> None:
    groups, splits = two_groups()
    changed = [groups[0].model_copy(update={"type": "paraphrase"}), groups[1]]

    assert canonical_bytes(changed, splits) != canonical_bytes(groups, splits)


def test_changing_a_split_changes_bytes() -> None:
    groups, splits = two_groups()

    assert canonical_bytes(groups, {"a": "test", "b": "test"}) != canonical_bytes(groups, splits)


def test_spanish_reviewed_flag_does_not_change_bytes() -> None:
    groups, splits = two_groups()
    flipped = [groups[0].model_copy(update={"spanish_reviewed": True}), groups[1]]

    assert canonical_bytes(flipped, splits) == canonical_bytes(groups, splits)
    assert b"spanish_reviewed" not in canonical_bytes(groups, splits)


def test_non_ascii_is_kept_as_utf8() -> None:
    group = make_group("g1", en="¿Es fuera de juego?")

    result = canonical_bytes([group], {"g1": "dev"})

    assert "¿Es fuera de juego?".encode() in result
    assert b"\\u" not in result


def test_group_without_split_raises() -> None:
    groups, _ = two_groups()

    with pytest.raises(ValueError, match="b"):
        canonical_bytes(groups, {"a": "dev"})


def test_split_for_unknown_group_raises() -> None:
    groups, splits = two_groups()

    with pytest.raises(ValueError, match="zzz"):
        canonical_bytes(groups, {**splits, "zzz": "dev"})


def test_evalset_hash_is_sha256_of_canonical_bytes() -> None:
    groups, splits = two_groups()

    expected = hashlib.sha256(canonical_bytes(groups, splits)).hexdigest()
    assert evalset_hash(groups, splits) == expected


def test_build_manifest_counts_and_hash() -> None:
    groups = [make_group("a"), make_group("b"), make_group("c")]
    splits: dict[str, Split] = {"a": "dev", "b": "test", "c": "dev"}

    manifest = build_manifest(
        groups,
        splits,
        seed=7,
        test_fraction=0.5,
        corpus_manifest_sha256="abc",
        reviewed_by="Mannie",
    )

    assert manifest.counts == {"dev": 2, "test": 1, "heldout": 0}
    assert manifest.held_out_laws == [3, 5, 16]
    assert manifest.hash == evalset_hash(groups, splits)
    assert (manifest.seed, manifest.test_fraction) == (7, 0.5)
    assert manifest.corpus_manifest_sha256 == "abc"
    assert manifest.reviewed_by == "Mannie"
    assert manifest.splits == splits


def test_manifest_is_frozen() -> None:
    groups, splits = two_groups()
    manifest = build_manifest(
        groups, splits, seed=0, test_fraction=0.5, corpus_manifest_sha256="", reviewed_by="x"
    )

    assert isinstance(manifest, EvalManifest)
    with pytest.raises(ValidationError):
        manifest.seed = 1
