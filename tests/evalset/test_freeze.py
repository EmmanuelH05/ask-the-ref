import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atr.evalset.freeze import (
    EvalManifest,
    build_manifest,
    canonical_bytes,
    evalset_hash,
    main,
    verify_frozen,
)
from atr.evalset.models import QuestionGroup, QuestionType, Split
from atr.evalset.store import corpus_pin


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
        corpus_pin="abc",
        reviewed_by="Mannie",
    )

    assert manifest.counts == {"dev": 2, "test": 1, "heldout": 0}
    assert manifest.held_out_laws == [3, 5, 16]
    assert manifest.hash == evalset_hash(groups, splits)
    assert (manifest.seed, manifest.test_fraction) == (7, 0.5)
    assert manifest.corpus_pin == "abc"
    assert manifest.reviewed_by == "Mannie"
    assert manifest.splits == splits


def test_manifest_is_frozen() -> None:
    groups, splits = two_groups()
    manifest = build_manifest(
        groups, splits, seed=0, test_fraction=0.5, corpus_pin="", reviewed_by="x"
    )

    assert isinstance(manifest, EvalManifest)
    with pytest.raises(ValidationError):
        manifest.seed = 1


def _write_frozen_inputs(tmp_path: Path, reviewed: bool) -> tuple[Path, Path]:
    draft = tmp_path / "draft"
    draft.mkdir()
    kinds = {
        "plain": ["L1.1"],
        "paraphrase": ["L1.1"],
        "scenario": ["L1.1"],
        "multi_rule": ["L1.1", "L2.1"],
        "false_premise": [],
        "out_of_scope": [],
    }
    lines = []
    for index, (kind, gold) in enumerate(kinds.items()):
        row = {
            "group_id": f"T-{index}",
            "type": kind,
            "gold": gold,
            "en": f"Invented question {index}?",
            "es": f"¿Pregunta inventada {index}?",
            "cs": f"¿Pregunta invented {index}?",
            "answer": "Invented.",
            "note": "Invented.",
            "spanish_reviewed": reviewed,
        }
        lines.append(json.dumps(row))
    (draft / "t.jsonl").write_text("\n".join(lines) + "\n")
    passages = [
        {"passage_id": i, "language": lang, "sha256": "aa", "word_count": 1}
        for i in ("L1.1", "L2.1")
        for lang in ("en", "es")
    ]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps({"edition": "e", "source": "s", "fetched_at": "t", "passages": passages})
    )
    return draft, corpus


def _freeze_args(tmp_path: Path, draft: Path, corpus: Path, *extra: str) -> list[str]:
    return [
        "--reviewed-by",
        "tester",
        "--draft-dir",
        str(draft),
        "--corpus-manifest",
        str(corpus),
        "--out",
        str(tmp_path / "questions.jsonl"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        *extra,
    ]


def test_cli_refuses_unreviewed_groups_and_writes_nothing(tmp_path: Path) -> None:
    draft, corpus = _write_frozen_inputs(tmp_path, reviewed=False)
    assert main(_freeze_args(tmp_path, draft, corpus)) == 1
    assert not (tmp_path / "questions.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_cli_allow_unreviewed_freezes_and_records_how_many_are_unreviewed(tmp_path: Path) -> None:
    draft, corpus = _write_frozen_inputs(tmp_path, reviewed=False)
    assert main(_freeze_args(tmp_path, draft, corpus, "--allow-unreviewed")) == 0
    manifest = EvalManifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert manifest.unreviewed_groups == 6
    assert manifest.reviewed_by == "tester"
    assert manifest.corpus_pin == corpus_pin(corpus)


def test_verify_frozen_passes_then_catches_an_edited_question_and_a_changed_corpus(
    tmp_path: Path,
) -> None:
    draft, corpus = _write_frozen_inputs(tmp_path, reviewed=True)
    assert main(_freeze_args(tmp_path, draft, corpus)) == 0
    questions, manifest = tmp_path / "questions.jsonl", tmp_path / "manifest.json"
    groups, loaded = verify_frozen(questions, manifest, corpus)
    assert len(groups) == 6 and loaded.unreviewed_groups == 0

    questions.write_text(questions.read_text().replace("Invented question 0?", "Edited?"))
    with pytest.raises(ValueError, match="hash"):
        verify_frozen(questions, manifest, corpus)

    assert main(_freeze_args(tmp_path, draft, corpus)) == 0
    corpus.write_text(corpus.read_text().replace('"aa"', '"bb"', 1))
    with pytest.raises(ValueError, match="corpus"):
        verify_frozen(questions, manifest, corpus)
