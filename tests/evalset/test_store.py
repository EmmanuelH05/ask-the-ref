from pathlib import Path

import pytest

from atr.corpus.build import CorpusManifest, ManifestEntry
from atr.evalset.models import QUESTION_TYPES, QuestionGroup
from atr.evalset.store import corpus_ids, load_groups, validate, write_groups

_GOLD = {
    "plain": ["L1.1"],
    "paraphrase": ["L1.1"],
    "scenario": ["L1.2"],
    "multi_rule": ["L1.1", "L1.2"],
    "false_premise": ["L1.1"],
    "out_of_scope": [],
}
EN_IDS = {"L1.1", "L1.2"}
ES_IDS = {"L1.1", "L1.2"}


def make_group(
    group_id: str, type_: str = "plain", *, en: str | None = None, gold: list[str] | None = None
) -> QuestionGroup:
    return QuestionGroup.model_validate(
        {
            "group_id": group_id,
            "type": type_,
            "gold": _GOLD[type_] if gold is None else gold,
            "en": en or f"Invented english question {group_id}?",
            "es": f"Pregunta inventada {group_id}?",
            "cs": f"Pregunta invented {group_id}?",
            "answer": "Invented answer.",
            "note": "",
        }
    )


def clean_set() -> list[QuestionGroup]:
    return [make_group(f"g-{t}", t) for t in QUESTION_TYPES]


def test_load_groups_reads_files_in_name_order_and_keeps_line_order(tmp_path: Path) -> None:
    write_groups([make_group("b1"), make_group("b2")], tmp_path / "b.jsonl")
    write_groups([make_group("a1"), make_group("a2")], tmp_path / "a.jsonl")

    ids = [g.group_id for g in load_groups(tmp_path)]

    assert ids == ["a1", "a2", "b1", "b2"]


def test_load_groups_skips_blank_lines(tmp_path: Path) -> None:
    line = make_group("g1").model_dump_json()
    (tmp_path / "x.jsonl").write_text(f"\n{line}\n   \n\n", encoding="utf-8")

    assert [g.group_id for g in load_groups(tmp_path)] == ["g1"]


def test_load_groups_malformed_json_names_file_and_line(tmp_path: Path) -> None:
    line = make_group("g1").model_dump_json()
    (tmp_path / "bad.jsonl").write_text(f"{line}\n{{not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_groups(tmp_path)


def test_load_groups_invalid_group_names_file_and_line(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text('\n{"group_id": "g1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_groups(tmp_path)


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    groups = clean_set()
    write_groups(groups, tmp_path / "all.jsonl")

    assert load_groups(tmp_path) == groups


def test_validate_clean_set_is_empty() -> None:
    assert validate(clean_set(), EN_IDS, ES_IDS) == []


def test_validate_duplicate_group_id() -> None:
    groups = [*clean_set(), make_group("g-plain", en="A different invented text?")]

    problems = validate(groups, EN_IDS, ES_IDS)

    assert len(problems) == 1
    assert "g-plain" in problems[0]
    assert not problems[0].startswith("warn: ")


def test_validate_gold_missing_from_en_ids() -> None:
    groups = [*clean_set(), make_group("g-x", gold=["L9.9"])]

    problems = validate(groups, EN_IDS, ES_IDS | {"L9.9"})

    assert len(problems) == 1
    assert "g-x" in problems[0]
    assert "L9.9" in problems[0]


def test_validate_question_type_with_no_group() -> None:
    groups = [g for g in clean_set() if g.type != "scenario"]

    problems = validate(groups, EN_IDS, ES_IDS)

    assert len(problems) == 1
    assert "scenario" in problems[0]


def test_validate_same_english_text_ignoring_case_and_spacing() -> None:
    groups = [
        *clean_set(),
        make_group("g-dup1", en="Is the  Zorp   legal?"),
        make_group("g-dup2", en="is the zorp legal?"),
    ]

    problems = validate(groups, EN_IDS, ES_IDS)

    assert len(problems) == 1
    assert "g-dup1" in problems[0]
    assert "g-dup2" in problems[0]


def test_validate_warns_for_gold_missing_from_es_ids() -> None:
    problems = validate(clean_set(), EN_IDS, {"L1.1"})

    assert problems
    assert all(p.startswith("warn: ") for p in problems)
    assert any("L1.2" in p for p in problems)


def test_corpus_ids_splits_by_language(tmp_path: Path) -> None:
    entries = [
        ManifestEntry(passage_id="L1.1", language="en", sha256="a", word_count=1),
        ManifestEntry(passage_id="L1.2", language="en", sha256="b", word_count=1),
        ManifestEntry(passage_id="L1.1", language="es", sha256="c", word_count=1),
    ]
    manifest = CorpusManifest(edition="e", source="s", fetched_at="t", passages=entries)
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")

    assert corpus_ids(path) == ({"L1.1", "L1.2"}, {"L1.1"})
