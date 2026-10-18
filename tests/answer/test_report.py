import pytest

from atr.answer.report import check_comparable
from atr.answer.run import AnswerRunMeta


def meta(system: str, **overrides: object) -> AnswerRunMeta:
    fields: dict[str, object] = {
        "system": system,
        "track": "T",
        "split": "dev",
        "serving_config": None,
        "refusal_source": None,
        "threshold": None,
        "generator": "generate:qwen2.5:14b",
        "code_stamp": "abc",
        "evalset_hash": "h",
        "corpus_pin": "p",
        "machine": "m",
        "created_at": "2026-10-18T00:00:00+00:00",
    }
    fields.update(overrides)
    return AnswerRunMeta.model_validate(fields)


def test_check_comparable_accepts_runs_from_the_same_code_and_refuses_a_stale_one() -> None:
    check_comparable({"closed_book": meta("closed_book"), "oracle": meta("oracle")})
    with pytest.raises(ValueError, match="code_stamp"):
        check_comparable({"full": meta("full"), "oracle": meta("oracle", code_stamp="old")})
    with pytest.raises(ValueError, match="generator"):
        check_comparable({"full": meta("full"), "oracle": meta("oracle", generator="generate:x")})


def test_report_main_refuses_the_barred_test_split() -> None:
    from atr.answer.report import main

    with pytest.raises(PermissionError, match="barred"):
        main(["--split", "test"])
