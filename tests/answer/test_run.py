from pathlib import Path

from atr.answer.run import context_passages, run_dir_name, serving_config
from atr.corpus.models import Passage, count_words
from atr.eval.refusal import RefusalChoice
from atr.pipeline import PipelineConfig


def passage(passage_id: str, language: str, text: str) -> Passage:
    return Passage.model_validate(
        {
            "passage_id": passage_id,
            "language": language,
            "law_number": 1,
            "law_title": "T",
            "section_index": 1,
            "section_title": "S",
            "subheading": None,
            "source_slug": "s",
            "text": text,
            "word_count": count_words(text),
        }
    )


def choice(source: str) -> RefusalChoice:
    return RefusalChoice.model_validate(
        {
            "track": "T",
            "config": {"label": "final", "dense": "bge-small-en", "translate": True},
            "source": source,
            "threshold": 0.6,
            "auroc_by_source": {"first_stage": 0.8, "reranker": 0.7},
            "balanced_accuracy": 0.8,
            "n_answerable": 10,
            "n_unanswerable": 2,
            "rows": [],
            "curve": [],
        }
    )


def test_serving_config_adds_the_reranker_only_when_phase_6_chose_its_score() -> None:
    final = PipelineConfig(label="final", dense="bge-small-en", translate=True)
    assert serving_config(final, choice("first_stage")) == final
    served = serving_config(final, choice("reranker"))
    assert served.reranker == "bge-m3" and served.dense == "bge-small-en"


def test_context_passages_prefer_the_question_language_and_fall_back_to_english() -> None:
    corpus = {
        ("en", "L1.1"): passage("L1.1", "en", "English one."),
        ("es", "L1.1"): passage("L1.1", "es", "Español uno."),
        ("en", "L1.2"): passage("L1.2", "en", "English two."),
    }
    assert [p.text for p in context_passages(["L1.1", "L1.2"], "es", corpus)] == [
        "Español uno.",
        "English two.",
    ]
    assert [p.text for p in context_passages(["L1.1"], "cs", corpus)] == ["English one."]
    assert context_passages(["L9.9"], "en", corpus) == []


def test_run_dir_name_encodes_system_track_and_split() -> None:
    assert run_dir_name("full", "T", "dev", Path("runs/answers")) == Path(
        "runs/answers/full__T__dev"
    )


def test_answer_oracle_refuses_a_gold_passage_missing_from_the_corpus() -> None:
    import pytest

    from atr.answer.run import answer_oracle, main
    from atr.evalset.models import Question

    question = Question(
        question_id="A.en", group_id="A", language="en", type="plain", text="q?", gold=["L9.9"]
    )
    corpus = {("en", "L1.1"): passage("L1.1", "en", "English one.")}
    with pytest.raises(KeyError, match=r"L9\.9"):
        answer_oracle([question], None, corpus, split="dev")  # type: ignore[arg-type]
    with pytest.raises(PermissionError, match="barred"):
        main(["--system", "closed_book", "--split", "test"])
