import pytest

from atr.llm import Completion
from atr.pipeline import Pipeline, PipelineConfig
from atr.retrieval.base import Hit


class Fixed:
    def __init__(self, name: str, ids: list[str]) -> None:
        self.name = name
        self.ids = ids
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append((query, k))
        return [Hit(i, float(len(self.ids) - n)) for n, i in enumerate(self.ids)][:k]


class Upper:
    name = "translate:fake"

    def rewrite(self, text: str, language: str) -> str:
        return text if language == "en" else text.upper()

    def cost_ms(self, text: str, language: str) -> float:
        return 0.0 if language == "en" else 900.0


class Expand:
    def expand(self, text: str) -> Completion:
        return Completion(f"{text}\nformal words", 30.0)


class Split:
    def split(self, text: str) -> tuple[list[str], float]:
        return ["part one?", "part two?"], 50.0


class ByLength:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return [float(len(text)) for text in texts]


def config(**overrides: object) -> PipelineConfig:
    fields: dict[str, object] = {"label": "t", "dense": "me5-small"}
    return PipelineConfig.model_validate({**fields, **overrides})


def test_config_id_ignores_the_label_and_changes_with_any_setting() -> None:
    assert config().config_id == config(label="other").config_id
    assert config().config_id != config(bm25=True).config_id
    assert len(config().config_id) == 10


def test_config_needs_at_least_one_retriever() -> None:
    with pytest.raises(ValueError):
        config(dense=None, bm25=False)


def test_single_retriever_passes_through_and_times_its_stage() -> None:
    dense = Fixed("dense", ["L1.1", "L1.2", "L1.3"])
    result = Pipeline(config(), retrievers=[dense]).search("q?", "en", 2)
    assert [h.passage_id for h in result.hits] == ["L1.1", "L1.2"]
    assert result.queries == ["q?"]
    assert dense.queries == [("q?", 50)]
    assert set(result.timings) == {"dense"}


def test_translation_changes_the_query_and_reports_its_stored_cost() -> None:
    dense = Fixed("dense", ["L1.1"])
    pipeline = Pipeline(config(translate=True), retrievers=[dense], translator=Upper())
    result = pipeline.search("hola?", "es", 5)
    assert dense.queries == [("HOLA?", 50)]
    assert result.timings["translate"] == 900.0


def test_two_retrievers_are_fused_by_rank() -> None:
    bm25 = Fixed("bm25", ["L1.1", "L1.2", "L1.3"])
    dense = Fixed("dense", ["L1.3", "L1.2", "L1.1"])
    result = Pipeline(config(bm25=True), retrievers=[bm25, dense]).search("q?", "en", 3)
    # L1.1 and L1.3 each get 1/61 + 1/63; L1.2 gets 2/62, which is smaller. Ties go by id.
    assert [h.passage_id for h in result.hits] == ["L1.1", "L1.3", "L1.2"]
    assert result.hits[0].score == pytest.approx(1 / 61 + 1 / 63)
    assert {"bm25", "dense", "fuse"} <= set(result.timings)


def test_rewrite_expands_the_search_query_but_not_the_rerank_query() -> None:
    dense = Fixed("dense", ["L1.1", "L1.2"])
    reranker = ByLength()
    pipeline = Pipeline(
        config(rewrite=True, reranker="minilm"),
        retrievers=[dense],
        rewriter=Expand(),
        reranker=reranker,
        rerank_texts={"L1.1": ["short"], "L1.2": ["a much longer chunk", "tiny"]},
    )
    result = pipeline.search("q?", "en", 2)
    assert dense.queries == [("q?\nformal words", 50)]
    assert reranker.calls == [("q?", ["short", "a much longer chunk", "tiny"])]
    assert [h.passage_id for h in result.hits] == ["L1.2", "L1.1"]
    assert result.timings["rewrite"] == 30.0 and "rerank" in result.timings


def test_decomposition_searches_the_whole_question_and_each_part() -> None:
    dense = Fixed("dense", ["L1.1"])
    pipeline = Pipeline(config(decompose=True), retrievers=[dense], decomposer=Split())
    result = pipeline.search("both?", "en", 5)
    assert [q for q, _ in dense.queries] == ["both?", "part one?", "part two?"]
    assert result.queries == ["both?", "part one?", "part two?"]
    assert result.timings["decompose"] == 50.0


def test_reranker_only_sees_the_top_candidates() -> None:
    ids = [f"L1.{n}" for n in range(1, 31)]
    reranker = ByLength()
    pipeline = Pipeline(
        config(reranker="minilm"),
        retrievers=[Fixed("dense", ids)],
        reranker=reranker,
        rerank_texts={i: [i] for i in ids},
    )
    pipeline.search("q?", "en", 10)
    assert len(reranker.calls[0][1]) == 20
