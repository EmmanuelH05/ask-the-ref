"""The retrieval pipeline: one configuration object, fixed stage order, a time per stage."""

import hashlib
import json
import time
from collections.abc import Mapping
from typing import Literal, NamedTuple, Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from atr.evalset.models import QuestionLanguage
from atr.llm import Completion
from atr.retrieval.base import Hit, Retriever
from atr.retrieval.chunking import ChunkStrategy
from atr.retrieval.fusion import rrf
from atr.retrieval.rerank import Reranker, rerank
from atr.retrieval.translate import QueryRewriter

# Fixed before any phase 4 run and not tuned (DECISIONS.md D29).
RRF_C = 60
FIRST_STAGE_DEPTH = 50
RERANK_DEPTH = 20


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    index: Literal["en", "en+es"] = "en"
    translate: bool = False
    dense: Literal["bge-small-en", "me5-small"] | None = None
    bm25: bool = False
    title_prefix: bool = False
    chunking: ChunkStrategy = "passage"
    reranker: Literal["minilm", "bge-m3"] | None = None
    rewrite: bool = False
    decompose: bool = False

    @model_validator(mode="after")
    def _has_a_retriever(self) -> Self:
        if self.dense is None and not self.bm25:
            raise ValueError("a pipeline needs BM25, a dense retriever, or both")
        return self

    @property
    def config_id(self) -> str:
        """Identifies the settings, not the label, so a cached run cannot be misattributed."""
        settings = self.model_dump(exclude={"label"})
        canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


class Expander(Protocol):
    def expand(self, text: str) -> Completion: ...


class Splitter(Protocol):
    def split(self, text: str) -> tuple[list[str], float]: ...


class PipelineResult(NamedTuple):
    hits: list[Hit]
    queries: list[str]
    timings: dict[str, float]


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        retrievers: list[Retriever],
        translator: QueryRewriter | None = None,
        rewriter: Expander | None = None,
        decomposer: Splitter | None = None,
        reranker: Reranker | None = None,
        rerank_texts: Mapping[str, list[str]] | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("a pipeline needs at least one retriever")
        if reranker is not None and rerank_texts is None:
            raise ValueError("a reranker needs the texts to score")
        self.config = config
        self._retrievers = retrievers
        self._translator = translator
        self._rewriter = rewriter
        self._decomposer = decomposer
        self._reranker = reranker
        self._rerank_texts = rerank_texts or {}

    def search(self, text: str, language: QuestionLanguage, k: int) -> PipelineResult:
        timings: dict[str, float] = {}
        question = text
        if self._translator is not None:
            question = self._translator.rewrite(text, language)
            timings["translate"] = self._translator.cost_ms(text, language)
        queries = self._queries(question, timings)
        rankings = [
            self._timed_search(retriever, query, timings)
            for query in queries
            for retriever in self._retrievers
        ]
        candidates = rankings[0]
        if len(rankings) > 1:
            started = time.perf_counter()
            candidates = rrf(rankings, c=RRF_C)
            timings["fuse"] = (time.perf_counter() - started) * 1000
        if self._reranker is None:
            return PipelineResult(candidates[:k], queries, timings)
        started = time.perf_counter()
        pairs = [
            (hit.passage_id, chunk)
            for hit in candidates[:RERANK_DEPTH]
            for chunk in self._rerank_texts[hit.passage_id]
        ]
        # The question as asked (or translated), not the expanded search query: the
        # cross-encoder judges relevance to what the person wants to know.
        hits = rerank(question, pairs, self._reranker, k)
        timings["rerank"] = (time.perf_counter() - started) * 1000
        return PipelineResult(hits, queries, timings)

    def _queries(self, question: str, timings: dict[str, float]) -> list[str]:
        search_query = question
        if self._rewriter is not None:
            expanded = self._rewriter.expand(question)
            search_query, timings["rewrite"] = expanded.text, expanded.ms
        if self._decomposer is None:
            return [search_query]
        parts, timings["decompose"] = self._decomposer.split(question)
        return [search_query, *parts]

    def _timed_search(
        self, retriever: Retriever, query: str, timings: dict[str, float]
    ) -> list[Hit]:
        started = time.perf_counter()
        hits = retriever.search(query, FIRST_STAGE_DEPTH)
        elapsed = (time.perf_counter() - started) * 1000
        timings[retriever.name] = timings.get(retriever.name, 0.0) + elapsed
        return hits
