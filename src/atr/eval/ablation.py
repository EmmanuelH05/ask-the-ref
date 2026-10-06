"""Phase 4: the pre-registered chain of experiments (DECISIONS.md D29) and its table."""

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from atr.cache import JsonCache
from atr.corpus.build import CorpusManifest
from atr.eval.bootstrap import Interval, paired_difference
from atr.eval.metrics import score_question
from atr.eval.report import (
    Record,
    format_rows,
    latency_summary,
    load_records,
    nearest_rank,
    summarize,
)
from atr.eval.run import CACHE_DIR, INDEXES, TOP_K, PassageCachedEncoder, machine_name
from atr.evalset.freeze import CORPUS_MANIFEST, EvalManifest, verify_frozen
from atr.evalset.models import Question, Split, expand
from atr.llm import OllamaLLM
from atr.pipeline import Pipeline, PipelineConfig
from atr.retrieval.base import Retriever
from atr.retrieval.bm25 import BM25Retriever
from atr.retrieval.decompose import Decomposer
from atr.retrieval.dense import ENCODERS, DenseRetriever, Encoder, SentenceTransformerEncoder
from atr.retrieval.docs import build_docs, load_passages, verify_passages
from atr.retrieval.rerank import CrossEncoderReranker, Reranker
from atr.retrieval.rewrite import RulebookRewriter
from atr.retrieval.translate import OllamaTranslator

PRIMARY = "recall_at_5"
Track = str
ALL_TRACKS: tuple[Track, ...] = ("T", "M")


class Variant(NamedTuple):
    name: str
    changes: dict[str, object]
    tracks: tuple[Track, ...] = ALL_TRACKS


class Step(NamedTuple):
    name: str
    tracks: tuple[Track, ...]
    variants: tuple[Variant, ...]


STARTS: dict[Track, PipelineConfig] = {
    "T": PipelineConfig(label="T0 dense-en + translate", translate=True, dense="bge-small-en"),
    "M": PipelineConfig(label="M0 dense-multi", dense="me5-small"),
}
STEPS: tuple[Step, ...] = (
    Step("both-editions", ("M",), (Variant("en+es", {"index": "en+es"}),)),
    Step("hybrid", ALL_TRACKS, (Variant("rrf", {"bm25": True}),)),
    Step("title-prefix", ALL_TRACKS, (Variant("titles", {"title_prefix": True}),)),
    Step(
        "reranker",
        ALL_TRACKS,
        (
            Variant("minilm", {"reranker": "minilm"}, ("T",)),
            Variant("bge-m3", {"reranker": "bge-m3"}),
        ),
    ),
    Step(
        "chunking",
        ALL_TRACKS,
        (Variant("blocks", {"chunking": "blocks"}), Variant("items", {"chunking": "items"})),
    ),
    Step("rewrite", ALL_TRACKS, (Variant("rulebook", {"rewrite": True}),)),
    Step("decompose", ALL_TRACKS, (Variant("subquestions", {"decompose": True}),)),
)


class Trial(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    track: Track
    step: str
    variant: str
    config: PipelineConfig
    base_label: str
    base_config_id: str
    base_recall_at_5: float
    recall_at_5: float
    difference: Interval
    p50_ms: float
    p95_ms: float
    kept: bool


def total_ms(record: Record) -> float:
    return sum(record.latency_ms.values())


def _percentiles(records: list[Record]) -> tuple[float, float]:
    ordered = sorted(total_ms(r) for r in records)
    return nearest_rank(ordered, 50), nearest_rank(ordered, 95)


def _primary(records: list[Record]) -> dict[str, float]:
    return {r.question_id: getattr(r.metrics, PRIMARY) for r in records if r.metrics is not None}


def _mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def choose(
    base: list[Record], candidates: dict[str, list[Record]], resamples: int, seed: int
) -> tuple[str | None, dict[str, Interval]]:
    """The variant to keep, if any, and every variant's paired difference against the base.

    A variant passes when its whole 95% interval is above zero. Among those that pass, the
    highest recall@5 wins and a tie goes to the lower median latency.
    """
    base_values = _primary(base)
    groups = {r.question_id: r.group_id for r in base}
    intervals = {
        name: paired_difference(
            base_values, _primary(records), clusters=groups, resamples=resamples, seed=seed
        )
        for name, records in candidates.items()
    }
    passing = [name for name, interval in intervals.items() if interval.low > 0]
    if not passing:
        return None, intervals
    best = min(
        passing,
        key=lambda name: (-_mean(_primary(candidates[name])), _percentiles(candidates[name])[0]),
    )
    return best, intervals


def run_chain(
    track: Track,
    start: PipelineConfig,
    steps: tuple[Step, ...] | list[Step],
    evaluate: Callable[[PipelineConfig], list[Record]],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[Trial], PipelineConfig]:
    current, trials = start, []
    for number, step in enumerate(steps):
        variants = [v for v in step.variants if track in v.tracks]
        if track not in step.tracks or not variants:
            continue
        base = evaluate(current)
        configs = {
            v.name: current.model_copy(
                update={**v.changes, "label": f"{track}{number} +{step.name}:{v.name}"}
            )
            for v in variants
        }
        results = {name: evaluate(config) for name, config in configs.items()}
        winner, intervals = choose(base, results, resamples, seed)
        for name, config in configs.items():
            p50, p95 = _percentiles(results[name])
            trials.append(
                Trial(
                    track=track,
                    step=step.name,
                    variant=name,
                    config=config,
                    base_label=current.label,
                    base_config_id=current.config_id,
                    base_recall_at_5=_mean(_primary(base)),
                    recall_at_5=_mean(_primary(results[name])),
                    difference=intervals[name],
                    p50_ms=p50,
                    p95_ms=p95,
                    kept=name == winner,
                )
            )
        if winner is not None:
            current = configs[winner]
    return trials, current


def evaluate_pipeline(
    questions: list[Question], pipeline: Pipeline, *, split: Split, k: int
) -> list[Record]:
    records = []
    for question in questions:
        result = pipeline.search(question.text, question.language, k)
        ranked = [hit.passage_id for hit in result.hits]
        records.append(
            Record(
                question_id=question.question_id,
                group_id=question.group_id,
                language=question.language,
                type=question.type,
                split=split,
                query=" | ".join(result.queries),
                gold=question.gold,
                ranked=ranked,
                scores=[hit.score for hit in result.hits],
                latency_ms=result.timings,
                metrics=score_question(ranked, set(question.gold)) if question.gold else None,
            )
        )
    return records


class RunMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config: PipelineConfig
    split: Split
    evalset_hash: str
    corpus_pin: str
    machine: str
    created_at: str


class Workbench:
    """Builds pipelines, reusing loaded models, and caches each configuration's records."""

    def __init__(
        self, questions: list[Question], manifest: EvalManifest, split: Split, out: Path
    ) -> None:
        self._questions = questions
        self._manifest = manifest
        self._split = split
        self._out = out
        self._encoders: dict[str, Encoder] = {}
        self._rerankers: dict[str, Reranker] = {}
        corpus = CorpusManifest.model_validate_json(CORPUS_MANIFEST.read_text(encoding="utf-8"))
        self._pins: dict[tuple[str, str], str] = {
            (e.language, e.passage_id): e.sha256 for e in corpus.passages
        }
        self._llm = OllamaLLM(cache=JsonCache(CACHE_DIR / "llm"))

    def run_dir(self, config: PipelineConfig) -> Path:
        return self._out / f"{config.config_id}__{self._split}"

    def evaluate(self, config: PipelineConfig) -> list[Record]:
        run_dir = self.run_dir(config)
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = RunMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
            current = (self._manifest.hash, self._manifest.corpus_pin)
            if (meta.evalset_hash, meta.corpus_pin) == current:
                return load_records(run_dir)
        print(f"  running {config.label} ({config.config_id}) ...", flush=True)
        pipeline = self._build(config)
        pipeline.search("warm up", "en", TOP_K)
        records = evaluate_pipeline(self._questions, pipeline, split=self._split, k=TOP_K)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "records.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
        )
        meta = RunMeta(
            config=config,
            split=self._split,
            evalset_hash=self._manifest.hash,
            corpus_pin=self._manifest.corpus_pin,
            machine=machine_name(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        meta_path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return records

    def _encoder(self, key: str) -> Encoder:
        if key not in self._encoders:
            self._encoders[key] = PassageCachedEncoder(
                SentenceTransformerEncoder(ENCODERS[key]), CACHE_DIR / "embeddings"
            )
        return self._encoders[key]

    def _build(self, config: PipelineConfig) -> Pipeline:
        passages = load_passages(Path("data/corpus"), INDEXES[config.index])
        verify_passages(passages, self._pins)
        plain = build_docs(passages, title_prefix=False, chunking=config.chunking)
        retrievers: list[Retriever] = []
        if config.bm25:
            retrievers.append(BM25Retriever(plain, name="bm25"))
        if config.dense is not None:
            docs = build_docs(passages, title_prefix=config.title_prefix, chunking=config.chunking)
            retrievers.append(DenseRetriever(docs, self._encoder(config.dense), name="dense"))
        reranker = None
        if config.reranker is not None:
            if config.reranker not in self._rerankers:
                self._rerankers[config.reranker] = CrossEncoderReranker(config.reranker)
            reranker = self._rerankers[config.reranker]
        rerank_texts: dict[str, list[str]] = {}
        for doc in plain:
            rerank_texts.setdefault(doc.passage_id, []).append(doc.text)
        return Pipeline(
            config,
            retrievers=retrievers,
            translator=OllamaTranslator(cache=JsonCache(CACHE_DIR / "translate"))
            if config.translate
            else None,
            rewriter=RulebookRewriter(self._llm) if config.rewrite else None,
            decomposer=Decomposer(self._llm) if config.decompose else None,
            reranker=reranker,
            rerank_texts=rerank_texts,
        )


def format_trials(trials: list[Trial]) -> str:
    lines = [
        f"{'step':<15}{'variant':<14}{'recall@5':>9}{'base':>7}  {'difference [95% interval]':<27}"
        f"{'p50 ms':>9}{'p95 ms':>9}  decision"
    ]
    for t in trials:
        d = t.difference
        lines.append(
            f"{t.step:<15}{t.variant:<14}{t.recall_at_5:>9.2f}{t.base_recall_at_5:>7.2f}  "
            f"{f'{d.mean:+.2f} [{d.low:+.2f}, {d.high:+.2f}]':<27}"
            f"{t.p50_ms:>9.1f}{t.p95_ms:>9.1f}  {'KEPT' if t.kept else 'dropped'}"
        )
    return "\n".join(lines)


def _stage_lines(records: list[Record]) -> list[str]:
    return [
        f"  {stage:<10} p50 {numbers['p50']:>8.2f}  p95 {numbers['p95']:>8.2f}"
        for stage, numbers in latency_summary(records).items()
    ]


def build_report(workbench: Workbench, resamples: int, seed: int, machine: str) -> str:
    parts = [
        "Ablation on dev. Primary metric: recall@5, all scored questions (DECISIONS.md D29).",
        "A change is KEPT only when the 95% interval of its paired difference against the "
        "current best is entirely above zero.",
        f"Paired bootstrap over question groups, {resamples} resamples, seed {seed}. "
        f"Latency is per question, all stages summed, on {machine}.",
    ]
    for track, start in STARTS.items():
        trials, final = run_chain(
            track, start, STEPS, workbench.evaluate, resamples=resamples, seed=seed
        )
        for title, config in ((f"start: {start.label}", start), (f"final: {final.label}", final)):
            records = workbench.evaluate(config)
            p50, p95 = _percentiles(records)
            parts.append(f"\n=== Track {track}, {title} ({config.config_id}) ===")
            parts.append(format_rows(summarize(records, resamples, seed)))
            parts.append(f"latency ms: total p50 {p50:.1f}, p95 {p95:.1f}")
            parts.extend(_stage_lines(records))
            if config is start:
                parts.append(f"\n--- Track {track} experiments, in order ---")
                parts.append(format_trials(trials))
        settings = final.model_dump(exclude={"label"})
        parts.append(f"final settings: {settings}")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run or regenerate the phase 4 ablation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--report", type=Path, default=Path("docs/ablation.txt"))
    args = parser.parse_args(argv)
    groups, manifest = verify_frozen(
        Path("evalset/questions.jsonl"), Path("evalset/manifest.json"), CORPUS_MANIFEST
    )
    questions = [q for g in groups if manifest.splits[g.group_id] == "dev" for q in expand(g)]
    workbench = Workbench(questions, manifest, "dev", args.out)
    started = time.perf_counter()
    report = build_report(workbench, args.resamples, args.seed, machine_name())
    header = (
        "Generated by `uv run python -m atr.eval.ablation`. Do not edit by hand.\n"
        f"Question set {manifest.hash[:12]}, {manifest.unreviewed_groups} of "
        f"{len(groups)} groups without a human review.\n\n"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(header + report + "\n", encoding="utf-8")
    print(header + report)
    print(f"\nWrote {args.report} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
