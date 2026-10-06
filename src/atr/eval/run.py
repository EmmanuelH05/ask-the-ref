import argparse
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from atr.cache import JsonCache
from atr.corpus.models import Language
from atr.eval.metrics import score_question
from atr.eval.report import Record, SliceRow, format_rows, latency_summary, summarize
from atr.evalset.freeze import CORPUS_MANIFEST, verify_frozen
from atr.evalset.models import Question, Split, expand
from atr.retrieval.base import Retriever, load_docs
from atr.retrieval.bm25 import BM25Retriever
from atr.retrieval.dense import (
    ENCODERS,
    CachedEncoder,
    DenseRetriever,
    Encoder,
    SentenceTransformerEncoder,
)
from atr.retrieval.translate import Identity, OllamaTranslator, QueryRewriter

TOP_K = 10
RETRIEVERS = {"bm25": None, "dense-en": "bge-small-en", "dense-multi": "me5-small"}
INDEXES: dict[str, tuple[Language, ...]] = {"en": ("en",), "en+es": ("en", "es")}
CACHE_DIR = Path("runs/cache")
WARMUP_QUERY = "warm up"


class RunInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    retriever: str
    rewriter: str
    index: str
    split: Split
    top_k: int
    evalset_hash: str
    corpus_pin: str
    unreviewed_groups: int
    seed: int
    resamples: int
    machine: str
    created_at: str
    questions: int
    scored: int
    latency_ms: dict[str, dict[str, float]]
    table: list[SliceRow]


class _PassageCachedEncoder:
    """Cache passage vectors but always compute query vectors, so query latency stays real."""

    def __init__(self, inner: Encoder, cache_dir: Path) -> None:
        self.name = inner.name
        self._inner = inner
        self._cached = CachedEncoder(inner, cache_dir)

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        return self._inner.encode_queries(texts)

    def encode_passages(self, texts: list[str]) -> NDArray[np.float32]:
        return self._cached.encode_passages(texts)


def check_split_allowed(split: Split, *, unreviewed_groups: int) -> None:
    if split == "test" and unreviewed_groups > 0:
        raise PermissionError(
            f"the test split is barred: {unreviewed_groups} question groups have had no human "
            "review (DECISIONS.md D18). Review the set and re-freeze first."
        )


def run_name(retriever: str, *, translate: bool, index: str, split: Split) -> str:
    return f"{retriever}{'+translate' if translate else ''}__{index}__{split}"


def machine_name() -> str:
    described = platform.platform()
    if platform.system() != "Darwin":
        return described
    result = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
        capture_output=True,
        text=True,
        check=False,
    )
    parts = result.stdout.split("\n")
    if result.returncode != 0 or len(parts) < 2:
        return described
    return f"{parts[0]}, {int(parts[1]) // 2**30} GB RAM, {described}"


def evaluate(
    questions: list[Question],
    retriever: Retriever,
    rewriter: QueryRewriter,
    *,
    split: Split,
    k: int,
) -> list[Record]:
    records: list[Record] = []
    for question in questions:
        query = rewriter.rewrite(question.text, question.language)
        started = time.perf_counter()
        hits = retriever.search(query, k)
        finished = time.perf_counter()
        ranked = [hit.passage_id for hit in hits]
        records.append(
            Record(
                question_id=question.question_id,
                group_id=question.group_id,
                language=question.language,
                type=question.type,
                split=split,
                query=query,
                gold=question.gold,
                ranked=ranked,
                scores=[hit.score for hit in hits],
                latency_ms={
                    "rewrite": rewriter.cost_ms(question.text, question.language),
                    "search": (finished - started) * 1000,
                },
                metrics=score_question(ranked, set(question.gold)) if question.gold else None,
            )
        )
    return records


def build_retriever(key: str, index: str, corpus_dir: Path) -> Retriever:
    docs = load_docs(corpus_dir, INDEXES[index])
    encoder_key = RETRIEVERS[key]
    if encoder_key is None:
        return BM25Retriever(docs, name=key)
    encoder = _PassageCachedEncoder(
        SentenceTransformerEncoder(ENCODERS[encoder_key]), CACHE_DIR / "embeddings"
    )
    return DenseRetriever(docs, encoder, name=key)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one retrieval configuration on one split.")
    parser.add_argument("--retriever", choices=sorted(RETRIEVERS), required=True)
    parser.add_argument("--translate", action="store_true", help="translate es/cs questions first")
    parser.add_argument("--index", choices=sorted(INDEXES), default="en")
    parser.add_argument("--split", choices=["dev", "heldout", "test"], default="dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--questions", type=Path, default=Path("evalset/questions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("evalset/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


def print_run(info: RunInfo) -> None:
    print(f"Run {info.name} | evalset {info.evalset_hash[:12]} | corpus pin {info.corpus_pin[:12]}")
    if info.unreviewed_groups:
        print(f"PROVISIONAL: {info.unreviewed_groups} question groups have had no human review.")
    print(
        f"{info.questions} questions, {info.scored} scored for retrieval "
        f"({info.questions - info.scored} have no gold passage). Top {info.top_k}."
    )
    print(
        "95% intervals: bootstrap over question groups, "
        f"{info.resamples} resamples, seed {info.seed}.\n"
    )
    print(format_rows(info.table))
    print(f"\nLatency per question in ms, on {info.machine}")
    for stage, numbers in info.latency_ms.items():
        print(
            f"  {stage:<8} p50 {numbers['p50']:.2f}  p95 {numbers['p95']:.2f}  n={numbers['n']:.0f}"
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    groups, manifest = verify_frozen(args.questions, args.manifest, CORPUS_MANIFEST)
    check_split_allowed(args.split, unreviewed_groups=manifest.unreviewed_groups)
    questions = [q for g in groups if manifest.splits[g.group_id] == args.split for q in expand(g)]
    retriever = build_retriever(args.retriever, args.index, args.corpus_dir)
    rewriter: QueryRewriter = (
        OllamaTranslator(cache=JsonCache(CACHE_DIR / "translate")) if args.translate else Identity()
    )
    retriever.search(WARMUP_QUERY, TOP_K)
    records = evaluate(questions, retriever, rewriter, split=args.split, k=TOP_K)
    name = run_name(args.retriever, translate=args.translate, index=args.index, split=args.split)
    info = RunInfo(
        name=name,
        retriever=args.retriever,
        rewriter=rewriter.name,
        index=args.index,
        split=args.split,
        top_k=TOP_K,
        evalset_hash=manifest.hash,
        corpus_pin=manifest.corpus_pin,
        unreviewed_groups=manifest.unreviewed_groups,
        seed=args.seed,
        resamples=args.resamples,
        machine=machine_name(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        questions=len(records),
        scored=sum(1 for r in records if r.metrics is not None),
        latency_ms=latency_summary(records),
        table=summarize(records, args.resamples, args.seed),
    )
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "records.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print_run(info)
    print(f"\nWrote {run_dir}/records.jsonl and run.json")


if __name__ == "__main__":
    main()
