"""Answer every question of a split with one of the three systems (D41).

closed_book: no passages, the model's memory (evaluation only).
full: the track's final retrieval pipeline, the phase 6 refusal rule, then generation.
oracle: the gold passages handed to the generator, the upper bound.
"""

import argparse
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from atr.answer.generate import Answer, Generator, split_claims
from atr.answer.records import SYSTEMS, AnswerRecord, System, write_answers
from atr.cache import JsonCache
from atr.corpus.models import Language, Passage
from atr.eval.ablation import RANKING_SOURCES, Workbench, code_stamp
from atr.eval.refusal import RefusalChoice, load_choice
from atr.eval.run import CACHE_DIR, check_split_allowed, machine_name
from atr.evalset.freeze import CORPUS_MANIFEST, verify_frozen
from atr.evalset.models import Question, QuestionLanguage, Split, expand
from atr.llm import OllamaLLM
from atr.pipeline import PipelineConfig
from atr.retrieval.docs import load_passages

CONTEXT_K = 5
ANSWERS_DIR = Path("runs/answers")
# Everything that can change a ranking, plus the prompts: two answer runs are comparable only
# when they share this stamp (D44).
ANSWER_SOURCES = [*RANKING_SOURCES, Path(__file__).with_name("generate.py")]
Corpus = Mapping[tuple[Language, str], Passage]


class AnswerRunMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system: System
    track: str
    split: Split
    serving_config: PipelineConfig | None
    refusal_source: str | None
    threshold: float | None
    generator: str
    code_stamp: str
    evalset_hash: str
    corpus_pin: str
    machine: str
    created_at: str


def serving_config(final: PipelineConfig, choice: RefusalChoice) -> PipelineConfig:
    """The pipeline the user gets: the final config, plus the reranker if its score is the gate."""
    if choice.source == "reranker":
        return final.model_copy(
            update={"reranker": "bge-m3", "label": f"{final.label} +refusal reranker"}
        )
    return final


def context_passages(ids: list[str], language: QuestionLanguage, corpus: Corpus) -> list[Passage]:
    """Passages in the question's language where the edition has them, else English."""
    preferred: Language = "es" if language == "es" else "en"
    found: list[Passage] = []
    for passage_id in ids:
        passage = corpus.get((preferred, passage_id)) or corpus.get(("en", passage_id))
        if passage is not None:
            found.append(passage)
    return found


def run_dir_name(system: System, track: str, split: Split, root: Path) -> Path:
    return root / f"{system}__{track}__{split}"


def load_corpus(corpus_dir: Path) -> dict[tuple[Language, str], Passage]:
    return {(p.language, p.passage_id): p for p in load_passages(corpus_dir, ("en", "es"))}


def _record(
    question: Question,
    *,
    split: Split,
    system: System,
    retrieved: list[str],
    top_score: float | None,
    threshold: float | None,
    answer: Answer,
    allowed: set[str],
    latency_ms: dict[str, float],
) -> AnswerRecord:
    return AnswerRecord(
        question_id=question.question_id,
        group_id=question.group_id,
        language=question.language,
        type=question.type,
        split=split,
        system=system,
        question=question.text,
        gold=list(question.gold),
        retrieved=retrieved,
        top_score=top_score,
        threshold=threshold,
        refused=answer.refused,
        answer=answer.text,
        citations=answer.citations,
        claims=[] if answer.refused else split_claims(answer.text, allowed),
        latency_ms={**latency_ms, "generate": answer.ms},
    )


def answer_closed_book(
    questions: list[Question], generator: Generator, *, split: Split, all_ids: set[str]
) -> list[AnswerRecord]:
    records = []
    for question in questions:
        answer = generator.closed_book(question.text, question.language, allowed=all_ids)
        records.append(
            _record(
                question,
                split=split,
                system="closed_book",
                retrieved=[],
                top_score=None,
                threshold=None,
                answer=answer,
                allowed=all_ids,
                latency_ms={},
            )
        )
    return records


def answer_oracle(
    questions: list[Question], generator: Generator, corpus: Corpus, *, split: Split
) -> list[AnswerRecord]:
    records = []
    for question in questions:
        missing = [g for g in question.gold if ("en", g) not in corpus]
        if missing:
            # A silent drop would turn a corpus gap into a generation failure of the upper bound.
            raise KeyError(f"{question.question_id}: gold passage(s) {missing} not in the corpus")
        passages = context_passages(question.gold, question.language, corpus)
        answer = generator.answer(question.text, question.language, passages)
        records.append(
            _record(
                question,
                split=split,
                system="oracle",
                retrieved=list(question.gold),
                top_score=None,
                threshold=None,
                answer=answer,
                allowed=set(question.gold),
                latency_ms={},
            )
        )
    return records


def answer_full(
    questions: list[Question],
    generator: Generator,
    corpus: Corpus,
    workbench: Workbench,
    config: PipelineConfig,
    choice: RefusalChoice,
    *,
    split: Split,
) -> list[AnswerRecord]:
    pipeline = workbench.build(config)
    for question in questions:
        pipeline.prime(question.text, question.language)
    pipeline.search("warm up", "en", CONTEXT_K)
    records = []
    for question in questions:
        result = pipeline.search(question.text, question.language, CONTEXT_K)
        ids = [hit.passage_id for hit in result.hits]
        top = result.hits[0].score if result.hits else None
        passages = (
            context_passages(ids, question.language, corpus)
            if top is not None and top >= choice.threshold
            else []
        )
        answer = generator.answer(question.text, question.language, passages)
        records.append(
            _record(
                question,
                split=split,
                system="full",
                retrieved=ids,
                top_score=top,
                threshold=choice.threshold,
                answer=answer,
                allowed=set(ids),
                latency_ms=dict(result.timings),
            )
        )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer a split with one system.")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--track", choices=["T", "M"], default="T")
    parser.add_argument("--split", choices=["dev", "heldout", "test"], default="dev")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=ANSWERS_DIR)
    args = parser.parse_args(argv)
    groups, manifest = verify_frozen(
        Path("evalset/questions.jsonl"), Path("evalset/manifest.json"), CORPUS_MANIFEST
    )
    check_split_allowed(
        args.split,
        unreviewed_groups=sum(1 for g in groups if not g.spanish_reviewed),
        manifest_unreviewed=manifest.unreviewed_groups,
    )
    questions = [q for g in groups if manifest.splits[g.group_id] == args.split for q in expand(g)]
    corpus = load_corpus(args.corpus_dir)
    all_ids = {pid for (lang, pid) in corpus if lang == "en"}
    generator = Generator(OllamaLLM(cache=JsonCache(CACHE_DIR / "llm")))
    config: PipelineConfig | None = None
    choice: RefusalChoice | None = None
    print(f"answering {len(questions)} {args.split} questions with {args.system} ...", flush=True)
    if args.system == "closed_book":
        records = answer_closed_book(questions, generator, split=args.split, all_ids=all_ids)
    elif args.system == "oracle":
        records = answer_oracle(questions, generator, corpus, split=args.split)
    else:
        choice = load_choice(args.track)
        config = serving_config(choice.config, choice)
        workbench = Workbench(questions, manifest, args.split, Path("runs/ablation"))
        records = answer_full(
            questions, generator, corpus, workbench, config, choice, split=args.split
        )
    run_dir = run_dir_name(args.system, args.track, args.split, args.out)
    write_answers(records, run_dir)
    meta = AnswerRunMeta(
        system=args.system,
        track=args.track,
        split=args.split,
        serving_config=config,
        refusal_source=choice.source if choice else None,
        threshold=choice.threshold if choice else None,
        generator=generator.name,
        code_stamp=code_stamp(ANSWER_SOURCES),
        evalset_hash=manifest.hash,
        corpus_pin=manifest.corpus_pin,
        machine=machine_name(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (run_dir / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    refused = sum(1 for r in records if r.refused)
    print(f"wrote {run_dir}: {len(records)} answers, {refused} refusals")


if __name__ == "__main__":
    main()
