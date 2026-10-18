"""Phase 8: every failure gets one cause from a fixed list, by rules in a fixed order (D42)."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from atr.answer.judge import ACCEPTED_PATH, labels_from_judgments, load_judgments
from atr.answer.records import AnswerRecord, load_answers
from atr.answer.run import serving_config
from atr.eval.ablation import Workbench
from atr.eval.refusal import load_choice
from atr.eval.report import Record
from atr.eval.run import check_split_allowed
from atr.evalset.freeze import CORPUS_MANIFEST, verify_frozen
from atr.evalset.models import QUESTION_LANGUAGES, QuestionLanguage, QuestionType, Split, expand
from atr.evalset.overlap import overlap
from atr.retrieval.docs import load_passages
from atr.text import fold

Cause = Literal[
    "vocabulary mismatch",
    "needs two passages",
    "table or list lookup",
    "glossary term",
    "negation or exception",
    "language mismatch",
    "wrong refusal",
    "unsupported claim",
]
CAUSES: tuple[Cause, ...] = (
    "vocabulary mismatch",
    "needs two passages",
    "table or list lookup",
    "glossary term",
    "negation or exception",
    "language mismatch",
    "wrong refusal",
    "unsupported claim",
)
# Folded (lowercase, accent-free) words that mark a rule stated as a negation or an exception.
NEGATION_MARKERS = frozenset(
    (
        "not",
        "unless",
        "except",
        "exception",
        "only",
        "never",
        "cannot",
        "no",
        "salvo",
        "excepto",
        "solo",
        "nunca",
    )
)
# "a menos (que)" is checked as a phrase; the article "a" on its own is in nearly every text.
REPORT_PATH = Path("docs/errors.txt")
EXAMPLES_PATH = Path("docs/errors/examples.md")
OVERRIDES_PATH = Path("docs/errors/overrides.json")
ClaimKey = tuple[str, int]


class Failure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    group_id: str
    language: QuestionLanguage
    type: QuestionType
    question: str
    gold: list[str]
    top5: list[str]
    overlap: float
    cause: Cause
    evidence: str


def is_list_dominated(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    items = sum(1 for line in lines if line.lstrip().startswith("- "))
    return items * 2 >= len(lines)


def has_negation(text: str) -> bool:
    """True when the text holds a negation or exception marker, in either language."""
    words = set(fold(text).replace(",", " ").replace(".", " ").split())
    return bool(words & NEGATION_MARKERS) or "a menos" in fold(text)


def _hit(record: Record) -> bool:
    return record.metrics is not None and record.metrics.recall_at_5 > 0


def _retrieval_failed(record: Record) -> bool:
    return record.metrics is not None and record.metrics.recall_at_5 < 1


def _answer_failed(
    record: Record, answer: AnswerRecord | None, labels: Mapping[ClaimKey, bool] | None
) -> str | None:
    """Why the answer stage failed, or None when it did not (or there is no answer record)."""
    if answer is None:
        return None
    if record.type == "out_of_scope":
        return "answered an out-of-scope question" if not answer.refused else None
    if answer.refused:
        return f"refused an answerable question (top score {answer.top_score})"
    if record.gold and not any(c in record.gold for c in answer.citations):
        return f"cited {answer.citations or 'nothing'}, gold is {record.gold}"
    if labels is not None:
        unsupported = [
            i for i in range(len(answer.claims)) if labels.get((answer.question_id, i)) is False
        ]
        if unsupported:
            return f"judge marked claim(s) {unsupported} unsupported"
    return None


def _label(
    record: Record,
    answer: AnswerRecord | None,
    labels: Mapping[ClaimKey, bool] | None,
    texts: Mapping[str, str],
    english_hit: bool,
    lexical: float,
) -> tuple[Cause, str] | None:
    problem = _answer_failed(record, answer, labels)
    retrieval_failed = _retrieval_failed(record)
    if problem is None and not retrieval_failed:
        return None
    if problem is not None and ("refused" in problem or "out-of-scope" in problem):
        return "wrong refusal", problem
    if problem is not None and _hit(record):
        return "unsupported claim", problem
    if record.language != "en" and english_hit:
        return "language mismatch", "the English version of this question found its gold passage"
    recall = record.metrics.recall_at_5 if record.metrics is not None else 0.0
    if record.type == "multi_rule" and 0 < recall < 1:
        return "needs two passages", f"found {recall:.2f} of the gold passages in the top 5"
    if any(g.startswith("G.") for g in record.gold):
        return "glossary term", "the gold passage is a glossary entry"
    gold_text = " ".join(texts.get(g, "") for g in record.gold)
    if any(is_list_dominated(texts.get(g, "")) for g in record.gold):
        return "table or list lookup", "the gold passage is mostly list items"
    if has_negation(record.query) or has_negation(gold_text):
        return (
            "negation or exception",
            "the question or the rule is stated as a negation or exception",
        )
    return (
        "vocabulary mismatch",
        f"question shares {lexical:.2f} of its content words with the gold",
    )


def label_failures(
    records: list[Record],
    answers: Mapping[str, AnswerRecord] | None,
    labels: Mapping[ClaimKey, bool] | None,
    texts: Mapping[str, str],
    *,
    overrides: Mapping[str, str] | None = None,
) -> list[Failure]:
    english_hits = {r.group_id: _hit(r) for r in records if r.language == "en"}
    if answers is not None:
        unanswered = sorted(r.question_id for r in records if r.question_id not in answers)
        if unanswered:
            raise ValueError(
                f"{len(unanswered)} retrieval records have no answer record ({unanswered[:3]}); "
                "the answer run is for another question set"
            )
    unused = set(overrides or ()) - {r.question_id for r in records}
    if unused:
        raise ValueError(f"overrides name unknown question ids: {sorted(unused)}")
    failures: list[Failure] = []
    for record in records:
        answer = answers.get(record.question_id) if answers is not None else None
        question = answer.question if answer is not None else record.query
        lexical = overlap(question, [texts[g] for g in record.gold if g in texts])
        labelled = _label(
            record, answer, labels, texts, english_hits.get(record.group_id, False), lexical
        )
        if labelled is None:
            continue
        cause, evidence = labelled
        if overrides and record.question_id in overrides:
            forced = overrides[record.question_id]
            if forced not in CAUSES:
                raise ValueError(f"override for {record.question_id} is not a cause: {forced!r}")
            cause, evidence = forced, f"override by mannie (rule said: {cause})"
        failures.append(
            Failure(
                question_id=record.question_id,
                group_id=record.group_id,
                language=record.language,
                type=record.type,
                question=question,
                gold=list(record.gold),
                top5=list(record.ranked[:5]),
                overlap=lexical,
                cause=cause,
                evidence=evidence,
            )
        )
    return failures


def counts_by_language(failures: list[Failure]) -> dict[str, dict[str, int]]:
    return {
        cause: {
            lang: sum(1 for f in failures if f.cause == cause and f.language == lang)
            for lang in QUESTION_LANGUAGES
        }
        for cause in CAUSES
    }


def format_counts(counts: Mapping[str, Mapping[str, int]]) -> str:
    lines = [
        f"{'cause':<24}" + "".join(f"{lang:>6}" for lang in QUESTION_LANGUAGES) + f"{'total':>7}"
    ]
    for cause in CAUSES:
        row = counts[cause]
        lines.append(
            f"{cause:<24}"
            + "".join(f"{row[lang]:>6}" for lang in QUESTION_LANGUAGES)
            + f"{sum(row.values()):>7}"
        )
    totals = {lang: sum(counts[c][lang] for c in CAUSES) for lang in QUESTION_LANGUAGES}
    lines.append(
        f"{'total':<24}"
        + "".join(f"{totals[lang]:>6}" for lang in QUESTION_LANGUAGES)
        + f"{sum(totals.values()):>7}"
    )
    return "\n".join(lines)


def pick_examples(failures: list[Failure], *, n: int) -> list[Failure]:
    """Round-robin over causes; within a cause the highest lexical overlap first.

    A failure whose question shares many words with its gold passage is the instructive kind:
    the words were there and retrieval still missed, so the cause is not just vocabulary.
    """
    queues = {
        cause: sorted(
            (f for f in failures if f.cause == cause), key=lambda f: (-f.overlap, f.question_id)
        )
        for cause in CAUSES
    }
    picked: list[Failure] = []
    while len(picked) < n and any(queues.values()):
        for cause in CAUSES:
            if queues[cause] and len(picked) < n:
                picked.append(queues[cause].pop(0))
    return picked


def render_examples(examples: list[Failure], texts: Mapping[str, str]) -> str:
    lines = ["# Failure examples", "", "Picked by `uv run python -m atr.eval.errors`.", ""]
    for number, f in enumerate(examples, start=1):
        lines += [
            f"## {number}. {f.cause} ({f.question_id}, {f.type}, {f.language})",
            "",
            f"**Question:** {f.question}",
            "",
            f"**Gold:** {', '.join(f.gold) or 'none (out of scope)'}  ",
            f"**Top 5 retrieved:** {', '.join(f.top5)}  ",
            f"**Word overlap with gold:** {f.overlap:.2f}  ",
            f"**Why this label:** {f.evidence}",
            "",
        ]
        for g in f.gold:
            if g in texts:
                lines += [f"> `{g}`: {texts[g][:400]}{'...' if len(texts[g]) > 400 else ''}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Label every failure with one cause.")
    parser.add_argument("--track", choices=["T", "M"], default="T")
    parser.add_argument("--split", choices=["dev", "heldout", "test"], default="dev")
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--answers-dir", type=Path, default=Path("runs/answers"))
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--examples", type=Path, default=EXAMPLES_PATH)
    args = parser.parse_args(argv)
    split: Split = args.split
    groups, manifest = verify_frozen(
        Path("evalset/questions.jsonl"), Path("evalset/manifest.json"), CORPUS_MANIFEST
    )
    check_split_allowed(
        split,
        unreviewed_groups=sum(1 for g in groups if not g.spanish_reviewed),
        manifest_unreviewed=manifest.unreviewed_groups,
    )
    questions = [q for g in groups if manifest.splits[g.group_id] == split for q in expand(g)]
    choice = load_choice(args.track)
    config = serving_config(choice.config, choice)
    # A reranker-scored serving config is not part of the pre-registered chain and must not
    # land in the ablation directory (D40); it lives beside the refusal runs.
    run_root = Path("runs/ablation") if config == choice.config else Path("runs/refusal/runs")
    workbench = Workbench(questions, manifest, split, run_root)
    records = workbench.evaluate(config)
    texts = {p.passage_id: p.text for p in load_passages(Path("data/corpus"), ("en",))}
    answers_dir = args.answers_dir / f"full__{args.track}__{split}"
    answers = None
    labels = None
    if (answers_dir / "records.jsonl").is_file():
        answers = {a.question_id: a for a in load_answers(answers_dir)}
        judgments = load_judgments(answers_dir)
        if judgments is not None and ACCEPTED_PATH.is_file():
            labels = labels_from_judgments(judgments)
    overrides = (
        json.loads(OVERRIDES_PATH.read_text(encoding="utf-8")) if OVERRIDES_PATH.is_file() else None
    )
    failures = label_failures(records, answers, labels, texts, overrides=overrides)
    counts = counts_by_language(failures)
    stage = "retrieval and answers" if answers else "retrieval only (no answer run yet)"
    judged = "with judge labels" if labels else "without judge labels"
    header = (
        f"Generated by `uv run python -m atr.eval.errors --track {args.track} --split {split}`. "
        "Do not edit by hand.\n"
        f"Track {args.track}, {config.label} ({config.config_id}), {split} split, "
        f"{len(records)} questions, {stage}, {judged}.\n"
        "A question fails when its gold is not all in the top 5, when the answer was wrongly "
        "refused or wrongly given, when it does not cite a gold passage, or when the judge marks "
        "a claim unsupported. Labels are rule-based (DECISIONS.md D42); "
        f"{OVERRIDES_PATH} can override one.\n"
    )
    body = format_counts(counts)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"{header}\n{body}\n", encoding="utf-8")
    examples = pick_examples(failures, n=args.n_examples)
    args.examples.parent.mkdir(parents=True, exist_ok=True)
    args.examples.write_text(render_examples(examples, texts) + "\n", encoding="utf-8")
    print(header)
    print(body)
    print(f"\n{len(failures)} failures; wrote {args.report}")
    print(f"{len(examples)} examples -> {args.examples}")


if __name__ == "__main__":
    main()
