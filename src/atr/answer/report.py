"""Phase 7 check: citation accuracy, refusal accuracy, supported claims, three-system split."""

import argparse
import json
from pathlib import Path

from atr.answer.grading import AGREEMENT_PATH
from atr.answer.judge import ACCEPTED_PATH, count_undecided, labels_from_judgments, load_judgments
from atr.answer.records import SYSTEMS, AnswerRecord, load_answers
from atr.answer.run import AnswerRunMeta
from atr.answer.score import ErrorSplit, error_split, format_answer_rows, summarize_answers
from atr.eval.run import check_split_allowed
from atr.evalset.freeze import CORPUS_MANIFEST, verify_frozen
from atr.evalset.models import Split

REPORT_PATH = Path("docs/answers.txt")
_COMPARABLE = ("split", "generator", "code_stamp", "evalset_hash", "corpus_pin")


def check_comparable(metas: dict[str, AnswerRunMeta]) -> None:
    """The systems paired in one table must come from the same code, prompts, questions and
    corpus; otherwise the three-system split compares two different generators."""
    for field in _COMPARABLE:
        values = {system: getattr(meta, field) for system, meta in metas.items()}
        if len(set(values.values())) > 1:
            raise ValueError(f"answer runs differ in {field}: {values}; rerun the stale ones")


def judge_status() -> str:
    if ACCEPTED_PATH.is_file():
        accepted = json.loads(ACCEPTED_PATH.read_text(encoding="utf-8"))
        a = accepted["agreement"]
        return (
            f"judge validated: agreement {a['percent_agreement']:.2f}, Cohen's kappa "
            f"{a['cohen_kappa']:.2f} on {a['n_claims']} hand-graded claims, accepted by "
            f"{accepted['accepted_by']}"
        )
    if AGREEMENT_PATH.is_file():
        a = json.loads(AGREEMENT_PATH.read_text(encoding="utf-8"))
        return (
            f"judge NOT yet accepted: agreement {a['percent_agreement']:.2f}, kappa "
            f"{a['cohen_kappa']:.2f} on {a['n_claims']} claims; run `atr.answer.grading accept`"
        )
    return (
        "judge not validated: the supported-claim column stays empty until mannie grades the sheet"
    )


def format_split(split: ErrorSplit) -> str:
    g = split.retrieval_gap
    v = split.retrieval_value
    return "\n".join(
        [
            f"Three systems on the {split.n} questions with a gold passage, citation accuracy "
            "(the answer cites a gold passage):",
            f"  closed-book {split.closed_book:.2f}   full {split.full:.2f}   "
            f"oracle {split.oracle:.2f}",
            f"  retrieval gap, oracle minus full:   {g.mean:+.2f} [{g.low:+.2f}, {g.high:+.2f}]"
            "  (what better retrieval could still add)",
            f"  generation gap, 1 minus oracle:     {split.generation_gap:+.2f}"
            "  (what the generator loses even when handed the gold)",
            f"  retrieval value, full minus closed: {v.mean:+.2f} [{v.low:+.2f}, {v.high:+.2f}]"
            "  (what retrieval adds over the model's memory)",
        ]
    )


def build_report(
    runs: dict[str, list[AnswerRecord]],
    judged: dict[str, dict[tuple[str, int], bool]],
    *,
    resamples: int,
    seed: int,
) -> str:
    parts = [judge_status()]
    for system in SYSTEMS:
        if system not in runs:
            parts.append(f"\n=== {system}: no run yet ===")
            continue
        rows = summarize_answers(runs[system], judged.get(system), resamples=resamples, seed=seed)
        parts.append(f"\n=== {system} ===")
        parts.append(format_answer_rows(rows))
    if all(system in runs for system in SYSTEMS):
        parts.append("")
        parts.append(
            format_split(
                error_split(
                    runs["closed_book"],
                    runs["full"],
                    runs["oracle"],
                    resamples=resamples,
                    seed=seed,
                )
            )
        )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 7 tables from the saved answer records.")
    parser.add_argument("--track", choices=["T", "M"], default="T")
    parser.add_argument("--split", choices=["dev", "heldout", "test"], default="dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--answers-dir", type=Path, default=Path("runs/answers"))
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
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
    runs: dict[str, list[AnswerRecord]] = {}
    metas: dict[str, AnswerRunMeta] = {}
    judged: dict[str, dict[tuple[str, int], bool]] = {}
    undecided: dict[str, int] = {}
    for system in SYSTEMS:
        run_dir = args.answers_dir / f"{system}__{args.track}__{split}"
        if (run_dir / "records.jsonl").is_file():
            runs[system] = load_answers(run_dir)
            metas[system] = AnswerRunMeta.model_validate_json(
                (run_dir / "meta.json").read_text(encoding="utf-8")
            )
            judgments = load_judgments(run_dir)
            if judgments is not None and ACCEPTED_PATH.is_file():
                judged[system] = labels_from_judgments(judgments)
                undecided[system] = count_undecided(judgments)
    check_comparable(metas)
    unreadable = (
        "Unreadable judge verdicts, counted as unsupported: "
        + ", ".join(f"{s} {n}" for s, n in undecided.items())
        + ".\n"
        if undecided
        else ""
    )
    header = (
        f"Generated by `uv run python -m atr.answer.report --track {args.track} --split {split}`. "
        "Do not edit by hand.\n"
        f"Paired bootstrap over question groups, {args.resamples} resamples, seed {args.seed}.\n"
        + unreadable
    )
    body = build_report(runs, judged, resamples=args.resamples, seed=args.seed)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"{header}\n{body}\n", encoding="utf-8")
    print(header)
    print(body)
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
