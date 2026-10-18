"""Phase 6: when to refuse. A score threshold picked on dev, reported as a curve (D40)."""

import argparse
import math
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from atr.eval.ablation import STARTS, STEPS, Workbench, run_chain
from atr.eval.report import Record
from atr.evalset.freeze import CORPUS_MANIFEST, verify_frozen
from atr.evalset.models import (
    QUESTION_LANGUAGES,
    QUESTION_TYPES,
    QuestionLanguage,
    QuestionType,
    expand,
)
from atr.pipeline import PipelineConfig

ScoreSource = Literal["first_stage", "reranker"]
REFUSAL_RERANKER = "bge-m3"
OUT_DIR = Path("runs/refusal")
REPORT_PATH = Path("docs/refusal.txt")
FIGURE_PATH = Path("docs/figures/refusal_curve.png")


class Candidate(NamedTuple):
    question_id: str
    group_id: str
    type: QuestionType
    language: QuestionLanguage
    score: float
    answerable: bool
    hit: bool


class RefusalPoint(NamedTuple):
    threshold: float
    answered_share: float
    accuracy_when_answered: float | None
    decision_accuracy: float


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RefusalRow(_Frozen):
    slice: str
    n: int
    refused_share: float
    decision_accuracy: float


class CurvePoint(_Frozen):
    threshold: float | None  # None at the two ends (answer everything, refuse everything)
    answered_share: float
    accuracy_when_answered: float | None
    decision_accuracy: float


class RefusalChoice(_Frozen):
    """The committed outcome of phase 6 for one track; phases 7 and 9 read it."""

    track: str
    config: PipelineConfig
    source: ScoreSource
    threshold: float
    auroc_by_source: dict[str, float]
    balanced_accuracy: float
    n_answerable: int
    n_unanswerable: int
    rows: list[RefusalRow]
    curve: list[CurvePoint]


def candidates(records: list[Record]) -> list[Candidate]:
    return [
        Candidate(
            r.question_id,
            r.group_id,
            r.type,
            r.language,
            r.scores[0] if r.scores else -math.inf,
            r.type != "out_of_scope",
            r.metrics is not None and r.metrics.recall_at_5 > 0,
        )
        for r in records
    ]


def auroc(positives: list[float], negatives: list[float]) -> float:
    """Probability that a random answerable question outscores a random unanswerable one."""
    if not positives or not negatives:
        raise ValueError("AUROC needs scores from both classes")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def _answered(members: list[Candidate], threshold: float) -> list[bool]:
    return [c.score >= threshold for c in members]


def curve(members: list[Candidate]) -> list[RefusalPoint]:
    """Answer-everything to refuse-everything, one point per distinct score."""
    if not members:
        raise ValueError("no candidates")
    thresholds = [-math.inf, *sorted({c.score for c in members}), math.inf]
    points: list[RefusalPoint] = []
    for threshold in thresholds:
        answered = _answered(members, threshold)
        n_answered = sum(answered)
        correct_answers = sum(1 for c, a in zip(members, answered, strict=True) if a and c.hit)
        points.append(
            RefusalPoint(
                threshold=threshold,
                answered_share=n_answered / len(members),
                accuracy_when_answered=correct_answers / n_answered if n_answered else None,
                decision_accuracy=sum(
                    1 for c, a in zip(members, answered, strict=True) if a == c.answerable
                )
                / len(members),
            )
        )
    return points


def balanced_accuracy(members: list[Candidate], threshold: float) -> float:
    """Mean of the answer rate on answerable questions and the refusal rate on the rest.

    The two classes are far from equal in size (out-of-scope is one type of six), so plain
    accuracy would be maximized by never refusing.
    """
    answered = _answered(members, threshold)
    answerable = [a for c, a in zip(members, answered, strict=True) if c.answerable]
    unanswerable = [a for c, a in zip(members, answered, strict=True) if not c.answerable]
    if not answerable or not unanswerable:
        raise ValueError("balanced accuracy needs both answerable and unanswerable questions")
    return (sum(answerable) / len(answerable) + (1 - sum(unanswerable) / len(unanswerable))) / 2


def pick_threshold(members: list[Candidate]) -> float:
    """The lowest threshold with the highest balanced accuracy on dev.

    Candidates are the observed scores: a threshold between two scores behaves exactly like
    the higher one. Ties go low so the system answers as much as the rule allows.
    """
    thresholds = sorted({c.score for c in members})
    scored = [(balanced_accuracy(members, t), -t) for t in thresholds]
    best = max(scored)
    return -best[1]


def _slices(members: list[Candidate]) -> list[tuple[str, list[Candidate]]]:
    slices = [("all", members)]
    slices += [(f"type={t}", [c for c in members if c.type == t]) for t in QUESTION_TYPES]
    slices += [(f"lang={x}", [c for c in members if c.language == x]) for x in QUESTION_LANGUAGES]
    return [(name, found) for name, found in slices if found]


def refusal_rows(members: list[Candidate], threshold: float) -> list[RefusalRow]:
    rows = []
    for name, found in _slices(members):
        answered = _answered(found, threshold)
        rows.append(
            RefusalRow(
                slice=name,
                n=len(found),
                refused_share=1 - sum(answered) / len(found),
                decision_accuracy=sum(
                    1 for c, a in zip(found, answered, strict=True) if a == c.answerable
                )
                / len(found),
            )
        )
    return rows


def format_rows(rows: list[RefusalRow]) -> str:
    lines = [f"{'slice':<20}{'n':>5}{'refused':>10}{'decision ok':>13}"]
    lines += [
        f"{r.slice:<20}{r.n:>5}{r.refused_share:>10.2f}{r.decision_accuracy:>13.2f}" for r in rows
    ]
    return "\n".join(lines)


def choose(
    track: str, config: PipelineConfig, by_source: dict[ScoreSource, list[Record]]
) -> RefusalChoice:
    """Pick the score source by AUROC on dev, then the threshold on that source."""
    members = {source: candidates(records) for source, records in by_source.items()}
    aurocs = {
        source: auroc(
            [c.score for c in found if c.answerable], [c.score for c in found if not c.answerable]
        )
        for source, found in members.items()
    }
    source = max(aurocs, key=lambda s: (aurocs[s], s == "first_stage"))
    chosen = members[source]
    threshold = pick_threshold(chosen)
    return RefusalChoice(
        track=track,
        config=config,
        source=source,
        threshold=threshold,
        auroc_by_source={str(k): v for k, v in aurocs.items()},
        balanced_accuracy=balanced_accuracy(chosen, threshold),
        n_answerable=sum(1 for c in chosen if c.answerable),
        n_unanswerable=sum(1 for c in chosen if not c.answerable),
        rows=refusal_rows(chosen, threshold),
        curve=[
            CurvePoint(
                threshold=p.threshold if math.isfinite(p.threshold) else None,
                answered_share=p.answered_share,
                accuracy_when_answered=p.accuracy_when_answered,
                decision_accuracy=p.decision_accuracy,
            )
            for p in curve(chosen)
        ],
    )


def format_choice(choice: RefusalChoice) -> str:
    false_premise = next((r for r in choice.rows if r.slice == "type=false_premise"), None)
    lines = [
        f"=== Track {choice.track}: {choice.config.label} ({choice.config.config_id}) ===",
        "AUROC of the top score, answerable against out-of-scope, on dev: "
        + ", ".join(f"{s} {a:.3f}" for s, a in sorted(choice.auroc_by_source.items())),
        f"chosen source: {choice.source}; threshold {choice.threshold:.4f} "
        f"(highest balanced accuracy on dev, {choice.balanced_accuracy:.3f}; "
        f"{choice.n_answerable} answerable, {choice.n_unanswerable} out-of-scope)",
        format_rows(choice.rows),
    ]
    if false_premise is not None:
        lines.append(
            f"false-premise questions: {false_premise.refused_share:.2f} refused "
            f"(n={false_premise.n}). They have a gold passage, so a refusal counts as a wrong "
            "decision here; the right answer corrects the premise."
        )
    return "\n".join(lines)


def draw(choices: list[RefusalChoice], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"T": "#2a78d6", "M": "#eb6834"}
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for choice in choices:
        xs = [p.answered_share for p in choice.curve if p.accuracy_when_answered is not None]
        ys = [
            p.accuracy_when_answered or 0.0
            for p in choice.curve
            if p.accuracy_when_answered is not None
        ]
        color = colors.get(choice.track, "#52514e")
        ax.plot(
            xs, ys, color=color, linewidth=2, label=f"track {choice.track}: {choice.source} score"
        )
        rows = {r.slice: r for r in choice.rows}
        answered = 1 - rows["all"].refused_share
        nearest = min(range(len(xs)), key=lambda i: abs(xs[i] - answered))
        ax.scatter([xs[nearest]], [ys[nearest]], color=color, s=80, zorder=3)
        ax.annotate(
            f"chosen: threshold {choice.threshold:.3f}",
            (xs[nearest], ys[nearest]),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=9,
        )
    ax.set_xlabel("share of dev questions answered (the rest refused)")
    ax.set_ylabel("accuracy among answered (gold in top 5; out-of-scope counts wrong)")
    ax.set_title("Refusal curve on dev, by the top retrieval score", loc="left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def load_choice(track: str, out_dir: Path = OUT_DIR) -> RefusalChoice:
    path = out_dir / f"{track}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `uv run python -m atr.eval.refusal` first")
    return RefusalChoice.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pick the refusal threshold on dev.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--ablation-dir", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args(argv)
    groups, manifest = verify_frozen(
        Path("evalset/questions.jsonl"), Path("evalset/manifest.json"), CORPUS_MANIFEST
    )
    questions = [q for g in groups if manifest.splits[g.group_id] == "dev" for q in expand(g)]
    workbench = Workbench(questions, manifest, "dev", args.ablation_dir)
    # The reranker-scored runs are not part of the pre-registered chain, so they live apart
    # from the ablation directory and never appear in its table or plot.
    scoring = Workbench(questions, manifest, "dev", args.out / "runs")
    choices = []
    for track, start in STARTS.items():
        _, final = run_chain(
            track, start, STEPS, workbench.evaluate, resamples=args.resamples, seed=args.seed
        )
        with_reranker = final.model_copy(
            update={"reranker": REFUSAL_RERANKER, "label": f"{final.label} +refusal reranker"}
        )
        by_source: dict[ScoreSource, list[Record]] = {
            "first_stage": workbench.evaluate(final),
            "reranker": scoring.evaluate(with_reranker),
        }
        choices.append(choose(track, final, by_source))
    args.out.mkdir(parents=True, exist_ok=True)
    for choice in choices:
        (args.out / f"{choice.track}.json").write_text(
            choice.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    draw(choices, args.figure)
    header = (
        "Generated by `uv run python -m atr.eval.refusal`. Do not edit by hand.\n"
        f"Question set {manifest.hash[:12]}, {manifest.unreviewed_groups} of {len(groups)} "
        "groups without a human review. Dev split only (DECISIONS.md D40).\n"
        "A question is refused when the top score of the chosen source is under the threshold.\n"
        "'refused' is the share refused; 'decision ok' is the share where answering or refusing "
        "was the right call (answerable answered, out-of-scope refused).\n"
    )
    body = "\n\n".join(format_choice(c) for c in choices)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"{header}\n{body}\n", encoding="utf-8")
    print(header)
    print(body)
    print(f"\nWrote {args.report}, {args.figure} and {args.out}/<track>.json")


if __name__ == "__main__":
    main()
