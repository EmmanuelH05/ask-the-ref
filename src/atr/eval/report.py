import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from atr.eval.bootstrap import Interval, bootstrap_ci, excludes_zero, paired_difference
from atr.eval.metrics import METRICS, QuestionScores
from atr.evalset.models import (
    QUESTION_LANGUAGES,
    QUESTION_TYPES,
    QuestionLanguage,
    QuestionType,
    Split,
)

HEADERS = {
    "recall_at_1": "recall@1",
    "recall_at_5": "recall@5",
    "mrr": "MRR",
    "ndcg_at_10": "nDCG@10",
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Record(_Frozen):
    """One question's result. A run's records are its cache: every table is rebuilt from them."""

    question_id: str
    group_id: str
    language: QuestionLanguage
    type: QuestionType
    split: Split
    query: str
    gold: list[str]
    ranked: list[str]
    scores: list[float]
    latency_ms: dict[str, float]
    metrics: QuestionScores | None


class SliceRow(_Frozen):
    slice: str
    n: int
    n_groups: int
    metrics: dict[str, Interval]


class DifferenceRow(_Frozen):
    slice: str
    n: int
    n_groups: int
    metrics: dict[str, Interval]
    decided: dict[str, bool]


def _slices(records: list[Record]) -> list[tuple[str, list[Record]]]:
    scored = [r for r in records if r.metrics is not None]
    slices = [("all", scored)]
    slices += [(f"type={t}", [r for r in scored if r.type == t]) for t in QUESTION_TYPES]
    slices += [(f"lang={x}", [r for r in scored if r.language == x]) for x in QUESTION_LANGUAGES]
    return [(name, members) for name, members in slices if members]


def _value(record: Record, metric: str) -> float:
    assert record.metrics is not None
    value: float = getattr(record.metrics, metric)
    return value


def summarize(records: list[Record], resamples: int, seed: int) -> list[SliceRow]:
    """Mean and interval per slice. Question groups are resampled whole (DECISIONS.md D23)."""
    slices = _slices(records)
    if not slices:
        raise ValueError("no scored records")
    rows: list[SliceRow] = []
    for name, members in slices:
        clusters = [r.group_id for r in members]
        intervals = {
            metric: bootstrap_ci(
                [_value(r, metric) for r in members],
                clusters=clusters,
                resamples=resamples,
                seed=seed,
            )
            for metric in METRICS
        }
        rows.append(
            SliceRow(slice=name, n=len(members), n_groups=len(set(clusters)), metrics=intervals)
        )
    return rows


def compare_runs(
    a: list[Record], b: list[Record], resamples: int, seed: int
) -> list[DifferenceRow]:
    """Paired difference b minus a per slice; `decided` is true when the interval excludes 0."""
    a_slices, b_slices = dict(_slices(a)), dict(_slices(b))
    if {r.question_id for r in a_slices.get("all", [])} != {
        r.question_id for r in b_slices.get("all", [])
    }:
        raise ValueError("the two runs do not score the same questions")
    rows: list[DifferenceRow] = []
    for name, members in a_slices.items():
        others = {r.question_id: r for r in b_slices[name]}
        clusters = {r.question_id: r.group_id for r in members}
        intervals = {
            metric: paired_difference(
                {r.question_id: _value(r, metric) for r in members},
                {key: _value(other, metric) for key, other in others.items()},
                clusters=clusters,
                resamples=resamples,
                seed=seed,
            )
            for metric in METRICS
        }
        rows.append(
            DifferenceRow(
                slice=name,
                n=len(members),
                n_groups=len(set(clusters.values())),
                metrics=intervals,
                decided={metric: excludes_zero(i) for metric, i in intervals.items()},
            )
        )
    return rows


def _cell(interval: Interval, *, signed: bool, mark: bool = False) -> str:
    form = "+.2f" if signed else ".2f"
    text = f"{interval.mean:{form}} [{interval.low:{form}}, {interval.high:{form}}]"
    return text + (" *" if mark else "")


def format_rows(rows: list[SliceRow] | list[DifferenceRow]) -> str:
    header = f"{'slice':<20}{'n':>5}{'groups':>8}  " + "".join(
        f"{HEADERS[metric]:<24}" for metric in METRICS
    )
    lines = [header]
    for row in rows:
        decided = row.decided if isinstance(row, DifferenceRow) else None
        cells = ""
        for metric in METRICS:
            marked = bool(decided and decided[metric])
            cell = _cell(row.metrics[metric], signed=decided is not None, mark=marked)
            cells += f"{cell:<24}"
        lines.append(f"{row.slice:<20}{row.n:>5}{row.n_groups:>8}  {cells}".rstrip())
    return "\n".join(lines)


def _nearest_rank(ordered: list[float], percent: int) -> float:
    return ordered[max(math.ceil(percent / 100 * len(ordered)), 1) - 1]


def latency_summary(records: list[Record]) -> dict[str, dict[str, float]]:
    stages: dict[str, list[float]] = {}
    for record in records:
        for stage, value in record.latency_ms.items():
            stages.setdefault(stage, []).append(value)
    return {
        stage: {
            "p50": _nearest_rank(sorted(values), 50),
            "p95": _nearest_rank(sorted(values), 95),
            "n": len(values),
        }
        for stage, values in stages.items()
    }


def load_records(run_dir: Path) -> list[Record]:
    path = run_dir / "records.jsonl"
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Record.model_validate_json(line) for line in lines if line.strip()]
