"""Citation accuracy, refusal accuracy, supported-claim rate, and the error split (D41)."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from atr.answer.records import AnswerRecord
from atr.eval.bootstrap import Interval, bootstrap_ci, paired_difference
from atr.evalset.models import QUESTION_LANGUAGES, QUESTION_TYPES

ClaimKey = tuple[str, int]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnswerRow(_Frozen):
    slice: str
    n: int
    n_with_gold: int
    citation_accuracy: Interval | None
    all_gold_cited: Interval | None
    refusal_accuracy: Interval
    supported_claims: Interval | None
    n_claims: int


class ErrorSplit(_Frozen):
    """Citation accuracy of the three systems on the questions with a gold passage."""

    n: int
    closed_book: float
    full: float
    oracle: float
    retrieval_gap: Interval  # oracle minus full: what better retrieval could still add
    generation_gap: float  # 1 minus oracle: what the generator loses even with the gold
    retrieval_value: Interval  # full minus closed-book: what retrieval adds over memory


def citation_correct(record: AnswerRecord) -> bool | None:
    """True when the answer cites a gold passage. None when the question has no gold."""
    if not record.gold:
        return None
    return not record.refused and any(c in record.gold for c in record.citations)


def all_gold_cited(record: AnswerRecord) -> bool | None:
    if not record.gold:
        return None
    return not record.refused and all(g in record.citations for g in record.gold)


def refusal_correct(record: AnswerRecord) -> bool:
    return record.refused == (record.type == "out_of_scope")


def _claim_labels(
    records: list[AnswerRecord], labels: Mapping[ClaimKey, bool]
) -> list[tuple[str, float]]:
    found = []
    for record in records:
        if record.refused:
            continue
        for index in range(len(record.claims)):
            label = labels.get((record.question_id, index))
            if label is not None:
                found.append((record.group_id, 1.0 if label else 0.0))
    return found


def supported_rate(records: list[AnswerRecord], labels: Mapping[ClaimKey, bool]) -> float | None:
    """Share of labeled claims the judge (or a human) marked supported; None without labels."""
    values = _claim_labels(records, labels)
    if not values:
        return None
    return sum(v for _, v in values) / len(values)


def _slices(records: list[AnswerRecord]) -> list[tuple[str, list[AnswerRecord]]]:
    slices = [("all", records)]
    slices += [(f"type={t}", [r for r in records if r.type == t]) for t in QUESTION_TYPES]
    slices += [(f"lang={x}", [r for r in records if r.language == x]) for x in QUESTION_LANGUAGES]
    return [(name, members) for name, members in slices if members]


def _interval(pairs: list[tuple[str, float]], resamples: int, seed: int) -> Interval | None:
    if not pairs:
        return None
    return bootstrap_ci(
        [v for _, v in pairs], clusters=[g for g, _ in pairs], resamples=resamples, seed=seed
    )


def summarize_answers(
    records: list[AnswerRecord],
    labels: Mapping[ClaimKey, bool] | None,
    *,
    resamples: int,
    seed: int,
) -> list[AnswerRow]:
    rows: list[AnswerRow] = []
    for name, members in _slices(records):
        with_gold = [r for r in members if r.gold]
        cited = [(r.group_id, float(bool(citation_correct(r)))) for r in with_gold]
        complete = [(r.group_id, float(bool(all_gold_cited(r)))) for r in with_gold]
        decisions = [(r.group_id, float(refusal_correct(r))) for r in members]
        claim_values = _claim_labels(members, labels) if labels is not None else []
        rows.append(
            AnswerRow(
                slice=name,
                n=len(members),
                n_with_gold=len(with_gold),
                citation_accuracy=_interval(cited, resamples, seed),
                all_gold_cited=_interval(complete, resamples, seed),
                refusal_accuracy=bootstrap_ci(
                    [v for _, v in decisions],
                    clusters=[g for g, _ in decisions],
                    resamples=resamples,
                    seed=seed,
                ),
                supported_claims=_interval(claim_values, resamples, seed),
                n_claims=sum(len(r.claims) for r in members if not r.refused),
            )
        )
    return rows


def _mean(values: Mapping[str, float]) -> float:
    return sum(values.values()) / len(values)


def _by_question(records: list[AnswerRecord]) -> dict[str, float]:
    return {r.question_id: float(bool(citation_correct(r))) for r in records if r.gold}


def error_split(
    closed_book: list[AnswerRecord],
    full: list[AnswerRecord],
    oracle: list[AnswerRecord],
    *,
    resamples: int,
    seed: int,
) -> ErrorSplit:
    c, f, o = _by_question(closed_book), _by_question(full), _by_question(oracle)
    if not (c.keys() == f.keys() == o.keys()):
        raise ValueError("the three systems do not answer the same questions")
    if not f:
        raise ValueError("no questions with a gold passage")
    clusters = {r.question_id: r.group_id for r in full if r.gold}
    return ErrorSplit(
        n=len(f),
        closed_book=_mean(c),
        full=_mean(f),
        oracle=_mean(o),
        retrieval_gap=paired_difference(f, o, clusters=clusters, resamples=resamples, seed=seed),
        generation_gap=1.0 - _mean(o),
        retrieval_value=paired_difference(c, f, clusters=clusters, resamples=resamples, seed=seed),
    )


def _cell(interval: Interval | None) -> str:
    if interval is None:
        return "-"
    return f"{interval.mean:.2f} [{interval.low:.2f}, {interval.high:.2f}]"


def format_answer_rows(rows: list[AnswerRow]) -> str:
    header = (
        f"{'slice':<20}{'n':>4}{'gold':>5}  {'cites a gold passage':<22}{'cites every gold':<22}"
        f"{'refusal decision ok':<22}{'supported claims':<22}{'claims':>6}"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.slice:<20}{r.n:>4}{r.n_with_gold:>5}  {_cell(r.citation_accuracy):<22}"
            f"{_cell(r.all_gold_cited):<22}{_cell(r.refusal_accuracy):<22}"
            f"{_cell(r.supported_claims):<22}{r.n_claims:>6}"
        )
    return "\n".join(lines)
