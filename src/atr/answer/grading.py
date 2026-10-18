"""The hand-grading sheet and the judge's agreement with it (D41).

`sheet` writes 60 answers for mannie to grade claim by claim. He marks each claim `[y]` or
`[n]`. `agree` runs the judge on the same claims and prints percent agreement and Cohen's
kappa. `accept` records that he accepts that number, which unlocks the judge on full sets.
"""

import argparse
import json
import random
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from atr.answer.judge import (
    ACCEPTED_PATH,
    Judge,
    english_texts,
    judge_signature,
    labels_from_judgments,
)
from atr.answer.records import AnswerRecord, load_answers, records_sha256
from atr.cache import JsonCache
from atr.eval.run import CACHE_DIR
from atr.llm import OllamaLLM

SHEET_SIZE = 60
SHEET_PATH = Path("docs/grading/sheet.md")
AGREEMENT_PATH = Path("runs/answers/judge_agreement.json")
_MARK = re.compile(r"^- \[(.*?)\] \(([^#()]+)#(\d+)\)")
_CLAIM_LINE = re.compile(r"^\s*-\s*\[")
_STAMP = re.compile(r"^<!-- answers-stamp: ([0-9a-f]{64}) -->$", re.MULTILINE)
ClaimKey = tuple[str, int]


class Agreement(NamedTuple):
    n: int
    percent_agreement: float
    kappa: float
    human_supported: float
    judge_supported: float


def sample_for_grading(records: list[AnswerRecord], *, n: int, seed: int) -> list[AnswerRecord]:
    """Round-robin over (type, language) cells, shuffled by seed, so every cell is graded."""
    cells: dict[tuple[str, str], list[AnswerRecord]] = {}
    for record in sorted(records, key=lambda r: r.question_id):
        if not record.refused and record.claims:
            cells.setdefault((record.type, record.language), []).append(record)
    rng = random.Random(seed)
    queues = [list(members) for _, members in sorted(cells.items())]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    chosen: list[AnswerRecord] = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop())
    return chosen


def render_sheet(
    records: list[AnswerRecord], texts: Mapping[str, str], *, stamp: str | None = None
) -> str:
    """The sheet; `stamp` is the hash of the records file it was sampled from (D41)."""
    lines = [
        "# Hand-grading sheet",
        "",
        *([f"<!-- answers-stamp: {stamp} -->", ""] if stamp else []),
        "For every claim: is it supported by the passage it cites? Replace `[ ]` with `[y]` "
        "(supported) or `[n]` (not supported, or no citation). Leave the `(id#k)` tag alone.",
        "",
    ]
    for number, record in enumerate(records, start=1):
        lines += [
            f"### {number}. {record.question_id} ({record.type}, {record.language})",
            "",
            f"**Question:** {record.question}",
            "",
            f"**Answer:** {record.answer}",
            "",
        ]
        for index, claim in enumerate(record.claims):
            lines.append(f"- [ ] ({record.question_id}#{index}) {claim.text}")
            if not claim.citations:
                lines.append("  - cites nothing")
            for cited in claim.citations:
                text = texts.get(cited, "(passage not in the English corpus)")
                lines.append(f"  - `{cited}`: {text}")
        lines.append("")
    return "\n".join(lines)


def parse_grades(sheet: str) -> dict[ClaimKey, bool]:
    """Marked claims. A claim line that cannot be read raises instead of vanishing."""
    grades: dict[ClaimKey, bool] = {}
    for number, line in enumerate(sheet.splitlines(), start=1):
        match = _MARK.match(line)
        if match is None:
            if _CLAIM_LINE.match(line):
                raise ValueError(
                    f"sheet line {number} looks like a claim but cannot be read: {line!r}"
                )
            continue
        mark, question_id, index = (
            match.group(1).strip().lower(),
            match.group(2),
            int(match.group(3)),
        )
        if mark == "":
            continue
        if mark not in ("y", "n"):
            raise ValueError(
                f"sheet line {number}: mark {mark!r} on ({question_id}#{index}) is not y or n"
            )
        grades[(question_id, index)] = mark == "y"
    return grades


def unmarked_claims(sheet: str) -> int:
    """Claim lines still carrying `[ ]`."""
    marks = (_MARK.match(line) for line in sheet.splitlines())
    return sum(1 for m in marks if m is not None and not m.group(1).strip())


def parse_stamp(sheet: str) -> str | None:
    match = _STAMP.search(sheet)
    return match.group(1) if match else None


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal, non-empty label lists")
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)
    p_a, p_b = sum(a) / len(a), sum(b) / len(b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement(human: Mapping[ClaimKey, bool], judge: Mapping[ClaimKey, bool]) -> Agreement:
    shared = sorted(set(human) & set(judge))
    if not shared:
        raise ValueError("no claim was labeled by both the human and the judge")
    h = [human[key] for key in shared]
    j = [judge[key] for key in shared]
    return Agreement(
        n=len(shared),
        percent_agreement=sum(1 for x, y in zip(h, j, strict=True) if x == y) / len(shared),
        kappa=cohen_kappa(h, j),
        human_supported=sum(h) / len(h),
        judge_supported=sum(j) / len(j),
    )


def _sheet(args: argparse.Namespace) -> None:
    run_dir = args.answers_dir / f"full__{args.track}__dev"
    records = load_answers(run_dir)
    chosen = sample_for_grading(records, n=args.n, seed=args.seed)
    texts = english_texts(args.corpus_dir)
    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet = render_sheet(chosen, texts, stamp=records_sha256(run_dir))
    args.sheet.write_text(sheet + "\n", encoding="utf-8")
    claims = sum(len(r.claims) for r in chosen)
    print(f"wrote {args.sheet}: {len(chosen)} answers, {claims} claims to grade (seed {args.seed})")


def _agree(args: argparse.Namespace) -> None:
    sheet = args.sheet.read_text(encoding="utf-8")
    run_dir = args.answers_dir / f"full__{args.track}__dev"
    if parse_stamp(sheet) != records_sha256(run_dir):
        raise SystemExit(
            f"{args.sheet} was written for other answers than {run_dir / 'records.jsonl'}; "
            "write a new sheet (`sheet`) and grade it"
        )
    human = parse_grades(sheet)
    if not human:
        raise SystemExit(f"no grades found in {args.sheet}; mark claims [y] or [n] first")
    unmarked = unmarked_claims(sheet)
    if unmarked:
        print(f"{unmarked} claims are still unmarked and are left out of the agreement")
    records = {r.question_id: r for r in load_answers(run_dir)}
    texts = english_texts(args.corpus_dir)
    judge = Judge(OllamaLLM(cache=JsonCache(CACHE_DIR / "llm")))
    judgments: dict[str, list[bool | None]] = {}
    for question_id in sorted({key[0] for key in human}):
        record = records[question_id]
        judgments[question_id] = [judge.judge(c, texts).supported for c in record.claims]
    result = agreement(human, labels_from_judgments(judgments))
    report = {
        "track": args.track,
        "sheet": str(args.sheet),
        **judge_signature(),
        "n_claims": result.n,
        "percent_agreement": result.percent_agreement,
        "cohen_kappa": result.kappa,
        "human_supported_rate": result.human_supported,
        "judge_supported_rate": result.judge_supported,
        "undecided_verdicts": sum(1 for v in judgments.values() for x in v if x is None),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    AGREEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGREEMENT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{result.n} claims graded by both. Agreement {result.percent_agreement:.2f}, "
        f"Cohen's kappa {result.kappa:.2f}. Human marked {result.human_supported:.2f} supported, "
        f"judge {result.judge_supported:.2f}."
    )
    print(
        f"wrote {AGREEMENT_PATH}. If you accept this number: "
        "`uv run python -m atr.answer.grading accept`"
    )


def _accept(args: argparse.Namespace) -> None:
    if not AGREEMENT_PATH.is_file():
        raise SystemExit(f"{AGREEMENT_PATH} missing; run `agree` first")
    report = json.loads(AGREEMENT_PATH.read_text(encoding="utf-8"))
    ACCEPTED_PATH.write_text(
        json.dumps(
            {
                "accepted_by": args.by,
                "agreement": report,
                "accepted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"recorded acceptance in {ACCEPTED_PATH}; the judge may now run on full sets")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hand-grading sheet and judge agreement.")
    parser.add_argument("command", choices=["sheet", "agree", "accept"])
    parser.add_argument("--track", choices=["T", "M"], default="T")
    parser.add_argument("--n", type=int, default=SHEET_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sheet", type=Path, default=SHEET_PATH)
    parser.add_argument("--answers-dir", type=Path, default=Path("runs/answers"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--by", default="mannie")
    args = parser.parse_args(argv)
    {"sheet": _sheet, "agree": _agree, "accept": _accept}[args.command](args)


if __name__ == "__main__":
    main()
