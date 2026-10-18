"""The automatic judge: is each claim supported by the passage it cites? (D41)

A different local model from the generator, so the generator never grades itself. The judge is
only trusted after its agreement with mannie's hand grades is reported (`atr.answer.grading`).
"""

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from atr.answer.generate import Claim
from atr.answer.records import SYSTEMS, AnswerRecord, load_answers, records_sha256
from atr.cache import JsonCache
from atr.eval.run import CACHE_DIR, check_split_allowed
from atr.evalset.freeze import CORPUS_MANIFEST, verify_frozen
from atr.llm import OllamaLLM
from atr.retrieval.docs import load_passages
from atr.retrieval.rewrite import LLM
from atr.text import fold

JUDGE_MODEL = "gpt-oss:20b"
JUDGE_THINK = "low"
JUDGE_PROMPT = (
    "You are checking an answer about the IFAB Laws of the Game. Below is a claim and the "
    "passage(s) it cites. Is the claim fully supported by the cited text? A claim that adds "
    "facts the passage does not state, or contradicts it, is not supported.\n\n"
    "Cited passage(s):\n{passages}\n\nClaim: {claim}\n\n"
    "Answer with one word, yes or no."
)
ACCEPTED_PATH = Path("runs/answers/judge_accepted.json")
_YES = ("yes", "si")
_NO = ("no",)
# Markdown and punctuation around the word, and a label the model may put in front of it.
_NOISE = re.compile(r"[*_`#\"'(),.;:!?\-]")
_LABELS = ("answer", "verdict", "respuesta", "supported", "claim")


class Verdict(NamedTuple):
    supported: bool | None
    ms: float


def parse_verdict(reply: str) -> bool | None:
    """yes/no from the first real word; None when the reply says neither (counts as unsupported)."""
    words = _NOISE.sub(" ", fold(reply)).split()
    while words and words[0] in _LABELS:
        words.pop(0)
    if not words:
        return None
    if words[0] in _YES:
        return True
    if words[0] in _NO:
        return False
    return None


def judge_signature(model: str = JUDGE_MODEL, think: str | None = JUDGE_THINK) -> dict[str, str]:
    """What the hand-grading accepted: the model, its think setting and the prompt's hash."""
    return {
        "judge": f"judge:{model}",
        "think": think or "",
        "prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest(),
    }


class Judge:
    def __init__(
        self, llm: LLM, *, model: str = JUDGE_MODEL, think: str | None = JUDGE_THINK
    ) -> None:
        self._llm = llm
        self._model = model
        self._think = think
        self.name = f"judge:{model}"

    def judge(self, claim: Claim, texts: Mapping[str, str]) -> Verdict:
        """A claim with no citation is unsupported by definition and costs no call."""
        missing = [pid for pid in claim.citations if pid not in texts]
        if missing:
            raise KeyError(f"cited passage(s) {missing} are not in the English corpus")
        cited = [f"[{pid}] {texts[pid]}" for pid in claim.citations]
        if not cited:
            return Verdict(False, 0.0)
        prompt = JUDGE_PROMPT.format(passages="\n\n".join(cited), claim=claim.text)
        reply = self._llm.complete(self._model, prompt, think=self._think)
        return Verdict(parse_verdict(reply.text), reply.ms)


def judge_records(
    records: list[AnswerRecord], judge: Judge, texts: Mapping[str, str]
) -> dict[str, list[bool | None]]:
    """Per question id, one verdict per claim (refused answers have none)."""
    return {
        r.question_id: [judge.judge(claim, texts).supported for claim in r.claims]
        for r in records
        if not r.refused
    }


def labels_from_judgments(
    judgments: Mapping[str, list[bool | None]],
) -> dict[tuple[str, int], bool]:
    """Undecided verdicts count as unsupported: the judge could not find the support."""
    return {
        (question_id, index): bool(verdict)
        for question_id, verdicts in judgments.items()
        for index, verdict in enumerate(verdicts)
    }


def write_judgments(run_dir: Path, judgments: Mapping[str, list[bool | None]]) -> None:
    """Verdicts stamped with the hash of the records they judged, so a rerun of the answers
    can never be read against the old verdicts."""
    payload = {
        "records_sha256": records_sha256(run_dir),
        "undecided": sum(1 for verdicts in judgments.values() for v in verdicts if v is None),
        "verdicts": dict(judgments),
    }
    (run_dir / "judgments.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_judgments(run_dir: Path) -> dict[str, list[bool | None]] | None:
    """The verdicts for this run, None when the judge has not run; stale verdicts raise."""
    path = run_dir / "judgments.json"
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or "verdicts" not in loaded:
        raise ValueError(f"{path} predates the stamped format; rerun the judge")
    if loaded.get("records_sha256") != records_sha256(run_dir):
        raise ValueError(
            f"{path} was produced for other answers than {run_dir / 'records.jsonl'}; "
            "rerun the judge"
        )
    verdicts: dict[str, list[bool | None]] = loaded["verdicts"]
    return verdicts


def count_undecided(judgments: Mapping[str, list[bool | None]]) -> int:
    return sum(1 for verdicts in judgments.values() for v in verdicts if v is None)


def english_texts(corpus_dir: Path) -> dict[str, str]:
    return {p.passage_id: p.text for p in load_passages(corpus_dir, ("en",))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the validated judge over every answer of a track and split."
    )
    parser.add_argument("--track", choices=["T", "M"], default="T")
    parser.add_argument("--split", choices=["dev", "heldout", "test"], default="dev")
    parser.add_argument("--answers-dir", type=Path, default=Path("runs/answers"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    args = parser.parse_args(argv)
    groups, manifest = verify_frozen(
        Path("evalset/questions.jsonl"), Path("evalset/manifest.json"), CORPUS_MANIFEST
    )
    check_split_allowed(
        args.split,
        unreviewed_groups=sum(1 for g in groups if not g.spanish_reviewed),
        manifest_unreviewed=manifest.unreviewed_groups,
    )
    if not ACCEPTED_PATH.is_file():
        raise SystemExit(
            f"{ACCEPTED_PATH} is missing: the judge may only run on a full set after mannie "
            "accepts its agreement with his grades (`uv run python -m atr.answer.grading accept`)."
        )
    accepted = json.loads(ACCEPTED_PATH.read_text(encoding="utf-8"))["agreement"]
    current = judge_signature()
    for key, value in current.items():
        if accepted.get(key) != value:
            raise SystemExit(
                f"the accepted judge has {key}={accepted.get(key)!r} but the current judge has "
                f"{value!r}; grade a new sheet and accept again"
            )
    texts = english_texts(args.corpus_dir)
    judge = Judge(OllamaLLM(cache=JsonCache(CACHE_DIR / "llm")))
    for system in SYSTEMS:
        run_dir = args.answers_dir / f"{system}__{args.track}__{args.split}"
        if not (run_dir / "records.jsonl").is_file():
            print(f"skipping {run_dir}: no records")
            continue
        records = load_answers(run_dir)
        print(f"judging {len(records)} answers in {run_dir} ...", flush=True)
        judgments = judge_records(records, judge, texts)
        write_judgments(run_dir, judgments)
        labels = labels_from_judgments(judgments)
        supported = sum(labels.values())
        undecided = count_undecided(judgments)
        print(
            f"  {supported} of {len(labels)} claims supported; {undecided} verdicts unreadable "
            "(counted as unsupported)"
        )


if __name__ == "__main__":
    main()
