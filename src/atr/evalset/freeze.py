import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from atr.evalset.models import HELD_OUT_LAWS, QuestionGroup, Split
from atr.evalset.split import assign_splits
from atr.evalset.store import corpus_ids, load_groups, validate, write_groups

SPLIT_NAMES: tuple[Split, ...] = ("dev", "test", "heldout")
CORPUS_MANIFEST = Path("corpus/manifest.json")


class EvalManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hash: str
    seed: int
    test_fraction: float
    held_out_laws: list[int]
    corpus_manifest_sha256: str
    reviewed_by: str
    counts: dict[str, int]
    splits: dict[str, Split]


def canonical_bytes(groups: list[QuestionGroup], splits: dict[str, Split]) -> bytes:
    known = {group.group_id for group in groups}
    unknown = sorted(set(splits) - known)
    if unknown:
        raise ValueError(f"split for unknown group: {unknown}")
    missing = sorted(known - set(splits))
    if missing:
        raise ValueError(f"group without a split: {missing}")
    rows = [
        {**group.model_dump(exclude={"spanish_reviewed"}), "split": splits[group.group_id]}
        for group in sorted(groups, key=lambda group: group.group_id)
    ]
    text = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8")


def evalset_hash(groups: list[QuestionGroup], splits: dict[str, Split]) -> str:
    return hashlib.sha256(canonical_bytes(groups, splits)).hexdigest()


def build_manifest(
    groups: list[QuestionGroup],
    splits: dict[str, Split],
    *,
    seed: int,
    test_fraction: float,
    corpus_manifest_sha256: str,
    reviewed_by: str,
) -> EvalManifest:
    return EvalManifest(
        hash=evalset_hash(groups, splits),
        seed=seed,
        test_fraction=test_fraction,
        held_out_laws=list(HELD_OUT_LAWS),
        corpus_manifest_sha256=corpus_manifest_sha256,
        reviewed_by=reviewed_by,
        counts={name: sum(1 for value in splits.values() if value == name) for name in SPLIT_NAMES},
        splits=dict(splits),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atr.evalset.freeze")
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--draft-dir", type=Path, default=Path("evalset/draft"))
    parser.add_argument("--out", type=Path, default=Path("evalset/questions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("evalset/manifest.json"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    groups = load_groups(args.draft_dir)
    en_ids, es_ids = corpus_ids(CORPUS_MANIFEST)
    problems = validate(groups, en_ids, es_ids)
    errors = [problem for problem in problems if not problem.startswith("warn: ")]
    for problem in problems:
        print(problem, file=sys.stderr)
    if errors:
        return 1
    unreviewed = [group.group_id for group in groups if not group.spanish_reviewed]
    if unreviewed:
        print(f"spanish_reviewed is false for: {unreviewed}", file=sys.stderr)
        return 1
    test_fraction = 0.5
    splits = assign_splits(groups, seed=args.seed, test_fraction=test_fraction)
    manifest = build_manifest(
        groups,
        splits,
        seed=args.seed,
        test_fraction=test_fraction,
        corpus_manifest_sha256=hashlib.sha256(CORPUS_MANIFEST.read_bytes()).hexdigest(),
        reviewed_by=args.reviewed_by,
    )
    write_groups(groups, args.out)
    args.manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(manifest.hash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
