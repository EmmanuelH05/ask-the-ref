import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from atr.evalset.models import HELD_OUT_LAWS, QuestionGroup, Split
from atr.evalset.split import assign_splits
from atr.evalset.store import (
    corpus_ids,
    corpus_pin,
    load_file,
    load_groups,
    validate,
    write_groups,
)

SPLIT_NAMES: tuple[Split, ...] = ("dev", "test", "heldout")
CORPUS_MANIFEST = Path("corpus/manifest.json")


class EvalManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hash: str
    seed: int
    test_fraction: float
    held_out_laws: list[int]
    corpus_pin: str
    reviewed_by: str
    unreviewed_groups: int
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
    corpus_pin: str,
    reviewed_by: str,
) -> EvalManifest:
    return EvalManifest(
        hash=evalset_hash(groups, splits),
        seed=seed,
        test_fraction=test_fraction,
        held_out_laws=list(HELD_OUT_LAWS),
        corpus_pin=corpus_pin,
        reviewed_by=reviewed_by,
        unreviewed_groups=sum(1 for group in groups if not group.spanish_reviewed),
        counts={name: sum(1 for value in splits.values() if value == name) for name in SPLIT_NAMES},
        splits=dict(splits),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="atr.evalset.freeze")
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--draft-dir", type=Path, default=Path("evalset/draft"))
    parser.add_argument("--out", type=Path, default=Path("evalset/questions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("evalset/manifest.json"))
    parser.add_argument("--corpus-manifest", type=Path, default=CORPUS_MANIFEST)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="freeze although some groups lack a human review; the manifest records how many",
    )
    return parser.parse_args(argv)


def verify_frozen(
    questions: Path, manifest_path: Path, corpus_manifest: Path
) -> tuple[list[QuestionGroup], EvalManifest]:
    """Load the frozen set and refuse it if it no longer matches its own manifest."""
    manifest = EvalManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    groups = load_file(questions)
    actual = evalset_hash(groups, manifest.splits)
    if actual != manifest.hash:
        raise ValueError(
            f"{questions} does not match its manifest: hash {actual}, frozen {manifest.hash}"
        )
    pin = corpus_pin(corpus_manifest)
    if pin != manifest.corpus_pin:
        raise ValueError(
            f"the corpus changed since the freeze: pin {pin}, frozen {manifest.corpus_pin}"
        )
    return groups, manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    groups = load_groups(args.draft_dir)
    en_ids, es_ids = corpus_ids(args.corpus_manifest)
    problems = validate(groups, en_ids, es_ids)
    errors = [problem for problem in problems if not problem.startswith("warn: ")]
    for problem in problems:
        print(problem, file=sys.stderr)
    if errors:
        return 1
    unreviewed = [group.group_id for group in groups if not group.spanish_reviewed]
    if unreviewed and not args.allow_unreviewed:
        print(f"spanish_reviewed is false for: {unreviewed}", file=sys.stderr)
        return 1
    test_fraction = 0.5
    splits = assign_splits(groups, seed=args.seed, test_fraction=test_fraction)
    manifest = build_manifest(
        groups,
        splits,
        seed=args.seed,
        test_fraction=test_fraction,
        corpus_pin=corpus_pin(args.corpus_manifest),
        reviewed_by=args.reviewed_by,
    )
    write_groups(groups, args.out)
    args.manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(manifest.hash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
