import argparse
import statistics
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from atr.corpus.models import Passage
from atr.evalset.freeze import CORPUS_MANIFEST, verify_frozen
from atr.evalset.models import QUESTION_LANGUAGES, QUESTION_TYPES, QuestionGroup, Split, expand
from atr.evalset.overlap import GroupOverlap, group_overlap
from atr.evalset.split import assign_splits
from atr.evalset.store import load_groups

SPLITS: tuple[Split, ...] = ("dev", "test", "heldout")
# What each column measures: the question language against the edition it is compared with.
OVERLAP_COLUMNS = ("en_vs_en", "es_vs_es", "es_vs_en", "cs_vs_en", "cs_vs_es")


class Summary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    minimum: float
    median: float
    mean: float
    maximum: float


def summarize(values: list[float]) -> Summary | None:
    if not values:
        return None
    return Summary(
        n=len(values),
        minimum=min(values),
        median=statistics.median(values),
        mean=statistics.fmean(values),
        maximum=max(values),
    )


def count_table(
    groups: list[QuestionGroup], splits: dict[str, Split]
) -> dict[tuple[str, str], int]:
    """Groups per (type, split). Every group is one question in each language."""
    table: dict[tuple[str, str], int] = {
        (kind, split): 0 for kind in QUESTION_TYPES for split in SPLITS
    }
    for group in groups:
        table[(group.type, splits[group.group_id])] += 1
    return table


def _texts(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {p.passage_id: p.text for p in map(Passage.model_validate_json, lines)}


def _print_counts(groups: list[QuestionGroup], splits: dict[str, Split]) -> None:
    table = count_table(groups, splits)
    print("Groups by type and split (each group is one question per language)")
    print(f"  {'type':<14}" + "".join(f"{split:>9}" for split in SPLITS) + f"{'all':>9}")
    for kind in QUESTION_TYPES:
        row = [table[(kind, split)] for split in SPLITS]
        print(f"  {kind:<14}" + "".join(f"{n:>9}" for n in row) + f"{sum(row):>9}")
    totals = [sum(table[(kind, split)] for kind in QUESTION_TYPES) for split in SPLITS]
    print(f"  {'all':<14}" + "".join(f"{n:>9}" for n in totals) + f"{sum(totals):>9}")
    print("\nQuestions by language and split")
    questions = [(q.language, splits[q.group_id]) for g in groups for q in expand(g)]
    for language in QUESTION_LANGUAGES:
        row = [questions.count((language, split)) for split in SPLITS]
        print(f"  {language:<14}" + "".join(f"{n:>9}" for n in row) + f"{sum(row):>9}")
    row = [sum(1 for _, s in questions if s == split) for split in SPLITS]
    print(f"  {'all':<14}" + "".join(f"{n:>9}" for n in row) + f"{sum(row):>9}")


def _print_overlap(groups: list[QuestionGroup], scores: dict[str, GroupOverlap]) -> None:
    print("\nWord overlap with the gold passages, per type (n, min, median, mean, max)")
    for kind in QUESTION_TYPES:
        print(f"  {kind}")
        for column in OVERLAP_COLUMNS:
            values = [
                value
                for g in groups
                if g.type == kind and (value := getattr(scores[g.group_id], column)) is not None
            ]
            s = summarize(values)
            cell = (
                "no gold passages"
                if s is None
                else f"n={s.n} min={s.minimum:.2f} median={s.median:.2f} "
                f"mean={s.mean:.2f} max={s.maximum:.2f}"
            )
            print(f"    {column:<9} {cell}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluation set statistics, the phase 2 check.")
    parser.add_argument("--questions", type=Path, default=Path("evalset/questions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("evalset/manifest.json"))
    parser.add_argument("--draft-dir", type=Path, default=Path("evalset/draft"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--corpus-manifest", type=Path, default=CORPUS_MANIFEST)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.manifest.exists():
        groups, manifest = verify_frozen(args.questions, args.manifest, args.corpus_manifest)
        splits = manifest.splits
        print(f"FROZEN set, hash {manifest.hash} (recomputed and matched)")
        print(f"Reviewed by: {manifest.reviewed_by}")
        print(f"Groups without a human review: {manifest.unreviewed_groups} of {len(groups)}")
        print(f"Seed {manifest.seed}, held-out laws {manifest.held_out_laws}\n")
    else:
        groups = load_groups(args.draft_dir)
        splits = assign_splits(groups, seed=args.seed)
        print(f"DRAFT set, not frozen. Provisional split with seed {args.seed}.\n")
    _print_counts(groups, splits)
    en, es = _texts(args.corpus_dir / "en.jsonl"), _texts(args.corpus_dir / "es.jsonl")
    _print_overlap(groups, {g.group_id: group_overlap(g, en, es) for g in groups})


if __name__ == "__main__":
    main()
