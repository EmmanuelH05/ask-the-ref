import argparse
import json
import math
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from atr.corpus.models import Alignment, BuildReport, Passage

DEFAULT_SEED = 0
DEFAULT_PAIRS = 5
DEFAULT_MAX_WORDS = 60


class Distribution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    total: int
    minimum: int
    p25: int
    median: int
    p75: int
    p90: int
    maximum: int
    mean: float


def _nearest_rank(ordered: list[int], percent: int) -> int:
    return ordered[max(math.ceil(percent / 100 * len(ordered)), 1) - 1]


def distribution(values: list[int]) -> Distribution:
    if not values:
        raise ValueError("distribution of an empty list")
    ordered = sorted(values)
    return Distribution(
        n=len(ordered),
        total=sum(ordered),
        minimum=ordered[0],
        p25=_nearest_rank(ordered, 25),
        median=_nearest_rank(ordered, 50),
        p75=_nearest_rank(ordered, 75),
        p90=_nearest_rank(ordered, 90),
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
    )


def counts_by_law(passages: list[Passage]) -> dict[int | None, int]:
    """Passages per law number, laws ascending, glossary (None) last."""
    counts: dict[int | None, int] = {}
    for passage in passages:
        counts[passage.law_number] = counts.get(passage.law_number, 0) + 1
    laws = sorted(law for law in counts if law is not None)
    ordered: dict[int | None, int] = {law: counts[law] for law in laws}
    if None in counts:
        ordered[None] = counts[None]
    return ordered


def sample_pairs(ids: list[str], k: int, *, seed: int) -> list[str]:
    if len(ids) <= k:
        return list(ids)
    return random.Random(seed).sample(ids, k)


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def _load_passages(path: Path) -> list[Passage]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Passage.model_validate_json(line) for line in lines]


def _label(passage: Passage) -> str:
    title = f"{passage.law_title} / {passage.section_title}"
    return f"{title} / {passage.subheading}" if passage.subheading else title


def _print_counts(en: list[Passage], es: list[Passage]) -> None:
    en_counts, es_counts = counts_by_law(en), counts_by_law(es)
    titles = {p.law_number: p.law_title for p in en if p.law_number is not None}
    print("Passages per law\n  law  en  es  title")
    for law in sorted(k for k in en_counts.keys() | es_counts.keys() if k is not None):
        print(f"  {law:>3} {en_counts.get(law, 0):>3} {es_counts.get(law, 0):>3}  {titles[law]}")
    print(f"  glo {en_counts.get(None, 0):>3} {es_counts.get(None, 0):>3}  Glossary")
    print(f"  all {len(en):>3} {len(es):>3}")


def _print_distribution(name: str, passages: list[Passage]) -> None:
    d = distribution([p.word_count for p in passages])
    print(
        f"  {name}: n={d.n} total={d.total} min={d.minimum} p25={d.p25} median={d.median} "
        f"p75={d.p75} p90={d.p90} max={d.maximum} mean={d.mean:.1f}"
    )


def _print_alignment(alignment: Alignment, en: list[Passage], es: list[Passage]) -> None:
    total = len(alignment.pairs) + len(alignment.unaligned_en) + len(alignment.unaligned_es)
    print(
        f"\nAlignment: {len(alignment.pairs)} of {total} passage ids paired ({alignment.rate:.1%})"
    )
    by_id = {"en": {p.passage_id: p for p in en}, "es": {p.passage_id: p for p in es}}
    for language, ids in (("en", alignment.unaligned_en), ("es", alignment.unaligned_es)):
        print(f"Unaligned {language}: {len(ids)}")
        for passage_id in ids:
            print(f"  {passage_id}  {_label(by_id[language][passage_id])}")
    print(f"Mismatches: {len(alignment.mismatches)}")
    for line in alignment.mismatches:
        print(f"  {line}")


def _print_report(report: BuildReport) -> None:
    print("\nDropped (both editions together)")
    print(f"  previous-edition fragments (deleted text): {report.deleted_fragments}")
    print(f"  change-rationale attributes: {report.explanation_attrs}")
    print(f"  images: {report.images}")
    print(f"  IFAB FAQ entries not used: {report.faq_entries_skipped}")
    print(f"  sections with no text after cleaning: {report.empty_passages}")
    print(f"  group headings carried onto the passages under them: {report.group_headings}")
    print(f"  glossary cross-reference stubs: {report.cross_references_dropped}")
    print(f"  pages not parsed: {', '.join(report.pages_skipped)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Corpus statistics, the phase 1 check.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--manifest", type=Path, default=Path("corpus/manifest.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    args = parser.parse_args(argv)
    en = _load_passages(args.corpus_dir / "en.jsonl")
    es = _load_passages(args.corpus_dir / "es.jsonl")
    alignment = Alignment.model_validate_json((args.corpus_dir / "alignment.json").read_text())
    report = BuildReport.model_validate_json((args.corpus_dir / "report.json").read_text())
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(f"Edition {manifest['edition']} | source: {manifest['source']}")
    print(f"Fetched {manifest['fetched_at']}\n")
    _print_counts(en, es)
    print("\nWords per passage (regex \\w+, not model tokens)")
    _print_distribution("en", en)
    _print_distribution("es", es)
    _print_alignment(alignment, en, es)
    _print_report(report)
    en_by_id = {p.passage_id: p for p in en}
    es_by_id = {p.passage_id: p for p in es}
    chosen = sample_pairs([pair[0] for pair in alignment.pairs], args.pairs, seed=args.seed)
    print(f"\n{len(chosen)} random aligned pairs (seed {args.seed}, cut at {args.max_words} words)")
    for passage_id in chosen:
        for passage in (en_by_id[passage_id], es_by_id[passage_id]):
            head = f"[{passage_id}] {passage.language} | {_label(passage)}"
            print(f"\n{head} | {passage.word_count} words")
            print(truncate_words(passage.text, args.max_words))


if __name__ == "__main__":
    main()
