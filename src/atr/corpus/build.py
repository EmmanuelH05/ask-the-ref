import argparse
import html
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from atr.corpus.align import align
from atr.corpus.fetch import EDITION_PAGE, GLOSSARY_PAGES, RawManifest, raw_filename, sha256_hex
from atr.corpus.glossary import glossary_entries, parse_glossary
from atr.corpus.models import Alignment, BuildReport, Language, Passage
from atr.corpus.parse import parse_law

SOURCE = "theifab.com page-data, laws/latest"
# Pages of the online Laws that carry no law number, so an answer could not cite them.
PAGES_NOT_PARSED = (
    "about-the-laws",
    "notes-on-the-laws",
    "video-assistant-referee-var-protocol",
    "other protocols and guidelines",
)
_EDITION = re.compile(r"(20\d\d/\d\d)\s+edition")
_TAG = re.compile(r"<[^>]*>")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ManifestEntry(_Frozen):
    passage_id: str
    language: Language
    sha256: str
    word_count: int


class CorpusManifest(_Frozen):
    edition: str
    source: str
    fetched_at: str
    passages: list[ManifestEntry]


class CorpusBuild(_Frozen):
    edition: str
    fetched_at: str
    en: list[Passage]
    es: list[Passage]
    alignment: Alignment
    report: BuildReport


def find_edition(notes_html: str) -> str:
    text = html.unescape(_TAG.sub(" ", notes_html)).replace("\xa0", " ")
    match = _EDITION.search(text)
    if match is None:
        raise ValueError("no 'YYYY/YY edition' marker found on the notes page")
    return match.group(1)


def corpus_manifest(
    en: list[Passage], es: list[Passage], *, edition: str, fetched_at: str
) -> CorpusManifest:
    entries = [
        ManifestEntry(
            passage_id=p.passage_id,
            language=p.language,
            sha256=sha256_hex(p.text.encode("utf-8")),
            word_count=p.word_count,
        )
        for p in [*en, *es]
    ]
    entries.sort(key=lambda entry: (entry.language, entry.passage_id))
    return CorpusManifest(edition=edition, source=SOURCE, fetched_at=fetched_at, passages=entries)


def _load(raw_dir: Path, language: Language, page: str) -> dict[str, object]:
    path = raw_dir / language / raw_filename(page)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: not a JSON object")
    return loaded


def _notes_html(page: dict[str, object]) -> str:
    result = page.get("result")
    context = result.get("pageContext") if isinstance(result, dict) else None
    blocks = context.get("contentBlocks") if isinstance(context, dict) else None
    if not isinstance(blocks, list):
        raise ValueError(f"{EDITION_PAGE}: contentBlocks missing")
    return " ".join(
        str(block["htmlContent"])
        for block in blocks
        if isinstance(block, dict) and block.get("htmlContent")
    )


def _parse_edition(
    raw_dir: Path, language: Language, law_slugs: list[str], english_terms: dict[str, list[str]]
) -> tuple[list[Passage], BuildReport]:
    passages: list[Passage] = []
    report = BuildReport()
    for slug in law_slugs:
        found, counts = parse_law(_load(raw_dir, language, slug), language)
        passages.extend(found)
        report = report + counts
    for page in GLOSSARY_PAGES:
        name = page.rsplit("/", 1)[-1]
        found, counts = parse_glossary(
            _load(raw_dir, language, page), language, name, english_terms.get(name)
        )
        passages.extend(found)
        report = report + counts
    return passages, report


def build_corpus(raw_dir: Path) -> CorpusBuild:
    raw = RawManifest.model_validate_json((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    english_terms = {
        page.rsplit("/", 1)[-1]: [term for term, _ in glossary_entries(_load(raw_dir, "en", page))]
        for page in GLOSSARY_PAGES
    }
    en, en_report = _parse_edition(raw_dir, "en", raw.law_slugs, {})
    es, es_report = _parse_edition(raw_dir, "es", raw.law_slugs, english_terms)
    report = en_report + es_report + BuildReport(pages_skipped=list(PAGES_NOT_PARSED))
    return CorpusBuild(
        edition=find_edition(_notes_html(_load(raw_dir, "en", EDITION_PAGE))),
        fetched_at=raw.fetched_at,
        en=en,
        es=es,
        alignment=align(en, es),
        report=report,
    )


def write_corpus(build: CorpusBuild, corpus_dir: Path, manifest_path: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for language, passages in (("en", build.en), ("es", build.es)):
        lines = "".join(p.model_dump_json() + "\n" for p in passages)
        (corpus_dir / f"{language}.jsonl").write_text(lines, encoding="utf-8")
    (corpus_dir / "alignment.json").write_text(build.alignment.model_dump_json(indent=2) + "\n")
    (corpus_dir / "report.json").write_text(build.report.model_dump_json(indent=2) + "\n")
    manifest = corpus_manifest(
        build.en, build.es, edition=build.edition, fetched_at=build.fetched_at
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parse data/raw into aligned passages.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--manifest", type=Path, default=Path("corpus/manifest.json"))
    args = parser.parse_args(argv)
    build = build_corpus(args.raw_dir)
    write_corpus(build, args.corpus_dir, args.manifest)
    print(
        f"edition {build.edition}: {len(build.en)} en passages, {len(build.es)} es passages, "
        f"{len(build.alignment.pairs)} aligned pairs -> {args.corpus_dir}, {args.manifest}"
    )


if __name__ == "__main__":
    main()
