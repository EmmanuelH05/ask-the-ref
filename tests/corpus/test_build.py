import hashlib
import json
from pathlib import Path

import pytest

from atr.corpus.build import build_corpus, corpus_manifest, find_edition, write_corpus
from atr.corpus.models import Passage, count_words


def passage(passage_id: str, language: str, text: str) -> Passage:
    return Passage.model_validate(
        {
            "passage_id": passage_id,
            "language": language,
            "law_number": 1,
            "law_title": "Invented",
            "section_index": 1,
            "section_title": "Invented",
            "subheading": None,
            "source_slug": "invented",
            "text": text,
            "word_count": count_words(text),
        }
    )


def law_page(number: int, title: str, body: str) -> dict[str, object]:
    section = {
        "sectionIndex": "1",
        "lawSectionCategoryName": "Scoring",
        "slug": "scoring",
        "content": f"<p>{body}</p>",
    }
    context = {
        "chapterIndex": number,
        "chapter": {"title": title, "sections": [section]},
        "questionsAndAnswers": [{"question": "q"}],
    }
    return {"result": {"pageContext": context}}


def glossary_page(title: str, html: str) -> dict[str, object]:
    return {"result": {"pageContext": {"title": title, "contentBlocks": [{"htmlContent": html}]}}}


def write_raw(raw_dir: Path) -> None:
    pages: dict[str, dict[str, object]] = {
        "en/zorb-one": law_page(1, "The Zorb", "A zorb is worth three points."),
        "es/zorb-one": law_page(1, "El zorb", "Un zorb vale tres puntos."),
        "en/glossary__football-terms": glossary_page("Terms", "<h4>Zorb</h4><p>A round thing.</p>"),
        "es/glossary__football-terms": glossary_page(
            "Términos", "<h4>Zorbo (&rarr;&nbsp;zorb)</h4><p>Una cosa redonda.</p>"
        ),
        "en/glossary__referee-terms": glossary_page("Refs", "<h4>Umpire</h4><p>Decides.</p>"),
        "es/glossary__referee-terms": glossary_page("Árbitros", "<h4>Juez</h4><p>Decide.</p>"),
        "en/glossary__football-bodies": glossary_page("Bodies", "<h4>ZFA</h4><p>Runs zorb.</p>"),
        "es/glossary__football-bodies": glossary_page("Órganos", "<h4>ZFA</h4><p>Dirige.</p>"),
        "en/notes-on-the-laws": {
            "result": {
                "pageContext": {
                    "contentBlocks": [
                        {"htmlContent": "<p>The template for the 2031/32 edition of the Laws.</p>"}
                    ]
                }
            }
        },
    }
    for name, page in pages.items():
        target = raw_dir / f"{name}.page-data.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(page))
    manifest = {"fetched_at": "2031-07-01T00:00:00+00:00", "law_slugs": ["zorb-one"], "files": []}
    (raw_dir / "manifest.json").write_text(json.dumps(manifest))


def test_find_edition_reads_the_season_before_the_word_edition() -> None:
    assert find_edition("<p>layout for the 2031/32&nbsp;edition of the Laws</p>") == "2031/32"


def test_find_edition_raises_when_absent() -> None:
    with pytest.raises(ValueError):
        find_edition("<p>no season here</p>")


def test_corpus_manifest_holds_hashes_and_no_text() -> None:
    en = [passage("L1.2", "en", "blue wombats juggle"), passage("L1.1", "en", "red wombats")]
    es = [passage("L1.1", "es", "wombats rojos")]
    manifest = corpus_manifest(en, es, edition="2031/32", fetched_at="2031-07-01T00:00:00+00:00")
    assert manifest.edition == "2031/32"
    assert [(e.language, e.passage_id) for e in manifest.passages] == [
        ("en", "L1.1"),
        ("en", "L1.2"),
        ("es", "L1.1"),
    ]
    assert manifest.passages[0].sha256 == hashlib.sha256(b"red wombats").hexdigest()
    assert manifest.passages[0].word_count == 2
    assert "wombats" not in manifest.model_dump_json()


def test_build_corpus_parses_aligns_and_reports(tmp_path: Path) -> None:
    write_raw(tmp_path)
    build = build_corpus(tmp_path)
    assert build.edition == "2031/32"
    assert [p.passage_id for p in build.en] == [
        "L1.1",
        "G.football-terms.zorb",
        "G.referee-terms.umpire",
        "G.football-bodies.zfa",
    ]
    assert [p.passage_id for p in build.es] == [p.passage_id for p in build.en]
    assert build.alignment.rate == 1.0
    assert build.report.faq_entries_skipped == 2
    assert build.es[1].section_title == "Zorbo"


def test_write_corpus_writes_text_under_data_and_only_hashes_to_manifest(tmp_path: Path) -> None:
    raw, corpus, manifest_path = tmp_path / "raw", tmp_path / "corpus", tmp_path / "m.json"
    write_raw(raw)
    write_corpus(build_corpus(raw), corpus, manifest_path)
    lines = (corpus / "en.jsonl").read_text().splitlines()
    assert Passage.model_validate_json(lines[0]).text == "A zorb is worth three points."
    assert len((corpus / "es.jsonl").read_text().splitlines()) == 4
    assert json.loads((corpus / "alignment.json").read_text())["unaligned_en"] == []
    assert json.loads((corpus / "report.json").read_text())["faq_entries_skipped"] == 2
    assert "three points" not in manifest_path.read_text()
    assert len(json.loads(manifest_path.read_text())["passages"]) == 8
