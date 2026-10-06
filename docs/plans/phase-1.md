# Phase 1 plan: corpus in two languages

Goal: turn the 2026/27 IFAB Laws of the Game, English and Spanish, into aligned structured
passages under `data/corpus/`, with a committed manifest of passage IDs and hashes (no text).

## What the source looks like (verified 2026-10-05)

theifab.com is a Gatsby site. Every law page has a JSON twin at
`https://www.theifab.com/page-data/{es/}laws/latest/<slug>/page-data.json` whose
`result.pageContext` holds:

- `chapterIndex` (1..17), `chapter.header` ("Law 11" / "Regla 11"), `chapter.title`
- `chapter.sections[]`: `sectionIndex` (1.., or `null` for an unnumbered introduction),
  `slug` (identical in both languages), `lawSectionCategoryName` (the section title in that
  language), `content` (CMS HTML), `images[]`
- `nextChapterPath`: the chain from Law 1 to Law 17
- `questionsAndAnswers[]`: IFAB's FAQ, which we do not use

Glossary pages (`glossary/football-terms`, `glossary/referee-terms`, `glossary/football-bodies`)
hold `contentBlocks[]` with `htmlContent` of repeated `<h4>Term</h4><p>definition</p>`.
Spanish glossary terms carry the English term as `Término (→ english term)`.

The `notes-on-the-laws` page names the edition ("the 2026/27 edition of the Laws") and says
English is authoritative when wordings diverge.

Section `content` markup, counted over all 17 laws in both languages: `p`, `ul`/`li`, `h3`
(sub-headings, 48 in each language), `strong`, `em`, `br`, and IFAB's change-tracking tags
`added`, `deleted`, `explanation`, `addedexplanation`, `deletedexplanation` (each with a
`data-explanation` attribute holding the rationale), plus `ResponsiveImage`,
`TheFieldInteractivePicture` and `img`. No `<table>` anywhere. Law 14's "Summary table"
section is a single PNG in both languages.

## Passage model

One passage = the finest unit IFAB's own structure gives. A section with no `h3` is one
passage. A section with `h3` sub-headings is split at them. Rationale: Law 12 section 4 is
about 23,000 characters, and a citation to it would point at five pages; the sub-headings
are IFAB's structure, not ours, and they come in equal numbers in both languages.

Passage ID, identical across languages:

- `L{law}.{section}` for a law section (`L11.1`); `L{law}.0` for an unnumbered introduction
- `L{law}.{section}.{k}` for the k-th `h3` block of a section (k from 1); `k = 0` for lead text
  before the first `h3` when it is non-empty
- `G.{page}.{slug}` for a glossary entry, slug from the English term (`G.football-terms.advantage`)

Alignment: law passages align on ID, checked by equal section `slug` and equal `h3` counts per
section; glossary football-terms align by the English term in the Spanish arrow annotation;
referee-terms and football-bodies align by position. Anything that fails is listed, not guessed.

## What is kept, what is dropped

Kept: text inside `added`, `addedexplanation` and `explanation` (that is the current law
text); list items as `- ` lines; paragraphs separated by a blank line; sub-heading text as
the passage title.
Dropped and counted: text inside `deleted` and `deletedexplanation` (previous edition's
wording); every `data-explanation` attribute (change rationale, not law); all images,
including the Law 14 summary table (there is no text version); the FAQ blocks (they would hand
the scenario questions their answers); `about-the-laws`, `notes-on-the-laws`, the VAR protocol
and the other protocols and guidelines (no law number, out of scope for "cite the law").

## Files

```
src/atr/corpus/models.py    Passage, Alignment, BuildReport (frozen pydantic, extra="forbid")
src/atr/corpus/html.py      pure: CMS HTML -> text; h3 split; drop counts
src/atr/corpus/parse.py     pure: one law page-data dict -> list[Passage]
src/atr/corpus/glossary.py  pure: one glossary page-data dict -> list[Passage]
src/atr/corpus/align.py     pure: en + es passages -> Alignment
src/atr/corpus/fetch.py     network: page-data JSON -> data/raw/{en,es}/, data/raw/manifest.json
src/atr/corpus/build.py     CLI: data/raw -> data/corpus/{en,es}.jsonl, alignment.json, report.json,
                            corpus/manifest.json (committed: ids + sha256 + word counts, no text)
src/atr/corpus/stats.py     CLI: the phase check
tests/corpus/test_*.py      one per pure module; fixtures are invented text, never IFAB text
```

Words are counted with the regex `\w+` on the cleaned text and reported as words, not model
tokens; the embedding tokenizer arrives in phase 3.

## Waves

Wave 0 (main session): `models.py`, its test, decisions log.
Wave 1: `html.py` + test, `align.py` + test, `fetch.py` + test of its pure URL helper.
Wave 2: `parse.py` + test, `glossary.py` + test.
Wave 3: `build.py`, `stats.py`.
Then: fetch for real, build, run the check, review the full diff, commit.

## Check

`uv run python -m atr.corpus.stats` prints passages per law per language, the word-length
distribution, the alignment rate, every unaligned passage, the drop counts, and 5 random
aligned pairs. `uv run pytest -q && uv run mypy --strict src && uv run ruff check .` is clean.
