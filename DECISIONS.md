# Decisions log

Rulings that keep builders consistent. Read before building. Append, never rewrite.

## D1. Passage IDs (phase 1)

`L{law}.{section}` for a law section; `L{law}.0` for an unnumbered introduction;
`L{law}.{section}.{k}` for the k-th `<h3>` block inside a section, k from 1, with k = 0 for
non-empty lead text before the first `<h3>`; `G.{page}.{slug}` for a glossary entry where
`page` is `football-terms`, `referee-terms` or `football-bodies` and `slug` is the English
term lowercased, non-alphanumerics collapsed to `-`, leading and trailing `-` stripped.
Spanish passages carry the same ID as their English twin.

## D2. Passage model (phase 1)

Defined in `src/atr/corpus/models.py`, frozen, `extra="forbid"`:

```python
class Passage(BaseModel):
    passage_id: str          # D1
    language: Literal["en", "es"]
    law_number: int | None   # 1..17, None for glossary
    law_title: str           # chapter.title, or the glossary page title
    section_index: int | None
    section_title: str       # lawSectionCategoryName, or the glossary term
    subheading: str | None   # the <h3> text, when the passage is an h3 block
    source_slug: str         # section slug, or the glossary English-term slug
    text: str                # cleaned text, see D3
    word_count: int          # len(re.findall(r"\w+", text))
```

`Alignment` holds `pairs: list[tuple[str, str]]` of (en passage_id, es passage_id) that are
equal strings, `unaligned_en: list[str]`, `unaligned_es: list[str]`, and
`mismatches: list[str]` (human-readable reasons, one per failed check).
`BuildReport` holds integer counts: `deleted_fragments`, `explanation_attrs`, `images`,
`faq_entries_skipped`, `pages_skipped` (list of slugs).

## D3. HTML cleaning (phase 1)

`clean_html(html: str) -> Cleaned` where `Cleaned` is frozen with `text: str`,
`deleted_fragments: int`, `images: int`, `explanation_attrs: int`.
Rules: drop `<deleted>` and `<deletedexplanation>` elements with their content; keep the
content of `<added>`, `<addedexplanation>`, `<explanation>`, `<strong>`, `<em>`; count and
drop `data-explanation` attributes; count and drop `ResponsiveImage`,
`TheFieldInteractivePicture` and `img`; `<li>` becomes a line starting with `- `; `<p>`
blocks are separated by one blank line; `<br>` is a newline; HTML entities are unescaped and
`\xa0` becomes a space; runs of spaces collapse; leading and trailing whitespace is stripped.
Nested `<ul>` inside `<li>` indents by two spaces per level.

`split_subsections(html: str) -> list[tuple[str | None, str]]` returns (h3 text or None,
raw html) blocks in document order: the lead block (None) only when it has non-whitespace
text, then one block per `<h3>`.

Implementation: `html.parser.HTMLParser` from the standard library. No new dependencies.

## D4. Build outputs (phase 1)

`data/corpus/en.jsonl` and `data/corpus/es.jsonl`: one `Passage.model_dump_json()` per
line, law passages in document order then glossary passages. `data/corpus/alignment.json`:
`Alignment.model_dump_json()`. `data/corpus/report.json`: `BuildReport`.
`corpus/manifest.json` (committed): `{"edition": "2026/27", "source": "theifab.com page-data,
laws/latest", "fetched_at": ..., "passages": [{"passage_id", "language", "sha256", "word_count"}]}`
sorted by (language, passage_id). No passage text anywhere under `corpus/`.

## D5: The TDD hook is not bypassed or disabled by builders (pending mannie)

- Question: the one-test-at-a-time TDD hook blocks task C's edits to `fetch.py` and its test;
  may the builder have it relaxed, or write the files through Bash heredocs instead?
- Ruling: neither, until mannie decides. No builder writes a hook-blocked file by another
  route (Bash heredoc, `cat >`, `tee`, a script), and no builder or arbiter edits hook or
  plugin settings. Task C is paused on the blocked edits. Allowed meanwhile: work the hook
  accepts as is (one new test per edit, then a stub, then the minimal implementation).
  Open question for mannie: turn tdd-guard off for this build, or keep it and wire up its
  pytest reporter so it can see test runs (that touches `pyproject.toml` and adds a dev
  dependency, which is outside any builder's files).
- Why: a hook is a control mannie set up; switching it off or going around it is his call,
  not a design question the plan settles. Note for him: `.claude/settings.json` already lists
  `tdd-guard@tdd-guard` as `false`, yet the hook still fires, which means it is enabled
  from somewhere else.
- Applies to: task C (`src/atr/corpus/fetch.py`, `tests/corpus/test_fetch.py`) and every
  other builder in waves 1 to 3, who will hit the same hook.

Resolved 2026-10-05 by mannie: tdd-guard is off for this repo (he restarted the session with
the plugin disabled). Tests first is still required of every builder and checked in review.

## D6. Law parser interface (phase 1)

`parse_law(page_data: dict[str, object], language: Language) -> tuple[list[Passage], BuildReport]`
in `src/atr/corpus/parse.py`. Reads `result.pageContext`: `chapterIndex` (int, the law number),
`chapter.title` (law_title), `chapter.sections[]` in the given order, `questionsAndAnswers`
(list, may be missing or null). Per section: `sectionIndex` is a numeric STRING ("1") or null;
null means the unnumbered introduction and maps to section_index 0. `lawSectionCategoryName`
is section_title, `slug` is source_slug, `content` is CMS HTML.
Splitting: `split_subsections(content)`. One block with heading None gives one passage
`L{law}.{section}` with subheading None. Otherwise the lead block (heading None) is
`L{law}.{section}.0` with subheading None, and the k-th h3 block (k from 1) is
`L{law}.{section}.{k}` with subheading set to the h3 text. Text is `clean_html(block).text`.
A passage whose cleaned text is empty is not emitted; its id goes in
`BuildReport.empty_passages`. Its h3 number is still consumed, which keeps ids equal across
languages. The report sums `deleted_fragments`, `images` and `explanation_attrs` over every
block cleaned, and `faq_entries_skipped` is `len(questionsAndAnswers or [])`.
Malformed input (missing keys, wrong types, a duplicate passage id within the law) raises
`ValueError` naming the law and the problem. No silent defaults.

## D7. Glossary parser interface (phase 1)

In `src/atr/corpus/glossary.py`:
`slugify(term: str) -> str`: lowercase, every run of characters that are not a-z or 0-9
becomes one `-`, leading and trailing `-` stripped. Brackets and quote marks therefore vanish,
which is what lets "Goal line technology [GLT]" meet "Goal line technology (GLT)".
`glossary_entries(page_data: dict[str, object]) -> list[tuple[str, str]]`: concatenates the
`htmlContent` of every block in `result.pageContext.contentBlocks` that has one (blocks
without it are single-letter dividers and are skipped), and returns (term, definition html)
per `<h4>`, the term cleaned with `clean_html`, the definition being the raw html up to the
next `<h4>`.
`parse_glossary(page_data, language, page, english_terms: list[str] | None = None)
-> tuple[list[Passage], BuildReport]`, `page` being `football-terms`, `referee-terms` or
`football-bodies`. The slug of an entry is chosen by the first rule that applies:
1. language "en": `slugify(term)`.
2. the term ends with an arrow annotation matching `\(\s*→\s*(.+?)\s*\)\s*$`: the slug is
   `slugify` of the captured English term and the annotation is removed from the displayed term.
3. `english_terms` is given: the slug is `slugify(english_terms[i])` for the entry at position i;
   a length mismatch raises `ValueError`.
4. otherwise `slugify(term)` (the entry will simply fail to align and be listed).
Rule 3 is only used for a page where NO Spanish term carries an arrow; on a page with any
arrows, rule 4 is the fallback. Passage fields: passage_id `G.{page}.{slug}`, law_number None,
law_title = `pageContext.title`, section_index None, section_title = the displayed term,
subheading None, source_slug = slug, text = `clean_html(definition).text`. Empty definitions
are not emitted and go in `empty_passages`. Duplicate ids are NOT an error here (two Spanish
synonyms can point at one English term); `align` reports them.
