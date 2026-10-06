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
