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
    passage_id: str  # D1
    language: Literal["en", "es"]
    law_number: int | None  # 1..17, None for glossary
    law_title: str  # chapter.title, or the glossary page title
    section_index: int | None
    section_title: str  # lawSectionCategoryName, or the glossary term
    subheading: str | None  # the <h3> text, when the passage is an h3 block
    source_slug: str  # section slug, or the glossary English-term slug
    text: str  # cleaned text, see D3
    word_count: int  # len(re.findall(r"\w+", text))
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

## D8. Glossary cross-reference stubs are dropped (phase 1, main session, 2026-10-05)

A glossary entry whose cleaned definition starts with "V. " (Spanish) or "See " (English) and
has at most 6 words only points at another entry. It is not emitted as a passage and is
listed in `BuildReport.cross_references_dropped` as `"{passage_id} ({term})"`.
Why: the Spanish edition lists synonyms twice, once with the definition and once as a stub,
and both carry the same English term, which made four duplicate ids. A stub has nothing to
retrieve or cite.

## D9. List items keep their bullet when the CMS wraps them in a paragraph (phase 1)

The real markup is `<li> <p>text</p> </li>`. Inside a list, `<p>` ends a line but adds no blank
line; the list as a whole is separated from surrounding paragraphs by one blank line. This
corrects the first build of `html.py`, whose tests only used `<li>text</li>`.

## D10. Fixes from the phase 1 review (main session, 2026-10-05)

- Group headings: an `<h3>` in capitals is a group heading. The plain `<h3>` blocks after it,
  up to the next group heading, get `subheading = "{GROUP} / {heading}"`. An empty group
  block is listed in `BuildReport.group_headings`, not in `empty_passages`. Ids do not change.
  Why: Law 12 section 4 repeats "Warning", "Caution" and "Sending-off" once for players and
  once for team officials, and without the group a citation could not tell them apart.
- Alignment: if any `L{law}.{section}.{k}` id of a section fails to pair, every pair in that
  section is unpaired and one mismatch line is added. Subsection ids are positional, so one
  missing block shifts the rest.
- Glossary: a paragraph opening with a bold-only line and a `<br>` is an entry boundary. The
  English edition has one entry typed that way instead of with an `<h4>`.
- Fetch: a response body is parsed as JSON before it is written to the cache.
- Lists: later paragraphs of one `<li>`, and text after a nested list, are indented two
  spaces under the item's bullet.
- Known and left: a lead block holding only an image (no text) is not counted in `images`.
  The real data has none.

## D11. Held-out laws are 3, 5 and 16 (mannie delegated the choice, 2026-10-05)

Law 16 is short, Law 3 is long, Law 5 is long and list-heavy. Law 14 was dropped as a
hold-out: its summary table is an image and it has only three passages, too few to carry a
slice. No law in the online edition has a text table. `HELD_OUT_LAWS` in
`src/atr/evalset/models.py` is the single source.

## D12. Word overlap (phase 2)

`src/atr/evalset/overlap.py`:
`content_tokens(text: str) -> list[str]`: lowercase; Unicode NFKD with combining marks
removed (so "árbitro" equals "arbitro"); tokens are `[a-z0-9]+` runs; tokens in `STOPWORDS`
(one frozenset holding common English and Spanish function words, written out in the module)
are dropped; a token longer than 3 characters loses one trailing "s". Order is kept and
duplicates are kept.
`overlap(question: str, passages: list[str]) -> float`: the share of DISTINCT question
content tokens that occur among the content tokens of the passages joined together. 0.0 when
the question has no content tokens or `passages` is empty.
`GroupOverlap` (frozen): `group_id`, and for each question language a pair of floats or None:
`en_vs_en`, `es_vs_es`, `cs_vs_en`, `cs_vs_es`, `es_vs_en`, `en_vs_es`. None when the group has
no gold or the gold passage is missing in that edition.
`group_overlap(group: QuestionGroup, en: Mapping[str, str], es: Mapping[str, str]) -> GroupOverlap`
where the mappings go from passage id to passage text.

## D13. Split (phase 2)

`src/atr/evalset/split.py`:
`assign_splits(groups: list[QuestionGroup], *, seed: int, test_fraction: float = 0.5)
-> dict[str, Split]`. A group for which `is_held_out` is true gets "heldout". The rest are
handled one type at a time: group ids sorted, shuffled with `random.Random(f"{seed}:{type}")`,
the first `round(n * test_fraction)` get "test" and the others "dev". The result has the same
key order as `groups`. Duplicate group ids raise `ValueError`. `test_fraction` outside (0, 1)
raises `ValueError`.

## D14. Store and validation (phase 2)

`src/atr/evalset/store.py`:
`load_groups(directory: Path) -> list[QuestionGroup]`: every `*.jsonl` file in the directory,
files in name order, lines in file order, blank lines skipped; a bad line raises `ValueError`
naming the file and line number.
`write_groups(groups: list[QuestionGroup], path: Path) -> None`: one `model_dump_json()` per line.
`validate(groups: list[QuestionGroup], en_ids: set[str], es_ids: set[str]) -> list[str]`:
returns problems, empty when clean. Errors: a duplicate `group_id`; a gold id not in
`en_ids`; a question type with no group; two groups with the same English text after
lowercasing and collapsing whitespace. Warnings, prefixed `warn: `: a gold id in `en_ids`
but not in `es_ids`.
`corpus_ids(manifest_path: Path) -> tuple[set[str], set[str]]`: the en and es passage ids
from `corpus/manifest.json`.

## D15. Freeze (phase 2)

`src/atr/evalset/freeze.py`:
`canonical_bytes(groups: list[QuestionGroup], splits: dict[str, Split]) -> bytes`: UTF-8 JSON
of a list sorted by `group_id`, each element the group's `model_dump()` minus
`spanish_reviewed`, plus `"split"`; `sort_keys=True`, `ensure_ascii=False`,
separators `(",", ":")`. A group with no split, or a split for an unknown group, raises
`ValueError`.
`evalset_hash(groups, splits) -> str`: sha256 hex of those bytes.
`EvalManifest` (frozen): `hash`, `seed`, `test_fraction`, `held_out_laws: list[int]`,
`corpus_manifest_sha256`, `reviewed_by`, `counts: dict[str, int]` (groups per split),
`splits: dict[str, Split]`.
`build_manifest(groups, *, seed, test_fraction, corpus_manifest_sha256, reviewed_by) -> EvalManifest`.
CLI `python -m atr.evalset.freeze --reviewed-by NAME [--draft-dir evalset/draft]
[--out evalset/questions.jsonl] [--manifest evalset/manifest.json] [--seed 0]`: loads,
validates (any error aborts with exit code 1 and prints the problems), refuses unless every
group has `spanish_reviewed` true, writes the questions file and the manifest, prints the hash.
`--reviewed-by` is required: the set is only frozen after a named human reviewed it.
