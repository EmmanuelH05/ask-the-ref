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

## D16. Overlap tokenizer amendments (phase 2, main session, 2026-10-05)

Amends D12 before any question is frozen: single-character tokens are dropped (a possessive
left a stray "s" that matched across texts), and "cannot", "antes", "después" and "durante"
join `STOPWORDS`. Found from false leak flags on the first review sheet.

## D17. Fixes from the phase 2 review (main session, 2026-10-05)

- Stemming (amends D12, D16): apostrophes are deleted before tokenizing, so contractions stay
  one stopword and possessives stem to the noun; a token of 4 or more characters loses a
  final "s" unless it ends in "ss", and then a final "e". That makes "jugadores" meet
  "jugador" and "offences" meet "offence".
- `build_manifest(groups, splits, *, seed, test_fraction, corpus_pin, reviewed_by)`: `splits` is
  positional (amends D15). The manifest field is `corpus_pin`, a hash of the sorted
  (language, passage id, text hash) entries of `corpus/manifest.json`, not of the file bytes,
  so a refetch of identical text keeps the pin.
- `verify_frozen(questions, manifest, corpus_manifest)` recomputes the hash and the pin and
  raises on any mismatch. The stats command and every later eval run go through it.
- The freeze CLI takes `--allow-unreviewed`; the manifest records `unreviewed_groups`.
- The blind-check flag fires for any group without gold, not only out_of_scope; an unchecked
  group is flagged "not blind-checked"; a missing blind directory, a repeated group and picks
  for unknown groups raise. Code-switched paraphrases are leak-checked against both editions.

## D18. Provisional freeze (main session, 2026-10-05)

mannie said to keep going before reviewing the questions. The set is frozen now with
`--allow-unreviewed` so that the split and the gold labels are locked before any retrieval
work. `reviewed_by` says the review is pending, and the manifest counts every group as
unreviewed. Rules that follow:
- No test-split result is produced or looked at until mannie has reviewed the set and it
  has been re-frozen under his name. Dev and held-out results before that are provisional.
- A re-freeze after his fixes keeps every existing group in its current split (the old
  manifest's splits are reused; only new groups are assigned), so edits cannot reshuffle test.
- Nothing may claim that the Spanish or code-switched text was written or reviewed by a
  native speaker until that re-freeze.

## D19. Retrieval core (phase 3)

`src/atr/retrieval/base.py` (written): `Hit(passage_id, score)` and `Doc(passage_id, language,
text)` are NamedTuples; `Retriever` is a Protocol with `name: str` and
`search(query: str, k: int) -> list[Hit]`; `top_hits(hits, k)` keeps the best score per
passage id, sorted by score descending then id; `load_docs(corpus_dir, languages)`.
Every retriever returns its result through `top_hits`, because several docs can share a
passage id (two editions now, chunks in phase 4). `src/atr/text.py` holds `fold`, `stem`,
`raw_tokens`, `search_tokens`. `src/atr/cache.py` holds `cache_key(*parts)` and `JsonCache`.

## D21. BM25 (phase 3)

`src/atr/retrieval/bm25.py`: `class BM25Retriever` with
`__init__(self, docs: list[Doc], *, k1: float = 1.5, b: float = 0.75, name: str = "bm25")` and
`search`. Tokens come from `atr.text.search_tokens` for docs and queries. For N docs, a term
with document frequency df has `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`. A doc of length dl
(token count) scores, summed over the DISTINCT query tokens,
`idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`. Docs scoring 0 are not returned.
An empty doc list raises `ValueError`. A query with no tokens returns `[]`. k1 and b are
fixed defaults and are not tuned in this project. The index is built once in `__init__`
(postings per term), and a search touches only the postings of the query terms.

## D22. Metrics (phase 3)

`src/atr/eval/metrics.py`, all pure, `ranked` being passage ids best first and `gold` a set:
`recall_at_k(ranked, gold, k) = |gold ∩ ranked[:k]| / |gold|`;
`reciprocal_rank(ranked, gold)` = 1 / (1-based rank of the first gold id), 0.0 if none;
`ndcg_at_k(ranked, gold, k)` with binary relevance: DCG = sum over positions i (1-based, up to
k) holding a gold id of `1 / log2(i + 1)`; IDCG = the same sum over the first `min(|gold|, k)`
positions; result DCG / IDCG.
`QuestionScores` (frozen pydantic): `recall_at_1`, `recall_at_5`, `mrr`, `ndcg_at_10`.
`score_question(ranked: list[str], gold: set[str]) -> QuestionScores`.
Every function raises `ValueError` on an empty gold set, on duplicate ids in `ranked`, and on
k < 1. `METRICS: tuple[str, ...] = ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_10")`.

## D23. Bootstrap (phase 3)

`src/atr/eval/bootstrap.py`, numpy with `numpy.random.default_rng(seed)`:
`Interval` (frozen pydantic): `mean`, `low`, `high`, `n` (values), `n_clusters`, `resamples`,
`seed`, `confidence`.
`bootstrap_ci(values: Sequence[float], *, clusters: Sequence[str] | None = None,
resamples: int = 10000, seed: int = 0, confidence: float = 0.95) -> Interval`: percentile
bootstrap of the mean. With `clusters` (one label per value) whole clusters are resampled
with replacement and the mean is taken over all values of the drawn clusters (a cluster drawn
twice counts twice); with `clusters=None` every value is its own cluster. Why clusters: the
English, Spanish and code-switched versions of one question are not independent, so a pooled
number must resample question groups. `mean` is the plain mean of `values`. `low` and `high`
are the `(1 - confidence) / 2` and `1 - (1 - confidence) / 2` quantiles of the resampled
means (`numpy.quantile`, default linear method).
`paired_difference(a: Mapping[str, float], b: Mapping[str, float], *,
clusters: Mapping[str, str] | None = None, resamples=10000, seed=0, confidence=0.95) -> Interval`:
the interval of the mean of `b[key] - a[key]`. The key sets must be equal or `ValueError`.
Values are taken in sorted key order, so the result does not depend on dict order.
`excludes_zero(interval: Interval) -> bool`: `low > 0 or high < 0`.
Errors (`ValueError`): no values; clusters of a different length; resamples < 1; confidence
outside (0, 1).

## D25. Dense retrieval (phase 3)

`src/atr/retrieval/dense.py`:
`Encoder` Protocol: `name: str`; `encode_queries(texts: list[str]) -> NDArray[np.float32]`;
`encode_passages(texts: list[str]) -> NDArray[np.float32]`; rows are L2-normalized.
`EncoderSpec` (frozen pydantic): `key`, `model_name`, `query_prefix`, `passage_prefix`.
`ENCODERS: dict[str, EncoderSpec]` with exactly:
`"bge-small-en"` -> `BAAI/bge-small-en-v1.5`, query prefix
`"Represent this sentence for searching relevant passages: "`, passage prefix `""`;
`"me5-small"` -> `intfloat/multilingual-e5-small`, query prefix `"query: "`, passage prefix
`"passage: "`.
`SentenceTransformerEncoder(spec: EncoderSpec, *, device: str | None = None)`: loads the model
lazily on first use (import `sentence_transformers` inside the method, so importing this
module stays fast and tests need no model); `normalize_embeddings=True`; `name` is `spec.key`.
`CachedEncoder(inner: Encoder, cache_dir: Path)`: same Protocol; each text's vector is stored
as `cache_dir/<key[:2]>/<key>.npy` with `key = cache_key(inner.name, kind, text)`, kind being
"query" or "passage"; only texts missing from the cache are sent to `inner`, in one call, and
the output keeps the input order.
`DenseRetriever(docs: list[Doc], encoder: Encoder, *, name: str)`: encodes all doc texts once
in `__init__`; `search` encodes the query, takes dot products with the doc matrix, and
returns `top_hits`. Empty docs raise `ValueError`.
Tests use a small fake encoder defined in the test file. No test loads a real model or
touches the network.

## D26. Query translation (phase 3)

`src/atr/retrieval/translate.py`:
`QueryRewriter` Protocol: `name: str`; `rewrite(text: str, language: QuestionLanguage) -> str`.
`Identity` rewriter: returns the text, `name = "none"`.
`OllamaTranslator(*, model: str = "qwen2.5:14b", host: str = "http://localhost:11434",
cache: JsonCache, timeout: float = 120.0)`: `name = f"translate:{model}"`. `rewrite` returns
English text unchanged (language "en"); otherwise it returns the cached translation or asks
Ollama: POST `{host}/api/generate` with JSON `{"model", "prompt", "stream": false,
"options": {"temperature": 0, "seed": 0}}` using `urllib.request` (no new dependency), reads
the `"response"` field, strips it, and caches it under
`cache_key("translate", model, PROMPT_TEMPLATE, text)`. `PROMPT_TEMPLATE` is one module
constant containing `{text}`; it asks for an English translation of a soccer question and
nothing else. An HTTP failure, a missing `"response"` or an empty translation raises
`RuntimeError` naming the model; nothing is cached on failure. The single network call sits in
one function `_post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]`
so tests can monkeypatch it. No test touches the network.

## D27. Run records and latency (phase 3, main session, 2026-10-05)

`src/atr/eval/run.py` writes `runs/<retriever>[+translate]__<index>__<split>/records.jsonl` and
`run.json`; both are committed, and every table is rebuilt from the records
(`src/atr/eval/report.py`). `runs/cache/` (embeddings, translations) is not committed.
`QueryRewriter` gained `cost_ms(text, language)`: the translator stores how long each
translation took beside the translation, and a run reports that stored time, so a rerun
served from the cache does not report a rewrite latency near zero. Passage vectors are
cached; query vectors never are, for the same reason. One warm-up search runs before timing.
The rewrite latency rows include English questions, which cost 0.
The `test` split is refused while the manifest counts any unreviewed group (D18).

## D28. Fixes from the phase 3 review (main session, 2026-10-05)

- Translation (amends D26): the first prompt made the model translate only the last sentence
  of a scenario, and one reply drifted into Chinese. The prompt now asks for every sentence.
  `translation_problem(source, translation)` rejects a reply with non-Latin script, a reply
  that repeats the prompt, and a reply whose word count is under 0.5 or over 2.0 times the
  source. A rejected reply goes to a fallback model (`gpt-oss:20b`, `think: low`); temperature
  is 0, so asking the same model again would change nothing. If both are rejected the run
  stops. The cache key includes both model names and the prompt.
  Measured on the 90 non-English dev questions with the new prompt: `qwen2.5:14b` alone was
  rejected on 1 (`uv run python <scratch>/translator_probe.py qwen2.5:14b none 90`, a
  scratch probe, not kept in the repo).
- The test-split bar (D18) counts unreviewed groups from the questions file itself and
  refuses a manifest that disagrees.
- Embedding cache keys use the encoder `fingerprint` (model name and both prefixes), not its
  short key (amends D25).
- A run checks every indexed passage's text hash against `corpus/manifest.json`.
- BM25 sums query terms in sorted order, so scores do not depend on the hash seed.
- All seven dev runs were regenerated with the committed code.

## D29. Phase 4 rule and order, fixed before any phase 4 run (2026-10-05)

Primary metric: recall@5, dev, slice "all". Keep a candidate only if the paired difference
(candidate minus current best; question groups resampled; 10000 resamples; seed 0) has a 95%
interval with `low > 0`. Two tracks, each with its own current best: T starts at
dense-en + translate on the English index, M starts at dense-multi on the English index.
Steps in order: (0, M only) index both editions; (1) hybrid RRF of BM25 and the track's dense
retriever; (2) title prefix on dense docs; (3) reranker over the top 20, variants `minilm`
(T only) and `bge-m3`; (4) chunking, variants `blocks` and `items`; (5) rulebook rewrite;
(6) decomposition. Where a step has variants, each is compared with the current best; among
those that pass, the highest recall@5 is kept, ties going to the lower total p50 latency.
Fixed and not tuned: RRF constant 60, first-stage depth 50, rerank depth 20, chunk limit 120
words. A reranker that passes but multiplies latency is still reported both ways for mannie
to choose.

## D30. Reciprocal rank fusion (phase 4)

`src/atr/retrieval/fusion.py`: `rrf(rankings: list[list[Hit]], *, c: int = 60) -> list[Hit]`.
The fused score of a passage id is the sum, over the rankings that contain it, of
`1 / (c + rank)` with rank 1-based. Input scores are ignored. Output holds every id that
appears, sorted by fused score descending then id. A ranking with a repeated id raises
`ValueError`; `c < 1` raises; an empty list of rankings or all-empty rankings give `[]`.

## D31. Chunking (phase 4)

`src/atr/retrieval/chunking.py`, pure. Passage text has the shape D3 and D9 produce:
paragraphs separated by one blank line; a list is a run of lines that start with `- ` (nested
lines start with spaces), and the paragraph immediately before a list is its lead-in.
`ChunkStrategy = Literal["passage", "blocks", "items"]`, `MAX_WORDS = 120`
(`atr.corpus.models.count_words`).
`chunk_text(text: str, strategy: ChunkStrategy) -> list[str]`:
- `passage`: `[text]`.
- `blocks`: the text is cut into units: a paragraph, or a lead-in paragraph together with
  its list. Units are packed greedily, in order, into chunks of at most MAX_WORDS words,
  joined by a blank line. A single unit over the limit is split if it is a list unit: its
  top-level items (each with its nested lines) are packed greedily into pieces of at most
  MAX_WORDS words including the lead-in, and the lead-in is repeated at the top of every
  piece. A plain paragraph over the limit stays whole.
- `items`: like `blocks`, except every top-level item of every list becomes its own chunk,
  `lead-in + "\n" + item lines`.
Guarantees, each with a test: no empty chunk; every non-blank line of the text appears in
at least one chunk; a text of at most MAX_WORDS words with no list gives `[text]` under
every strategy; chunks keep document order.

## D32. Docs from passages (phase 4)

`src/atr/retrieval/docs.py`:
`load_passages(corpus_dir: Path, languages: tuple[Language, ...]) -> list[Passage]`.
`title_line(passage: Passage) -> str`: for a law passage
`"Law {law_number}: {law_title} > {section_title}"` plus `" > {subheading}"` when there is
one; for a glossary passage `"{law_title} > {section_title}"`.
`build_docs(passages: list[Passage], *, title_prefix: bool, chunking: ChunkStrategy) -> list[Doc]`:
one Doc per chunk of each passage, in order, carrying the passage's id and language; with
`title_prefix` the Doc text is `title_line + "\n" + chunk`.
`verify_passages(passages, pins)`: the D28 hash check, on passages (docs are derived text).

## D33. Reranker (phase 4)

`src/atr/retrieval/rerank.py`:
`Reranker` Protocol: `name: str`; `score(query: str, texts: list[str]) -> list[float]`.
`RERANKERS: dict[str, str]` = `{"minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
"bge-m3": "BAAI/bge-reranker-v2-m3"}`.
`CrossEncoderReranker(key: str, *, device: str | None = None, max_length: int = 512,
batch_size: int = 32)`: `name = key`; the `sentence_transformers.CrossEncoder` is imported
and loaded lazily on first `score`; an empty `texts` returns `[]` without loading.
`rerank(query: str, candidates: list[tuple[str, str]], reranker: Reranker, k: int) -> list[Hit]`:
candidates are (passage id, text); every candidate is scored in ONE `score` call; the best
score per passage id is kept; result through `top_hits`. No candidates gives `[]`.
Tests use a fake reranker; no test loads a model.

## D34. Local LLM calls (phase 4)

`src/atr/llm.py`:
`Completion(NamedTuple)`: `text: str`, `ms: float`.
`OllamaLLM(*, cache: JsonCache, host: str = "http://localhost:11434", timeout: float = 300.0)`:
`complete(model: str, prompt: str, *, think: str | None = None) -> Completion`. POSTs
`{host}/api/generate` with `{"model", "prompt", "stream": false, "options": {"temperature": 0,
"seed": 0}}` plus `"think"` when given; takes the stripped `"response"`. The text and the
wall-clock ms are cached under `cache_key("llm", model, think or "", prompt)`, and a cache hit
returns the STORED ms, so a rerun reports what the call really cost. HTTP failure, bad JSON,
a missing or empty response raise `RuntimeError` naming the model, and nothing is cached. The
network call is one module function `_post(url, payload, timeout) -> dict[str, object]`
(urllib only) that tests monkeypatch. No test touches the network.

## D35. Rulebook rewrite (phase 4, main session)

`src/atr/retrieval/rewrite.py`: `RulebookRewriter(llm, model="qwen2.5:14b").expand(text)` asks
a local model to restate the question in the Laws' vocabulary and returns
`question + "\n" + restatement` (query expansion, so the original words are kept). A reply
under 0.3 or over 3.0 times the question's word count falls back to the question alone.
The restatement is a search query only; it never reaches the user or the answer.

## D36. Decomposition (phase 4, main session)

`src/atr/retrieval/decompose.py`: `Decomposer(llm, model).split(text)` returns two or three
sub-questions parsed from a JSON array, or an empty list for anything else (one element,
more than three, not JSON). The pipeline always searches the full question as well and
fuses all rankings with RRF.

## D37. Pipeline (phase 4, main session)

`src/atr/pipeline.py`: `PipelineConfig` (frozen) holds `label`, `index`, `translate`, `dense`
(encoder key or None), `bm25`, `title_prefix`, `chunking`, `reranker`, `rewrite`, `decompose`.
`config_id` is the first 10 hex characters of the sha256 of the config without its label,
and names the run directory, so a cached run can never be read under another configuration.
Stage order in `Pipeline.search`: translate; rewrite; decompose; first stage (each retriever
on each query, depth 50); RRF when there is more than one ranking; rerank the top 20
passages with the translated-or-original question (never the expanded one), scoring every
indexed chunk of those passages and keeping the best per passage. The title prefix applies to
dense docs only. Each stage's time is recorded under its own name.

## D38. Fixes from the phase 4 review, and what they do to the pre-registration (2026-10-07)

The review recomputed all 16 paired differences independently and matched the table, and
confirmed every run directory is the configuration its meta claims. It found defects in two
stages, which were fixed AFTER the first dev results had been seen. That is disclosed here
because D29 was written before any run:
- Decomposition: the reply parser threw away replies that were not valid JSON, 13 of 62
  attempted splits on track T and 18 of 87 on track M. It now recovers the quoted strings.
  This repairs a broken stage; the prompt, the model and the two-to-three rule are unchanged.
- Rewrite: a reply that leaves Latin script is now rejected and falls back to the question,
  the same check translation already had. The prompt is unchanged.
Both steps are rerun under the same rule, and the first results stay in git history
(commit c392276) for anyone who wants to compare.
Also fixed: records are now stamped with a hash of every source file that can change a
ranking or a stage time, and a cached run with another stamp is rerun, not reused; the report
names the machine stored with each run, not the machine it is regenerated on; the trade-off
front compares latencies to two significant figures.
Known and left: the first uncached LLM call of a run carries the model load in its stored
time (one outlier per run); the rewrite prompt says "soccer question" and so turns an
out-of-scope question into a soccer one, which matters if track M's final is ever used to
detect out-of-scope questions; `run_chain` would adopt a passing reranker automatically,
where D29 says mannie chooses (none passed).

## D39. Fine-tuning data, mining and loop (phase 5, main session, 2026-10-10)

`src/atr/finetune/`. `synth.py`: `training_passages(passages)` keeps English passages whose law
is not in `HELD_OUT_LAWS` (glossary entries have no law and stay). One cached `OllamaLLM` call
per passage (`qwen2.5:14b`, `SYNTH_PROMPT`) asks for two questions in each of en, es and cs as a
JSON object; `parse_synthetic(reply)` returns the three lists or None, dropping blanks and
duplicates, and a passage whose reply is unusable is listed in the committed manifest
`runs/finetune/synthetic.json` (counts, prompt hash, content hash, machine). The questions go to
`data/finetune/synthetic.jsonl` (gitignored). Dev and test questions are never read.
`negatives.py`: `mine_hard_negatives(ranked, gold, n)` takes the first n ranked ids outside the
gold passage's section (`section_key`: `L12.1.3` -> `L12.1`), because sibling subsections often
answer the same question and would be false negatives. The ranking comes from the untrained
me5-small over plain English passages (depth 50); the phase 4 LLM stages are not used for
mining because they cost seconds per query and do not change which passages sit near the gold.
`train.py`: a plain PyTorch loop (no new packages). `contrastive_loss` is cross-entropy over
`anchors @ [positives; negatives].T * scale`, the MultipleNegativesRanking loss with hard
negatives. Fixed and not tuned (`TrainConfig`): 3 epochs, batch 16, AdamW 2e-5, scale 20,
linear warm-up over 10% of steps then linear decay, 3 negatives per question, seed 0 (torch and
`random`). Weights go to `models/me5-small-ft/` (gitignored); `runs/finetune/train.json`
(committed) records every setting, the synthetic hash, loss per epoch, seconds, device, machine.
The result is a third encoder `me5-small-ft` and retriever `dense-multi-ft`; its cache
fingerprint appends `weights_stamp` (sha256 of the weight files), so retraining never reuses
stale vectors (amends D25, D28). Measurement is the existing `atr.eval.run` and
`atr.eval.compare` on dev and on heldout.

## D40. Refusal threshold (phase 6, main session, 2026-10-10)

`src/atr/eval/refusal.py`, dev split only. A question is answerable unless its type is
`out_of_scope`; false-premise questions have a gold passage and count as answerable (the right
answer corrects the premise), and are reported on their own row. Two score sources are compared
per track: the top score of the track's final first stage, and the top `bge-m3` reranker score
over the final's top 20 (the decisions table names the reranker; phase 4 dropped it for
accuracy, which does not say whether its score separates better). The source with the higher
AUROC (answerable against out-of-scope; ties count half) is chosen, then the threshold is the
observed score with the highest balanced accuracy (mean of the answer rate on answerable and the
refusal rate on out-of-scope), ties going to the lower threshold. Balanced, because out-of-scope
is one type of six and plain accuracy would never refuse. The curve is accuracy among answered
questions (gold in the top 5; an answered out-of-scope question counts wrong) against the share
answered. `runs/refusal/<track>.json` (committed, `RefusalChoice`) holds the source, the
threshold, the AUROCs, the by-type and by-language rows and the curve; phases 7 and 9 read it
with `load_choice(track)`. The reranker-scored runs are cached under `runs/refusal/runs/`, apart
from the ablation directory, so the ablation table and plot show only the pre-registered chain.
Known: track M's final rewrites the question as "a soccer question" (D38), which pushes
out-of-scope questions toward the Laws and can only hurt its separation; that is reported, not
patched, because the rewrite stage was kept under D29 on recall@5.

## D41. Generation, judge and grading (phase 7, main session, 2026-10-10)

`src/atr/answer/`. `generate.py`: `Generator(llm, model="qwen2.5:14b")`. `answer(question,
language, passages)` prompts with ONLY the given passages, each labelled `[id] title`, asks for at
most three sentences each ending in a bracketed id, in the question's language, and for the
fixed `REFUSAL_TEXT[language]` when the passages do not answer. No passages means an immediate
refusal with no call. `extract_citations(text, allowed)` keeps cited ids in order, deduplicated,
and only ids the model was given (a hallucinated id is dropped, never shown). `is_refusal` matches
the fixed text or its opening ("I don't know", "No lo sé"). `split_claims` makes one `Claim`
per sentence with the ids cited inside it. `closed_book` is evaluation only and may cite any
corpus id. The Spanish and code-switched refusal strings are drafts until mannie reviews them.
`run.py`: systems `closed_book`, `full`, `oracle` over a split; `full` serves the track's final
config, plus the `bge-m3` reranker only when phase 6 chose the reranker score (`serving_config`),
takes the top 5, refuses under the D40 threshold without calling the generator, and passes
passages in the question's language (Spanish edition for `es`, English otherwise, English when
the Spanish twin is missing). `oracle` gets the gold passages. Records (`AnswerRecord`) go to
`runs/answers/<system>__<track>__<split>/` with a meta file; the test split stays barred (D18).
`score.py`: citation accuracy = among questions with gold, the answer was given and cites a
gold passage (a wrong refusal counts 0); `all_gold_cited` too; refusal accuracy = refused exactly
when out-of-scope; supported-claim rate = share of judged claims marked supported, only with
labels. `error_split`: oracle minus full is the retrieval gap, 1 minus oracle the generation gap,
full minus closed-book the value of retrieval, paired over question groups.
`judge.py`: `gpt-oss:20b` with `think: low`, a different model from the generator so it never
grades itself; one yes/no call per claim with the cited passages' English text; an uncited claim
is unsupported without a call; an unreadable verdict is None and counts as unsupported. The
judge runs on a full set only after `runs/answers/judge_accepted.json` exists.
`grading.py`: `sheet` samples 60 non-refused answers round-robin over (type, language) cells
with a seed, writes `docs/grading/sheet.md` where mannie marks each claim `[y]` or `[n]`;
`agree` runs the judge on exactly those claims and writes percent agreement and Cohen's kappa to
`runs/answers/judge_agreement.json`; `accept` records his acceptance. `report.py` prints the
tables and the three-system split, with the judge's status next to the supported-claim column.

## D42. Failure causes (phase 8, main session, 2026-10-10; entry written 2026-10-18)

`src/atr/eval/errors.py`: one cause per failed question from a fixed list, by rules in a fixed
order; the first rule that applies wins. Input: the track's serving config evaluated on the split
(`Workbench.evaluate`, dev by default), the `full` answer run for that track when
`runs/answers/full__<track>__<split>/records.jsonl` exists, and the judge's labels only when
`runs/answers/judge_accepted.json` exists. A question fails when its gold is not all in the top 5
(recall@5 < 1), or when the answer stage failed: an out-of-scope question was answered, an
answerable question was refused, no gold passage was cited, or an accepted judge marked a claim
unsupported. Causes, in order: `wrong refusal` (the answer problem is a refusal or an answered
out-of-scope question); `unsupported claim` (an answer problem while retrieval found a gold
passage); `language mismatch` (a non-English question whose English version found its gold);
`needs two passages` (type `multi_rule` with 0 < recall@5 < 1); `glossary term` (any gold id
starts with `G.`); `table or list lookup` (a gold passage is list-dominated: at least half of its
non-blank lines are items); `negation or exception` (the question or the gold text holds a word
from `NEGATION_MARKERS`, English and Spanish); otherwise `vocabulary mismatch`, with the
question's content-word overlap with the gold (D12) as evidence. `docs/errors/overrides.json`
maps a question id to a cause mannie chose; an unknown cause raises. Output: `docs/errors.txt`
(counts by cause and language, committed) and `docs/errors/examples.md` (up to `--n-examples`,
default 10, round-robin over causes, highest overlap first, each with up to 400 characters of
gold text). `docs/errors/examples.md` and `docs/grading/` are gitignored because they quote
passage text; the numbers derived from them (`docs/errors.txt`, `runs/answers/judge_agreement.json`)
are committed.

## D43. The demo (phase 9, main session, 2026-10-10; entry written 2026-10-18)

`src/atr/web/app.py`, standard library only (`http.server`), bound to 127.0.0.1, one page of
inline HTML and JavaScript. `uv run python -m atr.web.app --track T --port 8000`. Routes: `GET /`
the question box; `GET /results` and `GET /api/results` the committed report files
(`docs/ablation.txt`, `finetune.txt`, `refusal.txt`, `answers.txt`, `errors.txt`, whichever
exist) verbatim; `GET /healthz`; `POST /api/ask` with `{"question": ...}`, trimmed to 500
characters, 400 on anything else. `build_service` loads the track's phase 6 choice
(`load_choice`), serves `serving_config(choice.config, choice)` through `Workbench.build` against
`runs/ablation`, and a plain BM25 pipeline for the keyword comparison; the generator is the phase
7 `Generator` over the same `OllamaLLM` cache. `Service.ask`: the language is detected from two
word lists (English words, Spanish words; both present means `cs`); the full pipeline runs at
k = 5; when the top score is under the D40 threshold the generator gets no passages and refuses
without a call; otherwise the passages go in the question's language (`context_passages`, D41).
The response carries the answer or the fixed refusal text, the citations, the top score and the
threshold, per-stage latency plus generation time, and for each shown passage its title, both
edition texts, the score, why it was found (`matched words: ...` from stems shared with the
search query, else `semantic match`), whether it was cited, and the passage sentence the answer
restates (`cited_sentence`: best Dice overlap of content words, ties to the earlier sentence).
Passages are shown only when the question was answered. Every response ends with `DISCLAIMER`.
Tests drive `Service` and `route` with fakes; no test loads a model or starts a server.

## D44. Fixes from the phase 5 to 9 review (main session, 2026-10-18)

Three review passes (general, silent failures, test coverage) ran on the uncommitted phase 5 to
9 code before its first commit. Fixed, each with a test:
- Training leak (amends D39): the ranking used for mining, the mined negatives and every text
  embedded during training come from `training_passages` only; a held-out passage receives no
  gradient, not even as a negative. One row per question holds all its negatives (replacing one
  triplet per negative), rows carry passage ids, and `false_negative_mask` drops from the softmax
  any column holding the row's own passage (another row's positive, or another row's mined
  negative). `train` checks the synthetic questions against the manifest hash. The report
  records `rows`, `negatives` and `rows_short_of_negatives`.
- `is_refusal` matched any answer opening "no se" (the Spanish impersonal "No se permite ...")
  or "I do not know of ..."; it now requires the refusal's first words followed by the end of the
  text or punctuation.
- `NEGATION_MARKERS` held the article "a" (meant for "a menos"), which labelled nearly every
  failure "negation or exception"; the phrase check for "a menos" stays.
- `detect_language` had "a" among its English words and sent plain Spanish questions down the
  code-switched path.
- `--split test` is refused by `atr.eval.errors`, `atr.answer.judge` and `atr.answer.report`,
  as it already was by `atr.answer.run` and `atr.eval.run` (D18).
- Hallucinated citation ids are stripped from the answer text the demo shows (D41: never shown).
- Stale verdicts: `judgments.json` carries the sha256 of the records it judged and refuses to
  load against other records; `write_answers` deletes a stale one. The grading sheet carries the
  same stamp and `agree` refuses a sheet written for other answers; a malformed claim line raises
  and unmarked ones are counted and printed. The acceptance records the judge model, its think
  setting and the prompt hash; `judge.main` refuses a judge that differs from the accepted one.
- `parse_verdict` reads through Markdown and a leading label ("**Yes**", "Answer: no").
  Unreadable verdicts are counted and printed in `docs/answers.txt` next to the tables.
- Answer runs are stamped with `ANSWER_SOURCES` (the ranking sources plus `answer/generate.py`)
  and `report` refuses to pair runs whose stamp, generator, question set, corpus or split differ.
- Cached records remember the fine-tuned weights: `RunMeta.weights_stamp` (Workbench) and
  `RunInfo.encoder_fingerprint` (`atr.eval.run`), so retraining never reuses records (amends D39).
- `fingerprint_of` raises on a missing `models/...` directory instead of asking the Hugging Face
  hub for a repo of that name.
- The oracle raises on a gold id missing from the corpus instead of refusing silently; the judge
  raises on a cited id missing from the corpus; `label_failures` raises when the answer run lacks
  records for the question set or an override names an unknown id; a serving config that
  carries the reranker is evaluated under `runs/refusal/runs`, not `runs/ablation` (D40).
- Demo: an error from the pipeline or Ollama returns 502 with the message instead of dropping
  the connection; a question over 500 characters or a body over 64 KiB gets 413; the server is
  the plain `HTTPServer`, one request at a time, because the pipeline and the cache are not
  thread-safe.
- `split_claims` drops a leading "¿" or "¡"; the demo's `cited_sentence` scores by Dice overlap.

Known and left: `AnswerRecord.answer` and its claims are committed under `runs/answers/` (model
restatements, at most three sentences each, not the Laws text); hallucinated ids are dropped but
not counted; "hit" for a multi_rule question means any gold in the top 5 on the refusal curve
(D40) but all gold for a failure (D42); a judge run that stops midway writes nothing.
