# Phase 2 plan: evaluation set

Goal: at least 300 questions with gold passage ids, in English, Spanish and code-switched
Spanish-English, reviewed by mannie, then split and frozen with a committed hash.

This phase is high-stakes: a wrong gold label or a leaky split still produces numbers that
look fine.

## Shape of the set

A question is written once as a `QuestionGroup` (`src/atr/evalset/models.py`): one type, one
list of gold passage ids, and three texts (`en`, `es`, `cs`). 112 groups give 336 questions.
The three versions of a group always land in the same split, because a Spanish question in
dev and its English twin in test would leak.

| type | groups | gold |
|---|---|---|
| plain | 24 | 1 or more passages |
| paraphrase | 24 | 1 or more, and the question avoids the passage's content words |
| scenario | 24 | 1 or more, the question is a match situation needing a ruling |
| multi_rule | 16 | 2 or more, each one needed |
| false_premise | 12 | the passage that corrects the premise, or none |
| out_of_scope | 12 | none |

Held-out laws are 3, 5 and 16 (D11). A group with any gold passage in those laws goes to the
`heldout` slice. Every other group is split dev/test, half each, stratified by type, seeded.

## Steps

1. Tooling (crew): overlap, split, load and validate, freeze, then review sheet and stats.
2. Drafting (agents, not builders): English questions with gold from the corpus, then Spanish
   and code-switched drafts. Drafts live in `evalset/draft/*.jsonl`.
3. Blind check: separate agents see only the English question and the corpus and name the
   answering passage. Disagreement with the drafter's gold is flagged on the review sheet,
   not resolved. An item and its gold label are never confirmed in the same pass.
4. STOP. mannie reviews `data/review/questions.md`: every question, its gold passages, the
   overlap numbers and the blind-check flags. Spanish and code-switched text is his to fix.
5. After his fixes: validate, split, freeze, commit the hash, run the stats check.

## Files

```
src/atr/evalset/models.py    QuestionGroup, Question, types, HELD_OUT_LAWS (done)
src/atr/evalset/overlap.py   pure: word overlap between a question and its gold passages
src/atr/evalset/split.py     pure: seeded, stratified split by group
src/atr/evalset/store.py     load draft jsonl files; validate against the corpus
src/atr/evalset/freeze.py    canonical bytes, hash, manifest; CLI
src/atr/evalset/review.py    CLI: the review sheet (contains law text, written under data/)
src/atr/evalset/stats.py     CLI: the phase check
tests/evalset/test_*.py
evalset/draft/*.jsonl        drafts, one QuestionGroup per line
evalset/questions.jsonl      the frozen set (after review)
evalset/manifest.json        split per group, hash, corpus manifest hash
```

## Check

`uv run python -m atr.evalset.stats` prints counts by type, language and split, and the
overlap distribution per type; the frozen hash is committed and pinned by a test.
