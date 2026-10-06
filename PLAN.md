# Phase 3 plan: harness and baselines

Goal: an eval harness with intervals and a paired test, and baselines scored on dev.

This phase is high-stakes in its metrics and bootstrap code: a wrong score still looks fine.

## Setup being measured

The index is the English edition unless a run says otherwise. English is IFAB's authoritative
text, and it makes Spanish and code-switched questions a real cross-language test. A second
index option holds both editions, which gives keyword search a fair chance on Spanish
questions. Gold labels are passage ids shared by both editions, so scoring is the same.

Every run retrieves the top 10 passages per question. Questions with no gold passage (all
out_of_scope, and false_premise without gold) are not scored for retrieval; their count is
printed. They come back in phase 6.

## Baselines (dev only; test is barred by D18 until the set is reviewed)

| name | what |
|---|---|
| `bm25` | Okapi BM25 over stemmed tokens |
| `dense-en` | `BAAI/bge-small-en-v1.5`, English-only, 33M parameters |
| `dense-multi` | `intfloat/multilingual-e5-small`, 118M parameters |
| `bm25+translate`, `dense-en+translate` | Spanish and code-switched questions are translated to English by a local Ollama model first |

## Files

```
src/atr/text.py              fold, stem, tokens (done)
src/atr/cache.py             hash-keyed disk cache (done)
src/atr/retrieval/base.py    Hit, Doc, Retriever, top_hits, load_docs (done)
src/atr/retrieval/bm25.py    D21
src/atr/retrieval/dense.py   D25
src/atr/retrieval/translate.py  D26
src/atr/eval/metrics.py      D22
src/atr/eval/bootstrap.py    D23
src/atr/eval/run.py          CLI: score one configuration on one split (wave 2)
src/atr/eval/compare.py      CLI: paired difference between two runs (wave 2)
runs/<name>/                 run.json and records.jsonl, committed; runs/cache/ is not
```

## Check

`uv run python -m atr.eval.run --retriever <name>` writes a results file and prints the
table for each baseline; `uv run python -m atr.eval.compare <a> <b>` prints the paired
difference with its interval.
