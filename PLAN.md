# Phase 4 plan: improve retrieval, one change at a time

Goal: test six changes in a fixed order on dev, keep a change only when the paired
difference says to, log every experiment, and regenerate the whole ablation table (accuracy
and latency) from saved records with one command.

High-stakes: the keep/drop logic decides the final system.

## Two tracks

The best phase 3 baseline translates the question to English first, which costs about a
second per question. The multilingual model answers without that cost. Which one mannie
wants is a latency trade-off that is his to choose, so every change is tested on both:

- Track T (translate first): starts from `dense-en`, English index, translated questions.
- Track M (multilingual, no translation): starts from `dense-multi`, English index.

## The rule, fixed before any phase 4 run (D29)

Primary metric: recall@5 on dev, all scored questions. A candidate (current best plus one
change) is kept only when the 95% interval of the paired difference against the current best
is entirely above zero. Other metrics, slices and latency are reported and do not decide.
Hyperparameters are fixed in advance and not tuned: dev has 37 scored groups, too few to
tune on.

## Order

0. (track M only) index both editions
1. hybrid: reciprocal rank fusion of BM25 and dense
2. law title and section name in front of each passage before embedding
3. cross-encoder reranker over the top 20 (variants: a small English one on track T, a
   multilingual one on both)
4. chunking (variants: word-limited blocks, one chunk per list item)
5. query rewrite into rulebook vocabulary (one method)
6. splitting a question into sub-questions

## Files

```
src/atr/retrieval/fusion.py     D30  reciprocal rank fusion
src/atr/retrieval/chunking.py   D31  chunk strategies
src/atr/retrieval/docs.py       D32  passages -> docs (title prefix, chunks)
src/atr/retrieval/rerank.py     D33  cross-encoder reranker
src/atr/llm.py                  D34  cached local LLM calls with replayed latency
src/atr/retrieval/rewrite.py    D35  rulebook-vocabulary rewrite
src/atr/retrieval/decompose.py  D36  sub-questions
src/atr/pipeline.py             PipelineConfig and Pipeline (stages and per-stage timings)
src/atr/eval/ablation.py        the experiment chain, the log and the table
src/atr/eval/plot.py            accuracy against latency
```

## Check

`uv run python -m atr.eval.ablation` regenerates the full table from cached results;
`uv run python -m atr.eval.plot` draws accuracy against latency with the best trade-offs marked.
