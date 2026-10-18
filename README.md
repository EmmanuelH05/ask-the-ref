# Ask the Ref

Bilingual question answering over the IFAB Laws of the Game (2026/27 edition): ask in English,
Spanish or a mix of both, get a short answer that cites the passage it came from, or a refusal
when the Laws do not cover it. Retrieval is measured the pre-registered way: a frozen and hashed
question set, a keep rule written before any run, paired bootstrap intervals, and every table
rebuilt from committed records by one command. Local models only.

Results: [`docs/phase-4-results.md`](docs/phase-4-results.md) says which retrieval change won
and why, with the accuracy-latency plot. The generated reports are `docs/ablation.txt`,
`docs/finetune.txt`, `docs/refusal.txt`, `docs/answers.txt` and `docs/errors.txt`; every number
in them names the command that produced it.

## Layout

| Path | What |
|---|---|
| `src/atr/corpus/` | Fetch the Laws page-data, clean the HTML, cut passages with ids shared by both editions |
| `src/atr/evalset/` | Question groups in three languages, word-overlap leak check, split, freeze and hash |
| `src/atr/retrieval/` | BM25, dense encoders, translation, fusion, chunking, reranking, rewrite, decomposition |
| `src/atr/pipeline.py` | One configuration record and the staged search with per-stage timings |
| `src/atr/eval/` | Metrics, cluster bootstrap, runs, the pre-registered ablation, the plot, refusal, failure causes |
| `src/atr/finetune/` | Synthetic questions, hard negatives, contrastive fine-tune of the multilingual encoder |
| `src/atr/answer/` | Cited generation, the judge, the hand-grading sheet, the three-system report |
| `src/atr/web/` | The local demo |
| `corpus/manifest.json` | One hash per passage; the text itself is fetched into `data/` and never committed |
| `evalset/` | The frozen question set and its manifest |
| `runs/` | One record per question per configuration; caches under `runs/cache/` are not committed |
| `DECISIONS.md` | The append-only log that specifies every module; read it before changing anything |

## Setup

Python 3.13 with [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) serving
`qwen2.5:14b` and `gpt-oss:20b` at `localhost:11434`. The sentence-transformers models download
from the Hugging Face hub on first use.

```sh
uv sync
uv run python -m atr.corpus.fetch      # the Laws page-data, both editions, into data/
uv run python -m atr.corpus.build      # passages, alignment and the committed manifest
```

The check that every phase ends with:

```sh
uv run pytest -q && uv run mypy --strict src && uv run ruff check .
```

## Running

```sh
uv run python -m atr.eval.run --retriever dense-multi --split dev   # score one retriever
uv run python -m atr.eval.compare runs/<a> runs/<b>                 # paired difference, B minus A
uv run python -m atr.eval.ablation                                  # the phase 4 chain, from cached records
uv run python -m atr.eval.plot                                      # recall@5 against latency
uv run python -m atr.finetune.synth && uv run python -m atr.finetune.train
uv run python -m atr.eval.refusal                                   # the refusal threshold, dev only
uv run python -m atr.answer.run --system full --track T             # also closed_book and oracle
uv run python -m atr.answer.grading sheet                           # then mark the sheet, then agree, accept
uv run python -m atr.answer.report --track T
uv run python -m atr.eval.errors --track T
uv run python -m atr.web.app --track T --port 8000                  # http://localhost:8000
```

The `test` split is refused by every command until the question set has been reviewed by a
named person and re-frozen (`DECISIONS.md` D18). Dev and held-out numbers are provisional until
then.

## Rules the repo keeps

- No number appears in a document or commit message unless a command in this repo produced it,
  and the command is named next to it.
- The split is frozen and hash-committed; tuning happens on dev only.
- Pydantic models are frozen with `extra="forbid"`; every pure function has tests first.
- Nothing derived from the Laws' full text is committed: `data/`, `models/`, the caches, the
  grading sheet and the failure examples are gitignored.
