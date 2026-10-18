# Phase 5 plan: fine-tune the multilingual embedding model

Goal: train `intfloat/multilingual-e5-small` on synthetic question-passage pairs from the
non-held-out laws and measure it against the same model untrained, on dev, on the held-out
laws (3, 5 and 16) and by language. If it does not help on the held-out laws, say so.

High-stakes: a leak from dev or test questions, or from the held-out laws, into training
would produce a gain that looks real and is not.

## Steps

1. Synthetic questions (`src/atr/finetune/synth.py`): for every English passage whose law is
   not held out (glossary passages included), one cached local-LLM call writes two questions
   in each of English, Spanish and code-switched Spanish-English. Dev and test questions are
   never read by this step. Output `data/finetune/synthetic.jsonl` (gitignored) plus a
   committed manifest with counts and a content hash.
2. Hard negatives (`src/atr/finetune/negatives.py`): the untrained me5-small ranks the
   English passages for each synthetic question; the top-ranked passages that are neither
   the source passage nor a sibling of its section become negatives.
3. Training (`src/atr/finetune/train.py`): a plain PyTorch loop, in-batch negatives plus the
   mined ones, cross-entropy over scaled cosine scores (the MultipleNegativesRanking loss),
   fixed seed, AdamW. Saves `models/me5-small-ft/` (gitignored) and a committed
   `runs/finetune/train.json` with every setting, the loss per epoch, the duration and the
   machine.
4. Measurement: the fine-tuned model is a third encoder, `me5-small-ft`, whose cache
   fingerprint includes a hash of its weights. `atr.eval.run --retriever dense-multi-ft`
   and the existing `atr.eval.compare` give the paired difference on dev and on heldout.

## Fixed in advance

Epochs 3, batch 16, learning rate 2e-5, score scale 20, warm-up 10% of steps, 3 hard
negatives per question, seed 0. Nothing is tuned: the held-out slice is the result.

## Check

`uv run python -m atr.finetune.train --seed 0` trains; `uv run python -m atr.eval.run
--retriever dense-multi-ft --split dev` then `--split heldout`; `uv run python -m
atr.eval.compare runs/dense-multi__en__heldout runs/dense-multi-ft__en__heldout`.
