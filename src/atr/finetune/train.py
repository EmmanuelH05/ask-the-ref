"""Contrastive fine-tuning with a fixed seed (DECISIONS.md D39).

A plain PyTorch loop rather than the sentence-transformers trainer: it needs no extra
packages, every setting is visible, and the loss is the one line the tests check.
"""

import argparse
import math
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict

from atr.eval.run import CACHE_DIR, machine_name
from atr.finetune.negatives import Example, build_examples
from atr.finetune.synth import (
    MANIFEST_PATH,
    SYNTHETIC_PATH,
    SynthExample,
    SynthManifest,
    load_examples,
    synthetic_hash,
    training_passages,
)
from atr.retrieval.base import Doc
from atr.retrieval.dense import ENCODERS, CachedEncoder, DenseRetriever, SentenceTransformerEncoder
from atr.retrieval.docs import load_passages

BASE_ENCODER = "me5-small"
OUT_DIR = Path("models/me5-small-ft")
REPORT_PATH = Path("runs/finetune/train.json")
MINING_DEPTH = 50


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    scale: float = 20.0
    warmup_fraction: float = 0.1
    negatives_per_example: int = 3
    seed: int = 0


class TrainReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str
    out_dir: str
    config: TrainConfig
    synthetic_hash: str
    examples: int
    rows: int
    negatives: int
    rows_short_of_negatives: int  # rows whose top 50 held fewer usable negatives than asked
    steps: int
    loss_per_epoch: list[float]
    seconds: float
    device: str
    machine: str
    created_at: str


def false_negative_mask(
    positive_ids: Sequence[str], candidate_ids: Sequence[str]
) -> torch.Tensor:
    """True where a candidate column holds the row's own passage under another column.

    Column i is row i's positive and stays. Every other column with the same passage id, whether
    another row's positive (several synthetic questions share one passage) or a mined negative
    of another row, would otherwise be scored as a negative of the very passage being taught.
    """
    if len(candidate_ids) < len(positive_ids):
        raise ValueError("every row's positive must be among the candidates")
    mask = torch.zeros(len(positive_ids), len(candidate_ids), dtype=torch.bool)
    for row, positive in enumerate(positive_ids):
        for column, candidate in enumerate(candidate_ids):
            if column != row and candidate == positive:
                mask[row, column] = True
    return mask


def contrastive_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    scale: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over scaled cosine scores: each anchor against every positive and negative.

    This is the MultipleNegativesRanking loss with hard negatives: the candidates are the
    batch's positives (in-batch negatives for every other anchor) plus every mined negative in
    the batch. `mask` (rows by candidates) removes false negatives from the softmax.
    """
    if anchors.shape[0] != positives.shape[0]:
        raise ValueError("anchors and positives must have the same batch size")
    candidates = torch.cat([positives, negatives], dim=0)
    scores = anchors @ candidates.T * scale
    if mask is not None:
        if tuple(mask.shape) != tuple(scores.shape):
            raise ValueError(f"mask shape {tuple(mask.shape)} is not {tuple(scores.shape)}")
        scores = scores.masked_fill(mask.to(scores.device), float("-inf"))
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    return torch.nn.functional.cross_entropy(scores, labels)


def batches(rows: list[Example], *, batch_size: int, seed: int) -> list[list[Example]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    order = list(rows)
    random.Random(seed).shuffle(order)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def check_synthetic(examples: list[SynthExample], manifest: SynthManifest) -> None:
    """The questions on disk must be the ones the committed manifest describes."""
    actual = synthetic_hash(examples)
    if actual != manifest.hash:
        raise ValueError(
            f"synthetic questions hash {actual[:12]} but the manifest says "
            f"{manifest.hash[:12]}; rerun `uv run python -m atr.finetune.synth`"
        )


def warmup_steps(*, total_steps: int, fraction: float) -> int:
    return math.ceil(total_steps * fraction)


def _embed(model: Any, texts: list[str], device: torch.device) -> torch.Tensor:
    from sentence_transformers.util import batch_to_device

    features = batch_to_device(model.tokenize(texts), device)
    embeddings: torch.Tensor = model(features)["sentence_embedding"]
    return torch.nn.functional.normalize(embeddings, dim=1)


def train(
    rows: list[Example], *, base_model: str, out_dir: Path, config: TrainConfig
) -> tuple[list[float], int, str]:
    """Fine-tune and save. Returns the mean loss per epoch, the step count and the device."""
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = SentenceTransformer(base_model)
    device: torch.device = model.device
    model.train()
    # Recompute activations in the backward pass instead of keeping them: the same loss and
    # gradients for a fraction of the memory. A batch embeds its anchors, positives and every
    # mined negative at once (up to 80 passages of up to 512 tokens), which without this
    # exceeded the 30 GiB the MPS backend allows.
    transformer: Any = model[0].auto_model  # Any: the module type is only known at run time
    transformer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    plan = [
        batches(rows, batch_size=config.batch_size, seed=config.seed + e)
        for e in range(config.epochs)
    ]
    total = sum(len(epoch) for epoch in plan)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    warmup = warmup_steps(total_steps=total, fraction=config.warmup_fraction)

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        remaining = max(total - warmup, 1)
        return max(0.0, (total - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    losses: list[float] = []
    step = 0
    for epoch in plan:
        total_loss = 0.0
        for batch in epoch:
            anchors = _embed(model, [r.anchor for r in batch], device)
            positives = _embed(model, [r.positive for r in batch], device)
            negatives = _embed(model, [n for r in batch for n in r.negatives], device)
            positive_ids = [r.positive_id for r in batch]
            candidate_ids = positive_ids + [n for r in batch for n in r.negative_ids]
            mask = false_negative_mask(positive_ids, candidate_ids)
            loss = contrastive_loss(anchors, positives, negatives, scale=config.scale, mask=mask)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            step += 1
        losses.append(total_loss / max(len(epoch), 1))
        print(f"  epoch {len(losses)}/{config.epochs}: mean loss {losses[-1]:.4f}", flush=True)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    return losses, step, str(device)


def rank_with_base(examples: list[SynthExample], docs: list[Doc]) -> dict[str, list[str]]:
    """Every synthetic question ranked by the untrained model, for negative mining."""
    encoder = CachedEncoder(
        SentenceTransformerEncoder(ENCODERS[BASE_ENCODER]), CACHE_DIR / "embeddings"
    )
    retriever = DenseRetriever(docs, encoder, name="dense-multi")
    return {
        e.question: [hit.passage_id for hit in retriever.search(e.question, MINING_DEPTH)]
        for e in examples
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune me5-small on synthetic pairs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    parser.add_argument("--synthetic", type=Path, default=SYNTHETIC_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    config = TrainConfig(seed=args.seed, epochs=args.epochs)
    examples = load_examples(args.synthetic)
    check_synthetic(
        examples, SynthManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    )
    # Only passages outside the held-out laws are ranked, mined or embedded during training
    # (D39): a held-out passage must never receive gradient, not even as a negative.
    passages = training_passages(load_passages(args.corpus_dir, ("en",)))
    docs = [Doc(p.passage_id, p.language, p.text) for p in passages]
    texts = {p.passage_id: p.text for p in passages}
    spec = ENCODERS[BASE_ENCODER]
    print(f"ranking {len(examples)} synthetic questions with the untrained model ...", flush=True)
    rankings = rank_with_base(examples, docs)
    rows = build_examples(
        examples,
        rankings,
        texts,
        negatives_per_example=config.negatives_per_example,
        query_prefix=spec.query_prefix,
        passage_prefix=spec.passage_prefix,
    )
    negatives = sum(len(r.negatives) for r in rows)
    print(
        f"training on {len(rows)} rows, {negatives} negatives, seed {config.seed} ...", flush=True
    )
    started = time.perf_counter()
    losses, steps, device = train(
        rows, base_model=spec.model_name, out_dir=args.out_dir, config=config
    )
    report = TrainReport(
        base_model=spec.model_name,
        out_dir=str(args.out_dir),
        config=config,
        synthetic_hash=synthetic_hash(examples),
        examples=len(examples),
        rows=len(rows),
        negatives=negatives,
        rows_short_of_negatives=sum(
            1 for r in rows if len(r.negatives) < config.negatives_per_example
        ),
        steps=steps,
        loss_per_epoch=losses,
        seconds=time.perf_counter() - started,
        device=device,
        machine=machine_name(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.out_dir} in {report.seconds:.0f} s on {device}; report {args.report}")


if __name__ == "__main__":
    main()
