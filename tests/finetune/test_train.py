import math

import pytest
import torch

from atr.finetune.negatives import Example
from atr.finetune.synth import SynthExample, SynthManifest, synthetic_hash
from atr.finetune.train import (
    TrainConfig,
    batches,
    check_synthetic,
    contrastive_loss,
    false_negative_mask,
    warmup_steps,
)


def _unit(*rows: list[float]) -> torch.Tensor:
    tensor = torch.tensor(rows, dtype=torch.float32)
    return tensor / tensor.norm(dim=1, keepdim=True)


def test_contrastive_loss_is_near_zero_when_anchors_match_their_positives_only() -> None:
    anchors = _unit([1.0, 0.0], [0.0, 1.0])
    positives = _unit([1.0, 0.0], [0.0, 1.0])
    negatives = _unit([-1.0, 0.0], [0.0, -1.0])
    loss = contrastive_loss(anchors, positives, negatives, scale=20.0)
    assert loss.item() < 1e-6


def test_contrastive_loss_is_log_of_candidate_count_when_nothing_is_distinguished() -> None:
    anchors = _unit([1.0, 0.0], [1.0, 0.0])
    same = _unit([0.0, 1.0], [0.0, 1.0])
    loss = contrastive_loss(anchors, same, same, scale=20.0)
    # Two anchors, four candidates each scoring equally: uniform softmax over four.
    assert loss.item() == pytest.approx(math.log(4), abs=1e-5)


def test_contrastive_loss_rejects_mismatched_batch_sizes() -> None:
    with pytest.raises(ValueError):
        contrastive_loss(
            _unit([1.0, 0.0]), _unit([1.0, 0.0], [0.0, 1.0]), _unit([1.0, 0.0]), scale=1
        )


def test_contrastive_loss_takes_every_mined_negative_in_the_batch_not_one_per_row() -> None:
    anchors = _unit([1.0, 0.0], [0.0, 1.0])
    positives = _unit([1.0, 0.0], [0.0, 1.0])
    negatives = _unit([-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0])  # three negatives for two rows
    assert contrastive_loss(anchors, positives, negatives, scale=20.0).item() < 1e-6


def test_false_negative_mask_hides_the_rows_own_passage_under_other_columns() -> None:
    mask = false_negative_mask(["A", "B", "A"], ["A", "B", "A", "B", "C"])
    assert mask.tolist() == [
        [False, False, True, False, False],
        [False, False, False, True, False],
        [True, False, False, False, False],
    ]
    with pytest.raises(ValueError):
        false_negative_mask(["A", "B"], ["A"])


def test_masked_loss_does_not_punish_two_questions_that_share_a_passage() -> None:
    anchors = _unit([1.0, 0.0], [1.0, 0.0])
    positives = _unit([1.0, 0.0], [1.0, 0.0])  # the same passage twice
    negatives = _unit([-1.0, 0.0])
    unmasked = contrastive_loss(anchors, positives, negatives, scale=20.0)
    assert unmasked.item() == pytest.approx(math.log(2), abs=1e-4)
    mask = false_negative_mask(["P", "P"], ["P", "P", "N"])
    masked = contrastive_loss(anchors, positives, negatives, scale=20.0, mask=mask)
    assert masked.item() < 1e-6
    with pytest.raises(ValueError):
        contrastive_loss(anchors, positives, negatives, scale=20.0, mask=mask[:, :2])


def _manifest(examples: list[SynthExample], hash_value: str) -> SynthManifest:
    return SynthManifest.model_validate(
        {
            "model": "m",
            "prompt_sha256": "p",
            "passages": 1,
            "passages_without_usable_reply": [],
            "examples": len(examples),
            "by_language": {"en": len(examples), "es": 0, "cs": 0},
            "cs_switching_within_a_sentence": 0,
            "held_out_laws": [3, 5, 16],
            "hash": hash_value,
            "machine": "test",
            "created_at": "2026-10-18T00:00:00+00:00",
        }
    )


def test_check_synthetic_accepts_the_manifest_hash_and_rejects_another() -> None:
    examples = [SynthExample(passage_id="L1.1", language="en", question="q?")]
    check_synthetic(examples, _manifest(examples, synthetic_hash(examples)))
    with pytest.raises(ValueError, match=r"finetune\.synth"):
        check_synthetic(examples, _manifest(examples, "0" * 64))


def test_batches_are_shuffled_by_seed_and_cover_every_row_once() -> None:
    rows = [Example(f"q{i}", f"p{i}", f"P{i}", (f"n{i}",), (f"N{i}",)) for i in range(7)]
    first = batches(rows, batch_size=3, seed=0)
    assert sorted(t for batch in first for t in batch) == sorted(rows)
    assert [len(batch) for batch in first] == [3, 3, 1]
    assert batches(rows, batch_size=3, seed=0) == first
    assert batches(rows, batch_size=3, seed=1) != first


def test_batches_rejects_a_non_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        batches([], batch_size=0, seed=0)


def test_warmup_steps_is_a_tenth_of_total_rounded_up() -> None:
    assert warmup_steps(total_steps=100, fraction=0.1) == 10
    assert warmup_steps(total_steps=7, fraction=0.1) == 1
    assert warmup_steps(total_steps=0, fraction=0.1) == 0


def test_train_config_is_frozen_with_fixed_defaults() -> None:
    config = TrainConfig()
    assert (config.epochs, config.batch_size, config.learning_rate) == (3, 16, 2e-5)
    assert (config.scale, config.negatives_per_example, config.seed) == (20.0, 3, 0)
    with pytest.raises(ValueError):
        TrainConfig.model_validate({"epochs": 3, "unknown": 1})
