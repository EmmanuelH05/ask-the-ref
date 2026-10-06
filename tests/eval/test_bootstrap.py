import time

import numpy as np
import pytest
from pydantic import ValidationError

from atr.eval.bootstrap import Interval, bootstrap_ci, excludes_zero, paired_difference

MIXED = [0.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.75, 0.5, 1.0, 0.0]


def _interval(low: float, high: float) -> Interval:
    return Interval(
        mean=(low + high) / 2,
        low=low,
        high=high,
        n=1,
        n_clusters=1,
        resamples=1,
        seed=0,
        confidence=0.95,
    )


def test_same_seed_gives_identical_interval() -> None:
    assert bootstrap_ci(MIXED, seed=3) == bootstrap_ci(MIXED, seed=3)


def test_different_seed_changes_an_endpoint() -> None:
    values = np.random.default_rng(9).random(30).tolist()
    a = bootstrap_ci(values, seed=1)
    b = bootstrap_ci(values, seed=2)
    assert (a.low, a.high) != (b.low, b.high)


def test_mean_is_the_plain_mean() -> None:
    assert bootstrap_ci(MIXED).mean == pytest.approx(sum(MIXED) / len(MIXED))


def test_all_equal_values_collapse_to_a_point() -> None:
    result = bootstrap_ci([0.3] * 12)
    assert result.low == pytest.approx(0.3)
    assert result.high == pytest.approx(0.3)
    assert result.mean == pytest.approx(0.3)


def test_mean_lies_inside_interval_on_mixed_sample() -> None:
    result = bootstrap_ci(MIXED)
    assert result.low <= result.mean <= result.high
    assert result.low < result.high


def test_half_width_matches_normal_theory() -> None:
    values = np.random.default_rng(12345).normal(0.0, 1.0, size=400).tolist()
    result = bootstrap_ci(values, seed=0)
    expected = 1.96 / np.sqrt(400)
    assert (result.high - result.low) / 2 == pytest.approx(expected, rel=0.15)


def test_wider_confidence_gives_wider_interval() -> None:
    narrow = bootstrap_ci(MIXED, confidence=0.80)
    wide = bootstrap_ci(MIXED, confidence=0.99)
    assert wide.high - wide.low > narrow.high - narrow.low


def test_reports_counts_and_settings() -> None:
    result = bootstrap_ci(MIXED, resamples=500, seed=7, confidence=0.9)
    assert (result.n, result.n_clusters) == (10, 10)
    assert (result.resamples, result.seed, result.confidence) == (500, 7, 0.9)


def test_interval_is_frozen_and_forbids_extras() -> None:
    result = bootstrap_ci(MIXED)
    with pytest.raises(ValidationError):
        result.mean = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Interval(**result.model_dump(), extra=1)  # type: ignore[arg-type]


def test_singleton_clusters_equal_no_clusters() -> None:
    labels = [f"c{i}" for i in range(len(MIXED))]
    assert bootstrap_ci(MIXED, clusters=labels, seed=5) == bootstrap_ci(MIXED, seed=5)


def test_clustered_interval_is_wider_than_naive_for_identical_clusters() -> None:
    cluster_means = np.random.default_rng(1).normal(0.0, 1.0, size=50)
    values = np.repeat(cluster_means, 4).tolist()
    labels = [f"c{i}" for i in range(50) for _ in range(4)]
    naive = bootstrap_ci(values, seed=0)
    clustered = bootstrap_ci(values, clusters=labels, seed=0)
    assert (clustered.high - clustered.low) >= 1.5 * (naive.high - naive.low)
    assert (clustered.n, clustered.n_clusters) == (200, 50)
    assert (naive.n, naive.n_clusters) == (200, 200)


def test_unequal_cluster_sizes_match_a_brute_force_pooled_bootstrap() -> None:
    values = [0.0, 1.0, 1.0, 1.0, 0.2, 0.4, 0.9, 0.9, 0.1]
    labels = ["a", "b", "b", "b", "c", "c", "d", "d", "e"]
    groups = [[0.0], [1.0, 1.0, 1.0], [0.2, 0.4], [0.9, 0.9], [0.1]]
    resamples, seed = 4000, 3
    # The same draws as the implementation (one rng.integers call), then a plain loop that
    # pools the VALUES of the drawn clusters. A mean of cluster means would give other numbers.
    drawn = np.random.default_rng(seed).integers(0, len(groups), size=(resamples, len(groups)))
    means = []
    for row in drawn:
        pooled = [value for index in row for value in groups[int(index)]]
        means.append(sum(pooled) / len(pooled))
    low, high = np.quantile(means, [0.025, 0.975])
    result = bootstrap_ci(values, clusters=labels, resamples=resamples, seed=seed)
    assert result.low == pytest.approx(float(low))
    assert result.high == pytest.approx(float(high))
    unweighted = [float(np.mean([np.mean(groups[int(i)]) for i in row])) for row in drawn]
    assert float(np.quantile(unweighted, 0.975)) != pytest.approx(result.high)


def test_cluster_bootstrap_is_fast() -> None:
    values = np.random.default_rng(0).random(500).tolist()
    labels = [f"c{i % 100}" for i in range(500)]
    start = time.perf_counter()
    bootstrap_ci(values, clusters=labels, resamples=10000)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(
    "kwargs",
    [{"resamples": 0}, {"confidence": 0.0}, {"confidence": 1.0}],
)
def test_bad_settings_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        bootstrap_ci(MIXED, **kwargs)  # type: ignore[arg-type]


def test_empty_values_raise() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_cluster_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0, 3.0], clusters=["a", "b"])


def test_paired_identical_systems() -> None:
    scores = {"q1": 0.2, "q2": 0.9, "q3": 0.0}
    result = paired_difference(scores, dict(scores))
    assert (result.mean, result.low, result.high) == (0.0, 0.0, 0.0)
    assert not excludes_zero(result)


def test_paired_constant_improvement() -> None:
    a = {f"q{i}": i / 10 for i in range(8)}
    b = {key: value + 0.2 for key, value in a.items()}
    result = paired_difference(a, b)
    assert result.mean == pytest.approx(0.2)
    assert result.low == pytest.approx(0.2)
    assert result.high == pytest.approx(0.2)
    assert excludes_zero(result)


def test_paired_worse_system_is_negative() -> None:
    a = {"q1": 1.0, "q2": 1.0}
    b = {"q1": 0.5, "q2": 0.5}
    result = paired_difference(a, b)
    assert result.mean == pytest.approx(-0.5)
    assert excludes_zero(result)


def test_paired_ignores_insertion_order() -> None:
    a = {"q1": 0.1, "q2": 0.7, "q3": 0.4, "q4": 0.0}
    b = {"q1": 0.3, "q2": 0.6, "q3": 0.9, "q4": 0.2}
    forward = paired_difference(a, b, seed=4)
    reverse = paired_difference(dict(reversed(a.items())), dict(reversed(b.items())), seed=4)
    assert forward == reverse


def test_paired_mismatched_keys_raise() -> None:
    with pytest.raises(ValueError):
        paired_difference({"q1": 1.0, "q2": 0.0}, {"q1": 1.0, "q3": 0.0})


def test_paired_clusters_are_honored() -> None:
    # Two clusters, each with a constant difference: -1 for g1, +1 for g2.
    a = {"a1": 1.0, "a2": 1.0, "b1": 0.0, "b2": 0.0}
    b = {"a1": 0.0, "a2": 0.0, "b1": 1.0, "b2": 1.0}
    clusters = {"a1": "g1", "a2": "g1", "b1": "g2", "b2": "g2"}
    clustered = paired_difference(a, b, clusters=clusters, resamples=2000)
    assert clustered.n_clusters == 2
    assert clustered.low == pytest.approx(-1.0)
    assert clustered.high == pytest.approx(1.0)
    assert paired_difference(a, b, resamples=2000).n_clusters == 4


def test_paired_cluster_key_missing_raises() -> None:
    a = {"q1": 1.0, "q2": 0.0}
    b = {"q1": 0.0, "q2": 1.0}
    with pytest.raises(ValueError):
        paired_difference(a, b, clusters={"q1": "g"})


def test_excludes_zero_cases() -> None:
    assert excludes_zero(_interval(0.1, 0.3))
    assert excludes_zero(_interval(-0.3, -0.1))
    assert not excludes_zero(_interval(-0.1, 0.2))
    assert not excludes_zero(_interval(0.0, 0.3))
    assert not excludes_zero(_interval(-0.3, 0.0))
