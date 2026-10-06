from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict


class Interval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float
    low: float
    high: float
    n: int
    n_clusters: int
    resamples: int
    seed: int
    confidence: float


def _cluster_ids(clusters: Sequence[str] | None, n: int) -> np.ndarray:
    if clusters is None:
        return np.arange(n)
    seen: dict[str, int] = {}
    return np.array([seen.setdefault(label, len(seen)) for label in clusters])


def bootstrap_ci(
    values: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    resamples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Interval:
    if len(values) == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if clusters is not None and len(clusters) != len(values):
        raise ValueError(f"{len(clusters)} cluster labels for {len(values)} values")
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be inside (0, 1), got {confidence}")

    data = np.asarray(values, dtype=np.float64)
    ids = _cluster_ids(clusters, len(data))
    sums = np.bincount(ids, weights=data)
    sizes = np.bincount(ids).astype(np.float64)
    n_clusters = len(sums)

    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, n_clusters, size=(resamples, n_clusters))
    # Weighted by cluster size: a resample's mean is over values, not over cluster means.
    means = sums[drawn].sum(axis=1) / sizes[drawn].sum(axis=1)

    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return Interval(
        mean=float(data.mean()),
        low=float(low),
        high=float(high),
        n=len(data),
        n_clusters=n_clusters,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )


def paired_difference(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    clusters: Mapping[str, str] | None = None,
    resamples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Interval:
    if a.keys() != b.keys():
        only = sorted(a.keys() ^ b.keys())
        raise ValueError(f"key sets differ: {only[:5]}")
    keys = sorted(a)
    labels: list[str] | None = None
    if clusters is not None:
        missing = [key for key in keys if key not in clusters]
        if missing:
            raise ValueError(f"keys missing from clusters: {missing[:5]}")
        labels = [clusters[key] for key in keys]
    return bootstrap_ci(
        [b[key] - a[key] for key in keys],
        clusters=labels,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )


def excludes_zero(interval: Interval) -> bool:
    return interval.low > 0 or interval.high < 0
