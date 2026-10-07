"""Accuracy against latency for every configuration the ablation tried."""

import argparse
from pathlib import Path
from typing import NamedTuple

from atr.eval.ablation import PRIMARY, RunMeta, total_ms
from atr.eval.report import load_records, nearest_rank

# Categorical slots 1 and 2 of the default palette, checked with the dataviz validator.
TRACK_COLORS = {"T": "#2a78d6", "M": "#eb6834"}
TRACK_NAMES = {"T": "translate first (track T)", "M": "multilingual, no translation (track M)"}
LATENCY_TOLERANCE = 0.05
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


class Point(NamedTuple):
    label: str
    track: str
    p50_ms: float
    recall_at_5: float


def pareto_front(points: list[Point]) -> list[Point]:
    """Points no other point beats on both latency and recall, fastest first.

    Latencies within LATENCY_TOLERANCE of each other count as equal: a half-millisecond gap
    between two runs is measurement noise, not a trade-off.
    """
    front: list[Point] = []
    for point in sorted(points, key=lambda p: (p.p50_ms, -p.recall_at_5, p.label)):
        if front and point.recall_at_5 <= front[-1].recall_at_5:
            continue
        if front and point.p50_ms <= front[-1].p50_ms * (1 + LATENCY_TOLERANCE):
            front.pop()
        front.append(point)
    return front


def load_points(ablation_dir: Path) -> list[Point]:
    points = []
    for meta_path in sorted(ablation_dir.glob("*__dev/meta.json")):
        meta = RunMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        records = load_records(meta_path.parent)
        scored = [getattr(r.metrics, PRIMARY) for r in records if r.metrics is not None]
        latency = nearest_rank(sorted(total_ms(r) for r in records), 50)
        label = meta.config.label
        points.append(Point(label, label[0], latency, sum(scored) / len(scored)))
    return points


def draw(points: list[Point], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    front = pareto_front(points)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for track, color in TRACK_COLORS.items():
        members = [p for p in points if p.track == track]
        ax.scatter(
            [p.p50_ms for p in members],
            [p.recall_at_5 for p in members],
            s=70,
            color=color,
            edgecolors=SURFACE,
            linewidths=2,
            label=TRACK_NAMES[track],
            zorder=3,
        )
    ax.plot(
        [p.p50_ms for p in front],
        [p.recall_at_5 for p in front],
        color=MUTED,
        linewidth=1.5,
        linestyle="--",
        label="best trade-offs (no config is both faster and better)",
        zorder=2,
    )
    for point in front:
        ax.annotate(
            point.label,
            (point.p50_ms, point.recall_at_5),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_xlabel("median latency per question, ms (log scale)", color=MUTED)
    ax.set_ylabel("recall@5 on dev", color=MUTED)
    ax.set_title(
        "Retrieval accuracy against latency, every configuration tried",
        loc="left",
        color=INK,
        fontsize=12,
    )
    ax.grid(True, color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot recall@5 against latency for the ablation.")
    parser.add_argument("--ablation-dir", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/accuracy_latency.png"))
    args = parser.parse_args(argv)
    points = load_points(args.ablation_dir)
    draw(points, args.out)
    print(f"{len(points)} configurations -> {args.out}")
    print("Best trade-offs (fastest first):")
    for point in pareto_front(points):
        print(f"  {point.label:<34} p50 {point.p50_ms:>8.1f} ms  recall@5 {point.recall_at_5:.2f}")


if __name__ == "__main__":
    main()
