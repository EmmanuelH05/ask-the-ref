from atr.eval.plot import Point, pareto_front


def test_pareto_front_keeps_points_no_other_point_beats_on_both_axes() -> None:
    points = [
        Point("fast-weak", "M", 5.0, 0.40),
        Point("slow-strong", "T", 900.0, 0.72),
        Point("slow-weak", "T", 950.0, 0.60),
        Point("mid", "M", 50.0, 0.50),
        Point("mid-dominated", "M", 60.0, 0.45),
    ]
    assert [p.label for p in pareto_front(points)] == ["fast-weak", "mid", "slow-strong"]


def test_pareto_front_keeps_one_of_two_identical_points_and_handles_none() -> None:
    twins = [Point("a", "T", 10.0, 0.5), Point("b", "T", 10.0, 0.5)]
    assert [p.label for p in pareto_front(twins)] == ["a"]
    assert pareto_front([]) == []


def test_latencies_equal_to_two_significant_figures_are_one_speed_class() -> None:
    points = [
        Point("a", "T", 929.5, 0.67),
        Point("b", "T", 930.0, 0.72),
        Point("c", "T", 2600.0, 0.74),
        Point("d", "M", 6.0, 0.43),
        Point("e", "M", 6.4, 0.44),
        Point("f", "M", 6.5, 0.50),
    ]
    assert [p.label for p in pareto_front(points)] == ["d", "e", "f", "b", "c"]
