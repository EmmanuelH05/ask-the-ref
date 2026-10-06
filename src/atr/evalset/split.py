import random

from atr.evalset.models import QUESTION_TYPES, QuestionGroup, Split, is_held_out


def assign_splits(
    groups: list[QuestionGroup], *, seed: int, test_fraction: float = 0.5
) -> dict[str, Split]:
    """Assign every group to dev, test or heldout (DECISIONS.md D13)."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    ids = [group.group_id for group in groups]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate group ids: {duplicates}")

    assigned: dict[str, Split] = {g.group_id: "heldout" for g in groups if is_held_out(g)}
    for question_type in QUESTION_TYPES:
        # Sorting before the shuffle makes the result independent of input order.
        pool = sorted(
            g.group_id for g in groups if g.type == question_type and g.group_id not in assigned
        )
        random.Random(f"{seed}:{question_type}").shuffle(pool)
        cut = round(len(pool) * test_fraction)
        for position, group_id in enumerate(pool):
            assigned[group_id] = "test" if position < cut else "dev"
    return {group_id: assigned[group_id] for group_id in ids}
