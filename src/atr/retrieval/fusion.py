from atr.retrieval.base import Hit


def rrf(rankings: list[list[Hit]], *, c: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: each ranking adds 1 / (c + rank) to an id's score.

    Only the order within each ranking matters; the input scores are ignored.
    """
    if c < 1:
        raise ValueError(f"c must be at least 1, got {c}")
    fused: dict[str, float] = {}
    for hits in rankings:
        ids = [hit.passage_id for hit in hits]
        if len(set(ids)) != len(ids):
            raise ValueError("a ranking contains a repeated passage id")
        for rank, passage_id in enumerate(ids, start=1):
            fused[passage_id] = fused.get(passage_id, 0.0) + 1 / (c + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(passage_id, score) for passage_id, score in ordered]
