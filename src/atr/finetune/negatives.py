"""Hard negatives: passages the untrained model ranks high that are not the answer."""

from collections.abc import Mapping
from typing import NamedTuple

from atr.finetune.synth import SynthExample


class Example(NamedTuple):
    """One training row: the question, its passage, and every mined negative, texts prefixed.

    The ids travel with the texts so the loss can mask a false negative: another row in the
    batch whose positive, or whose mined negative, is this row's own passage.
    """

    anchor: str
    positive: str
    positive_id: str
    negatives: tuple[str, ...]
    negative_ids: tuple[str, ...]


def section_key(passage_id: str) -> str:
    """`L12.1.3` and `L12.1` share a section; glossary ids are their own section."""
    if passage_id.startswith("G."):
        return passage_id
    parts = passage_id.split(".")
    return ".".join(parts[:2])


def mine_hard_negatives(ranked: list[str], gold: str, *, n: int) -> list[str]:
    """The first `n` ranked ids outside the gold passage's section.

    Siblings are skipped because neighbouring subsections of one section often answer the
    same question; calling them negatives would teach the model the wrong thing.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    forbidden = section_key(gold)
    return [pid for pid in ranked if section_key(pid) != forbidden][:n]


def build_examples(
    examples: list[SynthExample],
    rankings: Mapping[str, list[str]],
    texts: Mapping[str, str],
    *,
    negatives_per_example: int,
    query_prefix: str,
    passage_prefix: str,
) -> list[Example]:
    """One row per question holding all its negatives. A question without a ranking is a bug,
    not a skip; a question whose ranking yields no usable negative is skipped."""
    rows: list[Example] = []
    for example in examples:
        ranked = rankings[example.question]
        negatives = mine_hard_negatives(ranked, example.passage_id, n=negatives_per_example)
        if not negatives:
            continue
        rows.append(
            Example(
                query_prefix + example.question,
                passage_prefix + texts[example.passage_id],
                example.passage_id,
                tuple(passage_prefix + texts[negative] for negative in negatives),
                tuple(negatives),
            )
        )
    return rows
