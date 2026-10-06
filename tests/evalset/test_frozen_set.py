from pathlib import Path

from atr.evalset.freeze import verify_frozen
from atr.evalset.models import expand, is_held_out

ROOT = Path(__file__).resolve().parents[2]
# Changing this line is the visible act of re-freezing. See DECISIONS.md D18.
FROZEN_HASH = "1654d35d3dad2aab3bd0bdb0b22df9ea818c9fce83b6882605256e8fbdb325fa"


def test_shipped_set_matches_its_manifest_the_corpus_and_the_pinned_hash() -> None:
    groups, manifest = verify_frozen(
        ROOT / "evalset/questions.jsonl",
        ROOT / "evalset/manifest.json",
        ROOT / "corpus/manifest.json",
    )
    assert manifest.hash == FROZEN_HASH
    assert len([question for group in groups for question in expand(group)]) >= 300


def test_held_out_law_questions_are_only_in_the_heldout_split() -> None:
    groups, manifest = verify_frozen(
        ROOT / "evalset/questions.jsonl",
        ROOT / "evalset/manifest.json",
        ROOT / "corpus/manifest.json",
    )
    for group in groups:
        assert (manifest.splits[group.group_id] == "heldout") == is_held_out(group)
