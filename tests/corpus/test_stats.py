import pytest

from atr.corpus.models import Passage, count_words
from atr.corpus.stats import counts_by_law, distribution, sample_pairs, truncate_words


def passage(passage_id: str, law: int | None, text: str = "blue wombats juggle") -> Passage:
    return Passage.model_validate(
        {
            "passage_id": passage_id,
            "language": "en",
            "law_number": law,
            "law_title": "Invented",
            "section_index": None if law is None else 1,
            "section_title": "Invented",
            "subheading": None,
            "source_slug": "invented",
            "text": text,
            "word_count": count_words(text),
        }
    )


def test_distribution_of_known_values() -> None:
    result = distribution([1, 2, 3, 4, 5])
    assert (result.n, result.minimum, result.median, result.maximum) == (5, 1, 3, 5)
    assert result.mean == pytest.approx(3.0)
    assert result.total == 15
    assert (result.p25, result.p75) == (2, 4)


def test_distribution_percentile_is_nearest_rank() -> None:
    assert distribution(list(range(1, 11))).p90 == 9


def test_distribution_of_nothing_raises() -> None:
    with pytest.raises(ValueError):
        distribution([])


def test_counts_by_law_puts_glossary_under_none_and_sorts_laws() -> None:
    passages = [passage("L2.1", 2), passage("G.x.y", None), passage("L1.1", 1), passage("L2.2", 2)]
    assert counts_by_law(passages) == {1: 1, 2: 2, None: 1}
    assert list(counts_by_law(passages)) == [1, 2, None]


def test_sample_pairs_is_seeded_and_bounded() -> None:
    ids = [f"L1.{i}" for i in range(20)]
    assert sample_pairs(ids, 5, seed=3) == sample_pairs(ids, 5, seed=3)
    assert len(sample_pairs(ids, 5, seed=3)) == 5
    assert sample_pairs(ids[:2], 5, seed=3) == ids[:2]


def test_truncate_words_marks_the_cut() -> None:
    assert truncate_words("one two three four", 2) == "one two [...]"
    assert truncate_words("one two", 2) == "one two"
