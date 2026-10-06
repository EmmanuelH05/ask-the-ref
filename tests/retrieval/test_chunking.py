from atr.corpus.models import count_words
from atr.retrieval.chunking import MAX_WORDS, ChunkStrategy, chunk_text

STRATEGIES: tuple[ChunkStrategy, ...] = ("passage", "blocks", "items")
PACKING: tuple[ChunkStrategy, ...] = ("blocks", "items")


def words(tag: str, n: int) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def lines_of(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.strip()]


MIXED = "\n\n".join(
    [
        words("a", 30),
        "The zorbs rule:\n\n- "
        + words("b", 40)
        + "\n  - "
        + words("c", 5)
        + "\n- "
        + words("d", 40),
        words("e", 50),
        "Also:\n\n- " + words("f", 70) + "\n- " + words("g", 70),
        words("h", 20),
    ]
)


def test_passage_returns_text_unchanged() -> None:
    text = "Intro:\n\n- one\n- two\n\n" + words("x", 300)
    assert chunk_text(text, "passage") == [text]


def test_short_text_without_list_is_one_chunk_under_every_strategy() -> None:
    text = words("p", 40) + "\n\n" + words("q", 40)
    for strategy in STRATEGIES:
        assert chunk_text(text, strategy) == [text]


def test_blocks_packs_paragraphs_greedily() -> None:
    p1, p2, p3 = words("a", 50), words("b", 50), words("c", 50)
    assert MAX_WORDS == 120
    assert chunk_text("\n\n".join([p1, p2, p3]), "blocks") == [p1 + "\n\n" + p2, p3]


def test_lead_in_and_list_stay_together_when_they_fit() -> None:
    p1 = words("a", 50)
    unit = "Zorbs may:\n\n- " + words("b", 20) + "\n- " + words("c", 20)
    text = p1 + "\n\n" + unit
    assert chunk_text(text, "blocks") == [text]


def test_oversize_list_unit_splits_by_items_and_repeats_lead_in() -> None:
    lead = "Zorbs may:"
    items = [f"- {words(t, 50)}" for t in "xyz"]
    text = lead + "\n\n" + "\n".join(items)
    chunks = chunk_text(text, "blocks")
    assert chunks == [
        lead + "\n\n" + items[0] + "\n" + items[1],
        lead + "\n\n" + items[2],
    ]
    assert all(count_words(c) <= MAX_WORDS for c in chunks)


def test_nested_lines_travel_with_parent_item() -> None:
    lead = "Zorbs may:"
    first = "- " + words("x", 60) + "\n  - " + words("n", 10)
    second = "- " + words("y", 60)
    text = lead + "\n\n" + first + "\n" + second
    assert chunk_text(text, "blocks") == [lead + "\n\n" + first, lead + "\n\n" + second]


def test_plain_paragraph_over_limit_stays_whole() -> None:
    big = words("a", MAX_WORDS + 30)
    assert chunk_text(big, "blocks") == [big]
    assert chunk_text(big, "items") == [big]


def test_items_gives_one_chunk_per_item_and_packs_paragraphs() -> None:
    p1, p2 = words("a", 50), words("b", 50)
    text = "\n\n".join([p1, p2, "Zorbs may:\n\n- one two\n- three\n  - four", words("c", 10)])
    assert chunk_text(text, "items") == [
        p1 + "\n\n" + p2,
        "Zorbs may:\n- one two",
        "Zorbs may:\n- three\n  - four",
        words("c", 10),
    ]


def test_list_without_lead_in() -> None:
    text = "- alpha\n- beta\n  - gamma"
    assert chunk_text(text, "items") == ["- alpha", "- beta\n  - gamma"]
    assert chunk_text(text, "blocks") == [text]
    big = "- " + words("x", 80) + "\n- " + words("y", 80)
    assert chunk_text(big, "blocks") == big.split("\n")


def test_no_chunk_is_empty() -> None:
    for strategy in PACKING:
        chunks = chunk_text(MIXED, strategy)
        assert chunks
        assert all(c.strip() for c in chunks)


def test_every_non_blank_line_appears_in_a_chunk() -> None:
    for strategy in PACKING:
        seen = {line for c in chunk_text(MIXED, strategy) for line in lines_of(c)}
        assert set(lines_of(MIXED)) <= seen


def test_chunks_keep_document_order() -> None:
    tags = "abcdefgh"
    for strategy in PACKING:
        joined = "\n\n".join(chunk_text(MIXED, strategy))
        positions = [joined.index(f"{tag}0") for tag in tags]
        assert positions == sorted(positions)
