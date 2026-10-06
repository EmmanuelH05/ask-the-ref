from dataclasses import dataclass
from typing import Literal

from atr.corpus.models import count_words

ChunkStrategy = Literal["passage", "blocks", "items"]
MAX_WORDS = 120


@dataclass(frozen=True)
class _Unit:
    """A plain paragraph (no items) or a lead-in with its list."""

    lead_in: str
    items: tuple[str, ...]

    def render(self, items: tuple[str, ...] | None = None, sep: str = "\n\n") -> str:
        body = "\n".join(self.items if items is None else items)
        return f"{self.lead_in}{sep}{body}" if self.lead_in and body else self.lead_in or body

    def words(self) -> int:
        return count_words(self.render())


def _split_items(block: str) -> tuple[str, ...]:
    items: list[str] = []
    for line in block.split("\n"):
        if line.startswith("- ") or not items:
            items.append(line)
        else:
            items[-1] += "\n" + line
    return tuple(items)


def _units(text: str) -> list[_Unit]:
    units: list[_Unit] = []
    lead: str | None = None
    for block in (b for b in text.split("\n\n") if b.strip()):
        if block.startswith("- "):
            units.append(_Unit(lead or "", _split_items(block)))
            lead = None
            continue
        if lead is not None:
            units.append(_Unit(lead, ()))
        lead = block
    if lead is not None:
        units.append(_Unit(lead, ()))
    return units


def _split_list(unit: _Unit) -> list[str]:
    pieces: list[tuple[str, ...]] = []
    current: tuple[str, ...] = ()
    for item in unit.items:
        trial = (*current, item)
        if current and count_words(unit.render(trial)) > MAX_WORDS:
            pieces.append(current)
            trial = (item,)
        current = trial
    pieces.append(current)
    return [unit.render(piece) for piece in pieces]


def _pack(units: list[_Unit], strategy: ChunkStrategy) -> list[str]:
    chunks: list[str] = []
    pending: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal size
        if pending:
            chunks.append("\n\n".join(pending))
        pending.clear()
        size = 0

    for unit in units:
        if strategy == "items" and unit.items:
            flush()
            chunks.extend(unit.render((item,), "\n") for item in unit.items)
        elif unit.items and unit.words() > MAX_WORDS:
            flush()
            chunks.extend(_split_list(unit))
        else:
            if pending and size + unit.words() > MAX_WORDS:
                flush()
            pending.append(unit.render())
            size += unit.words()
    flush()
    return chunks


def chunk_text(text: str, strategy: ChunkStrategy) -> list[str]:
    if strategy == "passage":
        return [text]
    return _pack(_units(text), strategy)
