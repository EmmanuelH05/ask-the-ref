import re
from collections import Counter

from atr.corpus.models import Alignment, Language, Passage

_KEY_FIELDS = ("source_slug", "law_number", "section_index")
_SUBSECTION_ID = re.compile(r"^(L\d+\.\d+)\.\d+$")


def _unique_ids(passages: list[Passage]) -> list[str]:
    return list(dict.fromkeys(p.passage_id for p in passages))


def _field_mismatch(pid: str, en: Passage, es: Passage) -> str | None:
    for field in _KEY_FIELDS:
        en_value, es_value = getattr(en, field), getattr(es, field)
        if en_value != es_value:
            return f"{pid}: {field} differs (en={en_value!r}, es={es_value!r})"
    return None


def _check_language(passages: list[Passage], language: Language) -> None:
    for passage in passages:
        if passage.language != language:
            raise ValueError(
                f"{passage.passage_id}: expected language {language!r}, got {passage.language!r}"
            )


def _distrust_shifted_sections(all_ids: list[str], paired: set[str]) -> list[str]:
    """Unpair every subsection of a section where any subsection failed to pair.

    Subsection ids are positional, so one missing block in an edition shifts every id
    after it and the surviving pairs would join different text.
    """
    by_section: dict[str, list[str]] = {}
    for passage_id in dict.fromkeys(all_ids):
        match = _SUBSECTION_ID.match(passage_id)
        if match:
            by_section.setdefault(match.group(1), []).append(passage_id)
    reasons: list[str] = []
    for section, ids in by_section.items():
        if all(passage_id in paired for passage_id in ids):
            continue
        paired.difference_update(ids)
        reasons.append(
            f"{section}: subsection ids differ between editions, "
            "so no pair in this section is trusted"
        )
    return reasons


def align(en: list[Passage], es: list[Passage]) -> Alignment:
    _check_language(en, "en")
    _check_language(es, "es")
    en_ids = _unique_ids(en)
    es_ids = _unique_ids(es)
    en_counts = Counter(p.passage_id for p in en)
    es_counts = Counter(p.passage_id for p in es)
    en_by_id = {p.passage_id: p for p in en}
    es_by_id = {p.passage_id: p for p in es}

    mismatches: list[str] = []
    paired: set[str] = set()
    for pid in dict.fromkeys([*en_ids, *es_ids]):
        if en_counts[pid] > 1:
            mismatches.append(f"{pid}: duplicated in en")
        if es_counts[pid] > 1:  # Counter yields 0 for absent ids
            mismatches.append(f"{pid}: duplicated in es")
        if en_counts[pid] != 1 or es_counts[pid] != 1:
            continue
        problem = _field_mismatch(pid, en_by_id[pid], es_by_id[pid])
        if problem is None:
            paired.add(pid)
        else:
            mismatches.append(problem)

    mismatches.extend(_distrust_shifted_sections([*en_ids, *es_ids], paired))

    return Alignment(
        pairs=[(pid, pid) for pid in en_ids if pid in paired],
        unaligned_en=[pid for pid in en_ids if pid not in paired],
        unaligned_es=[pid for pid in es_ids if pid not in paired],
        mismatches=mismatches,
    )
