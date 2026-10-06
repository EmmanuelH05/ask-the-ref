from collections import Counter

from atr.corpus.models import Alignment, Language, Passage

_KEY_FIELDS = ("source_slug", "law_number", "section_index")


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

    return Alignment(
        pairs=[(pid, pid) for pid in en_ids if pid in paired],
        unaligned_en=[pid for pid in en_ids if pid not in paired],
        unaligned_es=[pid for pid in es_ids if pid not in paired],
        mismatches=mismatches,
    )
