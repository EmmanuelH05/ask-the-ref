from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from atr.corpus.build import CorpusManifest
from atr.evalset.models import QUESTION_TYPES, QuestionGroup

WARNING_PREFIX = "warn: "


def _parse_line(path: Path, number: int, line: str) -> QuestionGroup:
    try:
        return QuestionGroup.model_validate_json(line)
    except ValidationError as error:
        raise ValueError(f"{path.name}:{number}: {error}") from error


def load_groups(directory: Path) -> list[QuestionGroup]:
    groups: list[QuestionGroup] = []
    for path in sorted(directory.glob("*.jsonl")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                groups.append(_parse_line(path, number, line))
    return groups


def write_groups(groups: list[QuestionGroup], path: Path) -> None:
    lines = [group.model_dump_json() + "\n" for group in groups]
    path.write_text("".join(lines), encoding="utf-8")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _duplicate_problems(groups: list[QuestionGroup]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if group.group_id in seen:
            problems.append(f"duplicate group_id: {group.group_id}")
        seen.add(group.group_id)
    by_text: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        by_text[_normalize(group.en)].append(group.group_id)
    for ids in by_text.values():
        if len(ids) > 1:
            problems.append(f"same English text in groups: {', '.join(ids)}")
    return problems


def validate(groups: list[QuestionGroup], en_ids: set[str], es_ids: set[str]) -> list[str]:
    problems = _duplicate_problems(groups)
    for group in groups:
        for passage_id in group.gold:
            if passage_id not in en_ids:
                problems.append(f"{group.group_id}: gold id {passage_id} not in en corpus")
    present = {group.type for group in groups}
    for question_type in QUESTION_TYPES:
        if question_type not in present:
            problems.append(f"no group of type {question_type}")
    for group in groups:
        for passage_id in group.gold:
            if passage_id in en_ids and passage_id not in es_ids:
                problems.append(
                    f"{WARNING_PREFIX}{group.group_id}: gold id {passage_id} not in es corpus"
                )
    return problems


def corpus_ids(manifest_path: Path) -> tuple[set[str], set[str]]:
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    en = {e.passage_id for e in manifest.passages if e.language == "en"}
    es = {e.passage_id for e in manifest.passages if e.language == "es"}
    return en, es
