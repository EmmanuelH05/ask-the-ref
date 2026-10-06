import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from atr.corpus.models import Passage
from atr.evalset.models import QUESTION_TYPES, QuestionGroup
from atr.evalset.overlap import content_tokens, group_overlap
from atr.evalset.store import load_groups

DEFAULT_EXCERPT_WORDS = 60


def leak_tokens(question: str, passages: list[str]) -> list[str]:
    """Content words a question shares with its gold passages, the thing a paraphrase must avoid."""
    return sorted(set(content_tokens(question)) & set(content_tokens(" ".join(passages))))


def blind_flag(group: QuestionGroup, picked: list[str] | None) -> str | None:
    if picked is None:
        return None
    if group.type == "out_of_scope":
        return f"blind check found an answer in {', '.join(picked)}" if picked else None
    missed = [passage_id for passage_id in group.gold if passage_id not in picked]
    if not missed:
        return None
    return f"blind check missed {', '.join(missed)}; it picked {', '.join(picked) or 'nothing'}"


def _excerpt(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " [...]"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _flags(
    group: QuestionGroup, en: Mapping[str, str], es: Mapping[str, str], picked: list[str] | None
) -> list[str]:
    flags: list[str] = []
    if group.type == "paraphrase":
        for language, question, edition in (("en", group.en, en), ("es", group.es, es)):
            shared = leak_tokens(question, [edition[i] for i in group.gold if i in edition])
            if shared:
                flags.append(
                    f"paraphrase shares words with its gold passage ({language}): "
                    + ", ".join(shared)
                )
    blind = blind_flag(group, picked)
    if blind:
        flags.append(blind)
    return flags


def render_group(
    group: QuestionGroup,
    en: Mapping[str, str],
    es: Mapping[str, str],
    labels: Mapping[str, str],
    *,
    picked: list[str] | None,
    excerpt_words: int,
) -> str:
    scores = group_overlap(group, en, es)
    lines = [
        f"### {group.group_id} ({group.type})",
        f"- EN: {group.en}",
        f"- ES: {group.es}",
        f"- CS: {group.cs}",
        f"- Reference answer: {group.answer}",
        f"- Drafter note: {group.note}",
    ]
    if not group.gold:
        lines.append("- Gold: none")
    for passage_id in group.gold:
        text = _excerpt(en.get(passage_id, "(missing from the English corpus)"), excerpt_words)
        lines.append(f"- Gold `{passage_id}` {labels.get(passage_id, '?')}: {text}")
    lines.append(
        f"- Overlap: en→en {_fmt(scores.en_vs_en)}, es→es {_fmt(scores.es_vs_es)}, "
        f"es→en {_fmt(scores.es_vs_en)}, cs→en {_fmt(scores.cs_vs_en)}, "
        f"cs→es {_fmt(scores.cs_vs_es)}"
    )
    lines.extend(f"- FLAG {flag}" for flag in _flags(group, en, es, picked))
    return "\n".join(lines)


def load_blind(directory: Path) -> dict[str, list[str]]:
    """Blind-check picks by group id, from every jsonl file of {"group_id", "picked"} lines."""
    picks: dict[str, list[str]] = {}
    if not directory.is_dir():
        return picks
    for path in sorted(directory.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("group_id"), str) or not isinstance(row.get("picked"), list):
                raise ValueError(f"{path.name}:{number}: need group_id and picked")
            picks[row["group_id"]] = [str(item) for item in row["picked"]]
    return picks


def _label(passage: Passage) -> str:
    title = f"{passage.law_title} / {passage.section_title}"
    return f"{title} / {passage.subheading}" if passage.subheading else title


def _load_corpus(path: Path) -> list[Passage]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Passage.model_validate_json(line) for line in lines]


def render_sheet(
    groups: list[QuestionGroup],
    en: Mapping[str, str],
    es: Mapping[str, str],
    labels: Mapping[str, str],
    picks: Mapping[str, list[str]],
    excerpt_words: int,
) -> str:
    blocks = {
        g.group_id: render_group(
            g, en, es, labels, picked=picks.get(g.group_id), excerpt_words=excerpt_words
        )
        for g in groups
    }
    flagged = [group_id for group_id, block in blocks.items() if "- FLAG " in block]
    parts = [
        "# Question review sheet",
        f"{len(groups)} groups, {3 * len(groups)} questions. {len(flagged)} groups carry a flag: "
        + (", ".join(flagged) or "none"),
        "Fix the draft files under `evalset/draft/`, not this sheet. Set `spanish_reviewed` to "
        "true on a group once its Spanish and code-switched text is yours.",
    ]
    for question_type in QUESTION_TYPES:
        parts.append(f"## {question_type}")
        parts.extend(blocks[g.group_id] for g in groups if g.type == question_type)
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the question review sheet.")
    parser.add_argument("--draft-dir", type=Path, default=Path("evalset/draft"))
    parser.add_argument("--blind-dir", type=Path, default=Path("evalset/blind"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=Path("data/review/questions.md"))
    parser.add_argument("--excerpt-words", type=int, default=DEFAULT_EXCERPT_WORDS)
    args = parser.parse_args(argv)
    en_passages = _load_corpus(args.corpus_dir / "en.jsonl")
    en = {p.passage_id: p.text for p in en_passages}
    es = {p.passage_id: p.text for p in _load_corpus(args.corpus_dir / "es.jsonl")}
    labels = {p.passage_id: _label(p) for p in en_passages}
    groups = load_groups(args.draft_dir)
    sheet = render_sheet(groups, en, es, labels, load_blind(args.blind_dir), args.excerpt_words)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(sheet, encoding="utf-8")
    print(f"{len(groups)} groups -> {args.out} ({sheet.count('- FLAG ')} flags)")


if __name__ == "__main__":
    main()
