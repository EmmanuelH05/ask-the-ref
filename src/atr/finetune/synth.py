"""Synthetic training questions from the non-held-out English passages (DECISIONS.md D39).

Dev and test questions are never read here: the only inputs are passages. The held-out laws
are excluded so that the held-out slice stays an honest measure of generalization.
"""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from atr.cache import JsonCache
from atr.corpus.models import Passage
from atr.eval.run import CACHE_DIR, machine_name
from atr.evalset.models import HELD_OUT_LAWS, QUESTION_LANGUAGES, QuestionLanguage
from atr.llm import Completion, OllamaLLM
from atr.retrieval.docs import load_passages, title_line

QUESTIONS_PER_LANGUAGE = 2
SYNTH_MODEL = "qwen2.5:14b"
SYNTH_PROMPT = (
    "Below is a passage from the IFAB Laws of the Game (soccer). Write questions that a fan "
    "or a referee might ask whose answer is in this passage. Do not copy the passage's "
    "wording; ask the way a person would. Write exactly {n} questions in English, {n} in "
    "Spanish, and {n} that mix Spanish and English within one sentence, the way a bilingual "
    'speaker talks. Reply with a JSON object with keys "en", "es" and "cs", each a list '
    "of {n} strings, and nothing else.\n\n{title}\n{text}"
)
SYNTHETIC_PATH = Path("data/finetune/synthetic.jsonl")
MANIFEST_PATH = Path("runs/finetune/synthetic.json")
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
_WORD = re.compile(r"[a-záéíóúñü]+")
# Function words that mark a clause as English or Spanish; a sentence with both has a switch.
_EN_FUNCTION = frozenset(
    (
        "the",
        "is",
        "are",
        "was",
        "what",
        "when",
        "can",
        "does",
        "do",
        "how",
        "of",
        "to",
        "in",
        "if",
        "with",
        "for",
    )
)
_ES_FUNCTION = frozenset(
    (
        "el",
        "la",
        "los",
        "las",
        "es",
        "son",
        "que",
        "de",
        "en",
        "si",
        "un",
        "una",
        "cuando",
        "como",
        "por",
        "para",
        "con",
    )
)


class SynthExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str
    language: QuestionLanguage
    question: str

    @model_validator(mode="after")
    def _not_blank(self) -> Self:
        if not self.question.strip():
            raise ValueError("a synthetic question cannot be blank")
        return self


class SynthManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    prompt_sha256: str
    passages: int
    passages_without_usable_reply: list[str]
    examples: int
    by_language: dict[str, int]
    # How many "cs" drafts switch inside a sentence, not just between sentences (D39).
    cs_switching_within_a_sentence: int
    held_out_laws: list[int]
    hash: str
    machine: str
    created_at: str


def training_passages(passages: list[Passage]) -> list[Passage]:
    """English passages outside the held-out laws. Glossary entries have no law and stay."""
    return [p for p in passages if p.language == "en" and p.law_number not in set(HELD_OUT_LAWS)]


def _clean_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        text = item.strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or None


def parse_synthetic(reply: str) -> dict[str, list[str]] | None:
    """The questions per language, or None when the reply is not the asked-for object."""
    match = _OBJECT.search(reply)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    result: dict[str, list[str]] = {}
    for language in QUESTION_LANGUAGES:
        cleaned = _clean_list(parsed.get(language))
        if cleaned is None:
            return None
        result[language] = cleaned
    return result


def examples_from_reply(passage_id: str, reply: Completion) -> list[SynthExample]:
    parsed = parse_synthetic(reply.text)
    if parsed is None:
        return []
    return [
        SynthExample(passage_id=passage_id, language=language, question=question)
        for language in QUESTION_LANGUAGES
        for question in parsed[language]
    ]


def mixes_within_sentence(text: str) -> bool:
    """True when one sentence holds both an English and a Spanish function word."""
    for sentence in _SENTENCE_END.split(text):
        words = set(_WORD.findall(sentence.lower()))
        if words & _EN_FUNCTION and words & _ES_FUNCTION:
            return True
    return False


def synthetic_hash(examples: list[SynthExample]) -> str:
    rows = sorted(e.model_dump_json() for e in examples)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def generate(
    passages: list[Passage], llm: OllamaLLM, *, model: str = SYNTH_MODEL
) -> tuple[list[SynthExample], list[str]]:
    examples: list[SynthExample] = []
    failed: list[str] = []
    for passage in passages:
        prompt = SYNTH_PROMPT.format(
            n=QUESTIONS_PER_LANGUAGE, title=title_line(passage), text=passage.text
        )
        found = examples_from_reply(passage.passage_id, llm.complete(model, prompt))
        if not found:
            failed.append(passage.passage_id)
        examples.extend(found)
    return examples, failed


def load_examples(path: Path) -> list[SynthExample]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [SynthExample.model_validate_json(line) for line in lines if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write synthetic training questions.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=SYNTHETIC_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model", default=SYNTH_MODEL)
    args = parser.parse_args(argv)
    sources = training_passages(load_passages(args.corpus_dir, ("en",)))
    llm = OllamaLLM(cache=JsonCache(CACHE_DIR / "llm"))
    examples, failed = generate(sources, llm, model=args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8")
    manifest = SynthManifest(
        model=args.model,
        prompt_sha256=hashlib.sha256(SYNTH_PROMPT.encode("utf-8")).hexdigest(),
        passages=len(sources),
        passages_without_usable_reply=failed,
        examples=len(examples),
        by_language={x: sum(1 for e in examples if e.language == x) for x in QUESTION_LANGUAGES},
        cs_switching_within_a_sentence=sum(
            1 for e in examples if e.language == "cs" and mixes_within_sentence(e.question)
        ),
        held_out_laws=list(HELD_OUT_LAWS),
        hash=synthetic_hash(examples),
        machine=machine_name(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"{len(examples)} synthetic questions from {len(sources)} passages "
        f"({len(failed)} passages gave no usable reply) -> {args.out}"
    )
    print(f"by language: {manifest.by_language}")
    print(
        f"cs drafts that switch inside a sentence: {manifest.cs_switching_within_a_sentence} "
        f"of {manifest.by_language['cs']}"
    )
    print(f"hash {manifest.hash}")


if __name__ == "__main__":
    main()
