"""Split a question into sub-questions that are searched separately (DECISIONS.md D36)."""

import json
import re

from atr.retrieval.rewrite import LLM

DECOMPOSE_PROMPT = (
    "Some soccer rules questions need more than one rule to answer. If the question below "
    "needs two or three separate rules, write one short standalone question per rule, in "
    "the language of the question. If one rule is enough, return the question unchanged as "
    "the only element. Reply with a JSON array of strings and nothing else.\n\n"
    "Question:\n{text}"
)
MIN_PARTS = 2
MAX_PARTS = 3
_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class Decomposer:
    def __init__(self, llm: LLM, *, model: str = "qwen2.5:14b") -> None:
        self._llm = llm
        self._model = model
        self.name = f"decompose:{model}"

    def split(self, text: str) -> tuple[list[str], float]:
        """Sub-questions and what the call cost. Empty when the question should stay whole.

        A reply that is not a list of two or three strings also gives an empty list: the
        pipeline then searches with the original question only.
        """
        reply = self._llm.complete(self._model, DECOMPOSE_PROMPT.format(text=text))
        match = _ARRAY.search(reply.text)
        if match is None:
            return [], reply.ms
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], reply.ms
        usable = (
            isinstance(parsed, list)
            and MIN_PARTS <= len(parsed) <= MAX_PARTS
            and all(isinstance(part, str) and part.strip() for part in parsed)
        )
        return ([part.strip() for part in parsed], reply.ms) if usable else ([], reply.ms)
