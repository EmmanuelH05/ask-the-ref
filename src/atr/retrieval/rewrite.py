"""Expand a question with a restatement in the rulebook's own vocabulary (DECISIONS.md D35)."""

from typing import Protocol

from atr.llm import Completion
from atr.retrieval.translate import is_latin_script

REWRITE_PROMPT = (
    "Restate the soccer question below in one sentence, using the formal vocabulary of the "
    "IFAB Laws of the Game (for example: caution, sending-off, direct free kick, restart, "
    "offence, goalkeeper, penalty area). Keep the language or languages of the question. "
    "Do not answer it. Reply with the restated question only.\n\nQuestion:\n{text}"
)
MIN_LENGTH_RATIO = 0.3
MAX_LENGTH_RATIO = 3.0


class LLM(Protocol):
    def complete(self, model: str, prompt: str, *, think: str | None = None) -> Completion: ...


class RulebookRewriter:
    def __init__(self, llm: LLM, *, model: str = "qwen2.5:14b") -> None:
        self._llm = llm
        self._model = model
        self.name = f"rulebook:{model}"

    def expand(self, text: str) -> Completion:
        """The question followed by its restatement, so the original words are never lost.

        An unusable reply (far too short or long) falls back to the question alone: this
        stage is an optional boost, and a bad rewrite must not replace a good question.
        """
        reply = self._llm.complete(self._model, REWRITE_PROMPT.format(text=text))
        ratio = len(reply.text.split()) / max(len(text.split()), 1)
        if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO or not is_latin_script(reply.text):
            return Completion(text, reply.ms)
        return Completion(f"{text}\n{reply.text}", reply.ms)
