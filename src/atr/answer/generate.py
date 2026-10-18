"""Answers that come only from retrieved passages and cite them by id (DECISIONS.md D41)."""

import re

from pydantic import BaseModel, ConfigDict

from atr.corpus.models import Passage
from atr.evalset.models import QuestionLanguage
from atr.llm import Completion
from atr.retrieval.docs import title_line
from atr.retrieval.rewrite import LLM
from atr.text import fold

GENERATOR_MODEL = "qwen2.5:14b"
# Spanish and code-switched texts are drafts until mannie reviews them (decisions table).
REFUSAL_TEXT: dict[QuestionLanguage, str] = {
    "en": "I don't know. The Laws of the Game I have do not answer this question.",
    "es": "No lo sé. Las Reglas de Juego que tengo no responden esta pregunta.",
    "cs": "No lo sé. The Laws of the Game que tengo no responden esta pregunta.",
}
# A refusal opens with the fixed sentence's first words and then ends or punctuates. A bare
# prefix would also match ordinary answers: "No se permite ...", "I do not know of any rule ...".
_REFUSAL_OPENING = re.compile(r"^(?:i dont know|i do not know|no lo se)(?=$|[.,;:!?])")
LANGUAGE_NAMES: dict[QuestionLanguage, str] = {
    "en": "English",
    "es": "Spanish",
    "cs": "the same mix of Spanish and English as the question",
}
ANSWER_PROMPT = (
    "You answer questions about the IFAB Laws of the Game (soccer) using ONLY the passages "
    "below. Rules:\n"
    "- Every sentence must end with the id of the passage it comes from in square brackets, "
    "for example [L12.1] or [G.football-terms.offside]. Cite only ids listed below.\n"
    "- Answer in {language}.\n"
    "- At most three sentences. No preamble.\n"
    "- If the question assumes something the passages contradict, say so and cite the passage.\n"
    "- If the passages do not answer the question, reply with exactly this and nothing else: "
    "{refusal}\n\n"
    "Passages:\n{passages}\n\nQuestion: {question}\n\nAnswer:"
)
CLOSED_BOOK_PROMPT = (
    "Answer the question about the IFAB Laws of the Game (soccer) from your own knowledge, "
    "in {language}, in at most three sentences. After each sentence cite the law and section "
    "you believe it comes from as an id in square brackets, like [L12.1] (law 12, section 1). "
    "If you do not know, reply with exactly: {refusal}\n\nQuestion: {question}\n\nAnswer:"
)
_ID = r"L\d+\.\d+(?:\.\d+)?|G\.[a-z-]+\.[a-z0-9-]+"
_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_ID_TOKEN = re.compile(_ID)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Claim(_Frozen):
    text: str
    citations: list[str]


class Answer(_Frozen):
    text: str
    citations: list[str]
    refused: bool
    ms: float


def extract_citations(text: str, allowed: set[str]) -> list[str]:
    """Cited ids in order of first appearance, keeping only ids the model was allowed to cite."""
    found: list[str] = []
    for bracket in _BRACKET.findall(text):
        for passage_id in _ID_TOKEN.findall(bracket):
            if passage_id in allowed and passage_id not in found:
                found.append(passage_id)
    return found


def strip_unknown_citations(text: str, allowed: set[str]) -> str:
    """The text with every bracket that cites no allowed id removed (D41: never shown).

    A bracket that mixes allowed and unknown ids keeps only the allowed ones.
    """

    def keep(match: re.Match[str]) -> str:
        ids = [i for i in _ID_TOKEN.findall(match.group(1)) if i in allowed]
        return f"[{', '.join(ids)}]" if ids else ""

    stripped = _BRACKET.sub(keep, text)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return re.sub(r" +([.,;:!?])", r"\1", stripped).strip()


def is_refusal(text: str, language: QuestionLanguage) -> bool:
    folded = fold(text).strip()
    return folded == fold(REFUSAL_TEXT[language]) or _REFUSAL_OPENING.match(folded) is not None


def split_claims(text: str, allowed: set[str]) -> list[Claim]:
    """One claim per sentence, each with the ids cited inside that sentence."""
    claims: list[Claim] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        citations = extract_citations(sentence, allowed)
        bare = _BRACKET.sub("", sentence).strip().rstrip(".!?").lstrip("¿¡").strip()
        if bare:
            claims.append(Claim(text=bare, citations=citations))
    return claims


def passage_block(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.passage_id}] {title_line(p)}\n{p.text}" for p in passages)


class Generator:
    def __init__(self, llm: LLM, *, model: str = GENERATOR_MODEL) -> None:
        self._llm = llm
        self._model = model
        self.name = f"generate:{model}"

    def answer(self, question: str, language: QuestionLanguage, passages: list[Passage]) -> Answer:
        """Answer from the given passages only. No passages means an immediate refusal."""
        if not passages:
            return Answer(text=REFUSAL_TEXT[language], citations=[], refused=True, ms=0.0)
        prompt = ANSWER_PROMPT.format(
            language=LANGUAGE_NAMES[language],
            refusal=REFUSAL_TEXT[language],
            passages=passage_block(passages),
            question=question,
        )
        reply = self._llm.complete(self._model, prompt)
        return self._finish(reply, language, {p.passage_id for p in passages})

    def closed_book(
        self, question: str, language: QuestionLanguage, *, allowed: set[str]
    ) -> Answer:
        """Evaluation only, never shown to a user: the model's own memory of the Laws."""
        prompt = CLOSED_BOOK_PROMPT.format(
            language=LANGUAGE_NAMES[language], refusal=REFUSAL_TEXT[language], question=question
        )
        return self._finish(self._llm.complete(self._model, prompt), language, allowed)

    @staticmethod
    def _finish(reply: Completion, language: QuestionLanguage, allowed: set[str]) -> Answer:
        refused = is_refusal(reply.text, language)
        return Answer(
            text=reply.text,
            citations=[] if refused else extract_citations(reply.text, allowed),
            refused=refused,
            ms=reply.ms,
        )
