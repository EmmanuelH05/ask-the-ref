from atr.llm import Completion
from atr.retrieval.decompose import Decomposer
from atr.retrieval.rewrite import RulebookRewriter


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, model: str, prompt: str, *, think: str | None = None) -> Completion:
        self.prompts.append(prompt)
        return Completion(self.reply, 40.0)


def test_rewriter_appends_the_rewrite_to_the_original_question() -> None:
    llm = FakeLLM("May a zorb keeper hold the orb beyond the permitted time?")
    result = RulebookRewriter(llm).expand("Can the guy in the net keep it forever?")
    assert result.text == (
        "Can the guy in the net keep it forever?\n"
        "May a zorb keeper hold the orb beyond the permitted time?"
    )
    assert result.ms == 40.0
    assert "Can the guy in the net keep it forever?" in llm.prompts[0]


def test_rewriter_falls_back_to_the_original_when_the_reply_is_unusable() -> None:
    question = "Can the guy in the net keep it forever?"
    for reply in ("ok", "word " * 200):
        result = RulebookRewriter(FakeLLM(reply)).expand(question)
        assert result.text == question and result.ms == 40.0


def test_decomposer_returns_sub_questions_from_a_json_array() -> None:
    llm = FakeLLM('["Who restarts after a wobble?", "Where is the restart taken?"]')
    subs, ms = Decomposer(llm).split("After a wobble, who restarts and where?")
    assert subs == ["Who restarts after a wobble?", "Where is the restart taken?"]
    assert ms == 40.0


def test_decomposer_reads_an_array_wrapped_in_other_text() -> None:
    llm = FakeLLM('Here you go:\n```json\n["One thing?", "Another thing?"]\n```')
    assert Decomposer(llm).split("Two things?")[0] == ["One thing?", "Another thing?"]


def test_decomposer_gives_nothing_for_a_simple_question_or_a_bad_reply() -> None:
    for reply in ('["Only one?"]', "not json", "[1, 2]", '["a?", "b?", "c?", "d?"]', "[]"):
        assert Decomposer(FakeLLM(reply)).split("Simple?")[0] == []
