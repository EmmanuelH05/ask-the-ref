import pytest

from atr.answer.generate import (
    REFUSAL_TEXT,
    Answer,
    Claim,
    Generator,
    extract_citations,
    is_refusal,
    passage_block,
    split_claims,
)
from atr.corpus.models import Passage, count_words
from atr.llm import Completion


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def complete(self, model: str, prompt: str, *, think: str | None = None) -> Completion:
        self.prompts.append((model, prompt))
        return Completion(self.reply, 250.0)


def passage(passage_id: str, text: str, language: str = "en") -> Passage:
    return Passage.model_validate(
        {
            "passage_id": passage_id,
            "language": language,
            "law_number": 12,
            "law_title": "Fouls and Misconduct",
            "section_index": 1,
            "section_title": "Direct free kick",
            "subheading": None,
            "source_slug": "direct-free-kick",
            "text": text,
            "word_count": count_words(text),
        }
    )


def test_extract_citations_keeps_order_drops_duplicates_and_unknown_ids() -> None:
    text = "A foul [L12.1]. Also [L12.1] and [G.football-terms.offside] but not [L99.9]."
    allowed = {"L12.1", "G.football-terms.offside"}
    assert extract_citations(text, allowed) == ["L12.1", "G.football-terms.offside"]


def test_extract_citations_accepts_several_ids_in_one_bracket() -> None:
    assert extract_citations("Yes [L12.1, L12.3.2].", {"L12.1", "L12.3.2"}) == ["L12.1", "L12.3.2"]


def test_is_refusal_matches_the_fixed_text_in_each_language_and_its_opening() -> None:
    for language, text in REFUSAL_TEXT.items():
        assert is_refusal(text, language)
    assert is_refusal("I don't know, the passages do not cover it.", "en")
    assert is_refusal("No lo sé. Nada de eso aparece.", "es")
    assert is_refusal("No lo se, the Laws don't say.", "cs")
    assert not is_refusal("The ball must be round [L2.1].", "en")


def test_split_claims_gives_each_sentence_its_citations() -> None:
    text = "The ball is round [L2.1]. It weighs 410 to 450 g [L2.1, L2.2]. Bare claim here."
    assert split_claims(text, {"L2.1", "L2.2"}) == [
        Claim(text="The ball is round", citations=["L2.1"]),
        Claim(text="It weighs 410 to 450 g", citations=["L2.1", "L2.2"]),
        Claim(text="Bare claim here", citations=[]),
    ]


def test_split_claims_ignores_blank_fragments_and_refusals() -> None:
    assert split_claims("", set()) == []
    assert split_claims("   ", set()) == []
    assert split_claims("[L2.1]", {"L2.1"}) == []


def test_passage_block_labels_each_passage_with_its_id_and_title() -> None:
    block = passage_block([passage("L12.1", "A direct free kick is awarded.")])
    assert (
        block
        == "[L12.1] Law 12: Fouls and Misconduct > Direct free kick\nA direct free kick is awarded."
    )


def test_generator_answers_from_passages_and_keeps_only_cited_passages() -> None:
    llm = ScriptedLLM("A direct free kick is awarded [L12.1]. Unrelated [L7.1].")
    generator = Generator(llm, model="fake-model")
    answer = generator.answer("When is a free kick given?", "en", [passage("L12.1", "Text.")])
    assert answer == Answer(
        text="A direct free kick is awarded [L12.1]. Unrelated [L7.1].",
        citations=["L12.1"],
        refused=False,
        ms=250.0,
    )
    model, prompt = llm.prompts[0]
    assert model == "fake-model"
    assert "[L12.1] Law 12" in prompt and "When is a free kick given?" in prompt
    assert REFUSAL_TEXT["en"] in prompt


def test_generator_marks_a_model_refusal_as_refused() -> None:
    llm = ScriptedLLM(REFUSAL_TEXT["es"])
    answer = Generator(llm).answer("¿Cuánto cuesta una entrada?", "es", [passage("L1.1", "x")])
    assert answer.refused and answer.citations == []


def test_generator_refuses_without_a_call_when_there_are_no_passages() -> None:
    llm = ScriptedLLM("should not be used")
    answer = Generator(llm).answer("Anything?", "cs", [])
    assert answer == Answer(text=REFUSAL_TEXT["cs"], citations=[], refused=True, ms=0.0)
    assert llm.prompts == []


def test_closed_book_sends_no_passages_and_accepts_any_corpus_id() -> None:
    llm = ScriptedLLM("From memory, Law 12 [L12.1].")
    answer = Generator(llm).closed_book("Q?", "en", allowed={"L12.1"})
    assert answer.citations == ["L12.1"] and not answer.refused
    assert "Passages:" not in llm.prompts[0][1]


def test_answer_model_is_frozen() -> None:
    answer = Answer(text="x", citations=[], refused=False, ms=1.0)
    with pytest.raises(ValueError):
        answer.text = "y"  # type: ignore[misc]


def test_is_refusal_does_not_fire_on_answers_that_merely_open_with_no_se_or_i_do_not_know() -> None:
    assert not is_refusal("No se permite tocar el balón con la mano [L12.1].", "es")
    assert not is_refusal("No se puede marcar gol directamente de un saque de banda [L15.1].", "cs")
    assert not is_refusal("No second caution is given [L12.3].", "en")
    assert not is_refusal("I do not know of any rule that forbids it [L4.1].", "en")
    assert is_refusal("No lo sé.", "es")
    assert is_refusal("I do not know.", "en")


def test_strip_unknown_citations_removes_brackets_the_model_was_not_given() -> None:
    from atr.answer.generate import strip_unknown_citations

    text = "The ball is round [L2.1]. Eleven players [L99.9]."
    assert strip_unknown_citations(text, {"L2.1"}) == "The ball is round [L2.1]. Eleven players."
    assert strip_unknown_citations("Both [L2.1, L99.9].", {"L2.1"}) == "Both [L2.1]."
    assert strip_unknown_citations("Plain text.", set()) == "Plain text."


def test_split_claims_drops_the_spanish_opening_marks() -> None:
    claims = split_claims("¿Es falta? Sí, es falta [L12.1].", {"L12.1"})
    assert [c.text for c in claims] == ["Es falta", "Sí, es falta"]
