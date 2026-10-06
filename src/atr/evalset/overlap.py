import re
import unicodedata
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from atr.evalset.models import QuestionGroup

# Stopwords are removed BEFORE the trailing-s rule runs, so this set holds surface forms as
# typed (both "is" and "was", both "los" and "las"), lowercase and without accents.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but nor so if of at by for with about against between into through during
    before after above below to from up down in out on off over under again further then once
    here there when where why how all any both each few more most other some such no not only
    own same than too very can will just should now is are was were be been being am do does did
    doing have has had having i me my we our you your he him his she her it its they them their
    what which who whom this that these those would could may might must shall also as until
    while many much per whose whether either neither because since cannot
    doesnt dont didnt isnt arent wasnt werent cant wont wouldnt couldnt shouldnt hasnt havent
    theyre youre were im ive hes shes thats whats theres lets
    el la los las un una unos unas y e o u pero ni que de del al ante bajo con contra desde en
    entre hacia hasta para por segun sin sobre tras yo tu ella nosotros vosotros ellos
    ellas mi mis tus su sus te se lo le les nos
    os es son era eran fue fueron ser sido siendo somos estar esta estan estaba estoy
    estamos estas este esto estos ese esa eso esos esas aquel haber ha han
    hay habia hemos has tener tiene tienen tengo puede pueden poder podria debe deben deber como
    cuando donde cual cuales quien quienes cuanto cuanta cuantos cuantas porque si mas muy
    tambien ya otro otra otros otras todo toda todos todas algun alguna ningun cada
    antes despues durante
    """.split()  # noqa: SIM905 (a word list reads better than a quoted list)
)

_NON_TOKEN = re.compile(r"[^a-z0-9]+")
_MIN_STRIPPABLE_LENGTH = 4


class GroupOverlap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str
    en_vs_en: float | None
    es_vs_es: float | None
    cs_vs_en: float | None
    cs_vs_es: float | None
    es_vs_en: float | None
    en_vs_es: float | None


def _fold(text: str) -> str:
    # Apostrophes are deleted, not split on: "doesn't" must stay one (stop)word and
    # "keeper's" must stem back to "keeper".
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("'", "").replace("\u2019", ""))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _stem(token: str) -> str:
    """Drop a plural "s", then a final "e", so singular and plural meet in both languages.

    "offences" and "offence" both become "offenc"; "jugadores" and "jugador" both become
    "jugador". A double "s" is kept ("pass").
    """
    long_enough = len(token) >= _MIN_STRIPPABLE_LENGTH
    if long_enough and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) >= _MIN_STRIPPABLE_LENGTH and token.endswith("e"):
        token = token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    # Single letters are debris (the "s" of a possessive), never a content word.
    tokens = [token for token in _NON_TOKEN.split(_fold(text)) if len(token) > 1]
    return [_stem(token) for token in tokens if token not in STOPWORDS]


def overlap(question: str, passages: list[str]) -> float:
    wanted = set(content_tokens(question))
    if not wanted or not passages:
        return 0.0
    available = set(content_tokens(" ".join(passages)))
    return len(wanted & available) / len(wanted)


def _texts(gold: list[str], edition: Mapping[str, str]) -> list[str] | None:
    if not gold or any(passage_id not in edition for passage_id in gold):
        return None
    return [edition[passage_id] for passage_id in gold]


def _score(question: str, passages: list[str] | None) -> float | None:
    return None if passages is None else overlap(question, passages)


def group_overlap(
    group: QuestionGroup, en: Mapping[str, str], es: Mapping[str, str]
) -> GroupOverlap:
    en_texts = _texts(group.gold, en)
    es_texts = _texts(group.gold, es)
    return GroupOverlap(
        group_id=group.group_id,
        en_vs_en=_score(group.en, en_texts),
        es_vs_es=_score(group.es, es_texts),
        cs_vs_en=_score(group.cs, en_texts),
        cs_vs_es=_score(group.cs, es_texts),
        es_vs_en=_score(group.es, en_texts),
        en_vs_es=_score(group.en, es_texts),
    )
