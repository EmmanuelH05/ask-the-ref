"""Text normalization shared by the overlap measure and keyword search."""

import re
import unicodedata

_NON_TOKEN = re.compile(r"[^a-z0-9]+")
_MIN_STRIPPABLE_LENGTH = 4
_APOSTROPHES = {ord("'"): None, 0x2019: None}


def fold(text: str) -> str:
    # Apostrophes are deleted, not split on: "doesn't" must stay one word and "keeper's"
    # must stem back to "keeper".
    lowered = text.lower().translate(_APOSTROPHES)
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def stem(token: str) -> str:
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


def raw_tokens(text: str) -> list[str]:
    """Folded tokens before stemming. Single letters are debris, never a word."""
    return [token for token in _NON_TOKEN.split(fold(text)) if len(token) > 1]


def search_tokens(text: str) -> list[str]:
    """Tokens for keyword search: stemmed, function words kept (BM25's idf discounts them)."""
    return [stem(token) for token in raw_tokens(text)]
