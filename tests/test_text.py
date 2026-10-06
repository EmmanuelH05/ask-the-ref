from atr.text import fold, search_tokens, stem


def test_fold_lowercases_strips_accents_and_deletes_apostrophes() -> None:
    assert fold("Árbitro's Ñandú") == "arbitros nandu"


def test_stem_meets_singular_and_plural() -> None:
    assert stem("jugadores") == stem("jugador") == "jugador"
    assert stem("pass") == "pass"
    assert stem("bus") == "bus"


def test_search_tokens_keeps_function_words_but_drops_single_letters() -> None:
    assert search_tokens("¿Is the zorb's gate a goal?") == ["is", "the", "zorb", "gat", "goal"]
