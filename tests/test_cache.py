from pathlib import Path

from atr.cache import JsonCache, cache_key


def test_cache_key_is_stable_and_separates_parts() -> None:
    assert cache_key("a", "bc") == cache_key("a", "bc")
    assert cache_key("a", "bc") != cache_key("ab", "c")
    assert len(cache_key("x")) == 64


def test_json_cache_round_trips_and_misses(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path / "c")
    key = cache_key("model", "¿texto?")
    assert cache.get(key) is None
    cache.put(key, {"text": "ñ", "n": [1, 2]})
    assert JsonCache(tmp_path / "c").get(key) == {"text": "ñ", "n": [1, 2]}
    assert not list((tmp_path / "c").rglob("*.tmp"))
