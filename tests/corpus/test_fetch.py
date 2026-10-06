import hashlib
import json
from pathlib import Path

import pytest

from atr.corpus import fetch
from atr.corpus.models import Language


def test_page_data_url_english() -> None:
    assert (
        fetch.page_data_url("en", "the-ball")
        == "https://www.theifab.com/page-data/laws/latest/the-ball/page-data.json"
    )


def test_page_data_url_spanish() -> None:
    assert fetch.page_data_url("es", "glossary/referee-terms") == (
        "https://www.theifab.com/page-data/es/laws/latest/glossary/referee-terms/page-data.json"
    )


def test_raw_filename_replaces_slash() -> None:
    assert (
        fetch.raw_filename("glossary/football-terms") == "glossary__football-terms.page-data.json"
    )
    assert fetch.raw_filename("the-ball") == "the-ball.page-data.json"


def test_sha256_hex_known_value() -> None:
    assert fetch.sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()


def _page(next_path: object) -> dict[str, object]:
    return {"result": {"pageContext": {"nextChapterPath": next_path}}}


def test_next_law_slug_plain_and_trailing_slash() -> None:
    assert fetch.next_law_slug(_page("/laws/latest/the-ball")) == "the-ball"
    assert fetch.next_law_slug(_page("/laws/latest/the-ball/")) == "the-ball"


def test_next_law_slug_last_law() -> None:
    assert fetch.next_law_slug(_page(None)) is None
    assert fetch.next_law_slug({"result": {"pageContext": {}}}) is None


@pytest.mark.parametrize(
    "bad",
    [{}, {"result": []}, {"result": {"pageContext": 3}}, _page(7), _page("/")],
)
def test_next_law_slug_bad_structure(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"nextChapterPath|result|pageContext"):
        fetch.next_law_slug(bad)


def _slugs(count: int) -> list[str]:
    return [fetch.FIRST_LAW_SLUG] + [f"law-{i}" for i in range(2, count + 1)]


def _body(slugs: list[str], index: int, language: Language) -> bytes:
    nxt = f"/laws/latest/{slugs[index + 1]}/" if index + 1 < len(slugs) else None
    return json.dumps(_page(nxt) | {"lang": language}).encode()


def _serve(count: int) -> dict[str, bytes]:
    slugs = _slugs(count)
    served: dict[str, bytes] = {}
    for language in fetch.LANGUAGES:
        for i, slug in enumerate(slugs):
            served[fetch.page_data_url(language, slug)] = _body(slugs, i, language)
        for page in fetch.GLOSSARY_PAGES:
            served[fetch.page_data_url(language, page)] = b'{"glossary": true}'
    served[fetch.page_data_url("en", fetch.EDITION_PAGE)] = b'{"edition": true}'
    return served


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    served = _serve(17)

    def fake_get(url: str) -> bytes:
        log.append(url)
        return served[url]

    monkeypatch.setattr(fetch, "_get", fake_get)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    return log


def test_fetch_all_writes_files_and_manifest(tmp_path: Path, calls: list[str]) -> None:
    manifest = fetch.fetch_all(tmp_path)

    assert len(calls) == 41
    assert manifest.law_slugs == _slugs(17)
    assert len(manifest.files) == 41
    keys = [(f.language, f.page) for f in manifest.files]
    assert keys == sorted(keys)
    assert ("es", fetch.EDITION_PAGE) not in keys
    first = manifest.files[0]
    on_disk = (tmp_path / first.path).read_bytes()
    assert first.sha256 == fetch.sha256_hex(on_disk)
    assert first.bytes == len(on_disk)
    assert (tmp_path / "es" / "law-2.page-data.json").exists()
    assert (tmp_path / "en" / "glossary__football-terms.page-data.json").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["law_slugs"] == _slugs(17)
    assert len(saved["files"]) == 41


def test_second_call_downloads_nothing(tmp_path: Path, calls: list[str]) -> None:
    first = fetch.fetch_all(tmp_path)
    calls.clear()

    second = fetch.fetch_all(tmp_path)

    assert calls == []
    assert len(second.files) == 41
    assert second.law_slugs == first.law_slugs


def test_force_redownloads(tmp_path: Path, calls: list[str]) -> None:
    fetch.fetch_all(tmp_path)
    calls.clear()

    fetch.fetch_all(tmp_path, force=True)

    assert len(calls) == 41


def test_sixteen_law_chain_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served = _serve(16)
    monkeypatch.setattr(fetch, "_get", lambda url: served[url])
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="17"):
        fetch.fetch_all(tmp_path)
