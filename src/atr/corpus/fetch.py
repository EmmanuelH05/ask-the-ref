import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from atr.corpus.models import Language

BASE_URL = "https://www.theifab.com/page-data"
FIRST_LAW_SLUG = "the-field-of-play"
GLOSSARY_PAGES = ("glossary/football-terms", "glossary/referee-terms", "glossary/football-bodies")
EDITION_PAGE = "notes-on-the-laws"
LANGUAGES: tuple[Language, ...] = ("en", "es")
REQUEST_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 30
LAW_COUNT = 17
USER_AGENT = "atr-ask-the-ref/0.1 (student research project; local corpus build; polite 0.5s delay)"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RawFile(_Frozen):
    language: Language
    page: str
    url: str
    path: str
    sha256: str
    bytes: int


class RawManifest(_Frozen):
    fetched_at: str
    law_slugs: list[str]
    files: list[RawFile]


def page_data_url(language: Language, page: str) -> str:
    prefix = "/es" if language == "es" else ""
    return f"{BASE_URL}{prefix}/laws/latest/{page}/page-data.json"


def raw_filename(page: str) -> str:
    return page.replace("/", "__") + ".page-data.json"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def next_law_slug(page_data: dict[str, object]) -> str | None:
    result = page_data.get("result")
    if not isinstance(result, dict):
        raise ValueError("page data has no 'result' object")
    context = result.get("pageContext")
    if not isinstance(context, dict):
        raise ValueError("page data has no 'result.pageContext' object")
    next_path = context.get("nextChapterPath")
    if next_path is None:
        return None
    if not isinstance(next_path, str):
        raise ValueError(f"nextChapterPath is not a string: {next_path!r}")
    slug = next_path.rstrip("/").rsplit("/", 1)[-1]
    if not slug:
        raise ValueError(f"nextChapterPath has no slug: {next_path!r}")
    return slug


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body: bytes = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"request failed for {url}: {error}") from error
    return body


class _Fetcher:
    def __init__(self, raw_dir: Path, force: bool) -> None:
        self.raw_dir = raw_dir
        self.force = force
        self.files: list[RawFile] = []
        self.statuses: list[str] = []
        self._requested = False

    def load(self, language: Language, page: str) -> bytes:
        relative = f"{language}/{raw_filename(page)}"
        target = self.raw_dir / relative
        url = page_data_url(language, page)
        if target.exists() and not self.force:
            body = target.read_bytes()
            status = "cached"
        else:
            if self._requested:
                time.sleep(REQUEST_DELAY_SECONDS)
            self._requested = True
            body = _get(url)
            # Validate before caching: a cached error page would fail every later run.
            _parse_json(body, url)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            status = "fetched"
        self.files.append(
            RawFile(
                language=language,
                page=page,
                url=url,
                path=relative,
                sha256=sha256_hex(body),
                bytes=len(body),
            )
        )
        self.statuses.append(status)
        return body


def _parse_json(body: bytes, url: str) -> dict[str, object]:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON from {url}: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON from {url} is not an object")
    return parsed


def _walk_law_chain(fetcher: _Fetcher) -> list[str]:
    slugs: list[str] = []
    slug: str | None = FIRST_LAW_SLUG
    while slug is not None:
        if slug in slugs or len(slugs) > LAW_COUNT:
            raise RuntimeError(f"law chain does not end after {LAW_COUNT} laws: {slugs}")
        slugs.append(slug)
        body = fetcher.load("en", slug)
        slug = next_law_slug(_parse_json(body, page_data_url("en", slugs[-1])))
    if len(slugs) != LAW_COUNT:
        raise RuntimeError(f"expected {LAW_COUNT} law pages, found {len(slugs)}: {slugs}")
    return slugs


def _fetch_all_with_statuses(
    raw_dir: Path, force: bool
) -> tuple[RawManifest, list[tuple[RawFile, str]]]:
    fetcher = _Fetcher(raw_dir, force)
    law_slugs = _walk_law_chain(fetcher)
    for slug in law_slugs:
        fetcher.load("es", slug)
    for language in LANGUAGES:
        for page in GLOSSARY_PAGES:
            fetcher.load(language, page)
    fetcher.load("en", EDITION_PAGE)
    paired = sorted(zip(fetcher.files, fetcher.statuses, strict=True), key=_sort_key)
    manifest = RawManifest(
        fetched_at=datetime.now(UTC).isoformat(),
        law_slugs=law_slugs,
        files=[raw_file for raw_file, _ in paired],
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest, paired


def _sort_key(pair: tuple[RawFile, str]) -> tuple[str, str]:
    return (pair[0].language, pair[0].page)


def fetch_all(raw_dir: Path, *, force: bool = False) -> RawManifest:
    manifest, _ = _fetch_all_with_statuses(raw_dir, force)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download IFAB Laws page-data JSON.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest, paired = _fetch_all_with_statuses(args.raw_dir, args.force)
    for raw_file, status in paired:
        print(f"{raw_file.language} {raw_file.page} {raw_file.bytes} {status}")
    print(f"{len(manifest.files)} files")


if __name__ == "__main__":
    main()
