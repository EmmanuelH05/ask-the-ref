"""A disk cache keyed by content hash, so a rerun with the same inputs costs nothing."""

import hashlib
import json
from pathlib import Path

_SEPARATOR = "\x1f"


def cache_key(*parts: str) -> str:
    return hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


class JsonCache:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def _path(self, key: str) -> Path:
        return self._directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> object | None:
        path = self._path(key)
        if not path.exists():
            return None
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def put(self, key: str, value: object) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so an interrupted run never leaves a half-written entry.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
