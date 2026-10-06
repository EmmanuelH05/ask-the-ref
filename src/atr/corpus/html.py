import html as html_lib
import re
from html.parser import HTMLParser

from pydantic import BaseModel, ConfigDict

_DELETED_TAGS = frozenset({"deleted", "deletedexplanation"})
_IMAGE_TAGS = frozenset({"responsiveimage", "thefieldinteractivepicture", "img"})
_BLOCK_TAGS = frozenset({"p", "h3"})
_SPACE_RUN = re.compile(r"[ \t\r\n]+")
_TAG = re.compile(r"<[^>]*>")
_H3 = re.compile(r"<h3\b[^>]*>(.*?)</h3\s*>", re.IGNORECASE | re.DOTALL)
_INDENT = "  "


class Cleaned(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    deleted_fragments: int
    images: int
    explanation_attrs: int


class _Cleaner(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.deleted_fragments = 0
        self.images = 0
        self.explanation_attrs = 0
        self._deleted_depth = 0
        self._list_depth = 0
        self._prefix = ""
        self._buffer = ""
        # One entry per open <li>: the indent for its lines after the bulleted one.
        self._continuations: list[str] = []

    def _flush_line(self) -> None:
        text = _SPACE_RUN.sub(" ", self._buffer).strip()
        # The CMS wraps list item text in <p>, so a flush with nothing buffered must not
        # spend the bullet that the <li> just set.
        if text:
            self.lines.append(self._prefix + text)
            self._prefix = self._continuations[-1] if self._continuations else ""
        self._buffer = ""

    def _paragraph_break(self) -> None:
        self._flush_line()
        if self._list_depth:
            return
        if self.lines and self.lines[-1] != "":
            self.lines.append("")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.explanation_attrs += sum(1 for name, _ in attrs if name == "data-explanation")
        if tag in _IMAGE_TAGS:
            self.images += 1
        if tag in _DELETED_TAGS:
            if self._deleted_depth == 0:
                self.deleted_fragments += 1
            self._deleted_depth += 1
            return
        if self._deleted_depth:
            return
        if tag in _BLOCK_TAGS:
            self._paragraph_break()
        elif tag == "br":
            self._flush_line()
        elif tag == "ul":
            self._open_list()
        elif tag == "li":
            self._flush_line()
            indent = _INDENT * max(self._list_depth - 1, 0)
            self._prefix = indent + "- "
            self._continuations.append(indent + _INDENT)

    def _open_list(self) -> None:
        self._paragraph_break()
        self._list_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _DELETED_TAGS:
            self._deleted_depth = max(self._deleted_depth - 1, 0)
            return
        if self._deleted_depth:
            return
        if tag in _BLOCK_TAGS:
            self._paragraph_break()
        elif tag == "li":
            self._flush_line()
            if self._continuations:
                self._continuations.pop()
            self._prefix = self._continuations[-1] if self._continuations else ""
        elif tag == "ul":
            self._flush_line()
            self._list_depth = max(self._list_depth - 1, 0)
            self._paragraph_break()

    def handle_data(self, data: str) -> None:
        if not self._deleted_depth:
            self._buffer += data.replace("\xa0", " ")

    def result(self) -> Cleaned:
        self._flush_line()
        while self.lines and self.lines[-1] == "":
            self.lines.pop()
        return Cleaned(
            text="\n".join(self.lines),
            deleted_fragments=self.deleted_fragments,
            images=self.images,
            explanation_attrs=self.explanation_attrs,
        )


def clean_html(html: str) -> Cleaned:
    cleaner = _Cleaner()
    cleaner.feed(html)
    cleaner.close()
    return cleaner.result()


def _has_text(fragment: str) -> bool:
    return bool(html_lib.unescape(_TAG.sub("", fragment)).strip())


def split_subsections(html: str) -> list[tuple[str | None, str]]:
    headings = list(_H3.finditer(html))
    if not headings:
        return [(None, html)]
    blocks: list[tuple[str | None, str]] = []
    lead = html[: headings[0].start()]
    if _has_text(lead):
        blocks.append((None, lead))
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(html)
        title = clean_html(match.group(1)).text.strip()
        blocks.append((title, html[match.end() : end]))
    return blocks
