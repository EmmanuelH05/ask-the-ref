"""The demo: a question box, cited passages in both editions, keyword against full (D43).

Standard-library HTTP server, one page of inline HTML and JavaScript, JSON API underneath.
Everything the page shows comes from the same pipeline, threshold and generator the
evaluation used, so the demo cannot look better than the tables.
"""

import argparse
import json
import re
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from atr.answer.generate import REFUSAL_TEXT, Answer, Generator, strip_unknown_citations
from atr.answer.run import context_passages, serving_config
from atr.cache import JsonCache
from atr.corpus.models import Language, Passage
from atr.eval.ablation import Workbench
from atr.eval.refusal import load_choice
from atr.eval.run import CACHE_DIR
from atr.evalset.freeze import CORPUS_MANIFEST, verify_frozen
from atr.evalset.models import QuestionLanguage
from atr.evalset.overlap import content_tokens
from atr.llm import OllamaLLM
from atr.pipeline import PipelineConfig, PipelineResult
from atr.retrieval.docs import load_passages, title_line
from atr.text import raw_tokens, stem

DISCLAIMER = (
    "Not official refereeing guidance. Answers are generated from the IFAB Laws of the Game "
    "text and may be wrong; the cited passage is the authority."
)
TOP_K = 5
MAX_QUESTION_CHARS = 500
MAX_BODY_BYTES = 64 * 1024
RESULT_FILES = {
    "ablation": Path("docs/ablation.txt"),
    "finetune": Path("docs/finetune.txt"),
    "refusal": Path("docs/refusal.txt"),
    "answers": Path("docs/answers.txt"),
    "errors": Path("docs/errors.txt"),
}
# Function words of each language. "a" is left out of the English set: it is also the
# Spanish preposition and sent plain Spanish questions down the code-switched path.
_EN_WORDS = frozenset(
    (
        "the",
        "is",
        "are",
        "was",
        "what",
        "when",
        "can",
        "does",
        "do",
        "how",
        "of",
        "to",
        "in",
        "if",
        "with",
        "for",
        "an",
        "and",
        "it",
        "off",
        "out",
        "on",
        "this",
        "that",
        "be",
    )
)
_ES_WORDS = frozenset(
    (
        "el",
        "la",
        "los",
        "las",
        "es",
        "son",
        "que",
        "de",
        "en",
        "si",
        "un",
        "una",
        "cuando",
        "como",
        "por",
        "para",
        "con",
        "y",
        "se",
        "del",
        "al",
        "qué",
        "cuánto",
        "puede",
        "está",
    )
)
_WORD = re.compile(r"[a-záéíóúñü]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Searcher(Protocol):
    def search(self, text: str, language: QuestionLanguage, k: int) -> PipelineResult: ...


class Answering(Protocol):
    def answer(
        self, question: str, language: QuestionLanguage, passages: list[Passage]
    ) -> Answer: ...


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ShownPassage(_Frozen):
    passage_id: str
    title: str
    text_en: str
    text_es: str | None
    score: float
    why: str
    cited: bool
    marked_sentence: str | None


class AskResponse(_Frozen):
    question: str
    language: QuestionLanguage
    refused: bool
    answer: str
    citations: list[str]
    top_score: float | None
    threshold: float
    track: str
    passages: list[ShownPassage]
    keyword_only: list[ShownPassage]
    latency_ms: dict[str, float]
    disclaimer: str


def detect_language(text: str) -> QuestionLanguage:
    words = set(_WORD.findall(text.lower()))
    english, spanish = bool(words & _EN_WORDS), bool(words & _ES_WORDS)
    if english and spanish:
        return "cs"
    return "es" if spanish else "en"


def matched_words(query: str, passage_text: str) -> list[str]:
    """Query words whose stem occurs in the passage, in query order, without repeats."""
    available = set(content_tokens(passage_text))
    found: list[str] = []
    for word in raw_tokens(query):
        if stem(word) in available and word not in found:
            found.append(word)
    return found


def why_found(matched: list[str]) -> str:
    return (
        f"matched words: {', '.join(matched)}" if matched else "semantic match (no shared keywords)"
    )


def cited_sentence(answer_text: str, passage_text: str) -> str | None:
    """The passage sentence whose content words best match the answer's, if any share one.

    Scored by Dice overlap (shared words twice, over both word counts), so a short sentence
    the answer restates beats a long one that merely shares the topic words. Ties keep the
    earlier sentence.
    """
    wanted = set(content_tokens(answer_text))
    best, best_score = None, 0.0
    for sentence in _SENTENCE_END.split(passage_text):
        words = set(content_tokens(sentence))
        shared = len(wanted & words)
        if shared == 0:
            continue
        score = 2 * shared / (len(wanted) + len(words))
        if score > best_score:
            best, best_score = sentence.strip(), score
    return best


Corpus = Mapping[tuple[Language, str], Passage]


class Service:
    def __init__(
        self,
        full: Searcher,
        keyword: Searcher,
        generator: Answering,
        corpus: Corpus,
        *,
        threshold: float,
        track_label: str,
    ) -> None:
        self._full = full
        self._keyword = keyword
        self._generator = generator
        self._corpus = corpus
        self._threshold = threshold
        self._track = track_label

    def _shown(
        self, hits: list[tuple[str, float]], query: str, answer: Answer | None
    ) -> list[ShownPassage]:
        shown: list[ShownPassage] = []
        for passage_id, score in hits:
            english = self._corpus.get(("en", passage_id))
            if english is None:
                continue
            spanish = self._corpus.get(("es", passage_id))
            cited = answer is not None and passage_id in answer.citations
            shown.append(
                ShownPassage(
                    passage_id=passage_id,
                    title=title_line(english),
                    text_en=english.text,
                    text_es=spanish.text if spanish else None,
                    score=score,
                    why=why_found(matched_words(query, english.text)),
                    cited=cited,
                    marked_sentence=cited_sentence(answer.text, english.text)
                    if cited and answer is not None
                    else None,
                )
            )
        return shown

    def ask(self, question: str) -> AskResponse:
        language = detect_language(question)
        result = self._full.search(question, language, TOP_K)
        top = result.hits[0].score if result.hits else None
        ids = [hit.passage_id for hit in result.hits]
        passages = (
            context_passages(ids, language, self._corpus)
            if top is not None and top >= self._threshold
            else []
        )
        answer = self._generator.answer(question, language, passages)
        keyword = self._keyword.search(question, language, TOP_K)
        search_query = result.queries[0] if result.queries else question
        return AskResponse(
            question=question,
            language=language,
            refused=answer.refused,
            answer=strip_unknown_citations(answer.text, set(answer.citations))
            if not answer.refused
            else REFUSAL_TEXT[language],
            citations=answer.citations,
            top_score=top,
            threshold=self._threshold,
            track=self._track,
            passages=self._shown(
                [(h.passage_id, h.score) for h in result.hits], search_query, answer
            )
            if passages
            else [],
            keyword_only=self._shown(
                [(h.passage_id, h.score) for h in keyword.hits], question, None
            ),
            latency_ms={**result.timings, "generate": answer.ms},
            disclaimer=DISCLAIMER,
        )


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Ask the Ref</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e1;--accent:#2a78d6;--warn:#b4541c;--bg:#fcfcfb}
body{font:16px/1.5 system-ui,sans-serif;color:var(--ink);background:var(--bg);margin:0;
  padding:0 16px}
main{max-width:900px;margin:0 auto;padding:24px 0}
h1{font-size:1.6rem;margin:0 0 4px} .sub{color:var(--muted);margin:0 0 16px}
form{display:flex;gap:8px}
input{flex:1;font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px}
button{font:inherit;padding:10px 14px;border:1px solid var(--accent);background:var(--accent);
  color:#fff;border-radius:8px;cursor:pointer}
button.ghost{background:transparent;color:var(--accent)}
.card{border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0;
  background:#fff}
.answer{font-size:1.1rem} .refused{color:var(--warn)} .meta{color:var(--muted);font-size:.9rem}
.passage{border-top:1px solid var(--line);padding:10px 0} .passage:first-of-type{border-top:0}
.pid{font-weight:600} .why{color:var(--muted);font-size:.9rem} mark{background:#fff3b0}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){.cols{grid-template-columns:1fr}}
.note{color:var(--muted);font-size:.85rem;margin-top:24px} a{color:var(--accent)}
pre{white-space:pre-wrap;font-size:.8rem}
</style></head><body><main>
<h1>Ask the Ref</h1>
<p class="sub">Soccer rules questions in English, Spanish, or both at once.
Every answer cites the Laws of the Game passage it came from.</p>
<form id="f"><input id="q" placeholder="es offside si the ball comes off a defender?"
  autocomplete="off" required>
<button type="submit">Ask</button></form>
<div class="row"><label><input type="checkbox" id="cmp"> compare keyword-only search</label>
<label><input type="checkbox" id="other"> show passages in the other language</label>
<a href="/results">results tables</a></div>
<div id="out"></div>
<p class="note">__DISCLAIMER__</p>
</main>
<script>
const f=document.getElementById('f'),q=document.getElementById('q'),
  out=document.getElementById('out');
const cmp=document.getElementById('cmp'),other=document.getElementById('other');let last=null;
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function passage(p,lang,mark){
  let text=(other.checked&&lang!=='en'&&p.text_es)?p.text_es:p.text_en;
  let html=esc(text);
  if(mark&&p.marked_sentence&&text===p.text_en)
    html=html.replace(esc(p.marked_sentence),'<mark>'+esc(p.marked_sentence)+'</mark>');
  return `<div class="passage"><div><span class="pid">${esc(p.passage_id)}</span> `
    +`${p.cited?'(cited)':''} <span class="meta">score ${p.score.toFixed(3)}</span></div>`
    +`<div class="meta">${esc(p.title)}</div><div class="why">${esc(p.why)}</div>`
    +`<p>${html.replace(/\\n/g,'<br>')}</p></div>`;}
function render(r){
  const ans=r.refused?`<p class="answer refused">${esc(r.answer)}</p>`
    :`<p class="answer">${esc(r.answer)}</p>`;
  const lat=Object.entries(r.latency_ms).map(([k,v])=>k+' '+v.toFixed(0)+' ms').join(', ');
  const top=r.top_score==null?'-':r.top_score.toFixed(3);
  const meta=`<p class="meta">language detected: ${r.language} · track ${r.track} · `
    +`top score ${top} (refuses under ${r.threshold.toFixed(3)}) · ${lat}</p>`;
  const shown=r.passages.length?r.passages.map(p=>passage(p,r.language,true)).join('')
    :'<p class="meta">No passages shown: the top score was under the threshold.</p>';
  const full=`<div class="card"><h3>Full pipeline</h3>${ans}${meta}${shown}</div>`;
  const kwBody=r.keyword_only.map(p=>passage(p,r.language,false)).join('')
    ||'<p class="meta">no keyword matches</p>';
  const kw=`<div class="card"><h3>Keyword-only (BM25)</h3><p class="meta">`
    +`What plain keyword search returns for the same question.</p>${kwBody}</div>`;
  out.innerHTML=cmp.checked?`<div class="cols">${full}${kw}</div>`:full;}
f.addEventListener('submit',async e=>{e.preventDefault();
  out.innerHTML='<p class="meta">thinking…</p>';
  try{
    const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q.value})});
    if(!res.ok){out.innerHTML='<p class="refused">'+esc(await res.text())+'</p>';return;}
    last=await res.json();render(last);
  }catch(err){out.innerHTML='<p class="refused">'+esc('request failed: '+err)+'</p>';}});
cmp.addEventListener('change',()=>last&&render(last));
other.addEventListener('change',()=>last&&render(last));
</script></body></html>
"""

RESULTS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Ask the Ref: results</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px;
  color:#0b0b0b;background:#fcfcfb}
pre{white-space:pre;overflow-x:auto;font-size:.78rem;background:#fff;border:1px solid #e6e5e1;
  border-radius:8px;padding:12px}
a{color:#2a78d6}</style></head><body><h1>Results, straight from the results files</h1>
<p><a href="/">back to the question box</a></p>__SECTIONS__</body></html>
"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_results(results: Mapping[str, str]) -> str:
    sections = (
        "".join(
            f"<h2>{_escape(name)}</h2><pre>{_escape(text)}</pre>" for name, text in results.items()
        )
        or "<p>No results files yet.</p>"
    )
    return RESULTS_PAGE.replace("__SECTIONS__", sections)


def read_results(files: Mapping[str, Path] = RESULT_FILES) -> dict[str, str]:
    return {
        name: path.read_text(encoding="utf-8") for name, path in files.items() if path.is_file()
    }


def route(
    service: Service, method: str, path: str, body: bytes, *, results: Mapping[str, str]
) -> tuple[int, str, bytes]:
    """Status, content type and body for one request. Pure given the service."""
    if method == "GET" and path == "/":
        page = PAGE.replace("__DISCLAIMER__", _escape(DISCLAIMER))
        return 200, "text/html; charset=utf-8", page.encode("utf-8")
    if method == "GET" and path == "/results":
        return 200, "text/html; charset=utf-8", render_results(results).encode("utf-8")
    if method == "GET" and path == "/api/results":
        return 200, "application/json", json.dumps(dict(results)).encode("utf-8")
    if method == "GET" and path == "/healthz":
        return 200, "text/plain", b"ok"
    if method == "POST" and path == "/api/ask":
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return 400, "text/plain", b"body must be JSON"
        question = payload.get("question") if isinstance(payload, dict) else None
        if not isinstance(question, str) or not question.strip():
            return 400, "text/plain", b'expected {"question": "..."}'
        if len(question.strip()) > MAX_QUESTION_CHARS:
            return 413, "text/plain", f"question over {MAX_QUESTION_CHARS} characters".encode()
        response = service.ask(question.strip())
        return 200, "application/json", response.model_dump_json().encode("utf-8")
    return 404, "text/plain", b"not found"


def make_handler(service: Service) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body: bytes = b"") -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                status, content_type = 413, "text/plain"
                out = b"request body too large or malformed"
            else:
                payload = self.rfile.read(length) if length else b""
                try:
                    status, content_type, out = route(
                        service, self.command, self.path, payload, results=read_results()
                    )
                except (RuntimeError, ValueError, KeyError, FileNotFoundError) as error:
                    # A dead Ollama or a rejected translation must reach the page as text,
                    # not as a dropped connection that leaves it on "thinking".
                    print(f"error answering {self.path}: {error}", flush=True)
                    status, content_type, out = 502, "text/plain", str(error).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self) -> None:
            self._reply()

        def do_POST(self) -> None:
            self._reply()

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} {format % args}", flush=True)

    return Handler


class _PipelineSearcher:
    def __init__(self, workbench: Workbench, config: PipelineConfig) -> None:
        self._pipeline = workbench.build(config)
        self._pipeline.search("warm up", "en", TOP_K)

    def search(self, text: str, language: QuestionLanguage, k: int) -> PipelineResult:
        self._pipeline.prime(text, language)
        return self._pipeline.search(text, language, k)


def build_service(track: str, corpus_dir: Path) -> Service:
    _, manifest = verify_frozen(
        Path("evalset/questions.jsonl"), Path("evalset/manifest.json"), CORPUS_MANIFEST
    )
    choice = load_choice(track)
    config = serving_config(choice.config, choice)
    workbench = Workbench([], manifest, "dev", Path("runs/ablation"))
    corpus = {(p.language, p.passage_id): p for p in load_passages(corpus_dir, ("en", "es"))}
    keyword_config = PipelineConfig(label="keyword only", bm25=True)
    print(f"loading track {track}: {config.label} ({config.config_id}) ...", flush=True)
    return Service(
        _PipelineSearcher(workbench, config),
        _PipelineSearcher(workbench, keyword_config),
        Generator(OllamaLLM(cache=JsonCache(CACHE_DIR / "llm"))),
        corpus,
        threshold=choice.threshold,
        track_label=f"{track} ({config.label})",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Ask the Ref demo on localhost.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--track", choices=["T", "M"], default="T")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    args = parser.parse_args(argv)
    service = build_service(args.track, args.corpus_dir)
    # One request at a time: the pipeline, the model and the JSON cache are not thread-safe.
    server = HTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"Ask the Ref at http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
