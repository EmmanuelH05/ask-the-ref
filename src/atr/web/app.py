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


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Ask the Ref</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--ink:#14211a;--muted:#5d6b62;--line:#dfe6e1;--accent:#1b7f4c;--accent-ink:#0f5c36;
  --warn:#9a4b12;--bg:#f6f8f6;--card:#ffffff;--mark:#fff1a8;--chip:#edf3ef;--shadow:0 1px 2px
  rgba(20,33,26,.06)}
@media(prefers-color-scheme:dark){:root{--ink:#e8ede9;--muted:#9fb0a6;--line:#2a3630;
  --accent:#3fb97c;--accent-ink:#8ee0b4;--warn:#f0a35e;--bg:#0f1512;--card:#172019;
  --mark:#5a4a12;--chip:#1f2b24;--shadow:none}}
*{box-sizing:border-box}
body{margin:0;font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  color:var(--ink);background:var(--bg)}
main{max-width:760px;margin:0 auto;padding:44px 16px 64px}
h1{font-size:2rem;letter-spacing:-.02em;margin:0}
.sub{color:var(--muted);margin:6px 0 24px;font-size:1.05rem}
form{display:flex;gap:8px}
input{flex:1;min-width:0;font:inherit;font-size:1.05rem;padding:14px 16px;border:1px solid
  var(--line);border-radius:12px;background:var(--card);color:var(--ink);box-shadow:var(--shadow)}
input:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
button{font:inherit;font-weight:600;padding:14px 20px;border:0;border-radius:12px;
  background:var(--accent);color:#fff;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0}
.chip{font:inherit;font-size:.9rem;padding:6px 12px;border-radius:999px;border:1px solid
  var(--line);background:var(--chip);color:var(--ink);cursor:pointer;text-align:left}
.chip:hover{border-color:var(--accent)}
.chip small{color:var(--muted);margin-left:6px;font-size:.78rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px 24px;
  margin-top:28px;box-shadow:var(--shadow)}
.answer{font-size:1.35rem;line-height:1.45;margin:0 0 14px;letter-spacing:-.01em}
.answer.soft{font-size:1.15rem;color:var(--warn)}
.hint{margin:-6px 0 14px;color:var(--muted)}
.cite{display:inline-block;font-size:.72rem;font-weight:700;vertical-align:.18em;
  padding:1px 7px;border-radius:999px;background:var(--chip);color:var(--accent-ink);
  text-decoration:none;margin-left:2px;font-variant-numeric:tabular-nums}
.cite:hover{background:var(--accent);color:#fff}
.meta{display:flex;flex-wrap:wrap;gap:8px;color:var(--muted);font-size:.85rem}
.pill{padding:3px 10px;border-radius:999px;background:var(--chip)}
.loading{display:flex;align-items:center;gap:12px;color:var(--muted)}
.spin{width:18px;height:18px;border:2px solid var(--line);border-top-color:var(--accent);
  border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
section{margin-top:28px}
h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
  margin:0 0 10px;font-weight:600}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
label.toggle{font-size:.85rem;color:var(--muted);cursor:pointer}
details.p{border:1px solid var(--line);border-radius:12px;background:var(--card);margin:8px 0}
details.p summary{list-style:none;cursor:pointer;padding:12px 16px;display:flex;gap:10px;
  align-items:baseline;flex-wrap:wrap}
summary::-webkit-details-marker{display:none}
details.p summary::before{content:"\203A";color:var(--muted);display:inline-block;
  transition:transform .15s}
details.p[open] summary::before{transform:rotate(90deg)}
.pid{font-weight:700;font-variant-numeric:tabular-nums}
.ptitle{color:var(--muted);font-size:.9rem}
.tag{margin-left:auto;font-size:.72rem;font-weight:700;color:var(--accent-ink);
  text-transform:uppercase;letter-spacing:.06em}
.body{padding:0 16px 14px;white-space:pre-wrap;line-height:1.5}
.why{color:var(--muted);font-size:.82rem;padding:0 16px 14px}
mark{background:var(--mark);color:inherit;border-radius:3px;padding:0 2px}
details.cmp{margin-top:18px;color:var(--muted);font-size:.9rem}
details.cmp summary{cursor:pointer}
.cmp li{margin:4px 0}
.err{color:var(--warn)}
footer{margin-top:40px;color:var(--muted);font-size:.82rem;line-height:1.5}
footer a{color:var(--accent-ink)}
</style></head><body><main>
<h1>Ask the Ref</h1>
<p class="sub">Soccer rules questions in English, Spanish, or both at once.
Every answer cites the passage of the Laws of the Game it came from.</p>
<form id="f"><input id="q" placeholder="Ask about any rule…" autocomplete="off" required>
<button id="go" type="submit">Ask</button></form>
<div class="chips" id="chips"></div>
<div id="out"></div>
<footer>__DISCLAIMER__ · <a href="/results">How well this works, measured</a></footer>
</main>
<script>
const EXAMPLES=[
  ["Can a goal be scored directly from a goal kick?","English"],
  ["¿Cuánto dura el descanso del medio tiempo?","Español"],
  ["es offside si the ball comes off a defender?","Spanglish"]];
const LANG={en:"English",es:"Español",cs:"Spanglish"};
const f=document.getElementById('f'),q=document.getElementById('q'),
  out=document.getElementById('out'),go=document.getElementById('go');
let last=null,spanish=false;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const chips=document.getElementById('chips');
for(const [text,lang] of EXAMPLES){
  const b=document.createElement('button');b.type='button';b.className='chip';
  b.innerHTML=esc(text)+'<small>'+esc(lang)+'</small>';
  b.addEventListener('click',()=>{q.value=text;f.requestSubmit();});chips.appendChild(b);}
function cites(text,ids){
  return esc(text).replace(/\s*\[([^\]]+)\]/g,(m,inner)=>inner.split(/,\s*/)
    .map(id=>ids.includes(id)?'<a class="cite" href="#p-'+esc(id)+'">'+esc(id)+'</a>':'')
    .join(''));}
function passage(p,i,anyCited){
  const text=(spanish&&p.text_es)?p.text_es:p.text_en;
  let html=esc(text);
  if(p.marked_sentence&&text===p.text_en)
    html=html.replace(esc(p.marked_sentence),'<mark>'+esc(p.marked_sentence)+'</mark>');
  const open=p.cited||(!anyCited&&i===0);
  return '<details class="p" id="p-'+esc(p.passage_id)+'"'+(open?' open':'')+'>'
    +'<summary><span class="pid">'+esc(p.passage_id)+'</span>'
    +'<span class="ptitle">'+esc(p.title)+'</span>'
    +(p.cited?'<span class="tag">cited</span>':'')+'</summary>'
    +'<div class="body">'+html+'</div>'
    +'<div class="why">'+esc(p.why)+' · score '+p.score.toFixed(2)+'</div></details>';}
function render(r){
  const ids=r.passages.filter(p=>p.cited).map(p=>p.passage_id);
  let head;
  if(!r.refused)head='<p class="answer">'+cites(r.answer,ids)+'</p>';
  else if(r.passages.length)head='<p class="answer soft">I couldn&rsquo;t give a confident '
    +'answer.</p><p class="hint">The passages below are what the search found; the closest '
    +'is first.</p>';
  else head='<p class="answer soft">The Laws I have don&rsquo;t seem to cover that.</p>'
    +'<p class="hint">Nothing in the rulebook scored above the threshold for this question.</p>';
  const ms=r.latency_ms,total=Object.values(ms).reduce((a,b)=>a+b,0);
  const meta='<div class="meta"><span class="pill">'+esc(LANG[r.language]||r.language)+'</span>'
    +'<span class="pill">'+(total/1000).toFixed(1)+' s</span>'
    +(r.top_score!=null?'<span class="pill">match '+r.top_score.toFixed(2)
      +' · needs '+r.threshold.toFixed(2)+'</span>':'')+'</div>';
  const list=r.passages.map((p,i)=>passage(p,i,ids.length>0)).join('');
  const toggle=r.passages.some(p=>p.text_es)
    ?'<label class="toggle"><input type="checkbox" id="es"'+(spanish?' checked':'')
      +'> show in Spanish</label>':'';
  const kw=r.keyword_only.length
    ?'<details class="cmp"><summary>What plain keyword search would have returned</summary>'
      +'<ul class="cmp">'+r.keyword_only.map(p=>'<li><b>'+esc(p.passage_id)+'</b> '
      +esc(p.title)+'</li>').join('')+'</ul></details>':'';
  out.innerHTML='<div class="card">'+head+meta+'</div>'
    +(list?'<section><div class="row"><h2>From the Laws of the Game</h2>'+toggle+'</div>'
      +list+'</section>':'')+kw;
  const es=document.getElementById('es');
  if(es)es.addEventListener('change',()=>{spanish=es.checked;render(last);});}
f.addEventListener('submit',async e=>{e.preventDefault();
  go.disabled=true;
  const text=q.value.trim(),lang=/[áéíóúñ¿¡]|\b(el|la|que|de|es|un|una)\b/i.test(text);
  out.innerHTML='<div class="card"><div class="loading"><div class="spin"></div>'
    +(lang?'Translating, searching the Laws, writing an answer…':'Searching the Laws, writing'
    +' an answer…')+'</div></div>';
  try{
    const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:text})});
    if(!res.ok){out.innerHTML='<div class="card"><p class="err">'+esc(await res.text())
      +'</p></div>';return;}
    last=await res.json();render(last);
  }catch(err){out.innerHTML='<div class="card"><p class="err">'+esc('Request failed: '+err)
    +'</p></div>';}
  finally{go.disabled=false;}});
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
