# atr (Ask the Ref)

Python 3.13 + uv + pytest. Python is approved for this repo only. Never npm, bun or TypeScript here.
Package: `src/atr`. Task prompt: `ASK-THE-REF-PROMPT.md` (untracked, leave it in place).

Verify with: `uv run pytest -q && uv run mypy --strict src && uv run ruff check .`

Rules that get broken otherwise:

- No number goes into the README, a doc, or a commit message unless a command in this repo
  produced it in the current session. Put the command next to the number.
- The test split is frozen and hash-committed before any retrieval tuning. Tune on dev only.
  Looking at test results to make a choice means telling mannie and re-splitting.
- `data/` is gitignored. Never commit the IFAB Laws text, in full or in bulk.
- Local models only (sentence-transformers, Ollama). No paid AI API, ever (mannie, 2026-10-18).
  Ask before any `brew install`.
- Tests first for every pure function.
- Pydantic models are frozen with `extra="forbid"`.
- Commits: conventional format, `git commit --only <paths>`, as 05mannyhernandez@gmail.com.
  No remote, push or deploy without asking.
- Each phase ends with its check run, the raw output shown, a commit, and a stop for mannie's OK.
