# AI Policy Copilot — handoff spec

Context for picking this up in another tool (Codex or otherwise). Two
separate artifacts exist right now and they are **not connected to each
other yet**. That's the biggest structural gap.

## What this project is

A prototype for policymakers researching and drafting AI policy, built
against the OECD AI Policy Toolkit's pilot (a form-based self-assessment
survey that outputs a dense, unnavigable PDF). Portugal is the guinea-pig
country. The goal: replace the form-and-PDF flow with a conversation-driven
tool that shows a country's policy snapshot, compares it against other
countries' precedents, produces a prioritized roadmap, and helps draft policy
language, all grounded in retrieved evidence rather than static templates.

## Artifact 1: HTML mockup (`ai_policy_copilot_mockup.html`)

A single self-contained HTML/CSS/JS file, no build step, no backend. Opens
directly in a browser. This is a **design mockup with hardcoded data**, not
wired to any real pipeline.

Structure: a chat pane on the left with scripted suggestion chips, a canvas
on the right with four always-loaded tabs (no longer gated behind the chat,
that was an earlier design mistake, fixed):

- **Snapshot**: Portugal's stat summary + one comparable action plan (US)
  + a "Recommended actions" list (top 3 open gaps, Now/Next/Later).
- **Landscape**: a country/topic map. Countries are grouped into three
  loosely-geographic "landmasses" (not a real map), filterable by topic,
  clickable for case study detail.
- **Roadmap**: a Now/Next/Later/In-motion kanban rendered from a JS data
  model (`recommendations` array + `documents` registry, both defined inline
  in the `<script>` block). Each card shows Recommendation / Evidence /
  Related claims as three distinct blocks. Clicking a source pill expands a
  **mocked** "extracted passage", explicitly labeled as illustrative, not
  real document text, since only title/country/year/url were available from
  the OECD PDF exports, not full document contents.
- **Draft**: a multi-section policy document editor (Implementation
  mechanisms / Budget / Privacy drafted, Vision / Monitoring left as an
  honest "not started" empty state), with reset/concise/reframe buttons that
  swap between three hand-written text variants per section. Nothing here is
  generated live.

**Everything in this file is either hardcoded or hand-written mock text.**
The `documents` and `recommendations` objects in the JS are a good reference
for the target *shape* of data (see `schema.py` in the pipeline, which is a
Pydantic port of the same idea), but none of the mockup's content comes from
a real pipeline.

## Artifact 2: RAG pipeline (`rag_pipeline/`)

A minimal Python backend, unconnected to the HTML mockup, that does the real
retrieval + generation work the mockup is currently faking.

| File | Purpose | Status |
|---|---|---|
| `schema.py` | Pydantic `Recommendation` / `EvidenceItem` models; `schema_for_prompt()` renders the schema as JSON for prompt injection | Verified (validates good and bad input correctly) |
| `ingest.py` | Loads a PDF, extracts text per page (`pdfplumber`), guesses `pillar`/`country` from page text, upserts into ChromaDB with page-level chunking | Logic verified against the real Portugal PDF (correct page count, correct pillar/country tags); **live embedding download not verified** (network-blocked in my sandbox) |
| `retrieve.py` | `retrieve(query, k)` → top-k chunks with metadata | Mechanics verified via a stub embedding function; **real semantic retrieval quality not verified** |
| `generate.py` | Builds a prompt from retrieved chunks + `schema_for_prompt()`, calls GLM via NVIDIA's OpenAI-compatible API (streaming), strips code fences, parses JSON, validates against `Recommendation` | Parse/validate path verified against a hand-written fake response; **actual GLM API call never executed** (network-blocked) |
| `chat.py` | CLI loop: retrieve → generate → print | Not run live, but it's a thin wrapper around the two verified pieces above |

Data ingested so far: two of Portugal's six OECD toolkit pillars —
"Governance and institutions" (3 pages) and "Research, investment, and
commercialisation" (4 pages) — extracted from uploaded PDF exports, but
**never actually embedded into a persistent ChromaDB collection**, because
model download and the NVIDIA API were both blocked by my sandbox's network
allowlist (403 from the egress proxy on huggingface.co, ChromaDB's model
bucket, and `integrate.api.nvidia.com` — the proxy blocked it, not the real
servers). This is the single most important thing to redo first in an
environment with normal network access.

## What's missing (the real gap list)

**1. The pipeline has never been run for real, only mechanically simulated.**
Nothing has actually been embedded into ChromaDB with a real model, and GLM
has never actually returned a real response. First thing to do anywhere with
normal network access:
```
cd rag_pipeline
pip install -r requirements.txt
python ingest.py data/source_pdfs/2026-07_PRT_OECD_AI_Policy_Toolkit_Results-2.pdf
python chat.py "compute subsidies for startups and SMEs"
```
If this doesn't work cleanly, that's a real bug to fix, not a sandbox
artifact.

**2. The two artifacts aren't connected.** There's no API layer. The mockup's
`recommendations`/`documents` JS objects are hand-written; `rag_pipeline/`
can produce the same shape via `generate.py`, but nothing serves that over
HTTP for the HTML page to call. Needs something like a thin FastAPI/Flask
wrapper exposing `POST /recommend {query}` → `Recommendation` JSON, then the
mockup's hardcoded arrays get replaced with `fetch()` calls.

**3. Only 2 of 6 pillars, 1 of many countries, are ingested.** Governance and
Research/investment for Portugal only. The Landscape tab's other-country data
(US, UK, Spain, Austria, Canada, Malta, France, Israel, OECD/GPAI) is
currently hand-copied from the two source PDFs' "policy examples" listings,
not retrieved from an indexed corpus of those countries' own documents.
Realistically, the Landscape tab requires ingesting either more OECD toolkit
exports for other countries, or the actual primary-source documents it
references (America's AI Action Plan, UK's National AI Strategy, etc.) if
per-country evidence needs to be real rather than title/year/url metadata.

**4. Page-level chunking is coarser than the actual content.** At least one
page (page 1 of the Research pillar PDF) contains two distinct policy
priorities. Right now that's one chunk. Fine for a v0, but worth splitting
further (e.g. on the "POLICY EXAMPLES" delimiter or the priority-name header
pattern) if retrieval starts returning the wrong priority's context for a
query that should only match one of the two on that page.

**5. No reranking, no retrieval-quality eval.** `retrieve.py` is pure vector
similarity, top-k, no cross-encoder rerank, no relevance threshold (it'll
happily return the "closest" chunks even if none of them are actually
relevant). No test suite or eval set exists to measure whether retrieval is
actually surfacing the right evidence for a given query, or whether GLM's
output stays faithful to that evidence rather than drifting.

**6. No handling of a bad/borderline GLM response beyond one hard failure.**
`generate.py` raises on the first `JSONDecodeError` or `ValidationError`.
No retry-with-feedback loop (e.g., re-prompting the model with the validation
error to self-correct), no fallback behavior.

**7. Draft co-authoring isn't wired to the pipeline at all.** The mockup's
"Draft" tab is fully mock text with three static variants. Turning that into
something real means an additional generation call: retrieved evidence for a
section → GLM asked to draft prose (not structured JSON) grounded in that
evidence. Nothing like this exists yet in `rag_pipeline/`.

**8. Security housekeeping.** The real NVIDIA key lives in `rag_pipeline/.env`
so the code runs immediately. It's gitignored, but it was pasted in plaintext
in this conversation, so it should be treated as already exposed. Rotate it
on NVIDIA's console before this repo is pushed anywhere public or shared with
anyone outside this immediate context.

**9. No persistence of generated recommendations.** Every `chat.py`/`generate.py`
call regenerates from scratch. If the same query gets asked repeatedly, or
the app needs to feel instant rather than waiting on an LLM call each time,
some form of caching keyed on (country, pillar, priority) is worth adding.

## Suggested order of operations

1. Get the pipeline actually running end to end outside this sandbox (item 1
   above). This is the real unknown, everything else is scoped work.
2. Wrap `retrieve.py` + `generate.py` in a small HTTP API.
3. Point the mockup's Roadmap/Draft tabs at that API instead of hardcoded
   arrays, for Portugal's two already-ingested pillars first.
4. Ingest the remaining four pillars for Portugal, then decide whether the
   Landscape tab needs real per-country ingestion or can stay
   metadata-only for now.
5. Add the chunking split for multi-priority pages, then reranking/eval, in
   that order, only once the basic loop is proven reliable.
