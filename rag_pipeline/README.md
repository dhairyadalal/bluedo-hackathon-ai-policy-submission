# AI Policy Copilot — RAG pipeline (v0)

A minimal, real RAG pipeline behind the HTML mockup: page-level chunking of an
OECD AI Policy Toolkit results PDF into ChromaDB, retrieval, and ad hoc
structured generation via GLM (through NVIDIA's OpenAI-compatible API).

## Architecture

```
PDF (OECD toolkit results, one country/pillar per export)
  -> ingest.py: page-level chunks -> ChromaDB (local, persistent)
  -> retrieve.py: query -> top-k relevant chunks
  -> generate.py: chunks + Recommendation schema -> GLM -> validated JSON
  -> chat.py: ties retrieval + generation into a CLI loop
```

The core design choice, per your instruction: **we don't write a parser that
converts the PDF into the data model.** `ingest.py` only does page-level
chunking and light metadata tagging (pillar, country, page number). At query
time, `generate.py` hands the retrieved evidence *and* the JSON schema for
`Recommendation` (from `schema.py`) to GLM and asks it to populate the schema
ad hoc, grounded only in that evidence. `schema.py`'s Pydantic model then
validates the response, so a malformed or hallucinated output fails loudly
instead of silently corrupting data. That's the "data model as prompt"
pattern from your message.

## Embedding model: ChromaDB's default (all-MiniLM-L6-v2, ONNX, local)

For "fastest," the practical answer is: don't add anything on top of what
ChromaDB already ships. Its default embedding function is
`sentence-transformers/all-MiniLM-L6-v2` converted to ONNX and run locally via
`onnxruntime` — 22M parameters, 384 dimensions, no API key, no network call
per chunk, and it's already installed in ChromaDB's dependency tree. For a
project this size (a handful of pages per country/pillar export), that's
faster than any hosted embedding API purely because there's no round trip,
and faster than swapping in a bigger local model like `nomic-embed-text`
(137M params) for negligible retrieval-quality gain at this scale.

If retrieval quality becomes the bottleneck later (not speed), the natural
upgrades in order of effort are: `BAAI/bge-small-en-v1.5` (similar speed,
slightly better MTEB scores), or NVIDIA's own hosted embedding NIM if you want
everything on one vendor's infra (trades local speed for consistency).
`ingest.py` and `retrieve.py` don't pass a custom `embedding_function`, so
they use this default automatically.

## Setup

```bash
cd rag_pipeline
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env   # already has your real key filled in on this machine
```

**On the API key:** the NVIDIA key you pasted in chat is already sitting in
`.env` (gitignored) so the pipeline runs out of the box. Since it was shared
in plaintext in a conversation, treat it as already exposed — if this repo
is ever pushed to GitHub or shared with anyone, rotate the key on NVIDIA's
console first. Never remove `.env` from `.gitignore`.

## Running it

```bash
# 1. Ingest a results PDF (page-level chunking, first run downloads the
#    embedding model, ~80MB, one time)
python ingest.py data/source_pdfs/2026-07_PRT_OECD_AI_Policy_Toolkit_Results-2.pdf

# 2. Sanity-check retrieval on its own
python retrieve.py "compute subsidies for startups and SMEs"

# 3. Full loop: retrieval + GLM populating the Recommendation schema
python chat.py "compute subsidies for startups and SMEs"
# or just `python chat.py` for an interactive loop
```

Re-running `ingest.py` on the same file upserts (no duplicate chunks). Running
it again on a different country/pillar export just adds more chunks to the
same collection, tagged with their own `country`/`pillar` metadata, so the
collection accumulates across exports as you feed it more PDFs.

## What I could and couldn't verify here

I ran this in a sandboxed environment whose outbound network is restricted to
an allowlist that didn't include huggingface.co, ChromaDB's model bucket, or
`integrate.api.nvidia.com` (all returned `403` from the egress proxy, not from
the actual servers). That means I could not do a fully live end-to-end run
here. What I did verify directly, in this sandbox:

- PDF extraction and metadata guessing (`extract_pages`, `guess_pillar`,
  `guess_country`) against the real Portugal PDF you uploaded — correctly
  pulled 4 pages, all tagged `Research, investment, and commercialisation` /
  `Portugal`.
- ChromaDB ingestion and querying end-to-end, using a stub embedding function
  in place of the real one (to bypass the blocked model download) — chunking,
  metadata, `upsert`, and `query` all work mechanically.
- The generation pipeline's parsing and validation path (strip code fences,
  `json.loads`, `Recommendation.model_validate`) against both a realistic
  fake model response (passes) and a deliberately malformed one (correctly
  rejected).

What I could not verify here: the real embedding model download, and an
actual GLM API round trip. Both should work normally the first time you run
this outside this sandbox, since neither host is doing anything unusual, my
environment's proxy is just locked down. If `ingest.py` hangs or errors on
the model download on your machine too, that's a real bug worth flagging back
to me; if it's just slow the first time, that's the one-time model download.

## Next steps

- `schema.py`'s `related_claims` and `evidence[].excerpt` fields are the two
  things GLM has to actually generate, not just extract almost-verbatim — a
  chunk-relevance filter could catch cases where retrieval returns weak
  matches for a niche query before it reaches the model.
- Right now each `chat.py` turn produces one `Recommendation` for one query.
  Wiring this into the HTML mockup would mean the "Draft this section" /
  "Recommended actions" cards call this pipeline live instead of using the
  hand-written mock data.
- Ingesting more countries/pillars just means running `ingest.py` again per
  PDF export; nothing else changes.
