# AI Policy Copilot

Conversation-driven prototype for policymakers researching and drafting national AI strategy. Built against the [OECD AI Policy Toolkit](https://oecd.ai/en/policy-areas/ai-policy-toolkit) pilot, with Portugal as the reference case.

Instead of the toolkit's form → PDF flow, this tool lets you:
- See a country's policy snapshot compared against relevant international precedents
- Explore the global landscape of comparable AI strategies
- Get a prioritised, evidence-grounded roadmap of open gaps
- Draft policy document sections, grounded in retrieved case studies

The live AI features (free-text queries, "Analyze live from the pipeline" chips) are backed by a real RAG pipeline: ChromaDB for retrieval + GLM-5.2 (via NVIDIA's API) for generation.

---

## Running locally

```bash
cd rag_pipeline
pip install -r requirements.txt

# Copy and fill in your NVIDIA API key
cp .env.example .env

# Start the server (serves the app at http://localhost:8000)
python api.py
```

Open `http://localhost:8000` in your browser. The HTML mockup can also be opened directly as a file — it will automatically use `http://localhost:8000` as the API base when opened from `file://`.

### Re-ingesting the PDF

The ChromaDB is pre-seeded with Portugal's *Research, investment, and commercialisation* pillar (4 pages). To add more pillar PDFs:

```bash
cd rag_pipeline
python ingest.py data/source_pdfs/<your-pdf>.pdf
```

---

## Deploying to Render (free)

A `render.yaml` is included. Steps:

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → connect the repo
3. Render detects `render.yaml` automatically — confirm the settings
4. Add your `NVIDIA_API_KEY` under Environment Variables in the Render dashboard
5. Deploy — the app will be live at `https://ai-policy-copilot.onrender.com` (or your chosen name)

> **Note:** Render's free tier spins down after 15 minutes of inactivity. The first request after a cold start takes ~30 seconds. Open the URL yourself before a demo.

---

## What's in this repo

```
ai_policy_copilot_mockup.html   Self-contained frontend (no build step)
render.yaml                     Render deploy config
rag_pipeline/
  api.py          FastAPI server — serves the HTML at GET / and the API at POST /api/recommend
  ingest.py       PDF → ChromaDB (page-level chunking, auto-detects pillar/country)
  retrieve.py     Vector similarity retrieval from ChromaDB
  generate.py     GLM generation grounded in retrieved chunks
  schema.py       Pydantic Recommendation model (shared between pipeline and prompt)
  chat.py         CLI loop for testing without the UI
  chroma_db/      Pre-seeded vector store (Portugal Research pillar, 4 pages)
  data/           Source PDFs
```

---

## Current coverage

Only one of Portugal's six OECD toolkit pillars is ingested: **Research, investment, and commercialisation**. Queries about compute subsidies, R&D funding, local language AI resources, and AI research centres will return real evidence. Other queries will retrieve the closest available chunks, which may not be relevant — the model will say so.

To expand coverage, export additional pillar PDFs from the OECD toolkit and run `ingest.py` on each.

---

## Known gaps

- **5 of 6 pillars not yet ingested** for Portugal; no other countries are in the DB
- **Draft tab** is not wired to the API — text is hand-written mock content
- **No retry logic** if the model returns malformed JSON
- **Pure vector retrieval** — no reranking or relevance threshold
- **NVIDIA API key** in `.env` should be rotated before sharing the repo publicly
