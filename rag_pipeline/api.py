"""
FastAPI server for the AI Policy Copilot.

Serves the HTML mockup at GET / and exposes:
  POST /api/recommend  — retrieve + generate a Recommendation from ChromaDB
  POST /api/draft      — generate policy prose for a named section
  GET  /api/health     — liveness check

Run:
    uvicorn api:app --reload --port 8000
or:
    python api.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from generate import populate_recommendation
from retrieve import retrieve
from schema import Recommendation

load_dotenv()

MOCKUP_PATH = Path(__file__).parent.parent / "ai_policy_copilot_mockup.html"

app = FastAPI(title="AI Policy Copilot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    if not MOCKUP_PATH.exists():
        raise HTTPException(status_code=404, detail="Mockup HTML not found")
    return FileResponse(str(MOCKUP_PATH), media_type="text/html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    """Return which PDFs and pillars are currently ingested in ChromaDB."""
    from retrieve import get_collection
    col = get_collection()
    results = col.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    files: dict = {}
    for m in metadatas:
        src = m.get("source_file", "unknown")
        if src not in files:
            files[src] = {
                "source_file": src,
                "country": m.get("country", "unknown"),
                "pillars": set(),
                "pages": 0,
            }
        files[src]["pillars"].add(m.get("pillar", "unknown"))
        files[src]["pages"] += 1

    return {
        "total_chunks": len(metadatas),
        "files": [
            {**v, "pillars": sorted(v["pillars"])}
            for v in files.values()
        ],
    }


class RecommendRequest(BaseModel):
    query: str
    k: int = 4


@app.post("/api/recommend", response_model=Recommendation)
async def recommend(req: RecommendRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    chunks = retrieve(req.query, k=req.k)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No evidence found. Has ingest.py been run?",
        )
    try:
        rec = populate_recommendation(req.query, chunks, stream=False)
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
    return rec


# ── draft generation ──────────────────────────────────────────────────────────

_draft_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

DRAFT_SYSTEM = (
    "You are a policy writer helping a government team draft a section of their national AI strategy. "
    "Use only the provided recommendation and evidence excerpts. "
    "Write 2-4 crisp paragraphs of formal policy prose. "
    "Do not invent facts or cite sources not listed. "
    "Return only the prose — no headings, no JSON, no commentary."
)


class DraftRequest(BaseModel):
    section_title: str
    recommendation: str
    evidence_excerpts: list[str]
    tone: str = "formal"  # formal | concise | portugal


@app.post("/api/draft")
async def draft_section(req: DraftRequest):
    excerpts_block = "\n\n".join(
        f"Evidence {i + 1}: {e}" for i, e in enumerate(req.evidence_excerpts)
    )
    tone_note = {
        "concise": " Be as concise as possible — 1-2 short paragraphs.",
        "portugal": " Frame the prose specifically around Portugal's existing AI unit and stakeholder forum.",
    }.get(req.tone, "")

    user_msg = (
        f"Section: {req.section_title}\n\n"
        f"Recommendation: {req.recommendation}\n\n"
        f"Evidence excerpts:\n{excerpts_block}\n\n"
        f"Draft the section now.{tone_note}"
    )

    try:
        completion = _draft_client.chat.completions.create(
            model="z-ai/glm-5.2",
            messages=[
                {"role": "system", "content": DRAFT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=1024,
            stream=False,
        )
        text = completion.choices[0].message.content or ""
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Draft generation failed: {exc}") from exc

    return {"draft": text.strip()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
