"""
Ingest an OECD AI Policy Toolkit results PDF into ChromaDB, one chunk per page.

Chunking strategy: page-level. Each page of the generated PDF report covers
one (occasionally two) policy priorities under a single pillar, so a page is
already a reasonably coherent unit of evidence. We store the raw page text
plus lightweight metadata (source file, page number, pillar, country) and let
retrieval + the LLM do the finer-grained structuring at query time, rather
than writing a bespoke per-priority parser.

Embedding model: ChromaDB's built-in default embedding function
(sentence-transformers/all-MiniLM-L6-v2, run locally via ONNX). This is the
fastest practical option for a prototype like this: no network round trip per
chunk, ~22M parameters, and it ships with ChromaDB so there's nothing extra
to configure or download by hand.

Usage:
    python ingest.py data/source_pdfs/2026-07_PRT_OECD_AI_Policy_Toolkit_Results-2.pdf
"""

import sys
from pathlib import Path

import chromadb
import pdfplumber

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "ai_policy_evidence"


def extract_pages(pdf_path: Path) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [(page.extract_text() or "").strip() for page in pdf.pages]


def guess_pillar(page_text: str) -> str:
    """The pillar name is the first non-boilerplate line of each page."""
    for line in page_text.splitlines():
        line = line.strip()
        if not line or line.startswith("OECD AI Policy Toolkit") or line.startswith("Country/Territory"):
            continue
        return line
    return "unknown"


def guess_country(page_text: str) -> str:
    for line in page_text.splitlines():
        if line.startswith("Country/Territory:"):
            # e.g. "Country/Territory: Portugal Generated: 18/07/2026"
            rest = line.split("Country/Territory:", 1)[1].strip()
            return rest.split("Generated:")[0].strip()
    return "unknown"


def ingest(pdf_path: Path):
    pages = extract_pages(pdf_path)
    if not pages:
        print(f"No extractable text found in {pdf_path}")
        return

    country = guess_country(pages[0]) if pages else "unknown"

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids, docs, metadatas = [], [], []
    for i, text in enumerate(pages):
        if not text:
            continue
        pillar = guess_pillar(text)
        chunk_id = f"{pdf_path.stem}-p{i}"
        ids.append(chunk_id)
        docs.append(text)
        metadatas.append({
            "source_file": pdf_path.name,
            "page": i,
            "pillar": pillar,
            "country": country,
        })

    # Upsert so re-running ingestion on the same file doesn't create duplicates.
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    print(f"Ingested {len(ids)} page-level chunks from {pdf_path.name} into '{COLLECTION_NAME}'.")
    print(f"Country: {country}")
    print(f"Pillars seen: {sorted(set(m['pillar'] for m in metadatas))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-pdf>")
        sys.exit(1)
    ingest(Path(sys.argv[1]))
