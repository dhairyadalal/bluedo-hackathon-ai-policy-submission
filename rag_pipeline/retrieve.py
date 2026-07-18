"""
Retrieval step: given a natural-language query, return the top-k page-level
chunks from ChromaDB, with their metadata (source file, page, pillar, country).
"""

from pathlib import Path

import chromadb

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "ai_policy_evidence"


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 4) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    chunks = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        chunks.append({
            "id": id_,
            "text": doc,
            "page": meta.get("page"),
            "pillar": meta.get("pillar"),
            "country": meta.get("country"),
            "source_file": meta.get("source_file"),
            "distance": dist,
        })
    return chunks


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "compute subsidies for startups"
    for chunk in retrieve(query):
        print(f"--- page {chunk['page']} | pillar: {chunk['pillar']} | distance: {chunk['distance']:.3f} ---")
        print(chunk["text"][:300])
        print()
