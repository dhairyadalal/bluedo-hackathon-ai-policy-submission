"""
Simple agentic chat loop: ask a policy question in plain English, get back
a structured, evidence-grounded Recommendation.

    python chat.py

Each turn:
  1. Retrieve top-k page chunks from ChromaDB for the question (retrieve.py).
  2. Hand those chunks + the Recommendation schema to GLM (generate.py).
  3. Validate the response against schema.py and print it.

This is intentionally thin. The interesting work happens in retrieve.py
(what counts as relevant evidence) and generate.py (how the schema is
communicated to the model), not in this loop.
"""

import json
import sys

from pydantic import ValidationError

from generate import populate_recommendation
from retrieve import retrieve


def run_turn(query: str, k: int = 4):
    chunks = retrieve(query, k=k)
    if not chunks:
        print("No evidence found for that. Has ingest.py been run yet?")
        return

    print(f"\nRetrieved {len(chunks)} chunk(s):")
    for c in chunks:
        print(f"  - page {c['page']} ({c['pillar']}), distance {c['distance']:.3f}")
    print("\nAsking GLM to populate the recommendation...\n")

    try:
        rec = populate_recommendation(query, chunks)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"\nCouldn't validate the model's output against the schema: {e}")
        return

    print("\n--- Recommendation ---")
    print(rec.model_dump_json(indent=2))


def main():
    print("AI Policy Copilot — agentic chat (Ctrl+C to quit)")
    print("Ask about a policy priority, e.g. 'compute subsidies for startups'\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not query:
            continue
        run_turn(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_turn(" ".join(sys.argv[1:]))
    else:
        main()
