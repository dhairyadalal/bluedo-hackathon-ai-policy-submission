"""
Generation step: take retrieved evidence chunks and ask GLM (via NVIDIA's
OpenAI-compatible API) to populate the Recommendation schema ad hoc.

This is the "data model as prompt" pattern: we don't write a parser that
converts PDF text into a Recommendation object. Instead we hand the model
the JSON schema plus the retrieved evidence and ask it to fill the schema in,
grounded only in that evidence. schema.py's Recommendation model then
validates the result, so a malformed or hallucinated response fails loudly
instead of silently corrupting data.

Client setup follows the pattern provided for NVIDIA's integrate.api.nvidia.com
endpoint. The API key is read from the NVIDIA_API_KEY environment variable
(via .env), never hardcoded, since it's a live credential.
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from schema import Recommendation, schema_for_prompt

load_dotenv()

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""

MODEL = "z-ai/glm-5.2"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    # A non-secret placeholder keeps local demo mode importable. The API
    # server never calls this client unless a real NVIDIA_API_KEY is present.
    api_key=os.getenv("NVIDIA_API_KEY") or "not-configured",
)

SYSTEM_PROMPT = """You are a policy analysis assistant helping a government research manager \
turn retrieved evidence into a structured recommendation.

Rules:
- Use ONLY the evidence passages provided in the user message. Do not use outside knowledge \
about these countries or documents.
- Every excerpt you cite must be copied verbatim from the provided evidence, not paraphrased or invented.
- If the evidence does not clearly support a claim, do not include that claim.
- Return ONLY a single JSON object matching the schema below. No markdown fences, no commentary \
before or after it.

Schema:
{schema}
"""


def _format_evidence(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[page {c['page']} | pillar: {c['pillar']} | country: {c['country']}]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def populate_recommendation(query: str, chunks: list[dict], stream: bool = True) -> Recommendation:
    """
    Ask GLM to populate a Recommendation object grounded in the given chunks.
    Raises pydantic.ValidationError if the model's output doesn't match the schema.
    """
    evidence_block = _format_evidence(chunks)
    system = SYSTEM_PROMPT.format(schema=schema_for_prompt())
    user = f"""Policy question: {query}

Retrieved evidence (page-level chunks from an OECD AI Policy Toolkit results PDF):

{evidence_block}

Populate the recommendation object using only this evidence."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1,
        top_p=1,
        max_tokens=16384,
        seed=42,
        stream=stream,
    )

    full_text = ""
    if stream:
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None) is not None:
                print(delta.content, end="", flush=True)
                full_text += delta.content
        print()
    else:
        full_text = completion.choices[0].message.content

    cleaned = _strip_code_fence(full_text)
    data = json.loads(cleaned)
    return Recommendation.model_validate(data)


if __name__ == "__main__":
    from retrieve import retrieve

    query = " ".join(sys.argv[1:]) or "compute subsidies for startups and SMEs"
    chunks = retrieve(query, k=4)
    if not chunks:
        print("No evidence retrieved. Run ingest.py first.")
        sys.exit(1)

    try:
        rec = populate_recommendation(query, chunks)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"\n\nModel output did not match the schema: {e}")
        sys.exit(1)

    print("\n--- validated Recommendation ---")
    print(rec.model_dump_json(indent=2))
