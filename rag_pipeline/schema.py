"""
Data model for the AI Policy Copilot.

This is the same shape used in the HTML mockup (documents + recommendations),
ported to Pydantic so it can do double duty:

1. As a JSON schema we inject into the GLM prompt, so the model knows exactly
   what fields to fill in (this is the "data model as prompt" pattern).
2. As a validator we run the model's output through afterward, so a malformed
   or hallucinated response fails loudly instead of silently corrupting data.

We deliberately do NOT pre-populate this model from the PDF with a bespoke
parser. The RAG step retrieves raw evidence; GLM populates this schema ad hoc
per query, grounded only in what retrieval returned.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source_title: str = Field(description="Title of the case study or precedent document")
    country: Optional[str] = Field(default=None, description="Country or organisation the precedent is from")
    year: Optional[int] = Field(default=None, description="Publication year of the precedent, if known")
    url: Optional[str] = Field(default=None, description="Link to the precedent on oecd.ai, if present in the evidence")
    page: int = Field(description="Page number in the source PDF this evidence was retrieved from")
    excerpt: str = Field(
        description=(
            "A short passage copied verbatim from the retrieved evidence that supports "
            "this recommendation. Do not paraphrase or invent this text."
        )
    )


class Recommendation(BaseModel):
    pillar: str = Field(description="The OECD AI Policy Toolkit pillar this recommendation belongs to")
    priority_title: str = Field(description="The specific policy priority under review, e.g. 'Compute subsidies'")
    country: str = Field(description="The country this recommendation is for")
    status: Literal["gap", "planned", "implemented", "unknown"] = Field(
        description="Portugal's current maturity on this priority, if stated in the evidence, otherwise 'unknown'"
    )
    timeframe: Optional[Literal["now", "next", "later"]] = Field(
        default=None,
        description=(
            "Suggested delivery timeframe for gap priorities: 'now' (0–6 months, unblocks other work), "
            "'next' (6–12 months, depends on 'now' items), 'later' (parallel track or 12+ months). "
            "Set to null for planned or implemented priorities."
        ),
    )
    recommendation: str = Field(description="One to two sentence recommended action for the policymaker")
    related_claims: list[str] = Field(
        description="2-3 claims that justify the recommendation, each traceable back to the evidence list below"
    )
    evidence: list[EvidenceItem] = Field(description="The precedents grounding this recommendation")


def schema_for_prompt() -> str:
    """Render the Recommendation schema as JSON for embedding directly in a prompt."""
    import json
    return json.dumps(Recommendation.model_json_schema(), indent=2)
