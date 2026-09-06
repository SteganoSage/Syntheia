"""Classification agent — assigns multi-label categories to requirements.

Day 5: Two LLM-based baselines (zero-shot and few-shot).
Day 9-11: Replaced with fine-tuned BERT multi-label classifier.

The 13-category taxonomy:
  Business, Stakeholder, Functional, Security, Privacy, Regulatory,
  Performance, Availability, Usability, Data-management, Integration,
  Audit/Reporting, Operational
"""

from pydantic import BaseModel, Field

from src.models.requirement import Requirement
from src.utils.llm import call_structured

ALL_CATEGORIES = [
    "Business",
    "Stakeholder",
    "Functional",
    "Security",
    "Privacy",
    "Regulatory",
    "Performance",
    "Availability",
    "Usability",
    "Data-management",
    "Integration",
    "Audit/Reporting",
    "Operational",
]


class ClassificationResult(BaseModel):
    categories: list[str] = Field(
        description="One or more categories from the 13-category taxonomy"
    )


# ── Prompts ────────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = f"""\
You are a software requirements classifier for financial systems.

Given a software requirement statement, classify it into ONE OR MORE
categories from the following list (multi-label):

{chr(10).join(f'  - {c}' for c in ALL_CATEGORIES)}

Rules:
- Return ONLY categories from the list above — no new categories.
- A requirement may belong to multiple categories.
- Be precise: only assign a category if the requirement clearly relates to it.
"""

FEW_SHOT_PROMPT = f"""\
You are a software requirements classifier for financial systems.

Given a software requirement statement, classify it into ONE OR MORE
categories from the following list (multi-label):

{chr(10).join(f'  - {c}' for c in ALL_CATEGORIES)}

Rules:
- Return ONLY categories from the list above — no new categories.
- A requirement may belong to multiple categories.
- Be precise: only assign a category if the requirement clearly relates to it.

Here are examples of correct classification:

Example 1:
Requirement: "The system shall encrypt all applicant PII, including PAN and Aadhaar numbers, at rest using AES-256 encryption."
Categories: ["Security", "Privacy", "Data-management"]

Example 2:
Requirement: "The system shall route loan applications exceeding ₹50 lakhs to a senior loan officer for manual approval."
Categories: ["Business", "Functional", "Stakeholder"]

Example 3:
Requirement: "The system shall generate a monthly compliance report listing all loan applications that required manual override of automated approval rules."
Categories: ["Audit/Reporting", "Regulatory"]

Example 4:
Requirement: "The system shall support at least 500 concurrent loan application submissions without response times exceeding 5 seconds."
Categories: ["Performance"]

Example 5:
Requirement: "The system shall integrate with the UIDAI Aadhaar authentication API to perform real-time identity verification."
Categories: ["Integration", "Regulatory"]
"""


# ── Classification functions ───────────────────────────────────────────

def classify_zero_shot(statement: str) -> list[str]:
    """Classify a requirement using zero-shot prompting (no examples)."""
    result = call_structured(
        system_prompt=ZERO_SHOT_PROMPT,
        user_prompt=f'Requirement: "{statement}"',
        response_model=ClassificationResult,
    )
    # Filter to valid categories only
    return [c for c in result.categories if c in ALL_CATEGORIES]


def classify_few_shot(statement: str) -> list[str]:
    """Classify a requirement using few-shot prompting (with examples)."""
    result = call_structured(
        system_prompt=FEW_SHOT_PROMPT,
        user_prompt=f'Requirement: "{statement}"',
        response_model=ClassificationResult,
    )
    return [c for c in result.categories if c in ALL_CATEGORIES]


def classify_requirements(requirements: list[Requirement]) -> list[Requirement]:
    """Classify a batch of requirements (used in the LangGraph pipeline).

    Uses few-shot classification by default. Updates the `category` field
    on each requirement in-place and returns the list.
    """
    for req in requirements:
        req.category = classify_few_shot(req.statement)
    return requirements
