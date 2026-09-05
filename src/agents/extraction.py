"""Extraction agent — turns a free-text problem scenario into structured
Requirement objects via the LLM API.

Day 3: minimal single-shot version. Day 4 adds few-shot examples and explicit
handling for compound/implicit requirements and no-requirement edge cases.
Month 2 swaps this out for a fine-tuned Llama 3.1 8B (QLoRA) extractor.
"""

from src.models.requirement import ExtractionResult
from src.utils.llm import call_structured

SYSTEM_PROMPT = """You are a senior requirements engineer for financial software \
systems (banking, lending, payments, KYC). Read the problem scenario and extract \
every distinct software requirement as an atomic "The system shall ..." statement.

Rules:
- Split compound requirements into separate atomic requirements.
- Make implicit requirements explicit — do not skip requirements that are only implied.
- Assign each requirement a sequential id like REQ-001, REQ-002, ...
- Assign a priority of High, Medium, or Low based on business/regulatory importance.
- In category_hint, give a comma-separated best guess at applicable categories from: \
Business, Stakeholder, Functional, Security, Privacy, Regulatory, Performance, \
Availability, Usability, Data-management, Integration, Audit/Reporting, Operational.
- If the scenario contains no extractable requirements, return an empty list.
"""


def extract_requirements(scenario_text: str) -> ExtractionResult:
    return call_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Problem scenario:\n\n{scenario_text}",
        response_model=ExtractionResult,
    )
