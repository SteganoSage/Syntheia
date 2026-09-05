"""Extraction agent — turns a free-text problem scenario into structured
Requirement objects via the LLM API.

Day 3: minimal single-shot version. Day 4: this is the API baseline — few-shot
examples plus explicit handling for compound/implicit requirements and the
no-requirement edge case. Month 2 swaps this out for a fine-tuned Llama 3.1 8B
(QLoRA) extractor, benchmarked against this baseline.
"""

import json
import time
from pathlib import Path

from src.models.requirement import ExtractionResult
from src.utils.llm import call_structured

SYSTEM_PROMPT = """You are a senior requirements engineer for financial software \
systems (banking, lending, payments, KYC). Read the problem scenario and extract \
every distinct software requirement as an atomic "The system shall ..." statement.

Rules:
- Split compound requirements into separate atomic requirements — one system
  behaviour per requirement.
- Make implicit requirements explicit — do not skip requirements that are only
  implied by context (e.g. "apply from mobile phones" implies a mobile-responsive
  UI requirement).
- Assign each requirement a sequential id like REQ-001, REQ-002, ...
- Assign a priority of High, Medium, or Low based on business/regulatory importance.
- In category_hint, give a comma-separated best guess at applicable categories from: \
Business, Stakeholder, Functional, Security, Privacy, Regulatory, Performance, \
Availability, Usability, Data-management, Integration, Audit/Reporting, Operational.
- If the scenario contains no extractable software requirements (e.g. it's purely
  background/company information), return an empty requirements list.

Here are worked examples of correct extraction:

### Example 1 — compound + implicit requirements
Scenario: "We need an online loan application portal where customers can apply \
from their mobile phones. The application must verify identity through \
Aadhaar-based eKYC. Credit scores should be fetched from CIBIL in real-time. \
Applications above 50 lakhs need manual approval from a senior officer. All data \
must comply with RBI's digital lending guidelines."
Output:
[
  {"id": "REQ-001", "statement": "The system shall provide a mobile-responsive online loan application portal", "priority": "High", "category_hint": "Functional, Usability"},
  {"id": "REQ-002", "statement": "The system shall verify applicant identity through Aadhaar-based eKYC", "priority": "High", "category_hint": "Functional, Regulatory, Security"},
  {"id": "REQ-003", "statement": "The system shall fetch credit scores from CIBIL in real-time during application processing", "priority": "High", "category_hint": "Functional, Integration"},
  {"id": "REQ-004", "statement": "Loan applications exceeding INR 50 lakhs shall require manual approval from a senior officer", "priority": "Medium", "category_hint": "Business, Functional"},
  {"id": "REQ-005", "statement": "All data handling shall comply with RBI digital lending guidelines", "priority": "High", "category_hint": "Regulatory, Data-management"}
]
(Note the mobile-responsive UI requirement is implicit — "apply from mobile phones" \
never explicitly says the UI must be responsive, but it is made explicit here.)

### Example 2 — compound requirement split into atomic pieces
Scenario: "Customers shall be able to view, download, and print their account \
statements, and any failed download attempt shall be logged and automatically \
retried up to three times."
Output:
[
  {"id": "REQ-001", "statement": "The system shall allow customers to view their account statements", "priority": "Medium", "category_hint": "Functional, Usability"},
  {"id": "REQ-002", "statement": "The system shall allow customers to download their account statements", "priority": "Medium", "category_hint": "Functional, Usability"},
  {"id": "REQ-003", "statement": "The system shall allow customers to print their account statements", "priority": "Low", "category_hint": "Functional, Usability"},
  {"id": "REQ-004", "statement": "The system shall log any failed statement download attempt", "priority": "Medium", "category_hint": "Functional, Audit/Reporting"},
  {"id": "REQ-005", "statement": "The system shall automatically retry a failed statement download up to three times", "priority": "Medium", "category_hint": "Functional, Availability"}
]

### Example 3 — no extractable requirements
Scenario: "Our company was founded in 2010 and has grown to serve over one \
million customers across three countries."
Output:
[]
"""


def extract_requirements(scenario_text: str) -> ExtractionResult:
    return call_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Problem scenario:\n\n{scenario_text}",
        response_model=ExtractionResult,
    )


if __name__ == "__main__":
    # Day 4 DoD check: run the API-baseline extractor on ALL test scenarios
    # and log baseline metrics for later comparison against the fine-tuned
    # Llama extractor (Month 2).

    scenarios = [
        Path("data/test_cases/loan_processing/scenario_01.txt"),
        Path("data/test_cases/payment_system/scenario_01.txt"),
        Path("data/test_cases/kyc_onboarding/scenario_01.txt"),
    ]

    all_results = []

    for scenario_path in scenarios:
        scenario_text = scenario_path.read_text().strip()
        scenario_name = scenario_path.parent.name

        print(f"{'=' * 60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'=' * 60}")

        start = time.perf_counter()
        result = extract_requirements(scenario_text)
        elapsed = time.perf_counter() - start

        print(f"Extracted {len(result.requirements)} requirements in {elapsed:.2f}s:\n")
        for req in result.requirements:
            print(f"  [{req.id}] ({req.priority}) {req.statement}")
            print(f"      category_hint: {req.category_hint}")
        print()

        all_results.append({
            "scenario": scenario_name,
            "scenario_file": str(scenario_path),
            "num_requirements": len(result.requirements),
            "latency_seconds": round(elapsed, 3),
            "requirements": [r.model_dump() for r in result.requirements],
        })

    # Save combined baseline
    baseline = {
        "agent": "extraction_api_baseline",
        "total_scenarios": len(all_results),
        "total_requirements": sum(r["num_requirements"] for r in all_results),
        "avg_latency_seconds": round(
            sum(r["latency_seconds"] for r in all_results) / len(all_results), 3
        ),
        "results": all_results,
    }
    out_path = Path("evaluation/results/day4_extraction_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2))

    print(f"{'=' * 60}")
    print(f"SUMMARY: {baseline['total_requirements']} requirements across "
          f"{baseline['total_scenarios']} scenarios "
          f"(avg {baseline['avg_latency_seconds']}s per scenario)")
    print(f"Baseline saved to {out_path}")
