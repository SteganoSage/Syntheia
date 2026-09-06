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

SYSTEM_PROMPT = """\
You are a senior requirements engineer for financial software systems \
(banking, lending, payments, KYC). Read the problem scenario and extract \
every distinct software requirement as an atomic "The system shall ..." statement.

For EACH requirement you must provide ALL of the following fields:
- id: Sequential identifier like REQ-001, REQ-002, ...
- statement: Atomic "The system shall ..." requirement
- category: List of applicable categories from: Business, Stakeholder, \
Functional, Security, Privacy, Regulatory, Performance, Availability, \
Usability, Data-management, Integration, Audit/Reporting, Operational
- source_stakeholder: Who would need this (e.g. "Business Analyst", \
"Compliance Officer", "End User", "Security Team")
- business_justification: WHY this requirement matters — the reasoning
- priority: High, Medium, or Low
- dependencies: List of other requirement IDs this depends on (empty list if none)
- assumptions: What is assumed when stating this requirement
- acceptance_criteria: A testable condition to verify the requirement is met
- applicable_regulations: Relevant standards/regulations (e.g. ["RBI Digital Lending", "PCI-DSS"])
- risk_level: High, Medium, or Low — risk if this requirement is NOT met
- confidence_score: Your confidence in this extraction (0.0 to 1.0)
- approval_status: Always set to "Pending"

Rules:
- Split compound requirements into separate atomic requirements.
- Make implicit requirements explicit — do not skip implied requirements.
- If the scenario contains no extractable requirements, return an empty list.

### Example
Scenario: "We need a mobile loan portal with Aadhaar eKYC and RBI compliance."
Output (showing one requirement):
[
  {
    "id": "REQ-001",
    "statement": "The system shall provide a mobile-responsive online loan application portal",
    "category": ["Functional", "Usability"],
    "source_stakeholder": "End User",
    "business_justification": "Customers expect to apply from mobile devices; mobile-first design increases application completion rates",
    "priority": "High",
    "dependencies": [],
    "assumptions": "Users have smartphones with stable internet connectivity",
    "acceptance_criteria": "Portal renders correctly and is fully functional on screen widths from 320px to 1920px",
    "applicable_regulations": [],
    "risk_level": "High",
    "confidence_score": 0.95,
    "approval_status": "Pending"
  }
]
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
