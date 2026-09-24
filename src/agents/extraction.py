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

from src.models.requirement import ExtractionDraft, ExtractionResult
from src.utils.llm import call_structured

SYSTEM_PROMPT = """\
You are a senior requirements engineer for financial software systems \
(banking, lending, payments, KYC). Read the problem scenario and extract \
every distinct software requirement as an atomic "The system shall ..." statement.

For EACH requirement you must provide ALL of the following fields:
- id: Sequential identifier like REQ-001, REQ-002, ...
- statement: Atomic "The system shall ..." requirement
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
- Extract only what the SYSTEM must do, be, or enforce. Facts about how the \
PROJECT will be run are NOT system requirements: team size, staffing, delivery \
timeline, budget, funding, and development methodology all describe the project, \
not the software. A scenario saying "the team has 15 developers and 18 months" \
yields no requirement at all — do not write "the system shall be completed \
within 18 months" or "the system shall be developed using agile methodologies".
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



def extract_requirements(
    scenario_text: str,
    context: list[dict] | None = None,
    supplied: list[dict] | None = None,
) -> ExtractionResult:
    """Extract requirements, optionally grounded in retrieved source material.

    `context` is the output of `src.rag.retriever.retrieve_context` -- passages
    from the finance knowledge base. This is the supervisor's part 1: "feed a
    lightweight LLM existing requirement documents... then prompt it to generate
    the specific requirements for a particular functionality, along with its
    reasoning." Without it the model generates from parametric memory, which is
    what §6 of the problem statement exists to stop.

    Passing None runs the ungrounded path, which stays available deliberately --
    Day 20's headline result is grounded versus ungrounded, and that needs both.
    """
    # The LLM is asked for ExtractionDraft, which has no `category` field: the
    # classification node overwrites categories anyway, so generating them here
    # spends tokens on an answer that is thrown away one node later.
    user_prompt = f"Problem scenario:\n\n{scenario_text}"

    # Two kinds of evidence, treated differently on purpose.
    #
    #   context   passages similarity search found in the curated corpus.
    #             REFERENCE ONLY -- nobody asserted they apply to this project,
    #             and when they were treated as source a COBOL migration
    #             acquired "a mobile-responsive online loan application portal"
    #             from retrieved loan RFPs.
    #
    #   supplied  documents attached to this run by the person running it. That
    #             attachment IS the assertion of relevance, so requirements may
    #             be derived from them -- which is the supervisor's part 1 in
    #             its most literal reading.
    #
    # Ordering: evidence first, scenario last. A small model attends best to
    # whatever sits nearest the end, and the scenario is the instruction.
    blocks = []
    if supplied:
        from src.rag.retriever import format_supplied

        blocks.append(format_supplied(supplied))
    if context:
        from src.rag.retriever import format_context

        blocks.append(format_context(context))

    if blocks:
        closing = (
            "Extract the requirements of this project. They come from the "
            "problem scenario AND from the documents provided for it — a "
            "document's obligations are this project's requirements where they "
            "bear on the system described. The reference material is not a "
            "source of requirements."
            if supplied else
            "Extract requirements from the problem scenario above. The reference "
            "material is not a source of requirements."
        )
        user_prompt = "\n\n".join(blocks) + f"\n\n{user_prompt}\n\n{closing}"

    draft = call_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=ExtractionDraft,
    )
    # Promote to full Requirements with `category` empty; the classification
    # node fills it.
    return ExtractionResult.from_draft(draft)


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
            print(f"      {req.risk_level} risk | {req.source_stakeholder}")
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
