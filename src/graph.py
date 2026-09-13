"""LangGraph pipeline wiring for Phase 1.

Pipeline: Extraction (LLM) → Classification (fine-tuned) → Quality (LLM) → END

Classification is the one node not backed by the LLM API. It runs the
fine-tuned RoBERTa/BERT multi-label classifier from models/, which scores 80.2%
micro-F1 on the held-out ground truth against few-shot Gemini's 84.0%, at
roughly 1,450x lower latency. That trade is what makes the node practical:
classification runs per requirement, and 30 s each does not scale.

If no trained model is present the node falls back to the few-shot LLM
baseline, so the pipeline still runs on a fresh checkout.

Run `python -m src.graph` for a quick end-to-end demo, or
`python -m src.graph --classifier few_shot` to compare against the baseline.
"""

import argparse
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.classification import classify_requirements, resolve_model_dir
from src.agents.extraction import extract_requirements
from src.agents.quality import analyse_requirements
from src.models.quality import QualityReport
from src.models.requirement import Requirement


class PipelineState(TypedDict):
    text: str
    requirements: list[Requirement]
    quality_report: QualityReport | None
    classifier: str          # "auto" | "bert" | "few_shot" | "zero_shot"


def extraction_node(state: PipelineState) -> dict:
    result = extract_requirements(state["text"])
    return {"requirements": result.requirements}


def classification_node(state: PipelineState) -> dict:
    method = state.get("classifier") or "auto"
    return {"requirements": classify_requirements(state["requirements"], method=method)}


def quality_node(state: PipelineState) -> dict:
    report = analyse_requirements(state["requirements"])
    return {"quality_report": report}


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("extraction", extraction_node)
    graph.add_node("classification", classification_node)
    graph.add_node("quality", quality_node)
    graph.set_entry_point("extraction")
    graph.add_edge("extraction", "classification")
    graph.add_edge("classification", "quality")
    graph.add_edge("quality", END)
    return graph.compile()


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run the Phase 1 pipeline.")
    parser.add_argument(
        "--classifier", default="auto",
        choices=["auto", "bert", "few_shot", "zero_shot"],
        help="auto uses the fine-tuned model when one is trained, else few-shot",
    )
    parser.add_argument(
        "--scenario", type=Path,
        default=Path("data/test_cases/loan_processing/scenario_01.txt"),
    )
    args = parser.parse_args()

    SAMPLE_SCENARIO = args.scenario.read_text().strip()

    model_dir = resolve_model_dir()
    if args.classifier in ("auto", "bert") and model_dir:
        print(f"Classifier: fine-tuned ({model_dir.name})")
    elif args.classifier in ("auto", "bert"):
        print("Classifier: no trained model found, falling back to few-shot LLM")
    else:
        print(f"Classifier: {args.classifier} LLM baseline")

    # Load the classifier and pin the LLM before the clock starts. On a cold
    # process this is ~96% of a run's wall-clock (Day 13 measurement), and it
    # is one-off cost, not per-request -- so charging it to startup is the
    # difference between a 10-minute demo and a 1-minute one.
    from src.utils.warmup import warm
    warm()

    t_start = time.perf_counter()
    pipeline = build_pipeline()
    final_state = pipeline.invoke({
        "text": SAMPLE_SCENARIO,
        "requirements": [],
        "quality_report": None,
        "classifier": args.classifier,
    })

    elapsed = time.perf_counter() - t_start

    # ── Print requirements ──
    print(f"Extracted {len(final_state['requirements'])} requirements "
          f"in {elapsed:.1f}s (warm):\n")
    for req in final_state["requirements"]:
        print(f"[{req.id}] ({req.priority}) {req.statement}")
        print(f"    Categories:       {req.category}")
        print(f"    Stakeholder:      {req.source_stakeholder}")
        print(f"    Justification:    {req.business_justification}")
        print(f"    Acceptance:       {req.acceptance_criteria}")
        print(f"    Regulations:      {req.applicable_regulations}")
        print(f"    Risk:             {req.risk_level}")
        print(f"    Confidence:       {req.confidence_score}")
        print()

    # ── Print quality report ──
    report = final_state["quality_report"]
    if report:
        passed = sum(1 for r in report.result_list if r.passed)
        failed = len(report.result_list) - passed
        print(f"{'=' * 60}")
        print(f"QUALITY REPORT: {passed} passed, {failed} flagged\n")
        for qr in report.result_list:
            status = "✅" if qr.passed else "❌"
            print(f"  {status} [{qr.requirement_id}]")
            for issue in qr.issues:
                print(f"      ⚠️ {issue.issue_type} ({issue.severity}): {issue.description}")
                print(f"         → {issue.suggestion}")
        print()
