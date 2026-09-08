"""LangGraph pipeline wiring for Phase 1.

Pipeline: Extraction → Classification → Quality → END

Run `python -m src.graph` for a quick end-to-end demo.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.classification import classify_requirements
from src.agents.extraction import extract_requirements
from src.agents.quality import analyse_requirements
from src.models.quality import QualityReport
from src.models.requirement import Requirement


class PipelineState(TypedDict):
    text: str
    requirements: list[Requirement]
    quality_report: QualityReport | None


def extraction_node(state: PipelineState) -> dict:
    result = extract_requirements(state["text"])
    return {"requirements": result.requirements}


def classification_node(state: PipelineState) -> dict:
    return {"requirements": classify_requirements(state["requirements"])}


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

    SAMPLE_SCENARIO = Path(
        "data/test_cases/loan_processing/scenario_01.txt"
    ).read_text().strip()

    pipeline = build_pipeline()
    final_state = pipeline.invoke({
        "text": SAMPLE_SCENARIO,
        "requirements": [],
        "quality_report": None,
    })

    # ── Print requirements ──
    print(f"Extracted {len(final_state['requirements'])} requirements:\n")
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
