"""LangGraph pipeline wiring for Phase 1 (Extraction -> Classification -> ...).

Day 3: 2-node skeleton — Extraction (LLM API) -> Classification (placeholder).
Day 6 adds a Quality node. Day 11 swaps the Classification placeholder for the
fine-tuned BERT agent.

Run `python -m src.graph` for a quick end-to-end demo (requires a real
OPENAI_API_KEY in .env).
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.classification import classify_requirements
from src.agents.extraction import extract_requirements
from src.models.requirement import Requirement


class PipelineState(TypedDict):
    text: str
    requirements: list[Requirement]


def extraction_node(state: PipelineState) -> dict:
    result = extract_requirements(state["text"])
    return {"requirements": result.requirements}


def classification_node(state: PipelineState) -> dict:
    return {"requirements": classify_requirements(state["requirements"])}


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("extraction", extraction_node)
    graph.add_node("classification", classification_node)
    graph.set_entry_point("extraction")
    graph.add_edge("extraction", "classification")
    graph.add_edge("classification", END)
    return graph.compile()


if __name__ == "__main__":
    from pathlib import Path

    SAMPLE_SCENARIO = Path(
        "data/test_cases/loan_processing/scenario_01.txt"
    ).read_text().strip()

    pipeline = build_pipeline()
    final_state = pipeline.invoke({"text": SAMPLE_SCENARIO, "requirements": []})

    print(f"Extracted {len(final_state['requirements'])} requirements:\n")
    for req in final_state["requirements"]:
        print(f"[{req.id}] ({req.priority}) {req.statement}")
        print(f"    category_hint: {req.category_hint}\n")
