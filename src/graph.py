"""LangGraph pipeline wiring.

Pipeline: retrieval → extraction → grounding → classification → quality
          → conflicts → compliance → security → risk → clarification   (Phase B)
          → sdlc → END

Classification is the one node not backed by the LLM API. It runs the
fine-tuned RoBERTa/BERT multi-label classifier from models/, which scores 80.2%
micro-F1 on the held-out ground truth against few-shot Gemini's 84.0%, at
roughly 1,450x lower latency. That trade is what makes the node practical:
classification runs per requirement, and 30 s each does not scale.

If no trained model is present the node falls back to the few-shot LLM
baseline, so the pipeline still runs on a fresh checkout.

The CLI lives in `main.py`; `python -m src.graph` is an alias for it and
delegates there, so there is one argument parser and one renderer rather
than two that drift.
"""

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.classification import classify_requirements
from src.agents.extraction import extract_requirements
from src.agents.quality import analyse_requirements
from src.models.quality import QualityReport
from src.models.requirement import Requirement
from src.models.sdlc import SDLCRecommendation


class PipelineState(TypedDict):
    text: str
    requirements: list[Requirement]
    quality_report: QualityReport | None
    classifier: str          # "auto" | "bert" | "few_shot" | "zero_shot"
    use_rag: bool
    context: list[dict]      # passages retrieved for the scenario
    grounding: dict          # requirement id -> citations, confidence, needs_review
    sdlc: SDLCRecommendation | None
    run_sdlc: bool
    # True only when a person is at a terminal. Gates the follow-up questions:
    # a piped run or a --json eval loop must never block on input.
    interactive: bool
    # Temporary ChromaDB directories holding documents supplied for THIS run,
    # searched alongside the curated corpus. Empty for a normal run.
    extra_dirs: list
    # Extracted statements removed as project facts, not system requirements.
    dropped: list
    # Regulatory sector(s) the scenario is in ("banking", "insurance"), set by the
    # retrieval node and used to keep grounding inside that sector.
    sectors: list
    # Phase B analysis layer (src/pipeline_analysis.py). Off with --no-analysis.
    run_analysis: bool
    conflicts: list
    compliance: object        # ComplianceReport | None
    security: object          # SecurityReport | None
    risk_register: list
    clarifications: object    # ClarificationResult | None


def retrieval_node(state: PipelineState) -> dict:
    """Fetch source material for the scenario before anything is generated.

    §6's first step. Degrades rather than fails: with no index, or with
    --no-rag, the pipeline runs ungrounded exactly as it did before, which is
    also how the Day 20 comparison gets its control arm.
    """
    if not state.get("use_rag", True):
        return {"context": []}

    from src.rag.retriever import is_available, retrieve_context

    extra = state.get("extra_dirs") or []
    if not is_available() and not extra:
        print("  (no knowledge base indexed -- running ungrounded; "
              "build one with notebooks/09_build_knowledge_base.py)")
        return {"context": []}

    # Nodes do not print. This one used to dump the retrieved passages, which
    # landed between the submit marker and the results in the interactive app
    # and could not be suppressed without an argument threaded through the
    # state. The passages are in `context` -- whoever is rendering decides
    # whether to show them, which is the renderer's job and not the node's.
    from src.rag.sectors import scenario_sectors

    sectors = scenario_sectors(state["text"])
    return {"context": retrieve_context(state["text"], extra_dirs=extra,
                                        sectors=sectors),
            "sectors": sorted(sectors)}


def extraction_node(state: PipelineState) -> dict:
    # Attached documents are fetched on their own terms: a lower relevance floor
    # and more passages than the corpus gets, because they are source material
    # for this project rather than phrasing reference.
    supplied = []
    extra = state.get("extra_dirs") or []
    if extra and state.get("use_rag", True):
        from src.rag.retriever import retrieve_supplied

        supplied = retrieve_supplied(state["text"], extra)

    result = extract_requirements(state["text"], context=state.get("context"),
                                  supplied=supplied)
    if not result.requirements:
        # Reached only if every retry failed; the schema's min_length normally
        # forces a retry first. Loud, because everything downstream degrades
        # quietly from here -- grounding has nothing to ground, the SDLC engine
        # infers from an empty list, and the run still prints a recommendation.
        print("  !! extraction produced NO requirements -- the rest of this run "
              "is not meaningful. Check the model and the scenario.")
    return {"requirements": result.requirements, "dropped": result.dropped}


def grounding_node(state: PipelineState) -> dict:
    """Attach a citation and a confidence to each requirement.

    No LLM call: each statement is embedded and matched against the knowledge
    base, which costs milliseconds rather than the seconds a second generation
    pass would. §6 asks for citations *and* confidence *and* escalation of what
    is unsupported, and a similarity score gives all three honestly -- including
    admitting when nothing in the corpus backs a requirement.

    The limit is real and worth stating: similarity is not applicability. A high
    score means the passage is about the same subject, not that the rule governs
    this requirement. That is why nothing here overwrites a citation silently --
    low-confidence requirements are flagged for a human instead.
    """
    # Deliberately NOT gated on state["context"]. Scenario-level retrieval and
    # per-requirement grounding are different queries: the first asks which
    # documents cover this project, the second asks which regulation governs
    # this statement. A COBOL migration matches no document as a whole (0.555,
    # below the context floor) while its testing and integration requirements
    # still match real clauses. Gating citations on the scenario score threw
    # those away.
    if not state.get("use_rag", True):
        return {"grounding": {}}

    from src.rag.retriever import ground_requirement, is_available

    extra = state.get("extra_dirs") or []
    # Supplied documents alone are enough to ground against: the curated corpus
    # may be empty or may simply not cover this domain -- IRDAI is in the
    # `unavailable:` list, so an insurance run has nothing to cite without them.
    if not is_available() and not extra:
        return {"grounding": {}}

    sectors = set(state.get("sectors") or [])
    grounding: dict[str, dict] = {}
    for req in state["requirements"]:
        g = ground_requirement(req.statement, extra_dirs=extra, sectors=sectors)
        grounding[req.id] = g
        # Only overwrite what the model guessed when the corpus actually
        # supports something; an unsupported requirement keeps the model's own
        # answer and is flagged, rather than being silently emptied.
        if g["applicable_regulations"]:
            req.applicable_regulations = g["applicable_regulations"]
    return {"grounding": grounding}


def classification_node(state: PipelineState) -> dict:
    method = state.get("classifier") or "auto"
    return {"requirements": classify_requirements(state["requirements"], method=method)}


def quality_node(state: PipelineState) -> dict:
    # Never call the reviewer with nothing to review. Asked to analyse an empty
    # list, qwen2.5:14b invented fifteen requirements and critiqued them --
    # REQ-001 through REQ-015, none of which existed, one flagged Critical
    # Infeasibility. That propagated into `needs_review` and would have reached
    # a reader as a finding about their system.
    #
    # An empty list is not a degenerate case to paper over: it means extraction
    # failed, and the report should say so rather than fill the gap.
    if not state["requirements"]:
        return {"quality_report": None}
    return {"quality_report": analyse_requirements(state["requirements"])}


def _report_skipped_interview(framework, answers, weak=None) -> dict:
    """Not interactive: name the questions that would have been asked, then move on."""
    pivotal = framework.pivotal(answers, weak=weak)
    if pivotal:
        names = ", ".join(d["question"].id for d in pivotal[:4])
        print(f"  ({len(pivotal)} question(s) would settle this — {names}. "
              f"Run at a terminal to be asked, or pass them in.)")
    return {}


def sdlc_node(state: PipelineState) -> dict:
    """Phase 2: the requirements go back in, an SDLC recommendation comes out.

    The supervisor's part 2. Deliberately last, so it sees requirements that
    have already been classified and grounded -- the framework asks about
    regulatory exposure and integration, which live in `applicable_regulations`
    and `category`, not in the statement text.
    """
    if not state.get("run_sdlc", True) or not state["requirements"]:
        return {"sdlc": None}

    from src.agents.sdlc import recommend_sdlc

    ask = None
    if state.get("interactive"):
        from src.cli.interview_mode import fill_gaps

        ask = fill_gaps
    else:
        # Say so rather than skipping in silence. The interview was disabled by
        # a wrong tty test for a whole session and looked exactly like "there
        # was nothing to ask" -- which is the one thing it must never look like.
        ask = _report_skipped_interview
    return {"sdlc": recommend_sdlc(state["requirements"],
                                   scenario=state["text"], ask_user=ask)}


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("grounding", grounding_node)
    graph.add_node("classification", classification_node)
    graph.add_node("quality", quality_node)
    graph.add_node("sdlc", sdlc_node)
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "extraction")
    graph.add_edge("extraction", "grounding")
    graph.add_edge("grounding", "classification")
    graph.add_edge("classification", "quality")
    # quality → conflicts → compliance → security → risk → clarification → sdlc
    from src.pipeline_analysis import add_analysis_nodes

    add_analysis_nodes(graph, after="quality", before="sdlc")
    graph.add_edge("sdlc", END)
    return graph.compile()


if __name__ == "__main__":
    # Delegates to main.py rather than carrying a second copy of the CLI and
    # the renderer. `python -m src.graph` predates main.py, and the runbook,
    # CLAUDE.md and the eval loop all still use it -- so it stays, as an alias.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from main import main

    raise SystemExit(main())
