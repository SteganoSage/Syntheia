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


def retrieval_node(state: PipelineState) -> dict:
    """Fetch source material for the scenario before anything is generated.

    §6's first step. Degrades rather than fails: with no index, or with
    --no-rag, the pipeline runs ungrounded exactly as it did before, which is
    also how the Day 20 comparison gets its control arm.
    """
    if not state.get("use_rag", True):
        return {"context": []}

    from src.rag.retriever import is_available, retrieve_context

    if not is_available():
        print("  (no knowledge base indexed -- running ungrounded; "
              "build one with notebooks/09_build_knowledge_base.py)")
        return {"context": []}

    hits = retrieve_context(state["text"])
    if hits:
        print(f"Retrieved {len(hits)} passages "
              f"(top score {hits[0]['score']:.3f}):")
        for h in hits:
            m = h["metadata"]
            print(f"    {h['score']:.3f}  {m.get('citation', m.get('title',''))[:88]}")
    return {"context": hits}


def extraction_node(state: PipelineState) -> dict:
    result = extract_requirements(state["text"], context=state.get("context"))
    return {"requirements": result.requirements}


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

    if not is_available():
        return {"grounding": {}}

    grounding: dict[str, dict] = {}
    for req in state["requirements"]:
        g = ground_requirement(req.statement)
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
    report = analyse_requirements(state["requirements"])
    return {"quality_report": report}


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

    return {"sdlc": recommend_sdlc(state["requirements"], scenario=state["text"])}


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
    graph.add_edge("quality", "sdlc")
    graph.add_edge("sdlc", END)
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
    parser.add_argument(
        "--no-sdlc", action="store_true",
        help="skip the SDLC recommendation (Phase 2) -- faster when iterating on Phase 1",
    )
    parser.add_argument(
        "--json", type=Path, default=None, metavar="PATH",
        help="also write the run to PATH as JSON -- what notebooks/11_eval_sdlc.py "
             "reads, so a change can be measured instead of eyeballed in scrollback",
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="skip retrieval and generate from the model's own knowledge -- "
             "the control arm for the grounded-vs-ungrounded comparison",
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
    print(f"Grounding:  {'OFF (--no-rag)' if args.no_rag else 'retrieval-augmented'}")
    final_state = pipeline.invoke({
        "text": SAMPLE_SCENARIO,
        "requirements": [],
        "quality_report": None,
        "classifier": args.classifier,
        "use_rag": not args.no_rag,
        "context": [],
        "grounding": {},
        "sdlc": None,
        "run_sdlc": not args.no_sdlc,
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
        g = final_state.get("grounding", {}).get(req.id)
        if g and g["citations"]:
            print(f"    Regulations:      (grounded)")
            for c in g["citations"]:
                print(f"        [{c['confidence']:.2f}] {c['citation']}")
        elif g and g["needs_review"]:
            print(f"    Regulations:      {req.applicable_regulations}")
            print(f"        !! no regulation in the knowledge base supports this "
                  f"(best match {g['confidence']:.2f}) -- needs human review")
        else:
            print(f"    Regulations:      {req.applicable_regulations}")
        if g and g.get("precedents"):
            # Not regulations -- how comparable systems specified the same thing.
            print(f"    Precedent:        (how other institutions specified this)")
            for c in g["precedents"][:2]:
                print(f"        [{c['confidence']:.2f}] {c['citation'][:96]}")
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

    # ── Phase 2: SDLC recommendation ──
    rec = final_state.get("sdlc")
    if rec:
        print(f"{'=' * 60}")
        derived = "  [derived column -- not in the supplied framework]" if any(
            m.is_derived and m.name == rec.recommendation for m in rec.ranking) else ""
        print(f"SDLC RECOMMENDATION: {rec.recommendation}{derived}")
        if rec.tied_with:
            print(f"  ** TIED: {' = '.join(rec.tied_with)} -- the framework does "
                  f"not separate them.")
            print(f"     Listed first by name, which is not a preference. Answer "
                  f"an unassessed question to break it.")
        print(f"{'=' * 60}\n")

        print("  Framework scoring:")
        for s in rec.ranking:
            mark = " *" if s.name == rec.recommendation else "  "
            d = "  (derived)" if s.is_derived else ""
            print(f"   {mark} {s.name:<14}{s.score:>6.0%}  "
                  f"{len(s.matched)}/{len(s.matched) + len(s.missed)}{d}")

        print("\n  Inferred from the requirements:")
        for a in rec.criteria:
            ev = ", ".join(e[:44] for e in a.evidence) if a.evidence else "nothing cited"
            label = {"yes": "YES", "no": "NO", "not_stated": "--"}[a.answer]
            # An answer the evidence gate rejected reads as an abstention in the
            # score, but it is not one -- the model did commit, and a reader
            # checking the framework should see that it was overruled.
            if a.reasoning.startswith("UNSUPPORTED"):
                ev = a.reasoning.split("):")[0] + ")"
            print(f"    {a.question_id:<7} {label:<4} [{ev}]")

        print("\n  Framework coverage:")
        for c in rec.coverage.values():
            state_ = "assessed" if c.assessed else "NOT ASSESSED"
            print(f"    {c.name:<22} {c.answered}/{c.total:<3} {state_} ({c.source})")

        if rec.cross_check_ranking:
            agree = "AGREES" if rec.cross_check_agrees else "DISAGREES"
            print(f"\n  Cross-check ({agree}): the model ranked, without seeing scores,")
            print(f"    {' > '.join(rec.cross_check_ranking[:4])}")

        for j in rec.justifications:
            print(f"\n  --- {j.model} ---")
            print(f"    {j.justification}")
        if rec.summary:
            print(f"\n  {rec.summary}")
        print()

    # ── Machine-readable run, for the eval harness ──
    if args.json:
        import json

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "scenario": str(args.scenario),
            "scenario_name": args.scenario.parent.name,
            "requirements": [r.model_dump() for r in final_state["requirements"]],
            "sdlc": rec.model_dump() if rec else None,
        }, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.json}")
