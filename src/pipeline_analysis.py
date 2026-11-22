"""Phase B pipeline nodes: conflicts → compliance → security → risk → clarification.

Kept out of `src/graph.py` so the analysis layer joins the graph with one call
(`add_analysis_nodes`) and can be read, tested and switched off as a unit.

    ... → quality → [conflicts → compliance → security → risk → clarification] → sdlc

Placed after quality and before SDLC: every analysis needs classified, grounded,
quality-checked requirements, and the SDLC engine should see the requirements
as revised by clarification, not as first extracted.

Two rules for every node here:

  * **Degrade, never take the run down.** Phase B is additive. A failure in the
    compliance mapper must not cost the user the SDLC recommendation they
    already waited for, so each node reports the failure loudly and returns an
    empty result.
  * **`--no-analysis` skips all five**, so the Phase A runtime and outputs can
    be reproduced exactly.
"""

from __future__ import annotations

from src.utils.similarity import Similarity, default_similarity

ANALYSIS_NODES = ["conflicts", "compliance", "security", "risk", "clarification"]

_SIM: Similarity | None = None


def _sim() -> Similarity:
    global _SIM
    if _SIM is None:
        _SIM = default_similarity()
    return _SIM


def _active(state) -> bool:
    return bool(state.get("run_analysis", True) and state.get("requirements"))


def _guarded(name: str, fn, default):
    try:
        return fn()
    except Exception as e:                                   # noqa: BLE001
        print(f"  !! {name} failed ({type(e).__name__}: {e}) — continuing without it")
        return default


# ── Nodes ──────────────────────────────────────────────────────────────

def conflicts_node(state) -> dict:
    if not _active(state):
        return {"conflicts": []}
    from src.agents.conflict import detect_conflicts

    return {"conflicts": _guarded(
        "conflict detection", lambda: detect_conflicts(state["requirements"], _sim()), [])}


def compliance_node(state) -> dict:
    if not _active(state):
        return {"compliance": None}
    from src.agents.compliance import analyse_compliance

    return {"compliance": _guarded("compliance analysis", lambda: analyse_compliance(
        state["text"], state["requirements"], state.get("grounding"), sim=_sim(),
        use_rag=state.get("use_rag", True),
        extra_dirs=state.get("extra_dirs") or None), None)}


def security_node(state) -> dict:
    if not _active(state):
        return {"security": None}
    from src.agents.security import analyse_security

    return {"security": _guarded("security analysis", lambda: analyse_security(
        state["text"], state["requirements"], sim=_sim(),
        use_rag=state.get("use_rag", True)), None)}


def risk_node(state) -> dict:
    if not _active(state):
        return {"risk_register": []}
    from src.agents.risk import build_risk_register

    return {"risk_register": _guarded("risk analysis", lambda: build_risk_register(
        state["text"], state["requirements"], quality=state.get("quality_report"),
        conflicts=state.get("conflicts"), compliance=state.get("compliance"),
        security=state.get("security"), grounding=state.get("grounding")), [])}


def clarification_node(state) -> dict:
    if not _active(state):
        return {"clarifications": None}
    from src.agents.clarification import carry_over, clarify, plan_questions

    ask = None
    if state.get("interactive"):
        from src.cli.clarify_mode import ask_clarifications

        ask = ask_clarifications

    def run():
        result, updated, changed = clarify(
            state["text"], state["requirements"],
            quality=state.get("quality_report"), conflicts=state.get("conflicts"),
            compliance=state.get("compliance"), security=state.get("security"),
            ask=ask)
        if not changed:
            return {"clarifications": result}
        # §10: what failed goes back through the checks. Re-validate the
        # revised set, then re-plan, so the questions shown at the end describe
        # the requirements as they now stand.
        out = revalidate(state, updated, changed)
        merged = {**state, **out}
        out["clarifications"] = carry_over(result, plan_questions(
            state["text"], out["requirements"], merged.get("quality_report"),
            merged.get("conflicts"), merged.get("compliance"), merged.get("security")))
        return out

    return _guarded("clarification", run, {"clarifications": None})


# ── Re-validation after clarification ──────────────────────────────────

def revalidate(state, requirements, changed: set[str]) -> dict:
    """Re-run every check on what changed; reuse earlier verdicts for the rest.

    Grounding, classification and quality run on the changed requirements
    only. Conflicts re-judge only pairs touching a changed requirement.
    Compliance re-maps only changed requirements (the rule pack re-runs in
    full -- it is free). Security is recomputed. The risk register is rebuilt,
    reusing the scenario-level risks the LLM proposed, since the scenario has
    not changed.
    """
    from src.agents.classification import classify_requirements
    from src.agents.compliance import analyse_compliance
    from src.agents.conflict import detect_conflicts
    from src.agents.quality import analyse_requirements
    from src.agents.risk import build_risk_register
    from src.agents.security import analyse_security
    from src.models.quality import QualityReport

    use_rag = state.get("use_rag", True)
    extra = state.get("extra_dirs") or None
    subset = [r for r in requirements if r.id in changed]

    grounding = dict(state.get("grounding") or {})
    if use_rag:
        from src.rag.retriever import ground_requirement

        sectors = set(state.get("sectors") or [])
        for r in subset:
            g = ground_requirement(r.statement, extra_dirs=extra, sectors=sectors)
            grounding[r.id] = g
            if g["applicable_regulations"]:
                r.applicable_regulations = g["applicable_regulations"]

    classify_requirements(subset, method=state.get("classifier") or "auto")

    report = state.get("quality_report") or QualityReport()
    fresh = _guarded("quality re-check", lambda: analyse_requirements(subset), None)
    if fresh is not None:
        kept = [x for x in report.result_list if x.requirement_id not in changed]
        report = QualityReport(result_list=kept + fresh.result_list)

    sim = _sim()
    conflicts = detect_conflicts(requirements, sim, previous=state.get("conflicts") or [],
                                 changed=changed)
    compliance = analyse_compliance(state["text"], requirements, grounding, sim=sim,
                                    use_rag=use_rag, extra_dirs=extra,
                                    previous=state.get("compliance"), changed=changed)
    security = analyse_security(state["text"], requirements, sim=sim, use_rag=use_rag)
    llm_risks = [r for r in (state.get("risk_register") or [])
                 if r.source == "scenario:llm"]
    risks = build_risk_register(state["text"], requirements, quality=report,
                                conflicts=conflicts, compliance=compliance,
                                security=security, grounding=grounding,
                                reuse_llm_risks=llm_risks)
    return {"requirements": requirements, "grounding": grounding,
            "quality_report": report, "conflicts": conflicts,
            "compliance": compliance, "security": security, "risk_register": risks}


# ── Wiring ─────────────────────────────────────────────────────────────

NODES = {
    "conflicts": conflicts_node,
    "compliance": compliance_node,
    "security": security_node,
    "risk": risk_node,
    "clarification": clarification_node,
}


def add_analysis_nodes(graph, after: str, before: str) -> None:
    """Insert the analysis chain between two existing nodes."""
    chain = [after] + ANALYSIS_NODES + [before]
    for name in ANALYSIS_NODES:
        graph.add_node(name, NODES[name])
    for a, b in zip(chain, chain[1:]):
        graph.add_edge(a, b)
