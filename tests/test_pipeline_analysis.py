"""The Phase B nodes in the graph: order, the off switch, degradation, re-validation."""

from __future__ import annotations

import pytest

from src import pipeline_analysis as pa
from src.models.analysis import RequirementRevision, RevisionSet
from src.models.quality import QualityReport
from src.utils.similarity import LexicalSimilarity


def _chain(graph) -> list[str]:
    nxt = {}
    for e in graph.edges:
        assert e.source not in nxt, f"{e.source} branches -- the pipeline is linear"
        nxt[e.source] = e.target
    order, node = [], "__start__"
    while node in nxt:
        node = nxt[node]
        order.append(node)
    return order


def test_analysis_sits_between_quality_and_sdlc():
    from src.graph import build_pipeline

    assert _chain(build_pipeline().get_graph()) == [
        "retrieval", "extraction", "grounding", "classification", "quality",
        "conflicts", "compliance", "security", "risk", "clarification",
        "sdlc", "__end__"]


@pytest.fixture
def offline(monkeypatch):
    """Lexical similarity, no judge, no project-risk call."""
    monkeypatch.setattr(pa, "_SIM", LexicalSimilarity())

    def no_llm(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr("src.agents.conflict.judge", no_llm)
    monkeypatch.setattr("src.agents.risk.risks_from_llm", lambda *a, **k: [])


@pytest.fixture
def state(make_requirement):
    return {
        "text": "A KYC onboarding portal using Aadhaar. The team has two developers.",
        "requirements": [
            make_requirement(id="REQ-001", statement="The system shall verify identity via Aadhaar eKYC."),
            make_requirement(id="REQ-002", statement="The system shall respond within 2 seconds."),
        ],
        "grounding": {}, "quality_report": None, "use_rag": False,
        "run_analysis": True, "interactive": False, "classifier": "auto",
    }


def test_the_off_switch_skips_every_node(state):
    state["run_analysis"] = False
    for name, fn in pa.NODES.items():
        out = fn(state)
        assert all(v in (None, []) for v in out.values()), name


def test_nodes_produce_their_outputs_offline(state, offline):
    for name in pa.ANALYSIS_NODES:
        state.update(pa.NODES[name](state))
    assert state["compliance"].gaps
    assert any(d.kind == "aadhaar" for d in state["security"].data_inventory)
    assert any(r.source == "scenario:small_team" for r in state["risk_register"])
    assert state["clarifications"].questions
    assert state["clarifications"].changes == [], "nobody was asked"


def test_a_failing_agent_does_not_take_the_run_down(state, monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("pack broken")

    monkeypatch.setattr("src.agents.compliance.analyse_compliance", boom)
    assert pa.compliance_node(state) == {"compliance": None}
    assert "compliance analysis failed" in capsys.readouterr().out


def test_an_answer_triggers_revalidation(state, offline, monkeypatch):
    for name in ["conflicts", "compliance", "security", "risk"]:
        state.update(pa.NODES[name](state))

    state["interactive"] = True
    monkeypatch.setattr("src.cli.clarify_mode.ask_clarifications",
                        lambda qs: "Aadhaar numbers must be masked everywhere.")
    monkeypatch.setattr("src.agents.clarification.revise_requirements",
                        lambda r, a, reply: RevisionSet(revisions=[RequirementRevision(
                            requirement_id="REQ-001",
                            statement="The system shall verify identity via Aadhaar eKYC "
                                      "and mask Aadhaar numbers wherever displayed or stored.",
                            acceptance_criteria="Only the last four digits are ever shown.",
                            reason="Aadhaar numbers must be masked everywhere")]))
    classified = []
    monkeypatch.setattr("src.agents.classification.classify_requirements",
                        lambda reqs, method="auto": classified.extend(r.id for r in reqs) or reqs)
    rechecked = []
    monkeypatch.setattr("src.agents.quality.analyse_requirements",
                        lambda reqs: rechecked.extend(r.id for r in reqs) or QualityReport())

    out = pa.clarification_node(state)
    assert classified == ["REQ-001"], "only the changed requirement is re-classified"
    assert rechecked == ["REQ-001"], "only the changed requirement is re-reviewed"
    gap_ids = {g.rule_id for g in out["compliance"].gaps}
    assert "CMP-AADHAAR-01" not in gap_ids, "the revision now covers Aadhaar masking"
    assert out["clarifications"].changes
    assert "mask" in out["requirements"][0].statement
