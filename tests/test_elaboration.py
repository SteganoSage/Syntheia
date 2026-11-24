"""Phase C pipeline changes: implied requirements, batched quality, stated-only SDLC.

The elaboration agent is the one that adds requirements the stakeholder never
wrote, so what it is NOT allowed to add is the thing worth pinning down: project
facts, restatements, unbounded lists, and confidence it has not earned.
"""

from __future__ import annotations

import pytest

from src.agents import elaboration
from src.agents.elaboration import ElaborationDraft, elaborate
from src.models.requirement import ExtractedRequirement
from src.utils.similarity import LexicalSimilarity

SCENARIO = ("A neo-bank needs a digital onboarding platform. Customers upload PAN and Aadhaar; "
            "PAN is validated against NSDL. The team has two developers.")


def _cand(statement, confidence=0.95):
    return ExtractedRequirement(
        id="X", statement=statement, source_stakeholder="Analyst",
        business_justification="Implied by the onboarding flow.", priority="Medium",
        dependencies=[], assumptions="", acceptance_criteria="", applicable_regulations=[],
        risk_level="Medium", confidence_score=confidence)


@pytest.fixture
def stated(make_requirement):
    return [make_requirement(id="REQ-001", statement="The system shall validate PAN against the NSDL database"),
            make_requirement(id="REQ-004", statement="The system shall let customers upload Aadhaar documents")]


def _fake(monkeypatch, cands):
    monkeypatch.setattr("src.utils.llm.call_structured",
                        lambda **kw: ElaborationDraft(requirements=cands))


def test_keeps_new_requirements_and_rejects_facts_and_restatements(monkeypatch, stated):
    _fake(monkeypatch, [
        _cand("The system shall let a customer resume an interrupted onboarding session"),
        _cand("The project team of two developers shall deliver the platform"),
        _cand("The system shall validate the PAN against the NSDL database"),
    ])
    implied, rejected = elaborate(SCENARIO, stated, LexicalSimilarity())
    assert [r.statement for r in implied] == [
        "The system shall let a customer resume an interrupted onboarding session"]
    reasons = [x["reason"] for x in rejected]
    assert reasons[0].startswith("project fact")
    assert reasons[1].startswith("restates an existing requirement")


def test_implied_requirements_are_labelled_numbered_on_and_capped_in_confidence(monkeypatch, stated):
    _fake(monkeypatch, [_cand("The system shall send the customer an SMS when onboarding completes", 0.99),
                        _cand("The system shall lock an account after five failed login attempts", 0.40)])
    implied, _ = elaborate(SCENARIO, stated, LexicalSimilarity())
    assert [r.id for r in implied] == ["REQ-005", "REQ-006"]
    assert {r.origin for r in implied} == {"implied"}
    assert [r.confidence_score for r in implied] == [0.75, 0.40]
    assert all(r.approval_status == "Pending" for r in implied)


def test_candidates_that_repeat_each_other_are_kept_once(monkeypatch, stated):
    _fake(monkeypatch, [_cand("The system shall email the customer a welcome message"),
                        _cand("The system shall email the customer a welcome message.")])
    implied, rejected = elaborate(SCENARIO, stated, LexicalSimilarity())
    assert len(implied) == 1 and len(rejected) == 1


def test_list_is_capped(monkeypatch, stated):
    monkeypatch.setattr(elaboration, "MAX_IMPLIED", 2)
    _fake(monkeypatch, [_cand(s) for s in (
        "The system shall send an SMS once onboarding completes",
        "Customer sessions shall expire after fifteen minutes idle",
        "Operations staff shall export a daily report of rejected applications",
        "Uploaded images must be compressed below two megabytes")])
    implied, _ = elaborate(SCENARIO, stated, LexicalSimilarity())
    assert len(implied) == 2


def test_nothing_stated_means_nothing_implied(monkeypatch):
    monkeypatch.setattr("src.utils.llm.call_structured",
                        lambda **kw: pytest.fail("no call without stated requirements"))
    assert elaborate(SCENARIO, [], LexicalSimilarity()) == ([], [])


# ── graph wiring ───────────────────────────────────────────────────────

def test_elaboration_failure_is_not_fatal(monkeypatch, stated):
    from src import graph

    def boom(*a, **k):
        raise ConnectionError("endpoint gone")

    monkeypatch.setattr("src.agents.elaboration.elaborate", boom)
    monkeypatch.setattr("src.pipeline_analysis._sim", lambda: LexicalSimilarity())
    out = graph.elaboration_node({"text": SCENARIO, "requirements": stated})
    assert "requirements" not in out and out["implied_rejected"] == []


def test_elaboration_off_switch(stated):
    from src import graph

    out = graph.elaboration_node({"text": SCENARIO, "requirements": stated, "run_elaboration": False})
    assert out == {"implied_rejected": []}


def test_sdlc_sees_only_stated_requirements(monkeypatch, make_requirement):
    from src import graph

    seen = {}
    monkeypatch.setattr("src.agents.sdlc.recommend_sdlc",
                        lambda reqs, **k: seen.setdefault("ids", [r.id for r in reqs]))
    reqs = [make_requirement(id="REQ-001"),
            make_requirement(id="REQ-002", origin="implied"),
            make_requirement(id="REQ-003", origin="stated")]
    graph.sdlc_node({"text": SCENARIO, "requirements": reqs, "interactive": False})
    assert seen["ids"] == ["REQ-001", "REQ-003"]


def test_sdlc_falls_back_to_all_when_nothing_is_stated(monkeypatch, make_requirement):
    from src import graph

    seen = {}
    monkeypatch.setattr("src.agents.sdlc.recommend_sdlc",
                        lambda reqs, **k: seen.setdefault("ids", [r.id for r in reqs]))
    graph.sdlc_node({"text": SCENARIO, "interactive": False,
                     "requirements": [make_requirement(id="REQ-009", origin="implied")]})
    assert seen["ids"] == ["REQ-009"]


# ── quality batching ───────────────────────────────────────────────────

def test_quality_runs_in_batches_and_keeps_every_result(monkeypatch, make_requirement):
    from src.agents import quality
    from src.models.quality import QualityReport

    sizes = []

    def fake_batch(reqs):
        sizes.append(len(reqs))
        return QualityReport.model_validate({"result_list": [
            {"requirement_id": r.id, "issues": []} for r in reqs]})

    monkeypatch.setattr(quality, "_analyse_batch", fake_batch)
    reqs = [make_requirement(id=f"REQ-{i:03d}") for i in range(1, 20)]
    report = quality.analyse_requirements(reqs)
    assert sizes == [8, 8, 3]
    assert [r.requirement_id for r in report.result_list] == [r.id for r in reqs]
