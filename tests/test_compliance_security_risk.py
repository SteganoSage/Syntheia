"""Compliance mapping, the sensitive-data inventory, and the risk register.

LLM calls are stubbed; retrieval is switched off (`use_rag=False`) or stubbed,
so every test runs offline in milliseconds.
"""

from __future__ import annotations

import pytest

from src.agents import compliance, risk
from src.agents.compliance import analyse_compliance, map_requirement
from src.agents.risk import build_risk_register, rate, risks_from_scenario
from src.agents.security import analyse_security, data_inventory
from src.models.analysis import (
    ClauseMapping,
    ProjectRisk,
    ProjectRiskSet,
    RequirementRelation,
)

CLAUSE = ("(b) preserve the records pertaining to the identification of the customers "
          "and their addresses for at least five years after the business relationship "
          "is ended;")
CITATION = {"citation": "KYC Master Direction §(b)", "title": "KYC", "confidence": 0.9,
            "excerpt": CLAUSE[:80]}


# ── Compliance ─────────────────────────────────────────────────────────

def test_rule_pack_runs_without_retrieval(make_requirement):
    reqs = [make_requirement(statement="The system shall verify identity via Aadhaar eKYC.")]
    report = analyse_compliance("KYC onboarding portal", reqs, use_rag=False)
    assert report.mappings == [], "no clauses to map without retrieval"
    gap_ids = {g.rule_id for g in report.gaps}
    assert {"CMP-KYC-01", "CMP-AADHAAR-01"} <= gap_ids
    assert all(g.source is None for g in report.rule_findings)


def test_obligations_and_checkpoints_are_collected(make_requirement):
    report = analyse_compliance("KYC onboarding portal",
                                [make_requirement()], use_rag=False)
    kinds = {o.kind for o in report.obligations}
    assert "retention" in kinds
    assert any("sign-off" in c for c in report.checkpoints)


@pytest.fixture
def stub_mapping(monkeypatch):
    monkeypatch.setattr(compliance, "_clause", lambda c, s, e=None: (CLAUSE, {"jurisdiction": "IN"}))

    def install(**fields):
        base = dict(governs="yes", governs_reason="It regulates record retention.",
                    obligation="preserve the records pertaining to the identification "
                               "of the customers", obligation_types=["retention"],
                    coverage="partial", missing="State the five-year period.",
                    checkpoint="")
        base.update(fields)
        monkeypatch.setattr("src.utils.llm.call_structured",
                            lambda **kw: ClauseMapping(**base))
    return install


def test_a_mapping_quoting_the_clause_stands(make_requirement, stub_mapping):
    stub_mapping()
    m = map_requirement(make_requirement(), CITATION)
    assert m.coverage == "partial"
    assert m.source.jurisdiction == "IN", "jurisdiction comes from metadata, not the model"
    assert m.requires_legal_review


def test_an_invented_obligation_is_not_passed_off_as_the_clause(make_requirement, stub_mapping):
    stub_mapping(obligation="banks must encrypt all records with AES-256")
    m = map_requirement(make_requirement(), CITATION)
    assert (m.coverage, m.obligation) == ("unclear", "")
    assert "not in the clause text" in m.note


def test_not_applicable_flags_a_doubtful_citation(make_requirement, stub_mapping):
    stub_mapping(coverage="not_applicable", obligation="", governs="no",
                 governs_reason="It regulates update notices, not storage.")
    m = map_requirement(make_requirement(risk_level="High"), CITATION)
    assert "doubtful" in m.note
    assert not m.requires_legal_review


# ── Security and privacy ───────────────────────────────────────────────

def test_inventory_finds_classes_and_what_protects_them(make_requirement):
    reqs = [make_requirement(id="REQ-001", statement="The system shall mask Aadhaar numbers."),
            make_requirement(id="REQ-002", statement="The system shall validate the PAN.")]
    inv = {d.kind: d for d in data_inventory("Onboard customers with Aadhaar and PAN.", reqs)}
    assert inv["aadhaar"].protected_by == ["REQ-001"]
    assert inv["pan"].found_in == ["scenario", "REQ-002"]
    assert not inv["pan"].protected


def test_a_blanket_protection_covers_every_class(make_requirement):
    reqs = [make_requirement(id="REQ-001", statement="The system shall validate the PAN."),
            make_requirement(id="REQ-002", statement="All personal data shall be encrypted at rest.")]
    inv = {d.kind: d for d in data_inventory("PAN based onboarding", reqs)}
    assert inv["pan"].protected_by == ["REQ-002"]


def test_unprotected_data_sorts_first(make_requirement):
    reqs = [make_requirement(id="REQ-001", statement="Encrypt the mobile number.")]
    inv = data_inventory("Aadhaar and mobile number collected", reqs)
    assert inv[0].kind == "aadhaar" and not inv[0].protected


def test_missing_mfa_is_a_security_gap(make_requirement):
    reqs = [make_requirement(statement="Customers shall log in to the mobile app with a password.")]
    report = analyse_security("A mobile banking app", reqs, use_rag=False)
    assert "SEC-AUTH-01" in {g.rule_id for g in report.gaps}


# ── Risk ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score, rating", [(25, "Critical"), (20, "Critical"),
                                           (12, "High"), (9, "Medium"), (4, "Low")])
def test_rating_bands(score, rating):
    assert rate(score) == rating


def test_scenario_signals_quote_their_sentence():
    text = ("A bank is rewriting its COBOL core in Java. The team has two developers. "
            "It must go live within six months. Fair lending rules apply.")
    by = {r.source: r for r in risks_from_scenario(text)}
    assert "scenario:legacy" in by and "scenario:small_team" in by
    assert by["scenario:schedule"].evidence == ["It must go live within six months."]
    assert all("Fair lending" not in r.cause for r in by.values()), "'fair' is not a deadline"


def test_findings_become_scored_risks(make_requirement):
    reqs = [make_requirement(id="REQ-001", statement="Verify identity via Aadhaar eKYC.",
                             applicable_regulations=["RBI Digital Lending Guidelines"]),
            make_requirement(id="REQ-002", statement="Respond within 2 seconds.")]
    comp = analyse_compliance("KYC onboarding", reqs, use_rag=False)
    sec = analyse_security("KYC onboarding", reqs, use_rag=False)
    conflict = RequirementRelation(kind="conflict", requirement_ids=["REQ-001", "REQ-002"],
                                   similarity=0.7, confirmed=True, detected_by=["llm"],
                                   explanation="x", resolution="Decide.")
    register = build_risk_register(
        "KYC onboarding", reqs, conflicts=[conflict], compliance=comp, security=sec,
        grounding={"REQ-001": {"needs_review": True}}, use_llm=False)

    sources = {r.source for r in register}
    assert {"CMP-AADHAAR-01", "conflict", "grounding", "data:aadhaar"} <= sources
    aadhaar = next(r for r in register if r.source == "CMP-AADHAAR-01")
    assert (aadhaar.category, aadhaar.impact) == ("Compliance", 5)   # Critical rule
    assert [r.id for r in register] == [f"RISK-{i:03d}" for i in range(1, len(register) + 1)]
    assert [r.score for r in register] == sorted((r.score for r in register), reverse=True)


def test_model_proposed_risks_must_quote_the_scenario(monkeypatch):
    scenario = "The vendor contract ends in March and has not been renewed."
    monkeypatch.setattr("src.utils.llm.call_structured", lambda **kw: ProjectRiskSet(risks=[
        ProjectRisk(title="Vendor exit", category="Delivery",
                    evidence="vendor contract ends in March", consequence="Support stops."),
        ProjectRisk(title="Invented", category="Business",
                    evidence="the board has cut the budget", consequence="x"),
    ]))
    register = build_risk_register(scenario, [], use_llm=True)
    titles = {r.title for r in register if r.source == "scenario:llm"}
    assert titles == {"Vendor exit"}


def test_governs_no_overrides_a_coverage_grade(make_requirement, stub_mapping):
    # The model said the clause does not govern the requirement, then graded it
    # anyway. Step 1 wins: a self-contradiction resolves to not_applicable.
    stub_mapping(governs="no", governs_reason="It regulates update notices.",
                 coverage="not_addressed")
    m = map_requirement(make_requirement(), CITATION)
    assert m.coverage == "not_applicable"
    assert "update notices" in m.note
    assert not m.requires_legal_review


# ── Reading past a near-miss citation ──────────────────────────────────

def _verdicts(monkeypatch, *governs):
    """call_structured returns these governs answers in order."""
    seq = iter(governs)
    monkeypatch.setattr(compliance, "_clause", lambda c, s, e=None: (CLAUSE, {}))

    def fake(**kw):
        g = next(seq)
        return ClauseMapping(governs=g, governs_reason="reason",
                             obligation="preserve the records pertaining to the identification of the customers",
                             obligation_types=["retention"],
                             coverage="partial" if g == "yes" else "not_applicable",
                             missing="", checkpoint="")
    monkeypatch.setattr("src.utils.llm.call_structured", fake)


def _grounding(n):
    return {"REQ-001": {"citations": [dict(CITATION, citation=f"clause {i}") for i in range(1, n + 1)]}}


def test_the_next_citation_is_read_when_the_first_does_not_govern(make_requirement, monkeypatch):
    _verdicts(monkeypatch, "no", "yes")
    report = analyse_compliance("x", [make_requirement()], _grounding(3))
    [m] = report.mappings
    assert (m.source.citation, m.coverage, m.candidates_checked) == ("clause 2", "partial", 2)
    assert report.doubtful == []


def test_when_no_citation_governs_it_is_flagged(make_requirement, monkeypatch):
    _verdicts(monkeypatch, "no", "no", "no")
    report = analyse_compliance("x", [make_requirement()], _grounding(3))
    [m] = report.doubtful
    assert m.candidates_checked == 3 and "none of the 3" in m.note


def test_the_call_budget_is_respected_and_partial_reads_are_kept(make_requirement, monkeypatch):
    monkeypatch.setattr(compliance, "MAX_MAPPING_CALLS", 1)
    _verdicts(monkeypatch, "no")
    report = analyse_compliance("x", [make_requirement()], _grounding(3))
    [m] = report.doubtful
    assert m.candidates_checked == 1, "the one read is kept, not lost to the budget"


def test_a_doubtful_citation_reaches_the_review_list(make_requirement, monkeypatch):
    from main import needs_review

    _verdicts(monkeypatch, "no", "no")
    req = make_requirement()
    report = analyse_compliance("x", [req], _grounding(2))
    reasons = needs_review({"requirements": [req], "compliance": report})
    assert any("does not govern them" in r for r in reasons)


def test_budget_is_spent_breadth_first(make_requirement, monkeypatch):
    # Two requirements, two calls, neither first clause governs: each must get
    # its one read rather than the first requirement getting both.
    monkeypatch.setattr(compliance, "MAX_MAPPING_CALLS", 2)
    _verdicts(monkeypatch, "no", "no")
    reqs = [make_requirement(id="REQ-001"), make_requirement(id="REQ-002")]
    grounding = {r.id: _grounding(3)["REQ-001"] for r in reqs}
    report = analyse_compliance("x", reqs, grounding)
    assert sorted(m.requirement_id for m in report.mappings) == ["REQ-001", "REQ-002"]
    assert all(m.candidates_checked == 1 for m in report.mappings)
