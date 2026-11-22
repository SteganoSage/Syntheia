"""Project facts are dropped from extraction; real requirements never are."""

from __future__ import annotations

import pytest

from src.agents.project_facts import project_fact


@pytest.mark.parametrize("statement, reason", [
    ("The system shall be developed within 18 months", "delivery timeline"),
    ("The system shall be delivered in six months", "delivery timeline"),
    ("The system shall go live by Diwali", "delivery timeline"),
    ("The system shall be built by a team of 3 developers", "team and staffing"),
    ("The development team shall have 15 developers", "team and staffing"),
    ("The system shall be completed within the project budget", "budget and funding"),
    ("The system shall be developed using agile methodologies", "development methodology"),
    ("The system shall not require changes to requirements once signed off", "requirements process"),
    ("The requirements shall not change after sign-off", "requirements process"),
])
def test_project_facts_are_recognised(statement, reason):
    assert project_fact(statement) == reason


@pytest.mark.parametrize("statement", [
    "The system shall allow the compliance team to review flagged cases",
    "The system shall process loan applications within 30 days",
    "The system shall let users set a monthly budget for each spending category",
    "Loans above 10 lakh shall require sign-off from a senior officer",
    "The system shall retain KYC records for 5 years after the relationship ends",
    "The system shall return a loan decision within 10 seconds",
    "The system shall support an agile pricing engine for dynamic offers",
    "The system shall notify the customer within 2 minutes of a transaction",
    "The fraud team shall be alerted to suspicious applications",
])
def test_real_requirements_are_never_dropped(statement):
    assert project_fact(statement) == ""


def test_extraction_drops_project_facts_and_records_them(monkeypatch):
    from src.agents import extraction
    from src.models.requirement import ExtractedRequirement, ExtractionDraft

    def fake(**kw):
        assert kw["temperature"] == 0.0, "extraction must be greedy"
        return ExtractionDraft(requirements=[
            ExtractedRequirement(id="REQ-001", statement="The system shall provide a claims portal"),
            ExtractedRequirement(id="REQ-002", statement="The system shall be developed within 18 months"),
        ])
    monkeypatch.setattr(extraction, "call_structured", fake)
    result = extraction.extract_requirements("An insurer needs a claims portal.")
    assert [r.id for r in result.requirements] == ["REQ-001"]
    assert result.dropped == [{"id": "REQ-002", "reason": "delivery timeline",
                               "statement": "The system shall be developed within 18 months"}]


def test_extraction_never_filters_down_to_nothing(monkeypatch):
    from src.agents import extraction
    from src.models.requirement import ExtractedRequirement, ExtractionDraft

    monkeypatch.setattr(extraction, "call_structured", lambda **kw: ExtractionDraft(requirements=[
        ExtractedRequirement(id="REQ-001", statement="The system shall be developed within 18 months")]))
    result = extraction.extract_requirements("x")
    assert len(result.requirements) == 1 and result.dropped == []
