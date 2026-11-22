"""The clarification agent: what it asks, in what order, and what an answer may change.

The revision step is an LLM call and is stubbed. What is tested is the gate:
a revision is applied only when it quotes the person's reply, and every change
is recorded.
"""

from __future__ import annotations

import pytest

from src.agents import clarification
from src.agents.clarification import (
    MAX_NEW,
    apply_revisions,
    carry_over,
    clarify,
    plan_questions,
    uncovered_topics,
)
from src.agents.compliance import analyse_compliance
from src.models.analysis import (
    RequirementRelation,
    RequirementRevision,
    RevisionSet,
)
from src.models.quality import QualityIssue, QualityReport, QualityResult

REPLY = "Responses must come back within 3 seconds, and branch officers approve loans above 10 lakh."


@pytest.fixture
def reqs(make_requirement):
    return [
        make_requirement(id="REQ-001", statement="The system shall respond in real time."),
        make_requirement(id="REQ-002", statement="The system shall respond within 10 seconds."),
        make_requirement(id="REQ-003", statement="The system shall verify identity via Aadhaar eKYC."),
    ]


# ── Planning ───────────────────────────────────────────────────────────

def test_topics_nothing_mentions_become_questions(reqs):
    labels = {t["label"] for t in uncovered_topics("Build a loan portal.", reqs)}
    assert "Schedule and budget" in labels
    assert "Integration with existing systems" in labels


def test_contradictions_are_asked_first_and_in_words(reqs):
    conflict = RequirementRelation(kind="conflict", requirement_ids=["REQ-001", "REQ-002"],
                                   similarity=0.8, confirmed=True, detected_by=["llm"],
                                   explanation="Two response-time limits.")
    quality = QualityReport(result_list=[QualityResult(
        requirement_id="REQ-001", passed=False,
        issues=[QualityIssue(issue_type="Ambiguity", description="'real time' is not quantified.",
                             severity="High", suggestion="State a latency target")])])
    comp = analyse_compliance("KYC onboarding", reqs, use_rag=False)
    qs = plan_questions("KYC onboarding", reqs, quality=quality, conflicts=[conflict],
                        compliance=comp)

    assert qs[0].source == "conflict"
    assert "respond in real time" in qs[0].question, "quote the requirement, not just its id"
    assert [q.priority for q in qs] == sorted(q.priority for q in qs)
    assert any(q.source == "quality" and "For example" in q.question for q in qs)
    assert len({q.question for q in qs}) == len(qs)
    assert [q.id for q in qs][:2] == ["CQ-001", "CQ-002"]


def test_question_count_is_capped(reqs, monkeypatch):
    monkeypatch.setattr(clarification, "MAX_QUESTIONS", 3)
    assert len(plan_questions("x", reqs)) == 3


# ── Applying an answer ─────────────────────────────────────────────────

def _rev(rid, statement, reason, criteria="Measured at the 95th percentile."):
    return RequirementRevision(requirement_id=rid, statement=statement,
                               acceptance_criteria=criteria, reason=reason)


def test_a_revision_quoting_the_reply_is_applied_and_recorded(reqs):
    out, changes, changed = apply_revisions(reqs, RevisionSet(revisions=[
        _rev("REQ-001", "The system shall respond within 3 seconds.",
             "Responses must come back within 3 seconds")]), REPLY)
    r1 = next(r for r in out if r.id == "REQ-001")
    assert r1.statement == "The system shall respond within 3 seconds."
    assert r1.approval_status == "Pending" and r1.category == []
    assert changed == {"REQ-001"}
    assert {c.field for c in changes} == {"statement", "acceptance_criteria"}
    assert changes[0].before == "The system shall respond in real time."
    assert reqs[0].statement == "The system shall respond in real time.", "input not mutated"


def test_a_revision_the_reply_does_not_support_is_ignored(reqs):
    out, changes, changed = apply_revisions(reqs, RevisionSet(revisions=[
        _rev("REQ-001", "The system shall respond within 1 second.",
             "the regulator requires sub-second responses")]), REPLY)
    assert changes == [] and changed == set()


def test_a_new_requirement_gets_the_next_id(reqs):
    out, changes, changed = apply_revisions(reqs, RevisionSet(revisions=[
        _rev("NEW", "The system shall route loans above 10 lakh to a branch officer for approval.",
             "branch officers approve loans above 10 lakh")]), REPLY)
    assert changed == {"REQ-004"}
    assert out[-1].id == "REQ-004" and out[-1].source_stakeholder.startswith("Stakeholder")
    assert changes[0].field == "created"


def test_new_requirements_are_capped_and_deduplicated(reqs):
    revs = [_rev("NEW", f"The system shall do thing {i}.", "branch officers approve loans")
            for i in range(MAX_NEW + 2)]
    revs.append(_rev("NEW", reqs[0].statement, "branch officers approve loans"))
    out, _, changed = apply_revisions(reqs, RevisionSet(revisions=revs), REPLY)
    assert len(changed) == MAX_NEW


# ── The whole exchange ─────────────────────────────────────────────────

def test_no_one_to_ask_means_nothing_changes(reqs):
    result, out, changed = clarify("x", reqs)
    assert result.questions and all(q.status == "open" for q in result.questions)
    assert out is reqs and changed == set()


def test_an_empty_reply_marks_the_asked_questions_skipped(reqs):
    result, _, changed = clarify("x", reqs, ask=lambda qs: "")
    assert changed == set()
    assert {q.status for q in result.questions[:clarification.MAX_ASK]} == {"skipped"}


def test_a_reply_flows_through_to_changes(reqs, monkeypatch):
    monkeypatch.setattr(clarification, "revise_requirements", lambda r, a, reply: RevisionSet(
        revisions=[_rev("REQ-001", "The system shall respond within 3 seconds.",
                        "within 3 seconds")]))
    result, out, changed = clarify("x", reqs, ask=lambda qs: REPLY)
    assert changed == {"REQ-001"} and result.rounds == 1
    assert result.questions[0].status == "answered" and result.questions[0].answer == REPLY


def test_carry_over_keeps_answers_and_says_when_they_did_not_land(reqs):
    result, _, _ = clarify("x", reqs, ask=lambda qs: "")
    asked = result.questions[0]
    replanned = plan_questions("x", reqs)
    merged = carry_over(result, replanned)
    same = next(q for q in merged.questions if q.question == asked.question)
    assert same.status == "skipped"
