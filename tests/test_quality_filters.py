"""The deterministic filters applied to the quality agent's findings.

Each filter drops a category of complaint the model makes reliably and wrongly
(problem log §5). Each test pairs a complaint that must go with a nearby one
that must stay, because the failure mode of a filter is hiding real findings.
"""

from __future__ import annotations

from src.agents.quality import filter_issues
from src.models.quality import QualityIssue, QualityReport, QualityResult


def _issue(issue_type, description, severity="Medium"):
    return QualityIssue(issue_type=issue_type, description=description,
                        severity=severity, suggestion="fix it")


def _run(issues, requirement):
    report = QualityReport(result_list=[
        QualityResult(requirement_id=requirement.id, issues=issues, passed=False)])
    return filter_issues(report, [requirement]).result_list[0]


def test_a_complaint_about_a_citation_is_dropped(make_requirement):
    req = make_requirement(applicable_regulations=[
        "Master Direction on Outsourcing of Information Technology Services §3"])
    result = _run([_issue("Undefined Terminology",
                          "The term 'Master Direction' is not defined")], req)
    assert result.issues == []
    assert result.passed is True, "passed must be re-derived after filtering"


def test_a_machine_readable_reference_is_not_jargon(make_requirement):
    result = _run([_issue("Ambiguity", "RBI/DBR/2015-16/18 is unclear")],
                  make_requirement())
    assert result.issues == []


def test_genuinely_undefined_wording_stays(make_requirement):
    result = _run([_issue("Undefined Terminology",
                          "The term 'claim data' is not defined")], make_requirement())
    assert len(result.issues) == 1
    assert result.passed is False


def test_cross_requirement_labels_need_a_second_requirement(make_requirement):
    # "Inconsistency" with no other requirement named is a wording complaint.
    result = _run([_issue("Inconsistency", "'mobile-responsive' is not defined")],
                  make_requirement())
    assert result.issues[0].issue_type == "Ambiguity"


def test_a_real_cross_requirement_finding_keeps_its_label(make_requirement):
    # Regression: "REQ-004" matched the citation pattern, so every finding that
    # named a second requirement was dropped as a complaint about a citation.
    result = _run([_issue("Inconsistency", "Contradicts REQ-004 on retention period")],
                  make_requirement())
    assert len(result.issues) == 1
    assert result.issues[0].issue_type == "Inconsistency"


def test_a_duplication_finding_survives(make_requirement):
    result = _run([_issue("Duplication", "Duplicates REQ-2")], make_requirement())
    assert [i.issue_type for i in result.issues] == ["Duplication"]


def test_a_citation_complaint_is_still_dropped_when_it_names_a_requirement(make_requirement):
    result = _run([_issue("Ambiguity", "REQ-004 cites RBI/DBR/2015-16/18, which is unclear")],
                  make_requirement())
    assert result.issues == []


def test_vagueness_answered_by_a_number_in_acceptance_is_dropped(make_requirement):
    req = make_requirement(acceptance_criteria="Lookup completes within 2 seconds.")
    result = _run([_issue("Ambiguity", "'real-time' is not quantified")], req)
    assert result.issues == []


def test_a_role_is_not_answered_by_an_unrelated_number(make_requirement):
    # The acceptance criteria quantify the threshold, not who approves.
    req = make_requirement(acceptance_criteria="Applications above 50 lakhs are routed.")
    result = _run([_issue("Ambiguity", "'senior officer' is not defined")], req)
    assert len(result.issues) == 1


def test_an_established_term_is_not_undefined(make_requirement):
    result = _run([_issue("Undefined Terminology",
                          "The term 'Aadhaar-based eKYC' is not defined")],
                  make_requirement())
    assert result.issues == []


def test_failure_handling_is_not_incompleteness(make_requirement):
    result = _run([_issue("Incompleteness",
                          "What happens if the NSDL service is unavailable?")],
                  make_requirement())
    assert result.issues == []


def test_echoed_record_fields_are_dropped(make_requirement):
    result = _run([_issue("Ambiguity", "Assumptions: none are stated")],
                  make_requirement())
    assert result.issues == []
