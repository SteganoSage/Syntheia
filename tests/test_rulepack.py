"""The rule engine behind the compliance and security packs, and term matching.

The packs decide which obligations a project has and whether they are met --
deterministically, so these tests can pin exactly when a rule fires, when it
is satisfied, and when "satisfied" is not good enough (a 3-year retention
against a 5-year rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.agents.rulepack import (
    Rule,
    RulePackError,
    evaluate,
    evaluate_pack,
    load_rules,
    satisfying,
    years_stated,
)
from src.utils.evidence import find_terms, has_term, matches_groups, sentence_with

REPO = Path(__file__).resolve().parents[1]


# ── Term matching ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text, term, hit", [
    ("Data is encrypted at rest", "encrypt*", True),
    ("Encryption keys rotate", "encrypt*", True),
    ("the decrypt step", "encrypt*", False),          # prefix anchors at a word start
    ("multi-factor authentication", "multi factor", True),
    ("multi factor authentication", "multi-factor", True),
    ("multifactor", "multi factor", False),
    ("the login page", "log", False),                  # whole words unless starred
    ("Verify the PAN card", "pan", True),
    ("the company", "pan", False),
    ("Audit Trail of changes", "audit trail", True),
])
def test_has_term(text, term, hit):
    assert has_term(text, term) is hit


def test_find_terms_returns_the_terms_as_written():
    assert find_terms("Retain KYC records", ["retain*", "kyc", "pan"]) == ["retain*", "kyc"]


def test_groups_are_and_of_ors():
    groups = [["retain*", "preserv*"], ["record*"]]
    assert matches_groups("preserve records", groups)
    assert not matches_groups("preserve documents", groups)
    assert not matches_groups("anything", [])          # no groups never matches


def test_sentence_with():
    text = "We need a portal. Applicants upload Aadhaar. It must be fast."
    assert sentence_with(text, "aadhaar") == "Applicants upload Aadhaar."


# ── years_stated ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text, years", [
    ("retain for 5 years", [5]),
    ("retain for five (5) years", [5]),
    ("a 10-year archive", [10]),
    ("kept seven years, reviewed after 2 yrs", [7, 2]),
    ("retained for 90 days", []),
])
def test_years_stated(text, years):
    assert years_stated(text) == years


# ── Rule evaluation ────────────────────────────────────────────────────

def _rule(**kw):
    base = dict(
        id="T-1", title="Records kept five years", family="retention", severity="High",
        basis="regulation", obligation="retention",
        applies_terms=("kyc", "onboard*"), applies_categories=(),
        satisfied_groups=(("retain*", "preserv*", "store*"), ("record*", "document*")),
        proposed_requirement="The system shall retain records for five years.",
        min_years=5)
    base.update(kw)
    return Rule(**base)


def test_a_rule_that_does_not_apply_returns_none(make_requirement):
    assert evaluate(_rule(), "Build a payments reconciliation engine.",
                    [make_requirement()]) is None


def test_applicability_says_why(make_requirement):
    f = evaluate(_rule(), "A KYC onboarding portal.", [make_requirement()])
    assert f.triggered_by[0] == "'kyc' in the scenario"


def test_an_uncovered_rule_is_a_gap_with_a_proposal(make_requirement):
    f = evaluate(_rule(), "A KYC onboarding portal.", [make_requirement()])
    assert f.status == "gap"
    assert f.proposed_requirement.startswith("The system shall retain")
    assert f.requires_legal_review, "a High regulatory gap needs a compliance owner"


def test_all_groups_must_hit_the_same_requirement(make_requirement):
    reqs = [make_requirement(id="REQ-001", statement="The system shall retain data for 6 years."),
            make_requirement(id="REQ-002", statement="Records shall be searchable.")]
    f = evaluate(_rule(), "KYC onboarding", reqs)
    assert f.status == "gap"


def test_a_covering_requirement_with_enough_years_satisfies(make_requirement):
    req = make_requirement(statement="The system shall retain KYC records for seven years.")
    f = evaluate(_rule(), "KYC onboarding", [req])
    assert (f.status, f.satisfied_by, f.method) == ("satisfied", ["REQ-001"], "terms")
    assert f.proposed_requirement == ""
    assert not f.requires_legal_review


def test_too_short_a_period_is_weak(make_requirement):
    req = make_requirement(statement="The system shall store customer documents for 3 years.")
    f = evaluate(_rule(), "KYC onboarding", [req])
    assert f.status == "weak"
    assert "states 3 years; at least 5" in f.detail


def test_no_period_at_all_is_weak(make_requirement):
    req = make_requirement(statement="The system shall store customer documents securely.")
    f = evaluate(_rule(), "KYC onboarding", [req])
    assert f.status == "weak"
    assert "states no period" in f.detail


def test_acceptance_criteria_count_toward_coverage(make_requirement):
    req = make_requirement(statement="The system shall archive onboarding data.",
                           acceptance_criteria="Records preserved for 5 years after closure.")
    assert evaluate(_rule(), "KYC onboarding", [req]).status == "satisfied"


class _FixedSim:
    def __init__(self, value):
        self.value = value

    def cross(self, a, b):
        return [[self.value] * len(b) for _ in a]


def test_a_paraphrase_can_satisfy_semantically(make_requirement):
    req = make_requirement(statement="Customer onboarding evidence is held for five years.")
    ids, method = satisfying(_rule(min_years=None), [req], sim=_FixedSim(0.9))
    assert (ids, method) == (["REQ-001"], "semantic")
    ids, _ = satisfying(_rule(min_years=None), [req], sim=_FixedSim(0.5))
    assert ids == []


def test_category_triggers_a_rule(make_requirement):
    rule = _rule(applies_terms=(), applies_categories=("Privacy",))
    req = make_requirement(category=["Privacy"])
    f = evaluate(rule, "anything", [req])
    assert f is not None and "classified Privacy" in f.triggered_by[0]


def test_pack_puts_gaps_first_and_skips_resolution_when_asked(make_requirement):
    covered = _rule(id="A", severity="Low", min_years=None)
    missing = _rule(id="B", severity="Medium")
    req = make_requirement(statement="The system shall retain KYC records for 8 years.")
    other = _rule(id="C", severity="Critical",
                  satisfied_groups=(("tokeni*",),), min_years=None)
    out = evaluate_pack([covered, missing, other], "KYC onboarding", [req], resolve=False)
    # C is the only gap; A and B are both satisfied (8 years >= 5), Medium before Low.
    assert [(f.rule_id, f.status) for f in out] == [
        ("C", "gap"), ("B", "satisfied"), ("A", "satisfied")]
    assert all(f.source is None for f in out)


# ── The shipped packs ──────────────────────────────────────────────────

@pytest.mark.parametrize("pack", ["data/compliance/rules.yaml", "data/security/controls.yaml"])
def test_shipped_packs_load(pack):
    rules = load_rules(REPO / pack)
    assert rules
    assert len({r.id for r in rules}) == len(rules)
    for r in rules:
        if r.basis == "regulation":
            assert r.source_titles and r.source_query, r.id
        assert not r.verified, f"{r.id} claims verification -- has a person checked it?"


def test_kyc_scenario_raises_the_expected_compliance_gaps(make_requirement):
    rules = load_rules(REPO / "data/compliance/rules.yaml")
    reqs = [make_requirement(id="REQ-001",
                             statement="The system shall verify identity using Aadhaar-based eKYC."),
            make_requirement(id="REQ-002",
                             statement="The system shall retain KYC records for 3 years.")]
    out = {f.rule_id: f for f in evaluate_pack(
        rules, "Digital KYC onboarding for a bank.", reqs, resolve=False)}
    assert out["CMP-AADHAAR-01"].status == "gap"
    assert out["CMP-KYC-01"].status == "weak"
    assert "CMP-FRAUD-01" not in out, "nothing mentions fraud"


# ── Loader strictness ──────────────────────────────────────────────────

def _write(tmp_path, rule):
    p = tmp_path / "pack.yaml"
    p.write_text(yaml.safe_dump({"rules": [rule]}), encoding="utf-8")
    return p


_GOOD = dict(id="X-1", title="t", family="f", severity="High",
             applies_when={"terms": ["kyc"]},
             satisfied_when={"all_of": [["retain*"]]},
             proposed_requirement="p", basis="practice")


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.pop("title"), "has no 'title'"),
    (lambda r: r.update(applies_when={}), "can never apply"),
    (lambda r: r.update(satisfied_when={"all_of": []}), "can never be satisfied"),
    (lambda r: r.update(severity="Severe"), "severity"),
    (lambda r: r.update(basis="regulation"), "names no source"),
])
def test_loader_refuses_a_malformed_rule(tmp_path, mutate, message):
    rule = dict(_GOOD)
    mutate(rule)
    with pytest.raises(RulePackError, match=message):
        load_rules(_write(tmp_path, rule))


def test_loader_accepts_a_wellformed_rule(tmp_path):
    assert load_rules(_write(tmp_path, dict(_GOOD)))[0].id == "X-1"
