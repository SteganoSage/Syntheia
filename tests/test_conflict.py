"""Conflict and duplicate detection: the deterministic signals and the judge's gate.

The judge (an LLM) is replaced by a stub. What is tested is what the code does
with each kind of verdict -- confirmed, untraceable, overruled, unavailable --
because that is where a wrong finding is either caught or let through.
"""

from __future__ import annotations

import pytest

from src.agents import conflict
from src.agents.conflict import (
    candidate_pairs,
    detect_conflicts,
    numeric_clash,
    polarity_clash,
    quantities,
)
from src.models.analysis import PairJudgement
from src.utils.similarity import LexicalSimilarity

SIM = LexicalSimilarity()


# ── Quantities ─────────────────────────────────────────────────────────

def test_durations_are_normalised_to_seconds():
    assert quantities("respond within 2 seconds")["duration"] == {2.0}
    assert quantities("within 500 ms")["duration"] == {0.5}
    assert quantities("retain for 5 years")["duration"] == {5 * 31536000}


def test_money_and_percentages():
    q = quantities("loans above INR 50 lakh, availability 99.9%")
    assert q["money"] == {50 * 1e5}
    assert q["percent"] == {99.9}
    assert quantities("₹2,00,000")["money"] == {200000.0}


def test_same_quantity_in_different_units_is_not_a_clash():
    assert numeric_clash("within 2 seconds", "within 2000 ms") == ""
    assert numeric_clash("within 2 seconds", "within 10 seconds") == "duration"


def test_polarity():
    assert polarity_clash("The system shall store Aadhaar numbers.",
                          "The system shall not store Aadhaar numbers.")
    assert not polarity_clash("shall store", "must store")


# ── Candidates ─────────────────────────────────────────────────────────

def test_a_numeric_disagreement_on_the_same_subject_is_a_candidate(make_requirement):
    a = make_requirement(id="REQ-001",
                         statement="The system shall respond to balance enquiries within 2 seconds.")
    b = make_requirement(id="REQ-002",
                         statement="The system shall respond to balance enquiries within 10 seconds.")
    [c] = candidate_pairs([a, b], SIM)
    assert "rule:numeric:duration" in c["signals"]


def test_unrelated_requirements_are_not_candidates(make_requirement):
    a = make_requirement(id="REQ-001", statement="The system shall respond within 2 seconds.")
    b = make_requirement(id="REQ-002", statement="Customer records are retained for 5 years.")
    assert candidate_pairs([a, b], SIM) == []


# ── The judge's gate ───────────────────────────────────────────────────

@pytest.fixture
def pair(make_requirement):
    return [make_requirement(id="REQ-001", priority="High",
                             statement="The system shall respond to balance enquiries within 2 seconds."),
            make_requirement(id="REQ-002",
                             statement="The system shall respond to balance enquiries within 10 seconds.")]


def _judge(relation, qa="within 2 seconds", qb="within 10 seconds"):
    return lambda a, b: PairJudgement(relation=relation, quote_a=qa, quote_b=qb,
                                      explanation="Two limits for one response time.",
                                      resolution="Ask which limit applies.")


def test_without_the_judge_a_rule_hit_is_a_possible_conflict(pair):
    [f] = detect_conflicts(pair, SIM, use_llm=False)
    assert (f.kind, f.confirmed) == ("conflict", False)
    assert f.severity == "High", "one of the pair is High priority"


def test_a_traceable_verdict_confirms(pair, monkeypatch):
    monkeypatch.setattr(conflict, "judge", _judge("conflict"))
    [f] = detect_conflicts(pair, SIM)
    assert f.confirmed and "llm" in f.detected_by
    assert f.evidence == ["within 2 seconds", "within 10 seconds"]


def test_an_untraceable_verdict_does_not_confirm(pair, monkeypatch):
    monkeypatch.setattr(conflict, "judge", _judge("conflict", qa="within 5 minutes"))
    [f] = detect_conflicts(pair, SIM)
    assert not f.confirmed
    assert "llm:untraceable" in f.detected_by


def test_the_judge_cannot_erase_a_rule_detected_clash(pair, monkeypatch):
    # 14 Nov: the 8B called "within 30 seconds" vs "under 10 seconds" overlap.
    # The numeric clash is code-checkable, so it survives -- unconfirmed.
    monkeypatch.setattr(conflict, "judge", _judge("overlap"))
    [f] = detect_conflicts(pair, SIM)
    assert (f.kind, f.confirmed) == ("conflict", False)
    assert "llm:overlap" in f.detected_by
    assert "The judge read it as overlap" in f.explanation


def test_the_judge_does_settle_a_similarity_only_pair(make_requirement, monkeypatch):
    a = make_requirement(id="REQ-001", statement="The system shall notify customers of every transaction.")
    b = make_requirement(id="REQ-002", statement="The system shall notify customers of every transaction by SMS.")
    monkeypatch.setattr(conflict, "judge", _judge("overlap", qa="notify customers",
                                                  qb="notify customers"))
    assert detect_conflicts([a, b], SIM) == []


def test_a_failing_judge_degrades_to_rules(pair, monkeypatch):
    calls = []

    def boom(a, b):
        calls.append(1)
        raise RuntimeError("no endpoint")

    monkeypatch.setattr(conflict, "judge", boom)
    [f] = detect_conflicts(pair, SIM)
    assert not f.confirmed and len(calls) == 1


def test_unchanged_pairs_keep_their_verdict(pair, monkeypatch):
    monkeypatch.setattr(conflict, "judge", _judge("conflict"))
    first = detect_conflicts(pair, SIM)

    def never(a, b):
        raise AssertionError("an unchanged pair went back to the judge")

    monkeypatch.setattr(conflict, "judge", never)
    again = detect_conflicts(pair, SIM, previous=first, changed={"REQ-099"})
    assert again == first
