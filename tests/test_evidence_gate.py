"""The SDLC evidence gate and the helpers around it.

The gate is what stopped answers being fabricated wholesale on the Project Team
and User Community tables (problem log 6.5, 6.6). It is pure string matching,
so it can be tested exactly.
"""

from __future__ import annotations

from src.agents.sdlc import _gate_on_evidence, _quotes_the_source, relevant_sentences
from src.models.sdlc import CriterionAnswer

SCENARIO = ("A company is rewriting its payroll system. No new functionality will "
            "be added. The project team consists of 2 developers who have never "
            "built ML systems before. Users will give feedback every sprint.")


def _answer(answer="yes", evidence=None, qid="TEAM-2", reasoning="because"):
    return CriterionAnswer(question_id=qid, answer=answer,
                           evidence=evidence if evidence is not None else [],
                           reasoning=reasoning)


# ── CriterionAnswer.as_bool ────────────────────────────────────────────

def test_as_bool_distinguishes_all_three_answers():
    # Problem log 6.8: once `answer` became a Literal, every value was a truthy
    # string, and the justifier was told every answer was "yes".
    assert _answer("yes").as_bool is True
    assert _answer("no").as_bool is False
    assert _answer("not_stated").as_bool is None


# ── _quotes_the_source ─────────────────────────────────────────────────

def test_an_exact_quote_is_traceable():
    assert _quotes_the_source("never built ML systems before", SCENARIO)


def test_quoting_ignores_case_whitespace_and_curly_quotes():
    assert _quotes_the_source("NO NEW   functionality will be added", SCENARIO)
    source = "The team’s budget is approved."
    assert _quotes_the_source("the team's budget is approved", source)


def test_a_faithful_paraphrase_is_traceable():
    # >= 60% of the content words appear in the source.
    assert _quotes_the_source("payroll system rewriting, new functionality not added",
                              SCENARIO)


def test_an_invented_quote_is_not_traceable():
    assert not _quotes_the_source("funding is approved for 18 months", SCENARIO)


def test_an_empty_quote_is_not_traceable():
    assert not _quotes_the_source("", SCENARIO)
    assert not _quotes_the_source("a an of", SCENARIO)   # no content words


# ── _gate_on_evidence ──────────────────────────────────────────────────

def test_an_answer_citing_nothing_is_demoted():
    [a] = _gate_on_evidence([_answer("yes", [])], SCENARIO)
    assert a.answer == "not_stated"
    assert a.evidence == []
    assert a.reasoning.startswith("UNSUPPORTED (claimed yes")


def test_an_answer_citing_an_invented_quote_is_demoted():
    [a] = _gate_on_evidence([_answer("no", ["the team is highly experienced"])],
                            SCENARIO)
    assert a.answer == "not_stated"
    assert "claimed no" in a.reasoning


def test_a_supported_answer_survives_with_only_real_quotes():
    [a] = _gate_on_evidence([_answer("yes", [
        "never built ML systems before",
        "the team is highly experienced",     # invented -- dropped
    ])], SCENARIO)
    assert a.answer == "yes"
    assert a.evidence == ["never built ML systems before"]


def test_evidence_is_capped_at_two():
    quotes = ["payroll system", "No new functionality", "2 developers"]
    [a] = _gate_on_evidence([_answer("yes", quotes)], SCENARIO)
    assert len(a.evidence) == 2


def test_an_abstention_passes_through_untouched():
    original = _answer("not_stated", [], reasoning="not mentioned")
    [a] = _gate_on_evidence([original], SCENARIO)
    assert a.answer == "not_stated"
    assert a.reasoning == "not mentioned"


# ── relevant_sentences ─────────────────────────────────────────────────

def test_team_sentences_are_found():
    found = relevant_sentences(SCENARIO, "project_team")
    assert found == ["The project team consists of 2 developers who have never "
                     "built ML systems before."]


def test_user_sentences_are_found():
    assert relevant_sentences(SCENARIO, "user_community") == [
        "Users will give feedback every sprint."]


def test_a_scenario_silent_on_the_subject_yields_nothing():
    # Nothing to read means no LLM call -- which is how fabricated team answers
    # stopped (the model was asked to read a team out of a brief with none).
    silent = "Build a payments reconciliation engine. It must settle daily."
    assert relevant_sentences(silent, "project_team") == []


def test_unknown_table_returns_the_whole_scenario():
    assert relevant_sentences("One. Two.", "requirements") == ["One. Two."]
