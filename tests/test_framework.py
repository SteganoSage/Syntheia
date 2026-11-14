"""The SDLC framework: loading, scoring, ties, coverage, pivotal questions.

Scoring is the one step in the SDLC engine with no model in it, so it is the
one that has to be exactly right. The last test pins what the framework
currently concludes on every scenario: any edit to framework.yaml that changes
a result fails here, and has to be made on purpose.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from src.sdlc.framework import Framework, FrameworkError

FRAMEWORK_YAML = Path(__file__).resolve().parents[1] / "data/sdlc/framework.yaml"


# ── Loading ────────────────────────────────────────────────────────────

def test_loads_twenty_questions_over_seven_models(framework):
    assert len(framework.models) == 7
    assert len(framework.questions) == 20
    assert framework.version == "1.1"


def test_every_question_has_an_answer_for_every_model(framework):
    ids = {m.id for m in framework.models}
    for q in framework.questions:
        assert set(q.expects) == ids, q.id


def test_additions_are_marked_as_ours(framework):
    # Stripping the derived parts must give back the supervisor's 19 x 6.
    derived_models = [m.id for m in framework.models if m.is_derived]
    derived_questions = [q.id for q in framework.questions if q.is_derived]
    assert derived_models == ["agile"]
    assert derived_questions == ["PTR-6"]
    assert len(framework.questions) - len(derived_questions) == 19
    assert len(framework.models) - len(derived_models) == 6


def test_only_funding_is_scenario_only(framework):
    # PTR-4: requirements never testify about budgets. PTR-5 (lifespan) used to
    # be scenario-only too, and was relaxed in 897d8b7 -- what the system IS
    # answers it ("a payroll system"), so a requirement can support it.
    scenario_only = {q.id for q in framework.questions if q.evidence_from == "scenario"}
    assert scenario_only == {"PTR-4"}


def _write_framework(tmp_path, mutate) -> Path:
    data = yaml.safe_load(FRAMEWORK_YAML.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "framework.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_loader_refuses_a_model_missing_an_answer(tmp_path):
    # A model with no answer on a question would be scored on a smaller
    # denominator and could win on coverage rather than fit.
    def drop_one(data):
        del data["tables"][0]["questions"][0]["expects"]["agile"]

    with pytest.raises(FrameworkError, match="no answer for"):
        Framework.load(_write_framework(tmp_path, drop_one))


def test_loader_refuses_an_unknown_model(tmp_path):
    def add_stray(data):
        data["tables"][0]["questions"][0]["expects"]["devops"] = True

    with pytest.raises(FrameworkError, match="unknown model"):
        Framework.load(_write_framework(tmp_path, add_stray))


def test_loader_refuses_duplicate_model_ids(tmp_path):
    def duplicate(data):
        data["models"].append(dict(data["models"][0]))

    with pytest.raises(FrameworkError, match="duplicate"):
        Framework.load(_write_framework(tmp_path, duplicate))


# ── Scoring ────────────────────────────────────────────────────────────

def test_a_model_scores_one_for_each_matching_answer(framework):
    q = framework.question("REQ-1")
    ranked = framework.score({"REQ-1": True})
    for s in ranked:
        assert s.answered == 1
        assert s.score == (1.0 if q.expects[s.model.id] else 0.0)


def test_unanswered_questions_count_for_nobody(framework):
    ranked = framework.score({})
    assert all(s.answered == 0 and s.score == 0.0 for s in ranked)


def test_unknown_question_ids_are_ignored(framework):
    assert all(s.answered == 0 for s in framework.score({"NOPE-9": True}))


def test_ties_are_broken_by_name_so_reruns_are_stable(framework):
    ranked = framework.score({})
    names = [s.model.name for s in ranked]
    assert names == sorted(names)


def test_a_perfect_profile_ranks_its_model_first(framework):
    # Answer every question exactly as V-Shaped expects; it must win outright.
    answers = {q.id: q.expects["v_shaped"] for q in framework.questions}
    ranked = framework.score(answers)
    assert ranked[0].model.id == "v_shaped"
    assert ranked[0].score == 1.0
    assert ranked[1].score < 1.0


# ── Coverage and pivotal questions ─────────────────────────────────────

def test_coverage_reports_unassessed_tables(framework):
    cov = framework.coverage({"REQ-1": True, "REQ-2": False})
    assert cov["requirements"]["answered"] == 2
    assert cov["requirements"]["assessed"] is True
    assert cov["project_team"]["assessed"] is False
    assert cov["user_community"]["total"] == 4


def test_pivotal_only_offers_questions_that_can_move_the_result(framework):
    answers = {q.id: q.expects["waterfall"] for q in framework.questions
               if q.table in ("requirements", "project_type_risk")}
    pivotal = framework.pivotal(answers)
    for d in pivotal:
        assert d["question"].id not in answers
        # Each one must change the winner (or create/break a tie) for some answer.
        assert d["if_yes"] != d["if_no"] or not d["decisive"]


def test_pivotal_offers_weak_answers_too(framework):
    # An answered question with no evidence behind it is a guess, so it is
    # offered for asking -- but only if it could actually change something.
    answers = {"REQ-1": True}
    plain = {d["question"].id for d in framework.pivotal(answers)}
    weak = {d["question"].id for d in framework.pivotal(answers, weak={"REQ-1"})}
    assert "REQ-1" not in plain
    assert plain <= weak


# ── Outcomes on the scenario set, pinned ───────────────────────────────

# What the framework concludes from the EXPECTED answers (perfect inference, no
# LLM), as of 14 Nov 2026. Two of seven pass. This is not a statement that the
# failures are correct -- docs/roadmap.md records why several are not -- it is a
# tripwire: a framework change that alters any line here must update this table
# deliberately, in the same commit, with the reason.
PINNED_OUTCOMES = {
    "bank_migration":          {"Waterfall"},
    "credit_risk":             {"Spiral"},
    "fintech_startup":         {"Prototype"},
    "insurance_claims":        {"Waterfall"},
    "kyc_onboarding":          {"Incremental", "V-Shaped", "Waterfall"},
    "prof_payroll_conversion": {"Waterfall"},
    "prof_wristwatch":         {"Agile", "Prototype"},
}


def _expected_cases():
    ev = importlib.import_module("notebooks.11_eval_sdlc")
    return {c["path"].parent.name: ev.as_bools(c["answers"]) for c in ev.load_expected()}


def test_every_scenario_is_pinned():
    assert set(_expected_cases()) == set(PINNED_OUTCOMES)


@pytest.mark.parametrize("scenario", sorted(PINNED_OUTCOMES))
def test_framework_outcome_is_pinned(framework, scenario):
    ranked = framework.score(_expected_cases()[scenario])
    top = {s.model.name for s in ranked if abs(s.score - ranked[0].score) < 1e-9}
    assert top == PINNED_OUTCOMES[scenario]


def test_expected_answers_use_real_question_ids(framework):
    for name, answers in _expected_cases().items():
        unknown = [q for q in answers if framework.question(q) is None]
        assert not unknown, f"{name}: {unknown}"
