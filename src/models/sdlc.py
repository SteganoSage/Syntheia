# Schemas for SDLC recommendation (Phase 2 / deliverable 2).
#
# Split into two groups on purpose:
#
#   LLM-facing   CriterionAnswerSet, CrossCheckRanking, JustificationSet
#                Sent to the model as a decoding grammar, so they stay small --
#                every field name and description is billed as prompt tokens.
#   Assembled    SDLCRecommendation, never sent anywhere. Built in Python from
#                the LLM answers plus the deterministic scoring.
#
# As with requirement.py: docstrings on these classes become the JSON-schema
# `description` and are shipped to the model on every call, so rationale lives
# in comments.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── LLM-facing ─────────────────────────────────────────────────────────

# One framework question, answered from the requirements. `evidence` is what
# makes the answer checkable: the Definition of Done requires the justification
# to reference actual requirements, and an answer with no REQ id behind it
# cannot support that.
# Three states, not two. A framework question the source genuinely does not
# address must be declinable: forced to yes/no the model invents one, and an
# invented answer scores against every model in the table exactly as a real one
# would. Unanswered questions are simply not scored -- `coverage` then reports
# the partial assessment honestly instead of a confident half-measure.
Answer = Literal["yes", "no", "not_stated"]


class CriterionAnswer(BaseModel):
    question_id: str = Field(description="Framework question id, e.g. REQ-1 or PTR-3")
    answer: Answer = Field(
        description="yes, no, or not_stated when the source does not say"
    )
    # No default: a field with one is optional in the JSON schema, and Ollama's
    # grammar only requires what the schema marks required. Asking twice for
    # citations produced none; making the field required produces them, because
    # decoding cannot finish the object without it.
    evidence: list[str] = Field(
        description="What supports this answer: requirement ids like ['REQ-002'], "
                    "or a short quote from the scenario. Empty only when the "
                    "answer is not_stated.",
    )
    reasoning: str = Field(description="One sentence: what decided this answer")

    @property
    def as_bool(self) -> bool | None:
        """True/False for a real answer, None for an abstention."""
        return {"yes": True, "no": False}.get(self.answer)


class CriterionAnswerSet(BaseModel):
    answers: list[CriterionAnswer]


# The model's own ranking, produced WITHOUT seeing the framework scores. Kept
# separate from the scored ranking so the two can disagree visibly: agreement is
# evidence the recommendation is robust, disagreement is worth showing rather
# than averaging away.
class CrossCheckRanking(BaseModel):
    ranked_models: list[str] = Field(
        description="SDLC model names, most to least suitable"
    )
    reasoning: str = Field(description="Two sentences on why the top choice fits")


# Structure taken from the marked answer to CS305 Class Test-1 -- the shape an
# examiner actually rewards: name the model, list the project characteristics
# that favour it, map each advantage to THIS project, say what it costs, and
# close with a one-line signature of the decision.
#
# It is also the fix for a whole class of defect. One free-prose field made the
# model responsible for organising the argument, and it kept getting the
# organisation wrong: burying the winner's reasons under objections, inverting
# the direction of a mismatch, inventing totals. Splitting it into named fields
# removes those decisions from the model -- the order is the schema's, not its.
class Advantage(BaseModel):
    advantage: str = Field(description="The property, 3-6 words: 'Testing planned early'")
    relevance: str = Field(
        description="What it buys THIS project, one sentence, naming a "
                    "requirement or framework question id. A line that would "
                    "read the same for any project is not an answer."
    )


class ModelJustification(BaseModel):
    model: str = Field(description="SDLC model name")
    # NOT generated. Composed in Python from the questions this model actually
    # matched, because the model could not be trusted to select them: asked for
    # "the characteristics that favour this model" it listed Waterfall's own
    # mismatches -- "the requirements are not easily defined", "the requirements
    # cannot be defined early" -- as reasons Waterfall fitted, when Waterfall
    # expects the opposite of both.
    #
    # That is not a phrasing failure, it is a selection failure, and selection
    # is arithmetic we have already done. Third bug of this exact shape.
    why_it_fits: list[str] = Field(default_factory=list, exclude=True)
    advantages: list[Advantage] = Field(
        description="2-4 advantages of this model for this specific project"
    )
    gives_up: str = Field(
        description="One sentence: what this model costs. For a MISMATCH, say "
                    "the model assumes the opposite of what the project needs."
    )


class JustificationSet(BaseModel):
    justifications: list[ModelJustification]
    recommendation_summary: str = Field(
        description="Two or three sentences recommending the top model and "
                    "saying what would have to change for the runner-up to win"
    )
    shortcut: str = Field(
        description="One line, the signature of the decision, in the form "
                    "'trait + trait + trait -> Model'. Example: 'stable "
                    "requirements + as-is conversion + critical payroll system "
                    "+ strong testing requirement + fixed schedule -> V-Model'."
    )


# ── Assembled in Python, not generated ─────────────────────────────────

class ScoredModel(BaseModel):
    model_id: str
    name: str
    score: float
    matched: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    is_derived: bool = False          # true when the column is not the supervisor's


class TableCoverage(BaseModel):
    name: str
    source: str                       # inferred | supplied
    answered: int
    total: int
    assessed: bool


class SDLCRecommendation(BaseModel):
    recommendation: str
    ranking: list[ScoredModel]

    # Every model sharing the top score, when more than one does. `score()`
    # breaks ties by name so a rerun cannot reorder equal results, but a
    # reproducible tie-break is not a finding: across five scenarios two of the
    # recommendations were decided alphabetically (V-Shaped over Waterfall at
    # 10/11, Incremental over Waterfall at 8/11), and one of those happened to
    # be the right answer. Reporting the tie is the difference between a
    # framework that discriminated and a framework that did not.
    tied_with: list[str] = Field(default_factory=list)
    criteria: list[CriterionAnswer]
    justifications: list[ModelJustification] = Field(default_factory=list)
    summary: str = ""
    # The one-line signature of the decision, in the instructor's own form.
    # Worth its own field: it is the sentence a reader remembers and the one an
    # examiner looks for.
    shortcut: str = ""
    coverage: dict[str, TableCoverage] = Field(default_factory=dict)

    # The LLM's independent opinion and whether it agreed. Not used to pick the
    # recommendation -- the framework decides that -- but reported, because a
    # disagreement between the rubric and the model is information the reader
    # should have rather than something to resolve silently.
    cross_check_ranking: list[str] = Field(default_factory=list)
    cross_check_reasoning: str = ""
    cross_check_agrees: bool | None = None

    framework_version: str = ""
    # True when any model in the top 3 has a derived column, so the output can
    # say which parts of the ranking rest on our reading rather than his tables.
    uses_derived_models: bool = False
