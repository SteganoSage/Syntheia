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

from pydantic import BaseModel, Field


# ── LLM-facing ─────────────────────────────────────────────────────────

# One framework question, answered from the requirements. `evidence` is what
# makes the answer checkable: the Definition of Done requires the justification
# to reference actual requirements, and an answer with no REQ id behind it
# cannot support that.
class CriterionAnswer(BaseModel):
    question_id: str = Field(description="Framework question id, e.g. REQ-1 or PTR-3")
    answer: bool = Field(description="true for yes, false for no")
    # No default: a field with one is optional in the JSON schema, and Ollama's
    # grammar only requires what the schema marks required. Asking twice for
    # citations produced none; making the field required produces them, because
    # decoding cannot finish the object without it.
    evidence: list[str] = Field(
        description="Requirement ids supporting this answer, e.g. ['REQ-002']. "
                    "At least one, taken from the requirements shown.",
    )
    reasoning: str = Field(description="One sentence: why the requirements imply this answer")


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


class ModelJustification(BaseModel):
    model: str = Field(description="SDLC model name")
    justification: str = Field(
        description="Why this model suits the project. Cite framework question "
                    "ids (REQ-3, PTR-2) and requirement ids (REQ-002)."
    )


class JustificationSet(BaseModel):
    justifications: list[ModelJustification]
    recommendation_summary: str = Field(
        description="Two or three sentences recommending the top model and "
                    "saying what would have to change for the runner-up to win"
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
    criteria: list[CriterionAnswer]
    justifications: list[ModelJustification] = Field(default_factory=list)
    summary: str = ""
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
