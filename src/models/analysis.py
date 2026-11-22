"""Schemas for Phase B: conflict, compliance, security, risk, clarification.

Same convention as requirement.py and sdlc.py: rationale lives in comments,
never in class docstrings. Pydantic ships a docstring as the JSON-schema
`description` on every call, so it is billed as prompt tokens and read by the
model as instruction (problem log 3.1).

The classes marked LLM-FACING are sent to the model as response schemas. Their
fields carry no defaults on purpose: a field with a default is optional in the
generated grammar, and the model then omits it (problem log 3.2).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["Critical", "High", "Medium", "Low"]
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


# ── B1. Conflict and duplicate detection ───────────────────────────────

# LLM-FACING. One pair at a time: a 3B judging eight pairs in one output
# recycles one explanation across all of them, as the SDLC tables did (6.10).
class PairJudgement(BaseModel):
    relation: Literal["duplicate", "conflict", "overlap", "unrelated"] = Field(
        description="duplicate, conflict, overlap or unrelated")
    quote_a: str = Field(description="exact words from requirement A that decide it")
    quote_b: str = Field(description="exact words from requirement B that decide it")
    explanation: str = Field(description="one sentence")
    resolution: str = Field(description="one sentence a business analyst could act on")


class RequirementRelation(BaseModel):
    kind: Literal["duplicate", "conflict"]
    requirement_ids: list[str]
    similarity: float
    # True only when the LLM judged it AND both quotes trace to their statements.
    # A rule-only or similarity-only finding is reported, but as "possible".
    confirmed: bool
    detected_by: list[str]
    explanation: str
    evidence: list[str] = Field(default_factory=list)
    resolution: str = ""
    severity: Severity = "Medium"


# ── Rule packs, shared by compliance (B2) and security (B3) ────────────

class SourceCitation(BaseModel):
    citation: str
    title: str = ""
    clause_id: str = ""
    issuer: str = ""
    jurisdiction: str = ""
    effective_date: str = ""
    version: str = ""
    url: str = ""
    excerpt: str = ""
    confidence: float = 0.0


class RuleFinding(BaseModel):
    rule_id: str
    title: str
    family: str
    # satisfied: a requirement covers it.  gap: nothing does.
    # weak: something covers it, but short of what is required (a 3-year
    # retention against a 5-year rule, or a retention with no period at all).
    status: Literal["satisfied", "gap", "weak"]
    severity: Severity
    basis: Literal["regulation", "practice"] = "regulation"
    obligation: str = ""
    triggered_by: list[str] = Field(default_factory=list)
    satisfied_by: list[str] = Field(default_factory=list)
    method: str = ""
    detail: str = ""
    proposed_requirement: str = ""
    reference: str = ""
    source: SourceCitation | None = None
    requires_legal_review: bool = False
    rule_verified: bool = False


# ── B2. Compliance ─────────────────────────────────────────────────────

ObligationType = Literal["retention", "reporting", "consent", "notice",
                         "security_control", "audit", "approval", "continuity",
                         "customer_protection", "other"]


# LLM-FACING. The model records what a clause requires and whether the
# requirement as written addresses it. It does not make a legal determination
# (PS §11) -- `requires_legal_review` is set in code, not by the model.
#
# `governs` comes FIRST on purpose: fields are generated in order, so the model
# must decide whether the clause is even about this requirement before it
# judges coverage. Without it, every near-miss citation grounding produced was
# judged "not_addressed" -- five of five on the 8B, 14 Nov.
class ClauseMapping(BaseModel):
    governs: Literal["yes", "no"] = Field(
        description="does this clause regulate the activity the requirement describes, "
                    "including when the requirement does what the clause forbids")
    governs_reason: str = Field(description="one sentence: what the clause regulates, and whether that is what the requirement does")
    obligation: str = Field(description="the clause's own words stating what must be done, quoted exactly")
    obligation_types: list[ObligationType] = Field(description="what kind of obligation it is")
    coverage: Literal["addressed", "partial", "not_addressed", "not_applicable"] = Field(
        description="does the requirement as written do what the clause requires")
    missing: str = Field(description="if partial or not_addressed: what the requirement must add, one sentence; otherwise empty")
    checkpoint: str = Field(description="approval, sign-off, audit or report to a regulator the clause requires, a few words; otherwise empty")


class RequirementMapping(BaseModel):
    requirement_id: str
    source: SourceCitation
    obligation: str = ""
    obligation_types: list[str] = Field(default_factory=list)
    coverage: Literal["addressed", "partial", "not_addressed", "not_applicable", "unclear"]
    missing: str = ""
    checkpoint: str = ""
    requires_legal_review: bool = True
    note: str = ""
    # How many retrieved clauses were read before one governed (or all failed).
    candidates_checked: int = 1


class Obligation(BaseModel):
    kind: str                     # retention | reporting | approval | audit | ...
    text: str
    citation: str = ""
    origin: str = ""              # rule id or requirement id it came from


class ComplianceReport(BaseModel):
    rule_findings: list[RuleFinding] = Field(default_factory=list)
    mappings: list[RequirementMapping] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)

    @property
    def gaps(self) -> list[RuleFinding]:
        return [f for f in self.rule_findings if f.status != "satisfied"]

    @property
    def doubtful(self) -> list[RequirementMapping]:
        # Requirements whose retrieved clauses were ALL read and none governs
        # them: the regulation citation grounding attached is unsupported.
        return [m for m in self.mappings if m.coverage == "not_applicable"]


# ── B3. Security and privacy ───────────────────────────────────────────

class DataItem(BaseModel):
    kind: str
    label: str
    sensitivity: Literal["high", "medium"]
    found_in: list[str] = Field(default_factory=list)    # "scenario", "REQ-002"
    protected_by: list[str] = Field(default_factory=list)

    @property
    def protected(self) -> bool:
        return bool(self.protected_by)


class SecurityReport(BaseModel):
    controls: list[RuleFinding] = Field(default_factory=list)
    data_inventory: list[DataItem] = Field(default_factory=list)

    @property
    def gaps(self) -> list[RuleFinding]:
        return [f for f in self.controls if f.status != "satisfied"]


# ── B4. Risk ───────────────────────────────────────────────────────────

RiskCategory = Literal["Business", "Technical", "Compliance", "Security",
                       "Operational", "Delivery"]


class Risk(BaseModel):
    id: str
    title: str
    category: RiskCategory
    cause: str
    consequence: str
    linked_requirements: list[str] = Field(default_factory=list)
    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    score: int
    rating: Severity
    mitigation: str
    owner: str
    source: str                    # which finding produced it
    evidence: list[str] = Field(default_factory=list)


# LLM-FACING. Project-level risks read from the scenario, each tied to a quote.
class ProjectRisk(BaseModel):
    title: str = Field(description="short name of the risk")
    category: RiskCategory = Field(description="risk category")
    evidence: str = Field(description="the exact words from the problem statement that show this risk")
    consequence: str = Field(description="what goes wrong if it happens, one sentence")


class ProjectRiskSet(BaseModel):
    risks: list[ProjectRisk]


# ── B5. Clarification ──────────────────────────────────────────────────

class ClarificationQuestion(BaseModel):
    id: str
    question: str
    why: str
    topic: str
    source: Literal["conflict", "compliance", "security", "quality", "coverage"]
    priority: int                  # lower is asked first
    requirement_ids: list[str] = Field(default_factory=list)
    status: Literal["open", "answered", "skipped"] = "open"
    answer: str = ""


# LLM-FACING. A requirement rewritten in light of the user's answer.
class RequirementRevision(BaseModel):
    requirement_id: str = Field(description="id of the requirement being revised, or NEW for an added one")
    statement: str = Field(description="the revised requirement statement")
    acceptance_criteria: str = Field(description="revised acceptance criteria")
    reason: str = Field(description="the words from the reply that justify this change, quoted exactly")


class RevisionSet(BaseModel):
    revisions: list[RequirementRevision]


class ChangeRecord(BaseModel):
    requirement_id: str
    field: str
    before: str
    after: str
    reason: str
    source: str = "clarification"
    at: str = ""


class ClarificationResult(BaseModel):
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    changes: list[ChangeRecord] = Field(default_factory=list)
    rounds: int = 0

    @property
    def open(self) -> list[ClarificationQuestion]:
        return [q for q in self.questions if q.status == "open"]
