"""Elaboration agent — the requirements a scenario implies but does not state.

Extraction takes what the stakeholder wrote: on a 50-120 word brief that is
5-7 requirements, almost all functional. A business analyst in the first
workshop would add more -- status tracking, error handling, notifications,
response times, administration -- because a system of THIS kind needs them.
PS §8 asks for exactly that: "make implicit requirements explicit".

This is the supervisor's part 1 in its fullest form: the retrieved requirement
documents (precedent RFPs for comparable systems, from the knowledge base) are
given to the model as evidence of how such systems are specified, and it writes
the requirements this project implies.

Every requirement made here is marked `origin: implied` and is reviewable as
such -- in the SRS, an implied requirement says it was derived, not stated. Gates:

  * a near-duplicate of a stated requirement (or of another implied one) is
    dropped -- elaboration must add, never restate
  * a project fact (timeline, team, budget) is dropped, as for extraction
  * regulatory obligations are NOT made here: the compliance rule packs propose
    those with a clause citation, which a model's recollection cannot match
  * at most MAX_IMPLIED, so the list stays reviewable
"""

from __future__ import annotations

import os
import re

from pydantic import BaseModel, Field

from src.models.requirement import ExtractedRequirement, Requirement
from src.utils.similarity import Similarity

MAX_IMPLIED = int(os.getenv("SYNTHEIA_MAX_IMPLIED", "12"))
# A candidate this close to an existing requirement restates it.
DUPLICATE_SIM = {"EmbeddingSimilarity": 0.80, "LexicalSimilarity": 0.55}


# LLM-facing. No min_length, unlike ExtractionDraft: a scenario that already
# states everything legitimately yields nothing more.
class ElaborationDraft(BaseModel):
    requirements: list[ExtractedRequirement] = Field(
        description="Additional requirements the scenario implies but does not state.")


ELABORATE_PROMPT = """\
You are a senior requirements engineer for financial software (banking, \
lending, payments, KYC, insurance). You are given a problem scenario and the \
requirements already extracted from it -- what the stakeholder STATED.

Write the ADDITIONAL requirements a system of this kind needs that the scenario \
implies but does not state: the ones a business analyst would raise in the \
first requirements workshop. Look for gaps in:
  - user actions and the information users need (status tracking, history, \
notifications, cancellation, corrections)
  - exception handling (failed checks, unavailable external services, \
timeouts, rejected applications, retries)
  - administration and operations (configuration of thresholds, user and role \
management, monitoring)
  - performance, availability and capacity, with numbers where the scenario \
gives a basis for them
  - usability and accessibility for the users the scenario names
  - data management (what is stored, how it is corrected or archived)

Rules:
- Do NOT repeat or reword a stated requirement. Add only what is missing.
- Do NOT write regulatory or legal obligations (KYC rules, data protection, \
record retention periods, reporting to regulators) -- those are handled \
separately with citations.
- Do NOT write facts about the project (team, schedule, budget, methodology).
- Each requirement must be specific to THIS system -- it should not read the \
same for every system -- and testable: acceptance_criteria states a measurable \
condition.
- business_justification says what in the scenario implies the requirement.
- One atomic "The system shall ..." statement per requirement.
- If reference material from comparable systems is given, use it as evidence \
of what such systems need; never copy features this scenario does not imply.
- Between 6 and {n} requirements. Fewer only if the scenario truly implies fewer.
"""


def _next_number(reqs: list[Requirement]) -> int:
    nums = [int(m.group()) for r in reqs if (m := re.search(r"\d+", r.id))]
    return (max(nums) if nums else 0) + 1


def elaborate(scenario: str, stated: list[Requirement], sim: Similarity,
              context: list[dict] | None = None) -> tuple[list[Requirement], list[dict]]:
    """Implied requirements to add, and what was rejected (with why)."""
    if not stated or MAX_IMPLIED <= 0:
        return [], []
    from src.agents.project_facts import project_fact
    from src.utils.llm import call_structured

    listing = "\n".join(f"[{r.id}] {r.statement}" for r in stated)
    user = f"Problem scenario:\n{scenario}\n\nStated requirements:\n{listing}"
    if context:
        from src.rag.retriever import format_context

        user = format_context(context) + "\n\n" + user
    draft: ElaborationDraft = call_structured(
        system_prompt=ELABORATE_PROMPT.format(n=MAX_IMPLIED),
        user_prompt=user, response_model=ElaborationDraft, temperature=0.0)

    floor = DUPLICATE_SIM.get(type(sim).__name__, 0.80)
    kept: list[Requirement] = []
    rejected: list[dict] = []
    known = [r.statement for r in stated]
    for cand in draft.requirements:
        statement = " ".join(cand.statement.split())
        if not statement:
            continue
        reason = project_fact(statement)
        if reason:
            rejected.append({"statement": statement, "reason": f"project fact ({reason})"})
            continue
        scores = sim.cross([statement], known)[0] if known else []
        if scores and max(scores) >= floor:
            twin = known[scores.index(max(scores))]
            rejected.append({"statement": statement,
                             "reason": f"restates an existing requirement ({max(scores):.2f}): {twin}"})
            continue
        known.append(statement)
        kept.append(cand)
        if len(kept) >= MAX_IMPLIED:
            break

    start = _next_number(stated)
    out = []
    for i, cand in enumerate(kept):
        data = cand.model_dump()
        data.update(id=f"REQ-{start + i:03d}", statement=" ".join(cand.statement.split()),
                    origin="implied", approval_status="Pending",
                    # A derived requirement is less certain than a stated one,
                    # whatever the model says about itself.
                    confidence_score=min(float(cand.confidence_score), 0.75))
        out.append(Requirement(**data))
    return out, rejected
