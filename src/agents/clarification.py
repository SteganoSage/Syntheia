"""Clarification agent — turn every open finding into a question, and act on the answer.

PS agent table: "detects missing or ambiguous information". §10: "Requirements
that fail these checks should be returned to the clarification agent." §7: the
interview should be adaptive, covering thirteen named topics, and follow up
when an answer is incomplete, vague or inconsistent.

So this agent sits at the END of the analysis, where everything that is wrong
is known, and does two things:

  PLAN    Collect the findings -- conflicts, compliance gaps, quality issues,
          missing security controls, unprotected data, and §7 topics nothing
          has touched -- and phrase each as one question for a stakeholder,
          in words (quoting the requirement, never just its id). Ordered by
          what most needs a human: contradictions first.

  REVISE  Only when a person is at the terminal. The top questions are put in
          one prompt; their free-text reply goes to the LLM, which rewrites the
          affected requirements or adds new ones. Each revision must quote the
          reply that justifies it -- the evidence gate once more -- so the
          model cannot use "the user answered" as licence to invent. Every
          change is recorded (field, before, after, reason, time) for the audit
          trail §20 asks for.

When nobody is there, the plan IS the output: the open-issues list §12 names
as an artefact.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml

from src.models.analysis import (
    ChangeRecord,
    ClarificationQuestion,
    ClarificationResult,
    ComplianceReport,
    RequirementRelation,
    RevisionSet,
    SecurityReport,
)
from src.models.quality import QualityReport
from src.models.requirement import Requirement
from src.utils.evidence import find_terms, normalise, quotes_source

REPO = Path(__file__).resolve().parents[2]
TOPICS_PATH = REPO / "data/clarification/topics.yaml"

MAX_QUESTIONS = int(os.getenv("SYNTHEIA_MAX_CLARIFICATIONS", "15"))
MAX_ASK = int(os.getenv("SYNTHEIA_MAX_ASK", "5"))      # one prompt; more is a form
MAX_NEW = 3                                             # requirements added per reply

# Lower is asked first. A contradiction blocks building anything; an
# uncovered topic is a gap in the conversation, not yet a defect.
PRIORITY = {
    ("conflict", True): 10, ("compliance", "Critical"): 15, ("quality", "Critical"): 18,
    ("compliance", "High"): 20, ("quality", "High"): 22, ("compliance", "weak"): 25,
    ("conflict", False): 30, ("data", "high"): 32, ("security", "High"): 35,
    ("coverage", None): 40, ("quality", "Medium"): 45, ("compliance", "Medium"): 50,
    ("security", "Medium"): 55, ("compliance", "Low"): 60, ("security", "Low"): 60,
}

_ASKABLE_QUALITY = {"Ambiguity", "Incompleteness", "Lack of Testability",
                    "Undefined Terminology", "Missing Source", "Infeasibility",
                    "Unverifiable Requirement", "Conflicting Expectations"}


@lru_cache(maxsize=1)
def load_topics() -> list[dict]:
    data = yaml.safe_load(TOPICS_PATH.read_text(encoding="utf-8")) or {}
    topics = data.get("topics") or []
    for t in topics:
        for key in ("id", "label", "question"):
            if key not in t:
                raise ValueError(f"{TOPICS_PATH.name}: topic missing '{key}'")
    return topics


def uncovered_topics(scenario: str, requirements: list[Requirement],
                     topics: list[dict] | None = None) -> list[dict]:
    """§7 topics that neither the scenario nor any requirement touches."""
    topics = topics if topics is not None else load_topics()
    text = " ".join([scenario] + [r.statement for r in requirements])
    cats = {c for r in requirements for c in (r.category or [])}
    return [t for t in topics
            if not find_terms(text, t.get("terms", []))
            and not cats & set(t.get("categories", []))]


def _quote(text: str, limit: int = 140) -> str:
    text = text.strip().rstrip(".")
    return f"“{text if len(text) <= limit else text[:limit - 1] + '…'}”"


# ── Planning ───────────────────────────────────────────────────────────

def plan_questions(scenario: str, requirements: list[Requirement],
                   quality: QualityReport | None = None,
                   conflicts: list[RequirementRelation] | None = None,
                   compliance: ComplianceReport | None = None,
                   security: SecurityReport | None = None) -> list[ClarificationQuestion]:
    by_id = {r.id: r for r in requirements}
    raw: list[dict] = []

    for c in conflicts or []:
        a, b = (by_id.get(i) for i in c.requirement_ids[:2])
        if not (a and b):
            continue
        if c.kind == "conflict":
            q = (f"These two requirements disagree: {_quote(a.statement)} and "
                 f"{_quote(b.statement)}. Which is right, or how should they be reconciled?")
        else:
            q = (f"{_quote(a.statement)} and {_quote(b.statement)} look like the same "
                 f"requirement. Are they, or what distinguishes them?")
        raw.append(dict(question=q, source="conflict", topic="Consistency",
                        why=c.explanation, ids=c.requirement_ids,
                        priority=PRIORITY[("conflict", c.confirmed)]))

    for f in (compliance.gaps if compliance else []):
        cite = f.source.citation if f.source else f.reference
        if f.status == "weak":
            q = f"{f.title}: {f.detail}. Should this be corrected?"
            pr = PRIORITY[("compliance", "weak")]
        else:
            q = (f"{f.title} ({cite}). Nothing in the requirements covers this. "
                 f"Should the system include: {_quote(f.proposed_requirement, 220)}?")
            pr = PRIORITY.get(("compliance", f.severity), 50)
        raw.append(dict(question=q, source="compliance", topic="Regulatory and security constraints",
                        why=f"{f.rule_id}: triggered by {', '.join(f.triggered_by[:2])}",
                        ids=f.satisfied_by, priority=pr))

    # One question per requirement: its most severe askable issue.
    sev_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    for res in (quality.result_list if quality else []):
        issues = sorted((i for i in res.issues if i.issue_type in _ASKABLE_QUALITY
                         and i.severity != "Low"), key=lambda i: sev_rank[i.severity])
        req = by_id.get(res.requirement_id)
        if not issues or not req:
            continue
        i = issues[0]
        q = f"About {_quote(req.statement)}: {i.description.strip()}"
        if i.suggestion:
            q += f" (For example: {i.suggestion.strip().rstrip('.')}.)"
        raw.append(dict(question=q, source="quality", topic=i.issue_type,
                        why=f"{i.issue_type} ({i.severity})", ids=[req.id],
                        priority=PRIORITY.get(("quality", i.severity), 45)))

    for f in (security.gaps if security else []):
        q = (f"{f.title}. Should the system include: "
             f"{_quote(f.proposed_requirement, 220)}?")
        raw.append(dict(question=q, source="security", topic="Regulatory and security constraints",
                        why=f"{f.rule_id} ({f.basis})", ids=[],
                        priority=PRIORITY.get(("security", f.severity), 55)))
    for d in (security.data_inventory if security else []):
        if d.protected or d.sensitivity != "high":
            continue
        where = ", ".join(d.found_in[:3])
        raw.append(dict(
            question=(f"{d.label} is handled ({where}) but nothing says how it is "
                      f"protected. Should it be encrypted, masked, or restricted to "
                      f"certain roles?"),
            source="security", topic="Data collection and retention",
            why=f"unprotected {d.kind} data", ids=[x for x in d.found_in if x != "scenario"],
            priority=PRIORITY[("data", "high")]))

    for t in uncovered_topics(scenario, requirements):
        raw.append(dict(question=t["question"].strip(), source="coverage", topic=t["label"],
                        why=f"§7 topic not covered: {t['label']}", ids=[],
                        priority=PRIORITY[("coverage", None)]))

    seen, out = set(), []
    for item in sorted(raw, key=lambda x: (x["priority"], x["question"])):
        key = normalise(item["question"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ClarificationQuestion(
            id=f"CQ-{len(out) + 1:03d}", question=item["question"], why=item["why"],
            topic=item["topic"], source=item["source"], priority=item["priority"],
            requirement_ids=list(item["ids"])))
        if len(out) >= MAX_QUESTIONS:
            break
    return out


# ── Revising from an answer ────────────────────────────────────────────

REVISE_PROMPT = """\
You maintain a list of software requirements for a financial system. A \
stakeholder was asked the questions below and replied. Update the requirements \
to reflect ONLY what the reply actually says.

Rules:
- Revise a requirement only if the reply answers a question about it. Keep its id.
- Use requirement_id "NEW" only when the reply asks for something no \
requirement covers.
- Never invent numbers, roles, systems or regulations the reply does not mention.
- statement: one sentence beginning "The system shall". acceptance_criteria: \
measurable, taken from the reply.
- reason: copy the exact words from the reply that justify the change.
- If the reply changes nothing, return an empty list.
"""


def revise_requirements(requirements: list[Requirement],
                        asked: list[ClarificationQuestion], reply: str) -> RevisionSet:
    from src.utils.llm import call_structured

    linked = {i for q in asked for i in q.requirement_ids}
    shown = [r for r in requirements if r.id in linked] or requirements[:12]
    reqs = "\n".join(f"[{r.id}] {r.statement}\n    acceptance: "
                     f"{r.acceptance_criteria or '(none)'}" for r in shown)
    qs = "\n".join(f"{n}. {q.question}" for n, q in enumerate(asked, start=1))
    return call_structured(
        system_prompt=REVISE_PROMPT,
        user_prompt=(f"Requirements:\n{reqs}\n\nQuestions asked:\n{qs}\n\n"
                     f"Reply:\n\"{reply}\""),
        response_model=RevisionSet, temperature=0.0)


def _next_id(requirements: list[Requirement]) -> int:
    nums = [int(m.group()) for r in requirements if (m := re.search(r"\d+", r.id))]
    return (max(nums) if nums else 0) + 1


def apply_revisions(requirements: list[Requirement], revisions: RevisionSet,
                    reply: str) -> tuple[list[Requirement], list[ChangeRecord], set[str]]:
    """Apply the revisions that quote the reply. Returns (requirements, changes, changed ids)."""
    out = [r.model_copy(deep=True) for r in requirements]
    index = {r.id: i for i, r in enumerate(out)}
    existing = {normalise(r.statement) for r in out}
    now = datetime.now().isoformat(timespec="seconds")
    changes: list[ChangeRecord] = []
    changed: set[str] = set()
    next_num, added = _next_id(requirements), 0

    for rev in revisions.revisions:
        statement = rev.statement.strip()
        if not statement or not quotes_source(rev.reason, reply):
            continue                       # not grounded in what they said
        criteria = rev.acceptance_criteria.strip()

        if rev.requirement_id in index:
            r = out[index[rev.requirement_id]]
            for field, new in (("statement", statement), ("acceptance_criteria", criteria)):
                old = getattr(r, field) or ""
                if new and normalise(new) != normalise(old):
                    setattr(r, field, new)
                    changes.append(ChangeRecord(requirement_id=r.id, field=field,
                                                before=old, after=new,
                                                reason=rev.reason, at=now))
                    changed.add(r.id)
            if r.id in changed:
                # Categories and citations belong to the old wording; the
                # re-validation pass recomputes them. Approval resets: a
                # changed requirement is not the one anybody approved.
                r.category = []
                r.approval_status = "Pending"

        elif rev.requirement_id.strip().upper() == "NEW" and added < MAX_NEW:
            if normalise(statement) in existing:
                continue
            rid = f"REQ-{next_num:03d}"
            next_num += 1
            added += 1
            existing.add(normalise(statement))
            out.append(Requirement(
                id=rid, statement=statement,
                source_stakeholder="Stakeholder (clarification)",
                business_justification=f"Added during clarification: {rev.reason}",
                priority="Medium", dependencies=[], assumptions="",
                acceptance_criteria=criteria, applicable_regulations=[],
                risk_level="Medium", confidence_score=0.7,
                approval_status="Pending", category=[]))
            changes.append(ChangeRecord(requirement_id=rid, field="created", before="",
                                        after=statement, reason=rev.reason, at=now))
            changed.add(rid)
    return out, changes, changed


# ── Orchestration ──────────────────────────────────────────────────────

def clarify(scenario: str, requirements: list[Requirement], *,
            quality: QualityReport | None = None,
            conflicts: list[RequirementRelation] | None = None,
            compliance: ComplianceReport | None = None,
            security: SecurityReport | None = None,
            ask=None) -> tuple[ClarificationResult, list[Requirement], set[str]]:
    """Plan the questions; if someone can answer, ask and revise.

    `ask(questions) -> str` is supplied only when a person is at a terminal.
    Returns the result, the (possibly revised) requirements, and the ids that
    changed -- which the pipeline re-validates.
    """
    questions = plan_questions(scenario, requirements, quality, conflicts,
                               compliance, security)
    result = ClarificationResult(questions=questions)
    if ask is None or not questions:
        return result, requirements, set()

    asked = questions[:MAX_ASK]
    reply = (ask(asked) or "").strip()
    result.rounds = 1
    if not reply:
        for q in asked:
            q.status = "skipped"
        return result, requirements, set()
    for q in asked:
        q.status, q.answer = "answered", reply

    try:
        revisions = revise_requirements(requirements, asked, reply)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (could not revise from the reply -- {type(e).__name__}; "
              f"it is recorded against the questions)")
        return result, requirements, set()

    updated, changes, changed = apply_revisions(requirements, revisions, reply)
    result.changes = changes
    return result, updated, changed


def carry_over(old: ClarificationResult, new: list[ClarificationQuestion]) -> ClarificationResult:
    """After re-validation: re-planned questions keep their earlier answers.

    A question asked and answered that comes back on re-planning means the
    answer did not reach the requirements -- worth saying, not hiding.
    """
    answered = {normalise(q.question): q for q in old.questions if q.status != "open"}
    for q in new:
        prev = answered.get(normalise(q.question))
        if prev:
            q.status, q.answer = prev.status, prev.answer
            if prev.status == "answered":
                q.why += " — answered, but not yet reflected in the requirements"
    return ClarificationResult(questions=new, changes=old.changes, rounds=old.rounds)
