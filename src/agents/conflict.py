"""Conflict and duplicate detection across requirements (PS §10, agent table).

Every other check in the pipeline looks at one requirement at a time. This one
looks at pairs, which is the only way to find the three defects §10 names that
are relationships rather than properties: inconsistency, duplication and
conflicting stakeholder expectations.

Three stages, cheapest first, and the LLM only sees what the first two found:

    1. CANDIDATES   pairwise similarity over all statements (embedding, ms)
    2. RULES        deterministic signals inside similar pairs:
                      numeric  -- both state a duration/percentage/amount for
                                  the same subject and the values differ
                      polarity -- one says "shall", the other "shall not"
    3. JUDGE        the LLM classifies each candidate pair, one pair per call,
                    quoting the words from each statement that decide it.

A finding is `confirmed` only when the judge called it AND both quotes trace to
their statements -- the same evidence standard as the SDLC gate. A rule hit
the judge did not confirm is still reported, marked possible, because a 3B
model saying "overlap" is not proof that 2 seconds and 10 seconds agree.

Pairs are judged one at a time on purpose: given a batch, the model reuses one
explanation across unrelated pairs (problem log 6.10).
"""

from __future__ import annotations

import os
import re
from itertools import combinations

from src.models.analysis import PairJudgement, RequirementRelation
from src.models.requirement import Requirement
from src.utils.evidence import quotes_source
from src.utils.similarity import Similarity, content_words

DUP_SIM = float(os.getenv("SYNTHEIA_DUP_SIM", "0.92"))
CANDIDATE_SIM = float(os.getenv("SYNTHEIA_CONFLICT_CANDIDATE_SIM", "0.60"))
# A rule signal still needs the two statements to be about the same thing.
RULE_SIM = float(os.getenv("SYNTHEIA_CONFLICT_RULE_SIM", "0.45"))
MAX_PAIRS = int(os.getenv("SYNTHEIA_CONFLICT_MAX_PAIRS", "8"))


# ── Deterministic signals ──────────────────────────────────────────────

_NEGATION = re.compile(
    r"\b(?:shall|must|will|should|may|can)\s+not\b"
    r"|\b(?:shan|mustn|won|shouldn|can)['’]?t\b"
    r"|\bcannot\b|\bnever\b|\bprohibit\w*|\bforbid\w*|\bnot\s+(?:be\s+)?(?:allowed|permitted)\b",
    re.I)

_UNIT_SECONDS = {
    "millisecond": 0.001, "ms": 0.001, "second": 1, "sec": 1, "minute": 60, "min": 60,
    "hour": 3600, "hr": 3600, "day": 86400, "week": 604800, "month": 2592000,
    "year": 31536000,
}
_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?"
    r"|days?|weeks?|months?|years?)\b", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
_MONEY = re.compile(
    r"(?:₹|\brs\.?|\binr)\s*(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|crores?)?"
    r"|(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|crores?)\b", re.I)
_MONEY_SCALE = {"lakh": 1e5, "crore": 1e7}


def quantities(text: str) -> dict[str, set[float]]:
    """Durations (seconds), percentages and rupee amounts stated in a text."""
    out: dict[str, set[float]] = {}
    for num, unit in _DURATION.findall(text):
        u = unit.lower().rstrip("s") or unit.lower()
        u = {"m": "ms"}.get(u, u)               # "ms".rstrip("s") -> "m"
        out.setdefault("duration", set()).add(float(num) * _UNIT_SECONDS.get(u, 1))
    for num in _PERCENT.findall(text):
        out.setdefault("percent", set()).add(float(num))
    for a, scale_a, b, scale_b in _MONEY.findall(text):
        num, scale = (a, scale_a) if a else (b, scale_b)
        mult = _MONEY_SCALE.get(scale.lower().rstrip("s"), 1) if scale else 1
        out.setdefault("money", set()).add(float(num.replace(",", "")) * mult)
    return out


def numeric_clash(a: str, b: str) -> str:
    """Name of the quantity the two statements disagree on, or ''."""
    qa, qb = quantities(a), quantities(b)
    for family in qa.keys() & qb.keys():
        if not (qa[family] & qb[family]):
            return family
    return ""


def polarity_clash(a: str, b: str) -> bool:
    return bool(_NEGATION.search(a)) != bool(_NEGATION.search(b))


# ── Candidates ─────────────────────────────────────────────────────────

def candidate_pairs(requirements: list[Requirement],
                    sim: Similarity) -> list[dict]:
    """Pairs worth a closer look, most similar first.

    Each: {a, b, similarity, signals: ["similarity" | "rule:numeric:duration" |
    "rule:polarity" | "similarity:duplicate"]}.
    """
    if len(requirements) < 2:
        return []
    texts = [r.statement for r in requirements]
    matrix = sim.cross(texts, texts)
    out = []
    for i, j in combinations(range(len(requirements)), 2):
        a, b, s = requirements[i], requirements[j], matrix[i][j]
        signals = []
        shared = len(content_words(a.statement) & content_words(b.statement))
        if s >= RULE_SIM and shared >= 2:
            family = numeric_clash(a.statement, b.statement)
            if family:
                signals.append(f"rule:numeric:{family}")
            if polarity_clash(a.statement, b.statement) and shared >= 3:
                signals.append("rule:polarity")
        if s >= DUP_SIM:
            signals.append("similarity:duplicate")
        elif s >= CANDIDATE_SIM:
            signals.append("similarity")
        if signals:
            out.append({"a": a, "b": b, "similarity": round(float(s), 4),
                        "signals": signals})
    # Rule hits first (they are the specific claims), then by similarity.
    return sorted(out, key=lambda c: (not any(x.startswith("rule") for x in c["signals"]),
                                      -c["similarity"]))


# ── The judge ──────────────────────────────────────────────────────────

# The examples below must never come from data/eval/conflict_pairs.yaml: a
# prompt that contains the test answers measures nothing. Revised 14 Nov after
# the 8B called 5 of 6 duplicates "overlap" -- the old definition ("building
# one builds the other") read any rewording as a difference.
JUDGE_PROMPT = """\
You compare two software requirements from the same project and decide how \
they relate.

  duplicate  they ask for the same capability. Different wording, synonyms \
("send a password-reset link by email" / "email the user a link to reset \
their password") and small details one of them merely implies ("the \
customer's registered email" / "the customer's email") do NOT make them \
different. Test: if the team built A exactly as written, would B be done too, \
and the other way round? If yes, duplicate.
  conflict   both cannot be satisfied as written: different limits for the same \
quantity, one requires what the other forbids, or behaviours that cannot both hold
  overlap    related, but EACH asks for something the other does not -- a \
different scope, a different subject, or one is a narrower case of the other \
("notify the customer of every transaction" / "notify the customer by SMS of \
transactions above INR 10,000")
  unrelated  about different things

Rules:
- Different numbers for DIFFERENT things are not a conflict. "respond within 2 \
seconds" and "retain records for 5 years" are unrelated.
- A general requirement and a more specific one about the same thing are \
overlap, not conflict.
- Decide duplicate before overlap: overlap needs something one asks for and \
the other does not. Rewording is not that.
- quote_a and quote_b: copy the exact words from A and from B that decide your \
answer.
- resolution: what a business analyst should ask or decide. Empty for overlap \
or unrelated.
"""


def judge(a: Requirement, b: Requirement) -> PairJudgement:
    from src.utils.llm import call_structured

    return call_structured(
        system_prompt=JUDGE_PROMPT,
        user_prompt=f"A [{a.id}]: {a.statement}\n\nB [{b.id}]: {b.statement}",
        response_model=PairJudgement,
        temperature=0.0,
    )


def _severity(kind: str, a: Requirement, b: Requirement) -> str:
    if kind == "duplicate":
        return "Low"
    serious = any(r.priority == "High" or r.risk_level == "High"
                  or "Regulatory" in (r.category or []) for r in (a, b))
    return "High" if serious else "Medium"


def _from_rules(c: dict) -> RequirementRelation | None:
    """What the deterministic signals alone support, unconfirmed."""
    a, b, sig = c["a"], c["b"], c["signals"]
    rule = [s for s in sig if s.startswith("rule")]
    if rule:
        what = ("state different values for " + rule[0].split(":")[-1]
                if rule[0].startswith("rule:numeric") else
                "one requires what the other forbids")
        return RequirementRelation(
            kind="conflict", requirement_ids=[a.id, b.id],
            similarity=c["similarity"], confirmed=False, detected_by=sig,
            explanation=f"Possible conflict: the two statements {what}.",
            resolution="Confirm with the stakeholders which statement is correct.",
            severity=_severity("conflict", a, b))
    if "similarity:duplicate" in sig:
        return RequirementRelation(
            kind="duplicate", requirement_ids=[a.id, b.id],
            similarity=c["similarity"], confirmed=False, detected_by=sig,
            explanation="Possible duplicate: the statements are nearly identical.",
            resolution="Merge them or state what distinguishes them.",
            severity="Low")
    return None


def detect_conflicts(requirements: list[Requirement], sim: Similarity,
                     use_llm: bool = True,
                     previous: list[RequirementRelation] | None = None,
                     changed: set[str] | None = None) -> list[RequirementRelation]:
    """All duplicate and conflict findings across the requirement set.

    `previous` and `changed` support re-validation after clarification: pairs
    where neither requirement changed keep their earlier verdict, so only pairs
    touching a revised requirement go back to the judge.
    """
    reuse = {}
    if previous is not None and changed is not None:
        reuse = {tuple(sorted(r.requirement_ids)): r for r in previous
                 if not set(r.requirement_ids) & changed}

    findings: list[RequirementRelation] = []
    judged = 0
    llm_ok = use_llm
    for c in candidate_pairs(requirements, sim):
        a, b = c["a"], c["b"]
        key = tuple(sorted((a.id, b.id)))
        if key in reuse:
            findings.append(reuse[key])
            continue
        if previous is not None and changed is not None and not ({a.id, b.id} & changed):
            continue                       # unchanged pair with no earlier finding

        verdict = None
        if llm_ok and judged < MAX_PAIRS:
            try:
                verdict = judge(a, b)
                judged += 1
            except Exception as e:                           # noqa: BLE001
                print(f"  (conflict judge unavailable -- {type(e).__name__}; "
                      f"reporting rule-based findings only)")
                llm_ok = False

        traceable = bool(verdict) and quotes_source(verdict.quote_a, a.statement) \
            and quotes_source(verdict.quote_b, b.statement)

        if verdict and traceable:
            if verdict.relation in ("duplicate", "conflict"):
                findings.append(RequirementRelation(
                    kind=verdict.relation, requirement_ids=[a.id, b.id],
                    similarity=c["similarity"], confirmed=True,
                    detected_by=c["signals"] + ["llm"],
                    explanation=verdict.explanation,
                    evidence=[verdict.quote_a, verdict.quote_b],
                    resolution=verdict.resolution,
                    severity=_severity(verdict.relation, a, b)))
                continue
            # The judge says overlap/unrelated. That settles a pair found by
            # similarity alone -- similarity was only ever a guess. It does NOT
            # erase a clash the code detected: "within 30 seconds" vs "under 10
            # seconds" for the same decision is checkable, and on 14 Nov the 8B
            # called exactly that pair "overlap" after a prompt change. A
            # deterministic finding the model disagrees with is reported as
            # possible, with the model's view beside it, never dropped.
            if not any(s.startswith("rule:") for s in c["signals"]):
                continue
            fallback = _from_rules(c)
            if fallback:
                fallback.detected_by = fallback.detected_by + [f"llm:{verdict.relation}"]
                fallback.explanation += (f" The judge read it as {verdict.relation}: "
                                         f"{verdict.explanation}")
                findings.append(fallback)
            continue

        fallback = _from_rules(c)
        if fallback:
            if verdict and not traceable:
                fallback.detected_by = fallback.detected_by + ["llm:untraceable"]
            findings.append(fallback)

    return sorted(findings, key=lambda r: (not r.confirmed, r.kind != "conflict",
                                           -r.similarity))
