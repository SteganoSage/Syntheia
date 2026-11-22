"""Compliance agent — map requirements to obligations and find the gaps (PS §11).

§11 asks the agent to determine, per requirement: the applicable jurisdiction
and regulation, the control required, the evidence for the mapping, compliance
gaps, mandatory approval or audit checkpoints, and retention and reporting
obligations -- and to make no final legal determination.

Two halves, deliberately different in kind:

  GAPS (deterministic)   The rule pack (`data/compliance/rules.yaml`) says which
                         obligations apply to this project and whether any
                         requirement covers them. This is the half that finds
                         requirements that are MISSING, which nothing upstream
                         can: extraction only finds what the scenario says, and
                         grounding only cites what a requirement already claims.

  MAPPINGS (LLM, gated)  For each grounded requirement, the model reads the
                         retrieved clause and records what it requires and
                         whether the requirement does it. The obligation must be
                         quoted from the clause; a quote that is not there marks
                         the mapping `unclear` rather than letting a paraphrase
                         pass as the regulation's words.

A mapping of `not_applicable` is worth having in its own right: grounding is
similarity, and similarity is not applicability (`src/graph.py` says so). This
is the first place in the pipeline that can catch a citation that is on-topic
but does not govern the requirement.

Jurisdiction, issuer and effective date come from the manifest metadata, never
from the model. `requires_legal_review` is set in code (PS §11: "High-impact
interpretations must be approved by compliance or legal officers").
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from src.agents.rulepack import evaluate_pack, load_rules
from src.models.analysis import (
    ClauseMapping,
    ComplianceReport,
    Obligation,
    RequirementMapping,
    RuleFinding,
    SourceCitation,
)
from src.models.requirement import Requirement
from src.utils.evidence import quotes_source
from src.utils.similarity import Similarity

REPO = Path(__file__).resolve().parents[2]
RULES_PATH = REPO / "data/compliance/rules.yaml"

# One LLM call per mapped requirement. Capped so a 15-requirement scenario on
# the local 3B does not add ten minutes; the rest are still covered by the rule
# pack, which costs nothing.
MAX_MAPPINGS = int(os.getenv("SYNTHEIA_COMPLIANCE_MAX_MAPPINGS", "8"))
# Clauses read per requirement before giving up, and LLM calls in total. The
# total cap bounds runtime. 16 was measured at ~90 s extra on the 8B for one
# extra governing clause found (14 Nov); 10 keeps a run inside 3-5 minutes.
MAX_CANDIDATES = int(os.getenv("SYNTHEIA_COMPLIANCE_MAX_CANDIDATES", "3"))
MAX_MAPPING_CALLS = int(os.getenv("SYNTHEIA_COMPLIANCE_MAX_CALLS", "10"))
CLAUSE_CHARS = 1500

_TRACKED = {"retention", "reporting", "approval", "audit", "consent", "notice",
            "continuity"}
_HIGH_IMPACT = {"retention", "reporting", "consent", "approval"}


@lru_cache(maxsize=1)
def load_compliance_rules():
    return load_rules(RULES_PATH)


# ── Per-requirement mapping ────────────────────────────────────────────

MAPPING_PROMPT = """\
You are a compliance analyst at a bank. You are given one software requirement \
and one regulatory clause that a search engine retrieved for it. The search \
matches on similar words, so the clause is OFTEN about a neighbouring subject \
rather than about this requirement. You are not giving legal advice; a \
compliance officer reviews everything you write.

Work in two steps.

STEP 1 -- does the clause govern this requirement?
- governs: "yes" only if the clause regulates the very activity the requirement \
describes. A clause about notifying customers of KYC updates does not govern a \
requirement to store video recordings, even though both mention KYC records. \
Sharing a topic is not enough.
- **A requirement that does what the clause forbids or restricts IS governed by \
it** -- that is the most important case of all. A clause protecting customer \
data from access governs a requirement that lets staff see that data; answer \
governs "yes", coverage "not_addressed", and say in `missing` that the \
requirement contradicts the clause.
- governs_reason: one sentence saying what the clause regulates and whether that \
is what the requirement does.

STEP 2 -- only matters if it governs:
- obligation: copy the words from the CLAUSE that say what must be done. Copy \
them exactly; do not paraphrase.
- obligation_types: one or more of retention, reporting, consent, notice, \
security_control, audit, approval, continuity, customer_protection, other.
- coverage:
    addressed       the requirement commits the system to what the clause \
requires. It does not have to say HOW -- a requirement is not an implementation.
    partial         it commits to part of it (for example a shorter period, or \
some of the required items)
    not_addressed   the clause requires something the requirement does not \
commit to at all
    not_applicable  use this whenever governs is "no"
- missing: for partial or not_addressed, one sentence on what the requirement \
must add. Otherwise "".
- checkpoint: if the clause requires an approval, sign-off, audit, or a report \
to a regulator, name it in a few words. Otherwise "".
"""


def _clause(citation: dict, statement: str, extra_dirs=None) -> tuple[str, dict]:
    """Full text and metadata of a cited chunk.

    Grounding keeps a 300-character excerpt, which routinely cuts a clause off
    before the obligation. Re-querying is milliseconds, and returns the
    jurisdiction and effective date the grounding record does not carry.
    """
    try:
        from src.rag.retriever import _query_all

        for h in _query_all(statement, 12, None, None, extra_dirs):
            if h["metadata"].get("citation") == citation.get("citation"):
                return h["text"], h["metadata"]
    except Exception:                                       # noqa: BLE001
        pass
    return citation.get("excerpt", ""), {}


def _source(citation: dict, meta: dict, text: str) -> SourceCitation:
    return SourceCitation(
        citation=citation.get("citation", ""), title=citation.get("title", ""),
        clause_id=citation.get("clause_id", ""),
        issuer=meta.get("issuer") or citation.get("issuer", ""),
        jurisdiction=meta.get("jurisdiction", ""),
        effective_date=meta.get("effective_date", ""),
        version=citation.get("version", ""), url=citation.get("url", ""),
        excerpt=text[:600], confidence=float(citation.get("confidence", 0.0)))


def map_requirement(req: Requirement, citation: dict,
                    extra_dirs=None) -> RequirementMapping:
    from src.utils.llm import call_structured

    text, meta = _clause(citation, req.statement, extra_dirs)
    source = _source(citation, meta, text)
    result: ClauseMapping = call_structured(
        system_prompt=MAPPING_PROMPT,
        user_prompt=(f"Requirement {req.id}: {req.statement}\n"
                     f"Acceptance criteria: {req.acceptance_criteria or '(none)'}\n\n"
                     f"Clause ({source.citation}):\n{text[:CLAUSE_CHARS]}"),
        response_model=ClauseMapping,
        temperature=0.0,
    )

    coverage, note = result.coverage, ""
    obligation = result.obligation.strip()
    # Step 1 decides. A model that says the clause does not govern the
    # requirement and then grades coverage anyway is contradicting itself; the
    # first answer is the one the prompt made it think about.
    if result.governs == "no":
        coverage = "not_applicable"
    if coverage != "not_applicable" and not quotes_source(obligation, text):
        # The model's "obligation" is not in the clause. Keep the mapping --
        # the citation is real -- but do not let invented regulatory wording
        # stand, and do not trust a coverage verdict built on it.
        note = "the quoted obligation is not in the clause text; coverage unverified"
        obligation, coverage = "", "unclear"
    if coverage == "not_applicable":
        why = result.governs_reason.strip()
        note = ("the retrieved clause does not appear to govern this requirement "
                "— the grounding citation is doubtful" + (f" ({why})" if why else ""))

    types = sorted(set(result.obligation_types))
    return RequirementMapping(
        requirement_id=req.id, source=source, obligation=obligation,
        obligation_types=types, coverage=coverage,
        missing=result.missing.strip() if coverage in ("partial", "not_addressed") else "",
        checkpoint=result.checkpoint.strip(),
        requires_legal_review=(coverage != "not_applicable" and (
            coverage in ("partial", "not_addressed", "unclear")
            or req.risk_level == "High"
            or bool(set(types) & _HIGH_IMPACT))),
        note=note,
    )


def _mapping_order(requirements: list[Requirement], grounding: dict) -> list[Requirement]:
    """Grounded requirements, most consequential first, for the mapping budget."""
    rank = {"High": 0, "Medium": 1, "Low": 2}
    grounded = [r for r in requirements
                if (grounding.get(r.id) or {}).get("citations")]
    return sorted(grounded, key=lambda r: (
        "Regulatory" not in (r.category or []), rank.get(r.risk_level, 1),
        rank.get(r.priority, 1), r.id))


# ── Obligations and checkpoints ────────────────────────────────────────

def _collect(findings: list[RuleFinding],
             mappings: list[RequirementMapping]) -> tuple[list[Obligation], list[str]]:
    obligations, seen = [], set()
    for f in findings:
        if f.obligation in _TRACKED:
            cite = f.source.citation if f.source else f.reference
            key = (f.obligation, f.rule_id)
            if key not in seen:
                seen.add(key)
                obligations.append(Obligation(kind=f.obligation, text=f.title,
                                              citation=cite, origin=f.rule_id))
    for m in mappings:
        for kind in m.obligation_types:
            if kind in _TRACKED and m.obligation:
                key = (kind, m.source.citation)
                if key not in seen:
                    seen.add(key)
                    obligations.append(Obligation(kind=kind, text=m.obligation,
                                                  citation=m.source.citation,
                                                  origin=m.requirement_id))
    order = ["retention", "reporting", "approval", "audit", "consent", "notice",
             "continuity"]
    obligations.sort(key=lambda o: (order.index(o.kind) if o.kind in order else 99,
                                    o.origin))

    checkpoints = []
    for m in mappings:
        if m.checkpoint:
            checkpoints.append(f"{m.checkpoint} — {m.source.citation} ({m.requirement_id})")
    for f in findings:
        if f.requires_legal_review and f.status != "satisfied":
            checkpoints.append(f"Compliance/legal sign-off needed: {f.title} ({f.rule_id})")
    return obligations, list(dict.fromkeys(checkpoints))


# ── Orchestration ──────────────────────────────────────────────────────

def analyse_compliance(scenario: str, requirements: list[Requirement],
                       grounding: dict | None = None,
                       sim: Similarity | None = None,
                       use_rag: bool = True, use_llm: bool = True,
                       extra_dirs=None,
                       previous: ComplianceReport | None = None,
                       changed: set[str] | None = None) -> ComplianceReport:
    """Rule-pack gaps for the project, plus clause mappings per requirement.

    With `use_rag` off (the ungrounded control arm) the rule pack still runs --
    it is deterministic and owes nothing to retrieval -- but citations are not
    resolved and no clause mappings are made, since there are no clauses.
    """
    grounding = grounding or {}
    findings = evaluate_pack(load_compliance_rules(), scenario, requirements,
                             sim=sim, resolve=use_rag)

    mappings: list[RequirementMapping] = []
    if use_rag:
        keep = {}
        if previous is not None and changed is not None:
            keep = {m.requirement_id: m for m in previous.mappings
                    if m.requirement_id not in changed}
        ordered = _mapping_order(requirements, grounding)
        todo = [r for r in ordered if r.id not in keep][:MAX_MAPPINGS]
        # Read retrieved clauses until one governs each requirement. On 14 Nov
        # the top citation did not govern its requirement in 15 of 16 cases,
        # and some had the right clause second or third -- similarity ranks by
        # topic, the mapper checks applicability.
        #
        # Breadth-first: every requirement's first clause before anyone's
        # second. Depth-first spent the whole budget on the first few
        # requirements and left the rest unread; the budget exists to bound
        # runtime (16 calls added ~90 s on the 8B for one extra find), not to
        # decide which requirements get checked at all.
        track = {r.id: {"first": None, "chosen": None, "checked": 0} for r in todo}
        calls, llm_ok = MAX_MAPPING_CALLS, use_llm
        for rank in range(MAX_CANDIDATES):
            for req in todo:
                t = track[req.id]
                cands = grounding[req.id]["citations"]
                if not llm_ok or calls <= 0 or t["chosen"] or rank >= len(cands):
                    continue
                try:
                    m = map_requirement(req, cands[rank], extra_dirs)
                except Exception as e:                       # noqa: BLE001
                    print(f"  (clause mapping unavailable -- {type(e).__name__}; "
                          f"rule-pack findings only)")
                    llm_ok = False
                    continue
                calls -= 1
                t["checked"] += 1
                m.candidates_checked = t["checked"]
                t["first"] = t["first"] or m
                if m.coverage != "not_applicable":
                    t["chosen"] = m

        for req in ordered:
            if req.id in keep:
                mappings.append(keep[req.id])
                continue
            t = track.get(req.id)
            if not t or t["first"] is None:
                continue
            if t["chosen"] is not None:
                mappings.append(t["chosen"])
                continue
            # Every clause read failed to govern. Keep the record -- it is §6's
            # escalation case: a regulation citation nobody can stand behind.
            # The count says how many were actually read before the budget or
            # the candidates ran out.
            first = t["first"]
            first.candidates_checked = t["checked"]
            first.note = (f"none of the {t['checked']} retrieved clause(s) read governs "
                          f"this requirement — its regulation citation is "
                          f"unsupported. First read: {first.note}")
            mappings.append(first)

    obligations, checkpoints = _collect(findings, mappings)
    return ComplianceReport(rule_findings=findings, mappings=mappings,
                            obligations=obligations, checkpoints=checkpoints)
