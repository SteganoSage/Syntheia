"""Risk-analysis agent — a threat and risk register (PS agent table, §12).

"Assesses business, technical and compliance risks." The register is built
mostly from what the earlier agents already established, so each risk is
traceable to a finding rather than conjured:

    compliance gap / weak    -> Compliance risk
    missing security control -> Security risk
    unprotected sensitive data -> Security risk
    confirmed conflict / duplicate -> Delivery risk
    regulation named but not grounded -> Compliance risk
    Critical/High quality issue -> Delivery risk
    high-risk requirement on an external integration -> Technical risk
    scenario signals (legacy, deadline, small team...) -> project-level risks

Scoring is arithmetic from `data/risk/rubric.yaml`: likelihood x impact on a
5x5 matrix, every number written down with its reason. The LLM's only job is
optional: propose up to three further project-level risks, each of which must
quote the problem statement or it is dropped -- the evidence gate again.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

from src.models.analysis import (
    ComplianceReport,
    ProjectRiskSet,
    RequirementRelation,
    Risk,
    SecurityReport,
)
from src.models.quality import QualityReport
from src.models.requirement import Requirement
from src.utils.evidence import find_terms, quotes_source, sentence_with

REPO = Path(__file__).resolve().parents[2]
RUBRIC_PATH = REPO / "data/risk/rubric.yaml"
MAX_LLM_RISKS = int(os.getenv("SYNTHEIA_MAX_LLM_RISKS", "3"))

_REQ_ID = re.compile(r"\bREQ-\d+\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@lru_cache(maxsize=1)
def load_rubric() -> dict:
    data = yaml.safe_load(RUBRIC_PATH.read_text(encoding="utf-8")) or {}
    for key in ("rating", "severity_impact", "sources", "scenario_signals"):
        if key not in data:
            raise ValueError(f"{RUBRIC_PATH.name}: missing '{key}'")
    return data


def rate(score: int, rubric: dict | None = None) -> str:
    bands = (rubric or load_rubric())["rating"]
    for name in ("Critical", "High", "Medium"):
        if score >= bands[name]:
            return name
    return "Low"


def _risk(rubric, kind: str, title: str, cause: str, *, impact: int | None = None,
          linked=(), fmt: dict | None = None, evidence=(), source: str = "") -> Risk:
    spec = rubric["sources"][kind]
    fmt = {k: (v or "") for k, v in (fmt or {}).items()}
    likelihood = int(spec["likelihood"])
    impact = int(impact if impact is not None else spec.get("impact", 3))
    score = likelihood * impact
    return Risk(
        id="", title=title, category=spec["category"], cause=cause,
        consequence=spec["consequence"].format(**fmt) if fmt else spec["consequence"],
        linked_requirements=sorted(set(linked)), likelihood=likelihood,
        impact=impact, score=score, rating=rate(score, rubric),
        mitigation=(spec["mitigation"].format(**fmt) if fmt else spec["mitigation"]).strip()
                   or "Resolve with the stakeholders.",
        owner=spec["owner"], source=source or kind, evidence=list(evidence))


# ── From the earlier agents' findings ──────────────────────────────────

def risks_from_findings(requirements: list[Requirement],
                        quality: QualityReport | None,
                        conflicts: list[RequirementRelation],
                        compliance: ComplianceReport | None,
                        security: SecurityReport | None,
                        grounding: dict | None,
                        rubric: dict | None = None) -> list[Risk]:
    rubric = rubric or load_rubric()
    sev = rubric["severity_impact"]
    by_id = {r.id: r for r in requirements}
    out: list[Risk] = []

    for f in (compliance.gaps if compliance else []):
        kind = "compliance_weak" if f.status == "weak" else "compliance_gap"
        cite = f.source.citation if f.source else f.reference
        out.append(_risk(
            rubric, kind, f.title,
            cause=(f"{f.detail}." if f.detail else "No requirement covers this obligation.")
                  + (f" ({cite})" if cite else ""),
            impact=sev[f.severity],
            linked=_REQ_ID.findall(" ".join(f.triggered_by)) + f.satisfied_by,
            fmt={"proposed": f.proposed_requirement, "detail": f.detail},
            source=f.rule_id))

    for f in (security.gaps if security else []):
        out.append(_risk(
            rubric, "security_gap", f.title,
            cause=f"No requirement provides this control ({f.basis}"
                  + (f": {f.source.citation}" if f.source else
                     f": {f.reference}" if f.reference else "") + ").",
            impact=sev[f.severity],
            linked=_REQ_ID.findall(" ".join(f.triggered_by)),
            fmt={"proposed": f.proposed_requirement}, source=f.rule_id))

    spec = rubric["sources"]["unprotected_data"]
    for d in (security.data_inventory if security else []):
        if d.protected:
            continue
        impact = spec["impact_high"] if d.sensitivity == "high" else spec["impact_medium"]
        out.append(_risk(
            rubric, "unprotected_data", f"{d.label} handled without stated protection",
            cause=f"{d.label} appears in {', '.join(d.found_in)} and no requirement "
                  f"encrypts, masks or restricts it.",
            impact=impact, linked=[x for x in d.found_in if x.startswith("REQ-")],
            fmt={"label": d.label}, source=f"data:{d.kind}"))

    for c in conflicts or []:
        kind = c.kind if c.kind in ("conflict", "duplicate") else "conflict"
        label = "Contradictory" if kind == "conflict" else "Duplicated"
        out.append(_risk(
            rubric, kind, f"{label} requirements {' / '.join(c.requirement_ids)}",
            cause=c.explanation + ("" if c.confirmed else " (unconfirmed)"),
            linked=c.requirement_ids, evidence=c.evidence,
            fmt={"resolution": c.resolution or "Resolve with the stakeholders."},
            source="conflict"))

    for rid, g in (grounding or {}).items():
        req = by_id.get(rid)
        if req and g.get("needs_review") and req.applicable_regulations:
            out.append(_risk(
                rubric, "ungrounded_regulation",
                f"Unverified regulation on {rid}",
                cause=f"{rid} names {', '.join(req.applicable_regulations)}, which "
                      f"nothing in the knowledge base supports.",
                linked=[rid], source="grounding"))

    for res in (quality.result_list if quality else []):
        for issue in res.issues:
            if issue.severity not in ("Critical", "High"):
                continue
            out.append(_risk(
                rubric, "quality_issue", f"{issue.issue_type} in {res.requirement_id}",
                cause=issue.description, impact=sev[issue.severity],
                linked=[res.requirement_id] + _REQ_ID.findall(issue.description),
                fmt={"suggestion": issue.suggestion}, source="quality"))

    for r in requirements:
        if r.risk_level == "High" and "Integration" in (r.category or []):
            out.append(_risk(
                rubric, "integration_dependency",
                f"High-risk function depends on an external system ({r.id})",
                cause=f"{r.id} is high risk and classified Integration: {r.statement}",
                linked=[r.id], source="requirements"))
    return out


# ── From the scenario ──────────────────────────────────────────────────

def risks_from_scenario(scenario: str, rubric: dict | None = None) -> list[Risk]:
    rubric = rubric or load_rubric()
    out = []
    for sig in rubric["scenario_signals"]:
        quote = ""
        hit = find_terms(scenario, sig.get("terms", []))
        if hit:
            quote = sentence_with(scenario, hit[0])
        elif sig.get("pattern"):
            m = re.search(sig["pattern"], scenario, re.I)
            if m:
                quote = next((s.strip() for s in _SENTENCE.split(scenario)
                              if m.group(0).lower() in s.lower()), m.group(0))
        if not quote:
            continue
        score = int(sig["likelihood"]) * int(sig["impact"])
        out.append(Risk(
            id="", title=sig["title"], category=sig["category"],
            cause=f"The problem statement says: \"{quote}\"",
            consequence=sig["consequence"], likelihood=int(sig["likelihood"]),
            impact=int(sig["impact"]), score=score, rating=rate(score, rubric),
            mitigation=sig["mitigation"], owner=sig["owner"],
            source=f"scenario:{sig['id']}", evidence=[quote]))
    return out


PROJECT_RISK_PROMPT = """\
You are a risk analyst for a bank's software project. Read the problem \
statement and list up to three PROJECT risks it reveals that are not already \
in the list given -- things about the business, the delivery, the people or \
the technology that could make the project fail.

Rules:
- Every risk must rest on words in the problem statement. Put those exact \
words in `evidence`. A risk you cannot quote is not a risk here.
- Do not repeat a risk already listed, in different words.
- If there is nothing more, return an empty list.
"""


def risks_from_llm(scenario: str, known: list[Risk],
                   rubric: dict | None = None) -> list[Risk]:
    """Up to MAX_LLM_RISKS further project risks, each quoted from the scenario."""
    if not scenario.strip() or MAX_LLM_RISKS <= 0:
        return []
    from src.utils.llm import call_structured

    rubric = rubric or load_rubric()
    listed = "\n".join(f"- {r.title}" for r in known) or "(none)"
    result: ProjectRiskSet = call_structured(
        system_prompt=PROJECT_RISK_PROMPT,
        user_prompt=f"Problem statement:\n{scenario}\n\nAlready listed:\n{listed}",
        response_model=ProjectRiskSet, temperature=0.0)

    out, titles = [], {r.title.lower() for r in known}
    for p in result.risks[:MAX_LLM_RISKS]:
        if not quotes_source(p.evidence, scenario) or p.title.lower() in titles:
            continue
        # Model-proposed risks get a middling score on purpose: the model can
        # say a risk exists, but nothing it said earns a likelihood or impact.
        out.append(Risk(
            id="", title=p.title, category=p.category,
            cause=f"The problem statement says: \"{p.evidence}\"",
            consequence=p.consequence, likelihood=3, impact=3, score=9,
            rating=rate(9, rubric), mitigation="Review with the project manager.",
            owner="Project manager", source="scenario:llm", evidence=[p.evidence]))
    return out


# ── Orchestration ──────────────────────────────────────────────────────

def build_risk_register(scenario: str, requirements: list[Requirement],
                        quality: QualityReport | None = None,
                        conflicts: list[RequirementRelation] | None = None,
                        compliance: ComplianceReport | None = None,
                        security: SecurityReport | None = None,
                        grounding: dict | None = None,
                        use_llm: bool = True,
                        reuse_llm_risks: list[Risk] | None = None) -> list[Risk]:
    rubric = load_rubric()
    risks = risks_from_findings(requirements, quality, conflicts or [], compliance,
                                security, grounding, rubric)
    risks += risks_from_scenario(scenario, rubric)

    if reuse_llm_risks is not None:
        # Re-validation after clarification: the scenario has not changed, so
        # neither have the model's scenario-level risks. No second call.
        risks += [r.model_copy() for r in reuse_llm_risks]
    elif use_llm:
        try:
            risks += risks_from_llm(scenario, risks, rubric)
        except Exception as e:                               # noqa: BLE001
            print(f"  (project-risk proposals unavailable -- {type(e).__name__})")

    # One risk per (source, title); highest score first, then by title so the
    # numbering is reproducible.
    unique = {}
    for r in risks:
        unique.setdefault((r.source, r.title), r)
    ordered = sorted(unique.values(), key=lambda r: (-r.score, r.category, r.title))
    for i, r in enumerate(ordered, start=1):
        r.id = f"RISK-{i:03d}"
    return ordered
