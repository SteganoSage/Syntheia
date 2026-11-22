"""Deterministic rule packs: when does a rule apply, and is it satisfied?

The problem statement asks for "rule-based compliance validation" alongside the
LLM (preamble, §11, §18 "rule engine for compliance and SDLC logic"). This is
that engine, shared by the compliance agent (`data/compliance/rules.yaml`) and
the security and privacy agent (`data/security/controls.yaml`).

Why rules rather than asking the model "what is missing?":

  * A 3B model asked for gaps produces plausible ones -- and different ones
    every run. A rule produces the same gap on the same input, says exactly
    which words triggered it, and can be argued with line by line.
  * The rule carries its own source. Every rule names the document and the
    obligation it rests on, and the citation is resolved against the knowledge
    base at run time -- so a finding cites a clause a compliance officer can
    open, not a regulation the model remembers.
  * The packs are data. Adding an obligation is a YAML edit with its rationale
    beside it; a reviewer can audit the whole list without reading Python.

Each rule:

    applies_when    terms (searched in the scenario and every requirement) and/or
                    requirement categories. Any one is enough.
    satisfied_when  groups of terms. A single requirement must hit every group
                    (AND of ORs), so "retain" in one requirement and "records"
                    in another does not count.
    min_years       optional. A satisfying requirement that states a shorter
                    period, or no period at all, is `weak`, not satisfied.
    source          which document, and the obligation text to find in it.

A semantic fallback catches paraphrases the term lists miss: a requirement
close enough to the rule's proposed requirement counts as satisfying it. It is
reported as `method: semantic` so a reviewer can tell the two apart.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from src.models.analysis import RuleFinding, SourceCitation
from src.models.requirement import Requirement
from src.utils.evidence import find_terms, matches_groups
from src.utils.similarity import Similarity

REPO = Path(__file__).resolve().parents[2]

SEVERITIES = {"Critical", "High", "Medium", "Low"}

# A requirement this close to a rule's proposed requirement satisfies the rule
# even when the term lists miss it. Deliberately high: a false "satisfied"
# hides a gap, which is worse than a false gap a reader can dismiss.
SEMANTIC_SATISFIED = float(os.getenv("SYNTHEIA_RULE_SEMANTIC_SIM", "0.72"))

# A resolved citation below this is treated as not found. Same floor the
# grounding node uses, so "cited" means the same thing everywhere.
MIN_SOURCE_CONFIDENCE = float(os.getenv("SYNTHEIA_MIN_CONFIDENCE", "0.55"))


class RulePackError(ValueError):
    """A rule pack is malformed. Always fatal -- a half-loaded pack hides gaps."""


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    family: str
    severity: str
    basis: str
    obligation: str
    applies_terms: tuple[str, ...]
    applies_categories: tuple[str, ...]
    satisfied_groups: tuple[tuple[str, ...], ...]
    proposed_requirement: str
    rationale: str = ""
    reference: str = ""
    min_years: int | None = None
    source_titles: tuple[str, ...] = ()
    source_query: str = ""
    # Per-rule floor, for a clause that is in the corpus but diluted inside a
    # long chunk. Must be justified in the pack beside the value.
    source_min_confidence: float | None = None
    legal_review: bool = False
    verified: bool = False
    extra: dict = field(default_factory=dict, compare=False, hash=False)


def load_rules(path: Path) -> list[Rule]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rules, seen = [], set()
    for raw in data.get("rules", []):
        rid = raw.get("id", "<no id>")
        for key in ("id", "title", "family", "severity", "applies_when",
                    "satisfied_when", "proposed_requirement"):
            if key not in raw:
                raise RulePackError(f"{path.name}: rule {rid} has no '{key}'")
        if rid in seen:
            raise RulePackError(f"{path.name}: duplicate rule id {rid}")
        seen.add(rid)
        if raw["severity"] not in SEVERITIES:
            raise RulePackError(f"{path.name}: rule {rid} severity "
                                f"{raw['severity']!r} not in {sorted(SEVERITIES)}")
        applies = raw["applies_when"] or {}
        terms = tuple(applies.get("terms", []) or ())
        cats = tuple(applies.get("categories", []) or ())
        if not terms and not cats:
            raise RulePackError(f"{path.name}: rule {rid} can never apply "
                                f"(applies_when has no terms or categories)")
        groups = tuple(tuple(g) for g in (raw["satisfied_when"].get("all_of") or []) if g)
        if not groups:
            raise RulePackError(f"{path.name}: rule {rid} can never be satisfied "
                                f"(satisfied_when.all_of is empty)")
        src = raw.get("source") or {}
        titles = src.get("title_contains") or ()
        if isinstance(titles, str):
            titles = (titles,)
        basis = raw.get("basis", "regulation")
        if basis == "regulation" and not (titles and src.get("query")):
            raise RulePackError(f"{path.name}: rule {rid} claims a regulatory "
                                f"basis but names no source document and query")
        rules.append(Rule(
            id=rid, title=raw["title"], family=raw["family"],
            severity=raw["severity"], basis=basis,
            obligation=raw.get("obligation", ""),
            applies_terms=terms, applies_categories=cats,
            satisfied_groups=groups,
            proposed_requirement=raw["proposed_requirement"].strip(),
            rationale=(raw.get("rationale") or "").strip(),
            reference=raw.get("reference", ""),
            min_years=raw["satisfied_when"].get("min_years"),
            source_titles=tuple(titles), source_query=src.get("query", ""),
            source_min_confidence=src.get("min_confidence"),
            legal_review=bool(raw.get("legal_review", False)),
            verified=bool(raw.get("verified", False)),
        ))
    return rules


# ── Applicability ──────────────────────────────────────────────────────

def _req_text(r: Requirement) -> str:
    return f"{r.statement} {r.acceptance_criteria or ''}"


def applicability(rule: Rule, scenario: str,
                  requirements: list[Requirement]) -> list[str]:
    """Why this rule applies, in words. Empty means it does not apply."""
    why: list[str] = []
    hit = find_terms(scenario, rule.applies_terms)
    if hit:
        why.append(f"'{hit[0]}' in the scenario")
    for r in requirements:
        if len(why) >= 3:
            break
        hit = find_terms(r.statement, rule.applies_terms)
        if hit:
            why.append(f"'{hit[0]}' in {r.id}")
            continue
        cats = set(r.category or []) & set(rule.applies_categories)
        if cats:
            why.append(f"{r.id} is classified {sorted(cats)[0]}")
    return why


# ── Satisfaction ───────────────────────────────────────────────────────

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_YEARS = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
                    r"[\s-]*(?:\(\d+\)\s*)?(?:years?|yrs?)\b", re.I)


def years_stated(text: str) -> list[int]:
    out = []
    for m in _YEARS.finditer(text or ""):
        tok = m.group(1).lower()
        out.append(int(tok) if tok.isdigit() else _NUMBER_WORDS[tok])
    return out


def satisfying(rule: Rule, requirements: list[Requirement],
               sim: Similarity | None = None) -> tuple[list[str], str]:
    """Requirement ids that satisfy the rule, and how they were found."""
    by_terms = [r.id for r in requirements
                if matches_groups(_req_text(r), rule.satisfied_groups)]
    if by_terms:
        return by_terms, "terms"
    if sim is None or not requirements:
        return [], ""
    scores = sim.cross([rule.proposed_requirement],
                       [r.statement for r in requirements])[0]
    close = [r.id for r, s in zip(requirements, scores) if s >= SEMANTIC_SATISFIED]
    return close, ("semantic" if close else "")


def evaluate(rule: Rule, scenario: str, requirements: list[Requirement],
             sim: Similarity | None = None) -> RuleFinding | None:
    """One rule against one project. None when the rule does not apply."""
    why = applicability(rule, scenario, requirements)
    if not why:
        return None

    ids, method = satisfying(rule, requirements, sim)
    status, detail = ("satisfied" if ids else "gap"), ""

    if ids and rule.min_years:
        by_id = {r.id: r for r in requirements}
        stated = [y for rid in ids for y in years_stated(_req_text(by_id[rid]))]
        if not stated:
            status = "weak"
            detail = (f"{', '.join(ids)} covers this but states no period; "
                      f"at least {rule.min_years} years is required")
        elif max(stated) < rule.min_years:
            status = "weak"
            detail = (f"{', '.join(ids)} states {max(stated)} years; "
                      f"at least {rule.min_years} is required")

    return RuleFinding(
        rule_id=rule.id, title=rule.title, family=rule.family,
        status=status, severity=rule.severity,
        basis="practice" if rule.basis == "practice" else "regulation",
        obligation=rule.obligation, triggered_by=why,
        satisfied_by=ids, method=method, detail=detail,
        proposed_requirement=rule.proposed_requirement if status != "satisfied" else "",
        reference=rule.reference,
        # A gap in a binding regulation is an interpretation someone has to own.
        # PS §11: "High-impact interpretations must be approved by compliance or
        # legal officers." The model never decides this; the pack does.
        requires_legal_review=(rule.legal_review
                               or (status != "satisfied" and rule.basis == "regulation"
                                   and rule.severity in ("Critical", "High"))),
        rule_verified=rule.verified,
    )


# ── Source resolution ──────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _resolve(titles: tuple[str, ...], query: str,
             floor: float) -> SourceCitation | None:
    from src.rag.retriever import REGULATORY_TYPES, _query_all, is_available

    if not is_available():
        return None
    wanted = [t.lower() for t in titles]
    for h in _query_all(query, 30):
        m = h["metadata"]
        if m.get("doc_type") not in REGULATORY_TYPES:
            continue
        if wanted and not any(t in m.get("title", "").lower() for t in wanted):
            continue
        if h["score"] < floor:
            return None              # sorted by score: nothing better follows
        return SourceCitation(
            citation=m.get("citation", ""), title=m.get("title", ""),
            clause_id=m.get("clause_id", ""), issuer=m.get("issuer", ""),
            jurisdiction=m.get("jurisdiction", ""),
            effective_date=m.get("effective_date", ""),
            version=m.get("version", ""), url=m.get("url", ""),
            excerpt=h["text"][:600], confidence=round(h["score"], 4),
        )
    return None


def resolve_source(rule: Rule) -> SourceCitation | None:
    """The clause in the knowledge base this rule rests on, or None.

    Resolved live rather than written into the pack: clause ids depend on how
    the chunker split the document, and a citation typed by hand into YAML
    would silently go stale the next time the corpus is rebuilt.
    """
    if not rule.source_query:
        return None
    floor = (rule.source_min_confidence if rule.source_min_confidence is not None
             else MIN_SOURCE_CONFIDENCE)
    try:
        return _resolve(rule.source_titles, rule.source_query, floor)
    except Exception:                                       # noqa: BLE001
        return None


def evaluate_pack(rules: list[Rule], scenario: str,
                  requirements: list[Requirement],
                  sim: Similarity | None = None,
                  resolve: bool = True) -> list[RuleFinding]:
    """Every applicable rule, gaps first, then by severity."""
    from src.models.analysis import SEVERITY_ORDER

    out = []
    for rule in rules:
        f = evaluate(rule, scenario, requirements, sim)
        if f is None:
            continue
        if resolve:
            f.source = resolve_source(rule)
        out.append(f)
    status_order = {"gap": 0, "weak": 1, "satisfied": 2}
    return sorted(out, key=lambda f: (status_order[f.status],
                                      SEVERITY_ORDER[f.severity], f.rule_id))
