"""Deterministic facts the SRS states: where each requirement goes, how it is
verified, where it came from, which terms and documents it relies on.

No LLM here. Everything in this module is reproducible from the run and can be
argued with line by line -- the same split the SDLC engine makes between
inference (model) and scoring (code).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
GLOSSARY = REPO / "data/srs/glossary.yaml"
KB_DIR = REPO / "data/knowledge_base"

# ── Which section of the SRS a requirement belongs in ──────────────────
#
# The classifier gives each requirement several of 13 categories (multi-label,
# PS §9), but an SRS states each requirement once, in one section. The primary
# section is chosen by category AND by what the statement actually talks about:
# "respond within 2 seconds" tagged Functional+Performance is a performance
# requirement, "fetch credit scores from CIBIL" tagged Functional+Integration is
# an interface requirement. The other categories are still shown on the block.

SECTION_TITLES = {
    "functional": "Functional requirements",
    "performance": "Performance requirements",
    "user_interfaces": "User interfaces",
    "usability": "Usability",
    "software_interfaces": "Software interfaces",
    "database": "Logical database requirements",
    "compliance": "Standards compliance",
    "security": "Security",
    "availability": "Reliability and availability",
    "maintainability": "Maintainability and operations",
    "other": "Other requirements",
}

_MEASURE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|secs?|minutes?|mins?)\b"
    r"|\bper\s+(?:second|minute)\b|\btps\b|\bconcurren\w*|\bresponse\s+time\b"
    r"|\blatency\b|\bthroughput\b|\b\d[\d,]*\s+(?:transactions|requests|users|payments)\b", re.I)
_AVAIL = re.compile(
    r"\buptime\b|\bavailab\w*|\b\d+(?:\.\d+)?\s*%|\bdisaster\s+recovery\b|\bfailover\b"
    r"|\bbackup\w*|\brto\b|\brpo\b|\bfault[\s-]toleran\w*|\bredundan\w*", re.I)
_SEC = re.compile(
    r"\bencrypt\w*|\bauthenticat\w*|\bauthori[sz]\w*|\baccess\s+control\b|\bmask\w*"
    r"|\bconsent\b|\bpassword\w*|\botp\b|\bmulti[\s-]factor\b|\bmfa\b|\btls\b"
    r"|\baudit\s+trail\w*|\brole[\s-]based\b|\bprivacy\b|\bconfidential\w*|\bsecure\b", re.I)
_DATA = re.compile(
    r"\bstore[sd]?\b|\bstorage\b|\bretain\w*|\bretention\b|\barchiv\w*|\bdatabase\b"
    r"|\brecords?\b|\bdelete[sd]?\b|\bpurg\w*", re.I)
_UI = re.compile(
    r"\buser[\s-]friendly\b|\bscreen\w*|\bmobile[\s-]responsive\b|\bdashboard\w*"
    r"|\bdisplay\w*|\bui\b|\bux\b|\baccessib\w*|\blanguages?\b|\binterface\b", re.I)
_USABLE = re.compile(
    r"\buser[\s-]friendly\b|\beasy\b|\bintuitive\b|\bsimple\b|\baccessib\w*"
    r"|\blanguages?\b|\blearn\w*|\bhelp\b|\bguid\w*|\bvernacular\b|\blow[\s-]literacy\b", re.I)
_INTEG = re.compile(
    r"\bintegrat\w*|\bapis?\b|\binterface\s+with\b|\bfetch\w*\s+.{0,40}\bfrom\b"
    r"|\bconnect\w*\s+to\b|\bnsdl\b|\bcibil\b|\buidai\b|\bnpci\b|\bcore\s+banking\b"
    r"|\bbureaus?\b|\bgateway\w*|\bthird[\s-]party\b|\bexternal\s+system\w*", re.I)


def primary_section(statement: str, categories: list[str]) -> str:
    cats = set(categories or [])
    s = statement or ""
    if "Performance" in cats and _MEASURE.search(s):
        return "performance"
    if "Availability" in cats and _AVAIL.search(s):
        return "availability"
    if cats & {"Security", "Privacy"} and _SEC.search(s):
        return "security"
    if "Integration" in cats and _INTEG.search(s):
        return "software_interfaces"
    if "Data-management" in cats and _DATA.search(s) and "Functional" not in cats:
        return "database"
    if "Usability" in cats and "Functional" not in cats and _USABLE.search(s):
        return "usability"
    if "Usability" in cats and _UI.search(s) and "Functional" not in cats:
        return "user_interfaces"
    if cats & {"Functional", "Audit/Reporting"}:
        return "functional"
    for cat, key in (("Regulatory", "compliance"), ("Data-management", "database"),
                     ("Usability", "user_interfaces"), ("Operational", "maintainability"),
                     ("Security", "security"), ("Privacy", "security"),
                     ("Performance", "performance"), ("Availability", "availability"),
                     ("Integration", "software_interfaces")):
        if cat in cats:
            return key
    # Unclassified (no classifier output) is treated as functional: the SRS
    # still shows it, in the section most requirements belong to.
    return "functional" if not cats else "other"


# ── How a requirement is verified (29148 §6.5.2: inspection, analysis,
#    demonstration, test) ─────────────────────────────────────────────

_METHOD = {
    "functional": "Test",
    "performance": "Test (load / performance test)",
    "availability": "Test (failover and recovery test) and Analysis",
    "security": "Test (security test) and Inspection",
    "software_interfaces": "Test (integration test)",
    "user_interfaces": "Demonstration",
    "usability": "Demonstration (usability test with representative users)",
    "database": "Inspection and Test",
    "compliance": "Inspection (compliance review)",
    "maintainability": "Inspection and Analysis",
    "other": "Analysis",
}


def verification_method(section: str) -> str:
    return _METHOD.get(section, "Test")


# ── Where a requirement came from (PS §12: "linked to its original
#    stakeholder statement") ─────────────────────────────────────────────

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
TRACE_FLOOR = {"EmbeddingSimilarity": 0.45, "LexicalSimilarity": 0.15}


def scenario_sentences(scenario: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(scenario or "") if len(s.strip()) > 3]


def trace_sources(scenario: str, statements: dict[str, str], sim) -> dict[str, tuple[str, float] | None]:
    """The scenario sentence each requirement most plausibly came from.

    Similarity, not proof -- so it is reported with its score, and a
    requirement below the floor is reported as implicit (derived, not stated)
    rather than pinned to the least-bad sentence.
    """
    sents = scenario_sentences(scenario)
    if not sents or not statements:
        return {rid: None for rid in statements}
    ids = list(statements)
    matrix = sim.cross([statements[i] for i in ids], sents)
    floor = TRACE_FLOOR.get(type(sim).__name__, 0.45)
    out = {}
    for rid, row in zip(ids, matrix):
        best = max(range(len(sents)), key=lambda j: row[j])
        out[rid] = (sents[best], round(float(row[best]), 2)) if row[best] >= floor else None
    return out


# ── Definitions, acronyms, abbreviations ───────────────────────────────

@lru_cache(maxsize=1)
def _glossary() -> list[dict]:
    return (yaml.safe_load(GLOSSARY.read_text(encoding="utf-8")) or {}).get("entries", [])


_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,6}(?:-[A-Z0-9]{1,4})?\b")
# Uppercase tokens that are not acronyms a reader needs expanded.
_NOT_ACRONYMS = {"REQ", "RISK", "CQ", "CMP", "SEC", "PRIV", "II", "III", "IV", "VI", "NO",
                 "OK", "ID", "IDS", "AND", "OR", "THE", "TO", "IN", "OF", "IT", "IS", "AS",
                 "A", "I", "Q", "NA", "N", "S", "TBD"}


def glossary_for(text: str) -> tuple[list[dict], list[str]]:
    """Glossary entries occurring in `text`, and unexplained acronyms found in it."""
    from src.utils.evidence import has_term

    found, known = [], set()
    for e in _glossary():
        term = str(e["term"])
        known.add(term.upper())
        hit = (re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text)
               if e.get("kind") == "acronym" else has_term(text, term))
        if hit:
            found.append(e)
    unknown = sorted({a for a in _ACRONYM.findall(text)
                      if a.upper() not in known and a not in _NOT_ACRONYMS
                      and not re.fullmatch(r"[A-Z]+-\d+", a)})
    order = {"term": 0, "acronym": 1}
    found.sort(key=lambda e: (order.get(e.get("kind"), 2), str(e["term"]).lower()))
    return found, unknown


# ── References ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _manifests() -> dict[str, dict]:
    out = {}
    for m in KB_DIR.rglob("*.manifest.yaml"):
        try:
            data = yaml.safe_load(m.read_text(encoding="utf-8")) or {}
        except Exception:                                    # noqa: BLE001
            continue
        if data.get("title"):
            out[data["title"]] = data
    return out


def cited_titles(run: dict) -> list[str]:
    """Every knowledge-base document the run actually cites, in first-seen order."""
    titles: list[str] = []

    def add(t):
        if t and t not in titles:
            titles.append(t)

    for g in (run.get("grounding") or {}).values():
        for c in g.get("citations", []) or []:
            add(c.get("title"))
    comp = run.get("compliance") or {}
    for f in comp.get("rule_findings", []) or []:
        add((f.get("source") or {}).get("title"))
    for m in comp.get("mappings", []) or []:
        if m.get("coverage") != "not_applicable":
            add((m.get("source") or {}).get("title"))
    for f in (run.get("security") or {}).get("controls", []) or []:
        add((f.get("source") or {}).get("title"))
    return titles


def reference_entry(title: str) -> dict:
    """Bibliographic fields for a cited document, from its manifest (§5 metadata)."""
    m = _manifests().get(title, {})
    eff = m.get("effective_date")
    return {
        "title": title,
        "issuer": m.get("issuer", ""),
        "doc_id": "" if m.get("doc_id") in (None, "none") else str(m.get("doc_id")),
        "version": m.get("version", ""),
        "effective": str(eff) if eff else "",
        "jurisdiction": m.get("jurisdiction", ""),
        "type": m.get("doc_type", ""),
        "url": m.get("url", "") or "",
    }


# ── A product name when no model is available ──────────────────────────

_WANTS = re.compile(
    r"\b(?:needs?|wants?|requires?|build|building|develop|developing|launch|create|"
    r"decided\s+to\s+(?:build|replace|migrate|rewrite|develop|launch))\s+"
    r"(?:to\s+(?:build|launch|develop|create)\s+)?(?:the\s+entire|an?|the|its)\s+"
    r"([A-Za-z][\w\s\-/]{3,60}?)"
    r"(?=\s+(?:that|where|which|who|to|with|so|but|and|on|in|from)\b|[.,;:'’])", re.I)
_SMALL = {"for", "of", "and", "the", "a", "an", "to", "in", "on"}


def fallback_product_name(scenario: str) -> str:
    """A name lifted from the scenario's own words. Only used without an LLM."""
    m = _WANTS.search(scenario or "")
    if not m:
        return "Proposed System"
    words = m.group(1).strip().split()[:6]
    while words and words[-1].lower() in _SMALL:
        words.pop()
    return " ".join(w if w.isupper() or (i and w.lower() in _SMALL and w.islower())
                    else w[:1].upper() + w[1:] for i, w in enumerate(words)) or "Proposed System"


# ── Requirements proposed by the compliance and security analysis ──────
#
# A rule-pack gap is an obligation that applies to this system and that no
# requirement covers. Its proposed requirement is a requirement in everything
# but approval, so the SRS states it as one -- origin "regulatory", status
# Proposed, with the clause it rests on -- instead of leaving it in a matrix.
# Stated only in the SRS: they are not fed back into the run, so the SDLC
# recommendation and the analysis are unaffected by them.

_FAMILY_SECTION = {
    "record_retention": "database", "customer_due_diligence": "functional",
    "identity_data": "security", "personal_data": "security", "audit": "security",
    "continuity": "availability", "incident_reporting": "functional",
    "fraud": "functional", "aml_reporting": "functional", "outsourcing": "compliance",
    "customer_protection": "functional", "access_security": "security",
    "lending_disclosure": "functional", "lending_operations": "functional",
    "data_localisation": "compliance",
    # security control families
    "authentication": "security", "authorisation": "security", "cryptography": "security",
    "data_protection": "security", "monitoring": "security",
    "integration_security": "security", "secure_development": "security",
}
_FAMILY_CATS = {
    "record_retention": ["Regulatory", "Data-management"], "personal_data": ["Regulatory", "Privacy"],
    "identity_data": ["Regulatory", "Privacy", "Security"], "audit": ["Regulatory", "Audit/Reporting"],
    "continuity": ["Regulatory", "Availability"], "incident_reporting": ["Regulatory", "Audit/Reporting"],
    "aml_reporting": ["Regulatory", "Audit/Reporting"], "fraud": ["Regulatory", "Functional"],
    "data_localisation": ["Regulatory", "Data-management"],
}
_SEVERITY_PRIORITY = {"Critical": "High", "High": "High", "Medium": "Medium", "Low": "Low"}


def proposed_requirements(run: dict, start: int) -> list[dict]:
    """Requirement records for every compliance/security gap, numbered from `start`."""
    comp = (run.get("compliance") or {}).get("rule_findings") or []
    sec = (run.get("security") or {}).get("controls") or []
    out, seen, n = [], set(), start
    for kind, findings in (("compliance", comp), ("security", sec)):
        for f in findings:
            text = " ".join((f.get("proposed_requirement") or "").split())
            if f.get("status") not in ("gap", "weak") or not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            src = f.get("source") or {}
            family = f.get("family", "")
            section = _FAMILY_SECTION.get(family, "compliance" if kind == "compliance" else "security")
            cats = _FAMILY_CATS.get(family, ["Regulatory"] if kind == "compliance" else ["Security"])
            practice = f.get("basis") == "practice"
            amends = (f.get("satisfied_by") or []) if f.get("status") == "weak" else []
            out.append({
                "id": f"REQ-{n:03d}", "statement": text, "origin": "regulatory",
                "source_stakeholder": "Compliance Officer" if kind == "compliance"
                                      else "Information Security Team",
                "business_justification": f.get("title", "") + (
                    f". {f['detail']}." if f.get("detail") else "."),
                "priority": _SEVERITY_PRIORITY.get(f.get("severity"), "Medium"),
                "risk_level": _SEVERITY_PRIORITY.get(f.get("severity"), "Medium"),
                "dependencies": list(amends), "assumptions": "",
                "acceptance_criteria": "", "applicable_regulations": [],
                "confidence_score": 1.0, "approval_status": "Pending", "category": cats,
                # proposal-only fields
                "rule_id": f["rule_id"], "rule_kind": kind, "practice": practice,
                "rule_title": f.get("title", ""), "reference": f.get("reference", ""),
                "source_title": src.get("title", ""), "source_clause": src.get("clause_id", ""),
                "section_hint": section, "amends": list(amends),
                "legal_review": bool(f.get("requires_legal_review")),
            })
            n += 1
    return out
