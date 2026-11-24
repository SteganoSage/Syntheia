"""Assemble the SRS from a pipeline run.

Default layout: IEEE Std 830-1998 as taught on the course slides --

    1 Introduction            1.1 Purpose (and intended audience) · 1.2 Scope
                              (what the system will and will not do) · 1.3
                              Definitions, acronyms, abbreviations · 1.4
                              References · 1.5 Overview
    2 Overall description     2.1 Product perspective (business case, operational
                              concept, interfaces, memory, operations, site
                              adaptation) · 2.2 Product functions · 2.3 User
                              characteristics (technical skills per class) · 2.4
                              Constraints · 2.5 Assumptions and dependencies
    3 Specific requirements   3.1 External interfaces (inputs and outputs) · 3.2
                              Functions (use-case specifications) · 3.3
                              Performance · 3.4 Logical database (entities and
                              relationships) · 3.5 Design constraints · 3.6
                              Software system quality attributes
    Appendices, Index

Section 3.7 (object-oriented models / UML diagrams) is deliberately omitted --
the user's decision, 22 Nov 2026. Alternative layout: ISO/IEC/IEEE 29148:2018.

Rules the builder keeps:
  * Every requirement is stated ONCE, as a block of attributes, in the section
    its type belongs to: interfaces in 3.1, functional in 3.2, non-functional
    in 3.3-3.6, constraints in 3.5. Elsewhere it is referred to by id.
  * Every requirement says where it came from: STATED by the stakeholder,
    IMPLIED by analysis of what such a system needs, or PROPOSED from an
    applicable regulation or security control, with the clause.
  * Every section the outline names is present; an empty one says so.
  * Nothing is presented as more certain than it is.
"""

from __future__ import annotations

import re
from datetime import date

from src.generators.srs import derive
from src.generators.srs.analysis_models import DataModel, InterfaceSpec, UseCase
from src.generators.srs.model import Bullets, Para, Requirement, Section, SRSDocument, Table
from src.generators.srs.narrative import Narrative

LAYOUTS = {
    "ieee830": "IEEE Std 830-1998, IEEE Recommended Practice for Software Requirements Specifications",
    "iso29148": "ISO/IEC/IEEE 29148:2018, Systems and software engineering — Life cycle processes — Requirements engineering",
}

APPROVERS = ["Business / Product Owner", "Compliance Officer", "Information Security Officer",
             "Solution Architect", "QA / Test Lead", "Project Manager"]

TYPE_OF = {
    "functional": ("Functional", "Functional"),
    "user_interfaces": ("External interface", "External interface — user"),
    "software_interfaces": ("External interface", "External interface — software"),
    "performance": ("Non-functional", "Non-functional — performance"),
    "database": ("Non-functional", "Non-functional — logical database"),
    "availability": ("Non-functional", "Non-functional — reliability and availability"),
    "security": ("Non-functional", "Non-functional — security and privacy"),
    "usability": ("Non-functional", "Non-functional — usability"),
    "maintainability": ("Non-functional", "Non-functional — maintainability"),
    "other": ("Non-functional", "Non-functional — other quality / business"),
    "compliance": ("Design constraint", "Design constraint — standards compliance"),
}
TYPE_ORDER = ["Functional", "External interface", "Non-functional", "Design constraint"]

_PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}
_HARDWARE = re.compile(
    r"\b(device|hardware|printer|scanner|card\s+reader|atm|pos\s+terminal|camera|"
    r"wristwatch|biometric\s+(?:device|reader)|sensor|kiosk)\w*", re.I)
_COMMS = re.compile(
    r"\b(sms|e-?mail|notification|push\s+message|alert|whatsapp|otp|https?|network)\w*", re.I)
_PLATFORM = re.compile(
    r"\b(mobile\s+app\w*|android|ios|web\s+(?:portal|app\w*)|portal|browser|java|cobol|"
    r"mainframe|cloud|on[\s-]premise\w*|oracle|postgres\w*|mysql|db2|sql\s+server|"
    r"core\s+banking\s+system|internet\s+banking)\b", re.I)


def _short(text: str, n: int = 90) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _none(what: str) -> Para:
    return Para(f"No {what} were identified in the source material.", style="muted")


class _Ctx:
    """Everything derived from the run once, shared by the layouts."""

    def __init__(self, run: dict, narrative: Narrative, use_cases: list[UseCase],
                 data_model: DataModel, interfaces: InterfaceSpec, sim):
        self.run, self.n = run, narrative
        self.data_model, self.interfaces = data_model, interfaces
        self.scenario = run.get("scenario_text", "")
        base = [dict(r, origin=r.get("origin") or "stated") for r in run.get("requirements") or []]
        nums = [int(m.group()) for r in base if (m := re.search(r"\d+", r["id"]))]
        self.proposed = derive.proposed_requirements(run, start=(max(nums) if nums else 0) + 1)
        self.reqs: list[dict] = base + self.proposed
        self.by_id = {r["id"]: r for r in self.reqs}
        self.section = {r["id"]: (r["section_hint"] if r.get("origin") == "regulatory"
                                  else derive.primary_section(r["statement"], r.get("category") or []))
                        for r in self.reqs}
        self.trace = derive.trace_sources(self.scenario,
                                          {r["id"]: r["statement"] for r in base}, sim)
        self.grounding = run.get("grounding") or {}
        self.issues = {q["requirement_id"]: q.get("issues", [])
                       for q in ((run.get("quality_report") or {}).get("result_list") or [])}
        comp = run.get("compliance") or {}
        self.compliance = comp
        self.mapping = {m["requirement_id"]: m for m in comp.get("mappings", []) or []}
        self.security = run.get("security") or {}
        self.risks = run.get("risk_register") or []
        self.risks_by_req: dict[str, list[str]] = {}
        by_rule = {p["rule_id"]: p["id"] for p in self.proposed}
        for k in self.risks:
            for rid in k.get("linked_requirements", []):
                self.risks_by_req.setdefault(rid, []).append(k["id"])
            if k.get("source") in by_rule:
                self.risks_by_req.setdefault(by_rule[k["source"]], []).append(k["id"])
        self.proposal_for_rule = by_rule
        self.conflicts = run.get("conflicts") or []
        self.clar = run.get("clarifications") or {}
        self.sdlc = run.get("sdlc")
        self.dropped = run.get("dropped") or []
        self.where_map: dict[str, str] = {}
        self.ref_tags = {t: f"R{i}" for i, t in enumerate(derive.cited_titles(run), start=1)}
        self.cite_strings: dict[str, str] = {}
        for g in self.grounding.values():
            for x in g.get("citations") or []:
                self.cite_strings[x.get("citation", "")] = self.cite(x.get("title"), x.get("clause_id"))
        for f in (comp.get("rule_findings") or []) + (self.security.get("controls") or []):
            src = f.get("source") or {}
            if src.get("citation"):
                self.cite_strings[src["citation"]] = self.cite(src.get("title"), src.get("clause_id"))

        # Functional requirements proposed from regulation have no stakeholder
        # use case; they get one of their own, so 3.2 covers every function.
        self.use_cases = list(use_cases)
        reg_functional = [p["id"] for p in self.proposed if self.section[p["id"]] == "functional"]
        if reg_functional:
            self.use_cases.append(UseCase(
                id=f"UC-{len(self.use_cases) + 1:02d}", name="Meet regulatory obligations",
                primary_actor="Compliance Officer", supporting_actors=[],
                description="Functions the applicable regulations require of the system, "
                            "proposed by the compliance analysis rather than stated by a stakeholder.",
                preconditions=["The regulation applies to the system (see the compliance-control matrix)."],
                main_flow=[], alternative_flows=[],
                postconditions=["The obligation is met and evidence is retained for audit."],
                requirement_ids=reg_functional))
        assigned = {i for uc in self.use_cases for i in uc.requirement_ids}
        orphans = [r["id"] for r in self.reqs
                   if self.section[r["id"]] == "functional" and r["id"] not in assigned]
        if orphans:
            self.use_cases.append(UseCase(
                id=f"UC-{len(self.use_cases) + 1:02d}", name="Other functions", primary_actor="User",
                supporting_actors=[], description="Functional requirements not assigned to a use case above.",
                preconditions=[], main_flow=[], alternative_flows=[], postconditions=[],
                requirement_ids=orphans))

    # ── citations ──
    def cite(self, title, clause="", long=False) -> str:
        title, clause = title or "", (clause or "").strip()
        tag = self.ref_tags.get(title)
        sec = f"§{clause}" if clause else ""
        if not tag:
            return " ".join(x for x in (title, sec) if x)
        if long:
            return f"[{tag}] {title}" + (f", {sec}" if sec else "")
        return f"[{tag}]" + (f" {sec}" if sec else "")

    def untangle(self, text: str) -> str:
        for full, short in sorted(self.cite_strings.items(), key=lambda kv: -len(kv[0])):
            if full and full in text:
                text = text.replace(full, short)
        return text

    # ── requirement helpers ──
    def in_section(self, key: str) -> list[dict]:
        return sorted((r for r in self.reqs if self.section[r["id"]] == key),
                      key=lambda r: (_PRIORITY_RANK.get(r.get("priority"), 1), r["id"]))

    def type_group(self, rid: str) -> str:
        return TYPE_OF[self.section[rid]][0]

    def origin(self, r: dict) -> str:
        o = r.get("origin", "stated")
        if o == "implied":
            return "Implied — derived by analysis of what this kind of system needs; confirm with the stakeholder"
        if o == "regulatory":
            basis = (self.cite(r["source_title"], r["source_clause"]) if r.get("source_title")
                     else r.get("reference") or "security practice")
            kind = "security control" if r.get("practice") else "regulation"
            return f"Proposed — required by {basis} ({kind}, rule {r['rule_id']}); not stated by the stakeholder"
        return "Stated — extracted from the problem scenario"

    def regulatory_basis(self, r: dict) -> str:
        if r.get("origin") == "regulatory":
            if r.get("source_title"):
                out = self.cite(r["source_title"], r["source_clause"], long=True)
            else:
                out = (r.get("reference") or "Industry practice") + " — not a regulation"
            return out + (" — requires compliance/legal sign-off" if r.get("legal_review") else "")
        g = self.grounding.get(r["id"]) or {}
        cites = g.get("citations") or []
        parts = [f"{self.cite(c.get('title'), c.get('clause_id'), long=True)} "
                 f"(retrieval confidence {c.get('confidence', 0):.2f})" for c in cites[:2]]
        m = self.mapping.get(r["id"])
        if m:
            verdict = {"addressed": "clause review: requirement addresses the clause",
                       "partial": "clause review: partially addresses the clause",
                       "not_addressed": "clause review: does NOT address the clause",
                       "not_applicable": "clause review: cited clause does not govern this "
                                         "requirement — citation unsupported",
                       "unclear": "clause review: inconclusive"}.get(m.get("coverage"), "")
            if verdict:
                parts.append(verdict + (f" — {m['missing']}" if m.get("missing") else ""))
        if not cites and r.get("applicable_regulations"):
            parts.append(f"Names {', '.join(r['applicable_regulations'])}, which the "
                         f"knowledge base does not contain — unverified (see open issues)")
        return "; ".join(p for p in parts if p) or "None identified"

    def source(self, r: dict) -> str:
        who = r.get("source_stakeholder") or "Not stated"
        if r.get("origin") == "regulatory":
            return f"{who} (proposed by the {'security' if r.get('rule_kind') == 'security' else 'compliance'} analysis)"
        t = self.trace.get(r["id"])
        if t:
            return f"{who}. Derived from the scenario: “{t[0]}” (similarity {t[1]:.2f})"
        if r.get("origin") == "implied":
            return f"{who}. Implied by the scenario as a whole — " + (r.get("business_justification") or "")
        return f"{who}. Not stated directly in the scenario; derived by analysis"

    def acceptance(self, r: dict) -> str:
        crit = (r.get("acceptance_criteria") or "").strip()
        if crit:
            return crit
        if r.get("origin") == "regulatory":
            base = (f"Inspection against {self.cite(r['source_title'], r['source_clause'])}: the "
                    f"implemented behaviour meets the clause, and evidence is retained for audit"
                    if r.get("source_title") else
                    f"A security test confirms the control is in place ({r.get('reference') or 'practice'})")
            return base + (f". Amends {', '.join(r['amends'])}." if r.get("amends") else ".")
        return "TBD — no measurable criterion stated (see open issues)"

    def block(self, r: dict) -> Requirement:
        sec = self.section[r["id"]]
        if r.get("origin") == "regulatory":
            found = "Not reviewed — generated from the rule's template wording"
            method = ("Inspection (compliance review)" if not r.get("practice")
                      else "Test (security test) and Inspection")
            status = "Proposed — Pending approval"
        else:
            found = "; ".join(f"{i.get('issue_type')} ({i.get('severity')}): {i.get('description')}"
                              for i in self.issues.get(r["id"], [])) or "None"
            method = derive.verification_method(sec)
            status = (f"{r.get('approval_status', 'Pending')} · extraction confidence "
                      f"{float(r.get('confidence_score', 0)):.2f}")
        risk = r.get("risk_level", "Medium")
        if self.risks_by_req.get(r["id"]):
            risk += f" — see {', '.join(self.risks_by_req[r['id']][:4])}"
        return Requirement(id=r["id"], statement=r["statement"], attributes=[
            ("Type", TYPE_OF[sec][1]),
            ("Origin", self.origin(r)),
            ("Priority", r.get("priority", "Medium")),
            ("Rationale", self.untangle(r.get("business_justification") or "Not stated")),
            ("Source", self.source(r)),
            ("Acceptance criteria", self.acceptance(r)),
            ("Verification method", method),
            ("Dependencies", ", ".join(r.get("dependencies") or []) or "None"),
            ("Assumptions", r.get("assumptions") or "None stated"),
            ("Risk if not met", risk),
            ("Regulatory basis", self.regulatory_basis(r)),
            ("Classification", ", ".join(r.get("category") or []) or "Unclassified"),
            ("Quality findings", found),
            ("Status", status),
        ])

    def blocks(self, key: str, what: str) -> list:
        reqs = self.in_section(key)
        return [self.block(r) for r in reqs] if reqs else [_none(what)]

    def refs(self, rs: list[dict]) -> Bullets | None:
        if not rs:
            return None
        return Bullets([f"{r['id']} — {_short(r['statement'], 110)} (§{self.where(r['id'])})"
                        for r in rs])

    def where(self, rid: str) -> str:
        return self.where_map.get(rid, "3")


# ── Section content ────────────────────────────────────────────────────

def _purpose(c: _Ctx, standard: str) -> list:
    return [Para(c.n.purpose),
            Para("Intended audience. Business and product owners read sections 1 and 2 and "
                 "the use cases in 3.2; developers and architects read section 3 in full; "
                 "testers use the acceptance criteria and the verification matrix; "
                 "compliance and security reviewers read 2.4, 3.5, 3.6 and the "
                 "compliance-control matrix."),
            Para(f"This document follows {standard}. It was generated by Syntheia from the "
                 f"problem scenario [R0] and a curated, versioned knowledge base of financial "
                 f"regulations; it is a draft for review, and every requirement is Pending "
                 f"until approved.", style="muted")]


def _scope(c: _Ctx) -> list:
    by_origin = {o: sum(1 for r in c.reqs if r.get("origin", "stated") == o)
                 for o in ("stated", "implied", "regulatory")}
    out = [Para(c.n.scope),
           Para(f"This specification states {len(c.reqs)} requirements: {by_origin['stated']} "
                f"stated in the scenario, {by_origin['implied']} implied by analysis of what such a "
                f"system needs, and {by_origin['regulatory']} proposed from the regulations and "
                f"security controls that apply to it. Section 3 gives each one with its origin.")]
    if c.n.objectives:
        out += [Para("Business objectives stated in the scenario:"),
                Bullets([f"{t} (“{q}”)" for t, q in c.n.objectives])]
    out.append(Para("The system will:", lead=""))
    out.append(Bullets([f"{uc.name} — {uc.description}" for uc in c.use_cases]
                       or ["See section 3."]))
    out.append(Para("The system will not:", lead=""))
    if c.n.out_of_scope:
        out.append(Bullets([f"{t} — stated in the scenario (“{q}”)" if q
                            else f"{t} — assumed; to be confirmed" for t, q in c.n.out_of_scope]))
    else:
        out.append(Para("No exclusions are stated in the source scenario; the scope boundary "
                        "is to be confirmed with the stakeholders (TBD).", style="muted"))
    return out


def _definitions(c: _Ctx, only: str | None = None) -> list:
    text = " ".join([c.scenario] + [r["statement"] + " " + (r.get("acceptance_criteria") or "")
                                    for r in c.reqs])
    found, unknown = derive.glossary_for(text)
    if only:
        found = [e for e in found if e.get("kind") == only]
    rows = [[str(e["term"]), e["definition"]] for e in found]
    out = [Table(["Term", "Definition"], rows, widths=[1, 4])] if rows else [
        Para("No specialised terms were identified.", style="muted")]
    out.append(Para("Conventions. Requirements are identified REQ-nnn and use cases UC-nn; "
                    "\"shall\" marks a mandatory requirement; [Rn] refers to the references in "
                    "1.4; TBD marks information to be determined (Appendix F). Each requirement "
                    "states its origin: Stated (by the stakeholder), Implied (derived by "
                    "analysis), or Proposed (from an applicable regulation or security "
                    "control).", style="muted"))
    if unknown and only != "term":
        out.append(Para("Acronyms used in the source material with no agreed definition yet "
                        "(TBD): " + ", ".join(unknown) + ".", style="note"))
    return out


def _references(c: _Ctx, standard: str) -> list:
    rows = [["[R0]", "Problem scenario supplied for this specification",
             "Project stakeholder", "—", f"Input to this run ({c.run.get('scenario', 'stdin')})"],
            ["[S1]", standard, "IEEE / ISO / IEC", "—", "Structure of this document"]]
    for i, title in enumerate(derive.cited_titles(c.run), start=1):
        e = derive.reference_entry(title)
        ident = " · ".join(x for x in (e["doc_id"], e["version"]) if x) or "—"
        eff = " · ".join(x for x in (f"effective {e['effective']}" if e["effective"] else "",
                                     e["jurisdiction"]) if x) or "—"
        rows.append([f"[R{i}]", title, e["issuer"] or "—", ident, eff])
    return [Table(["Ref", "Title", "Issuer", "Identifier / version", "Effective / jurisdiction"],
                  rows, widths=[0.6, 3.4, 1.7, 2.2, 1.6]),
            Para("Regulatory references are the knowledge-base documents this specification "
                 "cites; their issuer, version, effective date and jurisdiction come from each "
                 "document's manifest (problem statement §5).", style="muted")]


def _overview(layout: str) -> list:
    if layout == "ieee830":
        return [Para("Section 2 describes the product as a whole: its business case and "
                     "operational concept, interfaces, functions, users, constraints, "
                     "assumptions and dependencies. Section 3 states every requirement once, "
                     "by type: external interfaces (3.1), functional requirements as use cases "
                     "(3.2), and non-functional requirements — performance (3.3), logical "
                     "database (3.4), design constraints (3.5) and quality attributes (3.6). "
                     "Each requirement carries its type, origin, priority, rationale, source, "
                     "acceptance criteria, verification method and regulatory basis. The "
                     "appendices give the traceability matrix, verification matrix, "
                     "compliance-control matrix, risk register, recommended development life "
                     "cycle, open issues and analysis notes, followed by an index.")]
    return [Para("Clause 2 lists references. Clause 3 states every requirement once with its "
                 "attributes. Clause 4 gives the verification approach. Clause 5 holds the "
                 "assumptions, acronyms, traceability, compliance, risk and life-cycle "
                 "appendices, followed by an index.")]


def _interfaces_overview(c: _Ctx) -> list:
    integ = c.in_section("software_interfaces")
    ui = c.in_section("user_interfaces")
    hw = [r for r in c.reqs if _HARDWARE.search(r["statement"])]
    comms = [r for r in c.reqs if _COMMS.search(r["statement"])]
    return [("System interfaces", [c.refs(integ)] if integ else [_none("interfaces to external systems")]),
            ("User interfaces", [c.refs(ui)] if ui else [_none("user-interface requirements")]),
            ("Hardware interfaces", [c.refs(hw)] if hw else [_none("hardware interfaces")]),
            ("Software interfaces", [c.refs(integ)] if integ else [_none("software interfaces")]),
            ("Communications interfaces", [c.refs(comms)] if comms else [_none("communications interfaces")])]


def _counts(c: _Ctx) -> Table:
    origins = [("stated", "Stated"), ("implied", "Implied"), ("regulatory", "Proposed")]
    rows = []
    for t in TYPE_ORDER:
        ids = [r for r in c.reqs if c.type_group(r["id"]) == t]
        rows.append([t] + [str(sum(1 for r in ids if r.get("origin", "stated") == o)) for o, _ in origins]
                    + [str(len(ids))])
    rows.append(["Total"] + [str(sum(1 for r in c.reqs if r.get("origin", "stated") == o)) for o, _ in origins]
                + [str(len(c.reqs))])
    return Table(["Requirement type", "Stated", "Implied", "Proposed", "Total"], rows,
                 widths=[2.2, 1, 1, 1, 1],
                 caption="Requirements by type and origin")


def _product_functions(c: _Ctx) -> list:
    out = [Para("The major functional capabilities of the system, each specified as a use "
                "case in section 3.2:"),
           Bullets([f"{uc.id} {uc.name} — {uc.description}" for uc in c.use_cases]
                   or ["No functions were identified."])]
    return out


def _user_characteristics(c: _Ctx) -> list:
    classes = list(c.n.user_classes)
    named = {u[0].lower() for u in classes}
    for uc in c.use_cases:                  # whoever starts a use case uses the system
        who = (uc.primary_actor or "").strip()
        lw = who.lower().rstrip("s")
        if who and lw not in ("user", "to be confirmed") and \
                not any(lw in n or n.rstrip("s") in lw for n in named):
            named.add(who.lower())
            classes.append((who, f"Primary actor of {uc.id} {uc.name}.", "To be confirmed",
                            "To be confirmed"))
    if not classes:
        return [_none("user classes")]
    return [Para("The classes of user the system serves, their technical skills and how "
                 "often they use it. Characteristics not stated in the source material are "
                 "marked to be confirmed."),
            Table(["User class", "Description", "Technical expertise", "Frequency of use"],
                  [list(u) for u in classes], widths=[1.6, 3.2, 1.6, 1.4])]


def _constraints(c: _Ctx) -> list:
    out = []
    regs, seen = [], set()
    for f in c.compliance.get("rule_findings") or []:
        src = f.get("source") or {}
        ref = (c.cite(src.get("title"), long=True) if src.get("title") else f.get("reference")) or ""
        if ref and ref not in seen:
            seen.add(ref)
            regs.append(ref)
    out.append(Para("Regulatory policies.", lead=""))
    out.append(Bullets([f"The system is subject to {r}." for r in regs]) if regs
               else Para("No regulatory instruments were identified.", style="muted"))
    platform = sorted({m.group(0).lower() for r in c.reqs for m in _PLATFORM.finditer(r["statement"])}
                      | {m.group(0).lower() for m in _PLATFORM.finditer(c.scenario)})
    out.append(Para("Target platform, databases and network.", lead=""))
    out.append(Para("Platform elements named in the source material: " + ", ".join(platform) + "."
                    if platform else
                    "The target platform, database and network are not specified in the source "
                    "material and are to be confirmed (TBD)."))
    if c.n.constraints:
        out += [Para("Constraints stated in the scenario.", lead=""),
                Bullets([f"{t} (“{q}”)" for t, q in c.n.constraints])]
    if c.dropped:
        out += [Para("Project constraints (stated in the scenario as facts about the project, "
                     "so not specified as system requirements):", lead=""),
                Bullets([d["statement"] for d in c.dropped])]
    out.append(Para("Development standards.", lead=""))
    out.append(Para("This specification follows the standard named in 1.1. Coding, security "
                    "and testing standards for development are to be confirmed (TBD)."))
    design = c.in_section("compliance")
    if design:
        out.append(Para(f"Design constraints arising from standards compliance are specified in "
                        f"section {c.where(design[0]['id'])}.", style="muted"))
    return out


def _assumptions(c: _Ctx) -> list:
    rows = [[r["id"], r["assumptions"]] for r in c.reqs if (r.get("assumptions") or "").strip()]
    out = []
    if rows:
        out += [Para("Assumptions recorded against individual requirements:"),
                Table(["Requirement", "Assumption"], rows, widths=[1, 5])]
    deps = c.in_section("software_interfaces")
    if deps:
        out += [Para("Dependencies on external systems and services:"), c.refs(deps)]
    internal = [[r["id"], ", ".join(r["dependencies"])] for r in c.reqs if r.get("dependencies")]
    if internal:
        out += [Para("Dependencies between requirements:"),
                Table(["Requirement", "Depends on"], internal, widths=[1, 5])]
    return out or [_none("assumptions or dependencies")]


def _interface_spec(c: _Ctx) -> list:
    out = [Para("All inputs and outputs of the system's external interfaces. They complement "
                "the interface overview in 2.1 rather than repeat it.")]
    if c.interfaces.rows:
        out.append(Table(["Interface", "Type", "Counterpart", "Inputs", "Outputs", "Format"],
                         [list(r) for r in c.interfaces.rows],
                         widths=[1.3, 0.9, 1.3, 1.6, 1.6, 1.2]))
    else:
        out.append(Para("The inputs and outputs of each interface were not derived; they are to "
                        "be specified (TBD).", style="muted"))
    return out


def _use_case_table(uc: UseCase) -> Table:
    def numbered(items):
        return "\n".join(f"{i}. {s}" for i, s in enumerate(items, start=1)) if items else "None stated"

    return Table(["Field", "Specification"], [
        ["Use case", f"{uc.id} {uc.name}"],
        ["Primary actor", uc.primary_actor or "To be confirmed"],
        ["Supporting actors", ", ".join(uc.supporting_actors) or "None"],
        ["Description", uc.description or "—"],
        ["Preconditions", numbered(uc.preconditions)],
        ["Main success scenario", numbered(uc.main_flow) if uc.main_flow
         else "Specified by the related requirements below."],
        ["Alternative flows", "\n".join(uc.alternative_flows) or "None identified"],
        ["Postconditions", numbered(uc.postconditions)],
        ["Related requirements", ", ".join(uc.requirement_ids)],
    ], widths=[1.4, 5.2])


def _functions(c: _Ctx, parent: Section) -> None:
    parent.add(Para("Functional requirements, organised by use case. Each use case is "
                    "specified, then followed by the functional requirements it realises."))
    if not c.use_cases:
        parent.add(_none("functional requirements"))
        return
    for uc in c.use_cases:
        sec = parent.sub(f"{uc.id} {uc.name}")
        sec.add(_use_case_table(uc))
        reqs = sorted((c.by_id[i] for i in uc.requirement_ids if i in c.by_id),
                      key=lambda r: (_PRIORITY_RANK.get(r.get("priority"), 1), r["id"]))
        sec.add(*[c.block(r) for r in reqs])


def _logical_database(c: _Ctx) -> list:
    out = [Para("The data entities the system stores and their relationships, followed by the "
                "requirements on data storage and retention.")]
    dm = c.data_model
    if dm.entities:
        out.append(Table(["Entity", "Description", "Key attributes", "Identifier"],
                         [[n, d, ", ".join(a), i] for n, d, a, i in dm.entities],
                         widths=[1.4, 2.4, 2.6, 1.2], caption="Data entities"))
        if dm.relationships:
            out.append(Table(["Entity", "Relationship", "Related entity", "Cardinality"],
                             [list(r) for r in dm.relationships], widths=[1.5, 1.5, 1.5, 2.1],
                             caption="Relationships"))
    else:
        out.append(Para("The data entities were not derived; the logical data model is to be "
                        "specified (TBD).", style="muted"))
    out += c.blocks("database", "data storage or retention requirements")
    return out


# ── Appendices ─────────────────────────────────────────────────────────

def _rtm(c: _Ctx) -> list:
    rows = []
    for r in sorted(c.reqs, key=lambda r: r["id"]):
        if r.get("origin") == "regulatory":
            src = "— proposed from regulation" if not r.get("practice") else "— proposed control"
            reg = (c.cite(r["source_title"], r["source_clause"]) if r.get("source_title")
                   else "practice")
        else:
            t = c.trace.get(r["id"])
            src = _short(t[0], 110) if t else ("implied by analysis" if r.get("origin") == "implied"
                                                else "implicit")
            cites = (c.grounding.get(r["id"]) or {}).get("citations") or []
            reg = c.cite(cites[0].get("title"), cites[0].get("clause_id")) if cites else "—"
            m = c.mapping.get(r["id"])
            if m and m.get("coverage") == "not_applicable":
                reg += " (unsupported)"
        rows.append([r["id"], r.get("origin", "stated").replace("regulatory", "proposed"),
                     src, reg, c.where(r["id"]),
                     derive.verification_method(c.section[r["id"]]).split(" (")[0],
                     ", ".join(c.risks_by_req.get(r["id"], [])[:3]) or "—"])
    return [Para("Each requirement traced backward to its origin, the scenario sentence it "
                 "derives from and its regulatory basis, and forward to the section that "
                 "specifies it, its verification method and its risks (problem statement §12)."),
            Table(["Req", "Origin", "Scenario source", "Regulatory basis", "Section",
                   "Verify by", "Risks"], rows, widths=[0.95, 0.85, 2.3, 1.2, 0.8, 0.9, 1.0])]


def _verification(c: _Ctx) -> list:
    rows = [[r["id"], r["statement"], c.block(r).attributes[6][1], c.acceptance(r)]
            for r in sorted(c.reqs, key=lambda r: r["id"])]
    return [Para("Verification methods follow ISO/IEC/IEEE 29148: inspection, analysis, "
                 "demonstration and test, chosen by the kind of requirement."),
            Table(["Req", "Requirement", "Method", "Acceptance criterion"], rows,
                  widths=[0.95, 2.6, 1.4, 2.6])]


def _compliance_matrix(c: _Ctx) -> list:
    rows = []
    status = {"satisfied": "Covered", "gap": "GAP", "weak": "Weak"}
    for f in (c.compliance.get("rule_findings") or []) + (c.security.get("controls") or []):
        src = f.get("source") or {}
        clause = (c.cite(src.get("title"), src.get("clause_id")) if src.get("title")
                  else (f.get("reference") or "—") + (" (practice)" if f.get("basis") == "practice" else ""))
        action = "—"
        if f["status"] != "satisfied":
            action = (f"Proposed as {c.proposal_for_rule[f['rule_id']]}"
                      if f["rule_id"] in c.proposal_for_rule else (f.get("detail") or "—"))
            if f.get("detail") and f["status"] == "weak":
                action += f" ({f['detail']})"
        rows.append([f["rule_id"], f["title"], clause,
                     status.get(f["status"], f["status"])
                     + (f" by {', '.join(f['satisfied_by'])}" if f.get("satisfied_by") else ""),
                     action + (" [legal review]" if f.get("requires_legal_review") else "")])
    if not rows:
        return [Para("Compliance analysis was not run for this specification.", style="muted")]
    gaps = sum(1 for r in rows if r[3].startswith(("GAP", "Weak")))
    return [Para(f"Obligations and security controls that apply to this system, the clause each "
                 f"rests on, and whether a requirement covers it. {gaps} of {len(rows)} were "
                 f"gaps or weak; each is stated in section 3 as a Proposed requirement. Items "
                 f"marked [legal review] need a compliance or legal officer's decision. Rules "
                 f"are drafted from the knowledge base and not yet verified by a person."),
            Table(["Rule", "Obligation / control", "Source", "Status", "Action"],
                  rows, widths=[1.75, 2.4, 1.2, 1.15, 1.9])]


def _risk_register(c: _Ctx) -> list:
    if not c.risks:
        return [Para("Risk analysis was not run for this specification.", style="muted")]
    rows = [[k["id"], k["title"], k["category"], f"{k['likelihood']}×{k['impact']}={k['score']}",
             k["rating"], k["owner"], c.untangle(k["mitigation"])] for k in c.risks]
    return [Para("Likelihood and impact on a 1–5 scale; the score is their product "
                 "(data/risk/rubric.yaml). Each risk is traced to the finding that raised it."),
            Table(["ID", "Risk", "Category", "L×I", "Rating", "Owner", "Mitigation"], rows,
                  widths=[1.05, 1.9, 0.95, 0.85, 0.8, 1.15, 2.4])]


def _sdlc(c: _Ctx) -> list:
    s = c.sdlc
    if not s:
        return [Para("The SDLC recommendation was not run for this specification.", style="muted")]
    out = [Para(f"Recommended model: {s['recommendation']}"
                + (f" (tied with {', '.join(m for m in s['tied_with'] if m != s['recommendation'])})"
                   if s.get("tied_with") else "") + ".", lead="Recommendation.")]
    if s.get("shortcut"):
        out.append(Para(s["shortcut"], style="quote"))
    if s.get("summary"):
        out.append(Para(s["summary"]))
    rank = s.get("ranking") or []
    if rank:
        out.append(Table(["Rank", "Model", "Fit", "Matched / answered"],
                         [[str(i), m["name"] + (" (derived)" if m.get("is_derived") else ""),
                           f"{m['score']:.0%}", f"{len(m.get('matched', []))}/"
                           f"{len(m.get('matched', [])) + len(m.get('missed', []))}"]
                          for i, m in enumerate(rank[:5], start=1)], widths=[0.6, 2, 1, 1.6]))
    for j in (s.get("justifications") or [])[:1]:
        adv = [f"{a['advantage']}: {a['relevance']}" for a in j.get("advantages", [])]
        if adv:
            out += [Para(f"Why {j['model']} suits this project:"), Bullets(adv)]
        if j.get("gives_up"):
            out.append(Para(j["gives_up"], lead="Trade-off."))
    cov = s.get("coverage") or {}
    if cov:
        out.append(Table(["Selection-framework table", "Questions answered"],
                         [[v["name"], f"{v['answered']}/{v['total']}"] for v in cov.values()],
                         widths=[3, 1.5]))
    out.append(Para("Scored against the supervisor's SDLC selection framework "
                    f"(version {s.get('framework_version', '?')}) from the stated requirements; "
                    "the final choice must be approved by the project manager, architect, "
                    "security team and compliance officer (PS §14).", style="muted"))
    return out


def _open_issues(c: _Ctx) -> list:
    rows = []
    for q in c.clar.get("questions") or []:
        if q.get("status") != "open":
            continue
        rule = re.match(r"[A-Z]+(?:-[A-Z0-9]+)+", q.get("why") or "")
        proposal = c.proposal_for_rule.get(rule.group(0)) if rule else None
        if proposal:                        # the gap is already stated in section 3
            p = c.by_id[proposal]
            rows.append([q["id"], f"Confirm proposed requirement {proposal} ({p['rule_title']}), "
                                  f"which no stated requirement covers", "Proposed requirement",
                         proposal])
        else:
            rows.append([q["id"], c.untangle(q["question"]), q.get("topic", ""),
                         ", ".join(q.get("requirement_ids") or []) or "—"])
    n = 0
    for r in c.reqs:
        if r.get("origin") == "regulatory":
            continue
        if not (r.get("acceptance_criteria") or "").strip():
            n += 1
            rows.append([f"TBD-{n:02d}", f"No acceptance criterion for {r['id']}: "
                                          f"{_short(r['statement'], 90)}", "Testability", r["id"]])
        g = c.grounding.get(r["id"]) or {}
        if g.get("needs_review") and r.get("applicable_regulations"):
            n += 1
            rows.append([f"TBD-{n:02d}", f"{r['id']} names {', '.join(r['applicable_regulations'])}, "
                                          f"which the knowledge base cannot confirm",
                         "Regulatory basis", r["id"]])
        m = c.mapping.get(r["id"])
        if m and m.get("coverage") == "not_applicable":
            n += 1
            rows.append([f"TBD-{n:02d}", f"The regulation cited for {r['id']} does not govern it "
                                          f"on review; its regulatory basis is to be confirmed",
                         "Regulatory basis", r["id"]])
    implied = [r["id"] for r in c.reqs if r.get("origin") == "implied"]
    if implied:
        n += 1
        rows.append([f"TBD-{n:02d}", "Confirm with the stakeholders the requirements derived by "
                                      "analysis rather than stated", "Implied requirements",
                     ", ".join(implied)])
    if not rows:
        return [Para("No open issues were recorded.", style="muted")]
    return [Para("Questions for stakeholders and items to be determined (TBD), contradictions "
                 "first."),
            Table(["ID", "Issue", "Topic", "Requirements"], rows, widths=[0.95, 4.2, 1.3, 1.15])]


def _analysis_notes(c: _Ctx) -> list:
    out = []
    if c.conflicts:
        out += [Para("Contradictions and duplicates between requirements:"),
                Table(["Kind", "Requirements", "Explanation", "Confirmed"],
                      [[x["kind"], " / ".join(x["requirement_ids"]), x["explanation"],
                        "yes" if x.get("confirmed") else "possible"] for x in c.conflicts],
                      widths=[0.9, 1.3, 4.0, 0.9])]
    if c.dropped:
        out += [Para("Statements removed from the requirements as project facts:"),
                Bullets([f"{d['statement']} ({d['reason']})" for d in c.dropped])]
    rej = c.run.get("implied_rejected") or []
    if rej:
        out += [Para("Implied requirements the elaboration gate rejected:"),
                Bullets([f"{x['statement']} — {x['reason']}" for x in rej[:10]])]
    if c.issues:
        flagged = sum(1 for v in c.issues.values() if v)
        out.append(Para(f"Quality analysis flagged {flagged} of {len(c.issues)} requirements; "
                        f"each finding is shown on its requirement block."))
    review = c.run.get("needs_review") or []
    if review:
        out += [Para("Items the pipeline flagged for human review:"), Bullets(review)]
    return out or [Para("No analysis notes.", style="muted")]


# ── Index ──────────────────────────────────────────────────────────────

def _section_text(sec: Section) -> str:
    parts = []
    for b in sec.blocks:
        if isinstance(b, Para):
            parts.append(b.text)
        elif isinstance(b, Bullets):
            parts += b.items
        elif isinstance(b, Table):
            parts += [" ".join(map(str, row)) for row in b.rows]
        elif isinstance(b, Requirement):
            parts.append(b.statement)
    return " ".join(parts)


def _index(doc: SRSDocument, c: _Ctx) -> Section:
    found, _ = derive.glossary_for(c.scenario + " " + " ".join(r["statement"] for r in c.reqs))
    terms = {str(e["term"]) for e in found if e["term"] != "shall"}
    terms |= {uc.name for uc in c.use_cases}
    terms |= {e[0] for e in c.data_model.entities}
    terms |= {a for uc in c.use_cases for a in [uc.primary_actor] + uc.supporting_actors if a}
    terms -= {"User", "To be confirmed", "Other functions"}      # placeholders, not terms
    texts =[(s.number, _section_text(s)) for _, s in doc.walk() if s.number]
    rows = []
    for term in sorted(terms, key=str.lower):
        pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                         0 if term.isupper() else re.I)
        where = [num for num, text in texts if pat.search(text)]
        if where:
            rows.append([term, ", ".join(where[:8]) + (" …" if len(where) > 8 else "")])
    sec = Section("Index", unnumbered=True)
    sec.add(Para("Terms, actors, use cases and data entities, with the sections in which they "
                 "appear.", style="muted"),
            Table(["Term", "Sections"], rows, widths=[2, 4]) if rows
            else Para("No index entries.", style="muted"))
    return sec


# ── Layouts ────────────────────────────────────────────────────────────

def _front(c: _Ctx, layout: str, today: str) -> SRSDocument:
    history = [["1.0 (Draft)", today, "Initial draft generated from the problem scenario", "Syntheia"]]
    changes = c.clar.get("changes") or []
    if changes:
        history.append(["1.0 (Draft)", today,
                        f"{len(changes)} change(s) from stakeholder clarification", "Syntheia"])
    return SRSDocument(
        title="Software Requirements Specification",
        product=c.n.product_name, version="1.0 (Draft)", date=today,
        status="DRAFT — generated, not approved",
        prepared_by="Syntheia — automated requirements engineering assistant",
        standard=LAYOUTS[layout], layout=layout, revision_history=history,
        approvals=[[role, "", "", ""] for role in APPROVERS],
        notice=("This specification was generated automatically from the problem scenario and a "
                "curated regulatory knowledge base. It is a draft for review: every requirement "
                "is Pending until approved, regulatory citations must be confirmed by a "
                "compliance officer, and every item marked TBD is listed in the open-issues "
                "appendix."))


def _finish(doc: SRSDocument, c: _Ctx, rtm_section: Section) -> SRSDocument:
    doc.number()
    for _, sec in doc.walk():
        for b in sec.blocks:
            if isinstance(b, Requirement):
                c.where_map[b.id] = sec.number
    rtm_section.blocks = _rtm(c)
    for _, sec in doc.walk():           # cross-references now that numbers exist
        for i, b in enumerate(sec.blocks):
            if isinstance(b, Bullets) and b.items and all(" (§" in x for x in b.items):
                rs = [c.by_id[x.split(" — ")[0]] for x in b.items if x.split(" — ")[0] in c.by_id]
                if len(rs) == len(b.items):
                    sec.blocks[i] = c.refs(rs)
    doc.sections.append(_index(doc, c))
    return doc


def build_ieee830(c: _Ctx, today: str) -> SRSDocument:
    std = LAYOUTS["ieee830"]
    doc = _front(c, "ieee830", today)

    intro = Section("Introduction")
    intro.sub("Purpose").add(*_purpose(c, std))
    intro.sub("Scope").add(*_scope(c))
    intro.sub("Definitions, acronyms, and abbreviations").add(*_definitions(c))
    intro.sub("References").add(*_references(c, std))
    intro.sub("Overview").add(*_overview("ieee830"))

    overall = Section("Overall description")
    pp = overall.sub("Product perspective")
    pp.add(Para(c.n.perspective, lead="Business case."))
    if c.n.operational_concept:
        pp.add(Para(c.n.operational_concept, lead="Operational concept."))
    for title, blocks in _interfaces_overview(c):
        pp.sub(title).add(*blocks)
    pp.sub("Memory constraints").add(Para("None specified in the source material.", style="muted"))
    ops = c.in_section("maintainability")
    pp.sub("Operations").add(c.refs(ops) if ops else _none("operational requirements"))
    pp.sub("Site adaptation requirements").add(Para("None specified in the source material.",
                                                    style="muted"))
    overall.sub("Product functions").add(*_product_functions(c))
    overall.sub("User characteristics").add(*_user_characteristics(c))
    overall.sub("Constraints").add(*_constraints(c))
    overall.sub("Assumptions and dependencies").add(*_assumptions(c))

    spec = Section("Specific requirements").add(
        Para("Requirements are grouped by type: external interfaces (3.1), functional "
             "requirements (3.2), and non-functional requirements — performance (3.3), logical "
             "database (3.4), design constraints (3.5) and software system quality attributes "
             "(3.6)."), _counts(c))
    ext = spec.sub("External interfaces").add(*_interface_spec(c))
    ext.sub("User interfaces").add(*c.blocks("user_interfaces", "user-interface requirements"))
    hw = [r for r in c.reqs if _HARDWARE.search(r["statement"])]
    ext.sub("Hardware interfaces").add(c.refs(hw) if hw else _none("hardware interface requirements"))
    ext.sub("Software interfaces").add(*c.blocks("software_interfaces", "software interface requirements"))
    comms = [r for r in c.reqs if _COMMS.search(r["statement"])]
    ext.sub("Communications interfaces").add(
        c.refs(comms) if comms else _none("communications interface requirements"))
    _functions(c, spec.sub("Functions"))
    spec.sub("Performance requirements").add(*c.blocks("performance", "performance requirements"))
    spec.sub("Logical database requirements").add(*_logical_database(c))
    dc = spec.sub("Design constraints")
    dc.sub("Standards compliance").add(*c.blocks("compliance", "standards-compliance requirements"))
    dc.sub("Software and hardware constraints").add(
        Para("Specific software or hardware the system must use is not stated in the source "
             "material beyond the platform elements listed in 2.4; to be confirmed (TBD).",
             style="muted"))
    qa = spec.sub("Software system quality attributes")
    qa.sub("Reliability and availability").add(*c.blocks("availability", "reliability or availability requirements"))
    qa.sub("Security and privacy").add(*c.blocks("security", "security or privacy requirements"))
    qa.sub("Usability").add(*c.blocks("usability", "usability requirements"))
    qa.sub("Maintainability").add(*c.blocks("maintainability", "maintainability requirements"))
    qa.sub("Portability").add(_none("portability requirements"))
    other = c.in_section("other")
    if other:
        qa.sub("Other quality and business requirements").add(*[c.block(r) for r in other])

    rtm = Section("Requirements traceability matrix", appendix=True)
    doc.sections = [intro, overall, spec, rtm]
    for title, content in (("Verification matrix", _verification(c)),
                           ("Compliance-control matrix", _compliance_matrix(c)),
                           ("Risk register", _risk_register(c)),
                           ("Recommended development life cycle", _sdlc(c)),
                           ("Open issues (TBD list)", _open_issues(c)),
                           ("Analysis notes", _analysis_notes(c))):
        doc.sections.append(Section(title, appendix=True).add(*content))
    return _finish(doc, c, rtm)


def build_iso29148(c: _Ctx, today: str) -> SRSDocument:
    std = LAYOUTS["iso29148"]
    doc = _front(c, "iso29148", today)

    intro = Section("Introduction")
    intro.sub("Purpose").add(*_purpose(c, std))
    intro.sub("Scope").add(*_scope(c))
    po = intro.sub("Product overview")
    pp = po.sub("Product perspective").add(Para(c.n.perspective, lead="Business case."))
    if c.n.operational_concept:
        pp.add(Para(c.n.operational_concept, lead="Operational concept."))
    for title, blocks in _interfaces_overview(c):
        pp.add(Para(title + ":"), *blocks)
    po.sub("Product functions").add(*_product_functions(c))
    po.sub("User characteristics").add(*_user_characteristics(c))
    po.sub("Limitations").add(*_constraints(c))
    intro.sub("Definitions").add(*_definitions(c, only="term"))

    refs = Section("References").add(*_references(c, std))

    req = Section("Requirements")
    _functions(c, req.sub("Functions"))
    req.sub("Performance requirements").add(*c.blocks("performance", "performance requirements"))
    req.sub("Usability requirements").add(*c.blocks("usability", "usability requirements"))
    req.sub("Interface requirements").add(
        *_interface_spec(c), *c.blocks("software_interfaces", "software interface requirements"),
        *[c.block(r) for r in c.in_section("user_interfaces")])
    req.sub("Logical database requirements").add(*_logical_database(c))
    req.sub("Design constraints").add(*c.blocks("compliance", "design constraints"))
    attrs = req.sub("Software system attributes")
    attrs.sub("Reliability and availability").add(*c.blocks("availability", "reliability or availability requirements"))
    attrs.sub("Security and privacy").add(*c.blocks("security", "security or privacy requirements"))
    attrs.sub("Maintainability").add(*c.blocks("maintainability", "maintainability requirements"))
    attrs.sub("Portability").add(_none("portability requirements"))
    req.sub("Supporting information").add(*c.blocks("other", "other requirements"))

    ver = Section("Verification").add(*_verification(c))

    app = Section("Appendices")
    app.sub("Assumptions and dependencies").add(*_assumptions(c))
    app.sub("Acronyms and abbreviations").add(*_definitions(c, only="acronym"))
    rtm = app.sub("Requirements traceability matrix")
    app.sub("Compliance-control matrix").add(*_compliance_matrix(c))
    app.sub("Risk register").add(*_risk_register(c))
    app.sub("Recommended development life cycle").add(*_sdlc(c))
    app.sub("Open issues (TBD list)").add(*_open_issues(c))
    app.sub("Analysis notes").add(*_analysis_notes(c))

    doc.sections = [intro, refs, req, ver, app]
    return _finish(doc, c, rtm)


def build(run: dict, narrative: Narrative, use_cases: list[UseCase], sim,
          layout: str = "ieee830", today: str | None = None,
          data_model: DataModel | None = None,
          interfaces: InterfaceSpec | None = None) -> SRSDocument:
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {sorted(LAYOUTS)}")
    c = _Ctx(run, narrative, use_cases, data_model or DataModel(), interfaces or InterfaceSpec(), sim)
    today = today or date.today().isoformat()
    return build_ieee830(c, today) if layout == "ieee830" else build_iso29148(c, today)
