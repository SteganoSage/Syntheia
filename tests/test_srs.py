"""The SRS generator: the course's IEEE 830 outline, completeness, traceability.

Built from a real 8B run (tests/fixtures/run_kyc_8b.json) with the LLM parts
replaced by their deterministic fallbacks, so every assertion is about what the
code guarantees -- not about prose a model wrote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.generators.srs import builder, derive
from src.generators.srs.analysis_models import (
    DataModelDraft,
    EntityDraft,
    InterfaceDraft,
    InterfaceSet,
    RelationshipDraft,
    UseCaseFlow,
    UseCaseHead,
    UseCaseList,
    _goal,
    fallback_use_cases,
    write_data_model,
    write_interfaces,
    write_use_cases,
)
from src.generators.srs.model import Bullets, Para, Requirement, Table
from src.generators.srs.narrative import (
    QuotedItem,
    SRSNarrative,
    UserClassDraft,
    fallback_narrative,
    named_user_classes,
    write_narrative,
)
from src.utils.similarity import LexicalSimilarity

RUN = json.loads((Path(__file__).parent / "fixtures/run_kyc_8b.json").read_text())
SIM = LexicalSimilarity()
STATED = [r["id"] for r in RUN["requirements"]]
PROPOSED = derive.proposed_requirements(RUN, start=len(STATED) + 1)
ALL_IDS = STATED + [p["id"] for p in PROPOSED]


def _doc(layout="ieee830", run=RUN):
    reqs = run["requirements"]
    functional = [r for r in reqs if derive.primary_section(r["statement"], r["category"]) == "functional"]
    return builder.build(run, fallback_narrative(run["scenario_text"], reqs),
                         fallback_use_cases(functional), SIM, layout=layout, today="2026-11-22")


def _titles(doc, depth):
    return [f"{s.number} {s.title}" for d, s in doc.walk() if d == depth and not s.unnumbered]


def _blocks(doc):
    return [(s, b) for _, s in doc.walk() for b in s.blocks if isinstance(b, Requirement)]


def _attr(block, name):
    return dict(block.attributes)[name]


# ── Structure: the course slides, 3.7 deliberately absent ─────────────

def test_ieee830_outline_matches_the_course_slides():
    doc = _doc()
    assert _titles(doc, 1)[:3] == ["1 Introduction", "2 Overall description", "3 Specific requirements"]
    level2 = [t for t in _titles(doc, 2) if t[0] in "123"]
    assert level2 == [
        "1.1 Purpose", "1.2 Scope", "1.3 Definitions, acronyms, and abbreviations",
        "1.4 References", "1.5 Overview",
        "2.1 Product perspective", "2.2 Product functions", "2.3 User characteristics",
        "2.4 Constraints", "2.5 Assumptions and dependencies",
        "3.1 External interfaces", "3.2 Functions", "3.3 Performance requirements",
        "3.4 Logical database requirements", "3.5 Design constraints",
        "3.6 Software system quality attributes"]
    pp = [s.title for d, s in doc.walk() if s.number.startswith("2.1.") and d == 3]
    assert pp == ["System interfaces", "User interfaces", "Hardware interfaces", "Software interfaces",
                  "Communications interfaces", "Memory constraints", "Operations",
                  "Site adaptation requirements"]
    ext = [s.title for d, s in doc.walk() if s.number.startswith("3.1.") and d == 3]
    assert ext == ["User interfaces", "Hardware interfaces", "Software interfaces",
                   "Communications interfaces"]
    dc = [s.title for d, s in doc.walk() if s.number.startswith("3.5.") and d == 3]
    assert dc == ["Standards compliance", "Software and hardware constraints"]


def test_no_object_oriented_models_section():
    doc = _doc()
    assert not any(s.number.startswith("3.7") for _, s in doc.walk())
    assert not any("object" in s.title.lower() or "uml" in s.title.lower() for _, s in doc.walk())


def test_appendices_then_an_unnumbered_index():
    doc = _doc()
    apps = [s for s in doc.sections if s.appendix]
    assert [s.number for s in apps] == list("ABCDEFG")
    assert apps[0].title == "Requirements traceability matrix"
    last = doc.sections[-1]
    assert last.title == "Index" and last.unnumbered and last.number == ""
    assert last.label() == "Index"


def test_index_entries_point_at_real_sections():
    doc = _doc()
    numbers = {s.number for _, s in doc.walk() if s.number}
    table = next(b for b in doc.sections[-1].blocks if isinstance(b, Table))
    assert {"KYC", "PAN", "NSDL"} <= {row[0] for row in table.rows}
    for term, where in table.rows:
        for n in where.replace(" …", "").split(", "):
            assert n in numbers, (term, n)


def test_iso29148_outline_is_exact():
    doc = _doc("iso29148")
    assert _titles(doc, 1) == ["1 Introduction", "2 References", "3 Requirements",
                               "4 Verification", "5 Appendices"]
    req = [s.title for d, s in doc.walk() if d == 2 and s.number.startswith("3.")]
    assert req == ["Functions", "Performance requirements", "Usability requirements",
                   "Interface requirements", "Logical database requirements",
                   "Design constraints", "Software system attributes", "Supporting information"]
    overview = [s.title for d, s in doc.walk() if s.number.startswith("1.3.")]
    assert overview == ["Product perspective", "Product functions", "User characteristics", "Limitations"]


def test_unknown_layout_is_refused():
    with pytest.raises(ValueError):
        _doc("ieee1233")


# ── Every requirement once, in the section its type belongs to ────────

@pytest.mark.parametrize("layout", ["ieee830", "iso29148"])
def test_every_requirement_is_stated_exactly_once(layout):
    ids = [b.id for _, b in _blocks(_doc(layout))]
    assert sorted(ids) == sorted(ALL_IDS)


def test_proposals_exist_for_this_run():
    # The fixture has compliance and security gaps; without them most of the
    # assertions below would hold vacuously.
    assert len(PROPOSED) >= 10
    assert {p["origin"] for p in PROPOSED} == {"regulatory"}


def test_requirements_sit_under_the_section_their_type_belongs_to():
    expected = {"3.1": "External interface", "3.2": "Functional", "3.3": "Non-functional",
                "3.4": "Non-functional", "3.5": "Design constraint", "3.6": "Non-functional"}
    for sec, b in _blocks(_doc()):
        assert _attr(b, "Type").startswith(expected[sec.number[:3]]), (b.id, sec.number)


def test_every_block_carries_the_full_attribute_set():
    names = None
    for _, b in _blocks(_doc()):
        got = [k for k, _ in b.attributes]
        names = names or got
        assert got == names and len(got) == 14
    assert names[:7] == ["Type", "Origin", "Priority", "Rationale", "Source",
                         "Acceptance criteria", "Verification method"]


def test_origin_says_where_each_requirement_came_from():
    for _, b in _blocks(_doc()):
        origin = _attr(b, "Origin")
        if b.id in STATED:
            assert origin.startswith("Stated")
        else:
            assert origin.startswith("Proposed") and "rule " in origin
            assert _attr(b, "Status") == "Proposed — Pending approval"


def test_implied_requirements_are_labelled_and_queued_for_confirmation():
    run = json.loads(json.dumps(RUN))
    run["requirements"].append(dict(run["requirements"][0], id="REQ-050", origin="implied",
                                    statement="The system shall let customers resume an "
                                              "interrupted onboarding session"))
    doc = _doc(run=run)
    block = next(b for _, b in _blocks(doc) if b.id == "REQ-050")
    assert _attr(block, "Origin").startswith("Implied")
    issues = next(s for s in doc.sections if s.title.startswith("Open issues"))
    rows = next(b for b in issues.blocks if isinstance(b, Table)).rows
    assert any(r[2] == "Implied requirements" and "REQ-050" in r[3] for r in rows)


def test_counts_table_adds_up():
    doc = _doc()
    spec = next(s for s in doc.sections if s.title == "Specific requirements")
    table = next(b for b in spec.blocks if isinstance(b, Table))
    total = table.rows[-1]
    assert total[0] == "Total" and int(total[-1]) == len(ALL_IDS)
    assert int(total[1]) == len(STATED) and int(total[3]) == len(PROPOSED)
    assert sum(int(r[-1]) for r in table.rows[:-1]) == len(ALL_IDS)


def test_scope_states_the_final_count_and_what_is_excluded():
    doc = _doc()
    scope = next(s for _, s in doc.walk() if s.number == "1.2")
    text = " ".join(b.text for b in scope.blocks if isinstance(b, Para))
    assert f"states {len(ALL_IDS)} requirements" in text
    assert "The system will:" in text and "The system will not:" in text


# ── 3.2: use cases cover every functional requirement ─────────────────

def test_use_cases_cover_every_functional_requirement_once():
    doc = _doc()
    funcs = next(s for _, s in doc.walk() if s.number == "3.2")
    related = []
    for uc_sec in funcs.children:
        spec = uc_sec.blocks[0]
        assert isinstance(spec, Table) and spec.rows[0][0] == "Use case"
        ids = dict(spec.rows)["Related requirements"].split(", ")
        related += ids
        assert [b.id for b in uc_sec.blocks if isinstance(b, Requirement)] != []
        assert {b.id for b in uc_sec.blocks if isinstance(b, Requirement)} == set(ids)
    functional_blocks = [b.id for s, b in _blocks(doc) if s.number.startswith("3.2")]
    assert sorted(related) == sorted(functional_blocks)
    assert len(related) == len(set(related))


def test_functional_proposals_get_a_regulatory_use_case():
    doc = _doc()
    reg = next(s for _, s in doc.walk() if s.title.endswith("Meet regulatory obligations"))
    ids = [b.id for b in reg.blocks if isinstance(b, Requirement)]
    assert ids and all(i not in STATED for i in ids)


def test_user_classes_are_users_not_requirement_sources():
    doc = _doc()
    sec = next(s for _, s in doc.walk() if s.number == "2.3")
    names = [row[0] for row in next(b for b in sec.blocks if isinstance(b, Table)).rows]
    assert "Customer" in names
    assert "Business Analyst" not in names, "a requirement's source is not a user class"


# ── Cross-references, matrices, proposals linked through ───────────────

def test_cross_references_point_where_the_requirement_is():
    doc = _doc()
    where = {b.id: s.number for s, b in _blocks(doc)}
    refs = [item for _, s in doc.walk() for b in s.blocks if isinstance(b, Bullets) for item in b.items
            if re.match(r"REQ-\d+ — .*\(§", item)]
    assert refs, "section 2 refers to requirements by id"
    for item in refs:
        rid, sec = re.match(r"(REQ-\d+) — .*\(§([\dA-Z.]+)\)$", item).groups()
        assert where[rid] == sec, item


def test_empty_sections_say_so_rather_than_vanish():
    doc = _doc()
    hw = next(s for _, s in doc.walk() if s.number == "3.1.2")
    assert hw.title == "Hardware interfaces"
    assert isinstance(hw.blocks[0], Para) and "No hardware" in hw.blocks[0].text


def test_rtm_and_verification_cover_every_requirement():
    doc = _doc()
    for title in ("Requirements traceability matrix", "Verification matrix"):
        sec = next(s for s in doc.sections if s.title == title)
        table = next(b for b in sec.blocks if isinstance(b, Table))
        assert sorted(row[0] for row in table.rows) == sorted(ALL_IDS)


def test_compliance_gaps_point_at_their_proposal_and_are_not_asked_twice():
    doc = _doc()
    by_rule = {p["rule_id"]: p["id"] for p in PROPOSED}
    matrix = next(s for s in doc.sections if s.title == "Compliance-control matrix")
    rows = next(b for b in matrix.blocks if isinstance(b, Table)).rows
    for rule, status, action in ((r[0], r[3], r[4]) for r in rows):
        if rule in by_rule:
            assert f"Proposed as {by_rule[rule]}" in action
    issues = next(s for s in doc.sections if s.title.startswith("Open issues"))
    text = " ".join(" ".join(r) for r in next(b for b in issues.blocks if isinstance(b, Table)).rows)
    for p in PROPOSED:
        assert p["statement"][:60] not in text, f"{p['id']} re-asked as a question"


def test_references_and_tags_agree():
    doc = _doc()
    refs = next(s for _, s in doc.walk() if s.title == "References")
    table = next(b for b in refs.blocks if isinstance(b, Table))
    tags = [row[0] for row in table.rows]
    assert tags[:2] == ["[R0]", "[S1]"]
    assert [row[1] for row in table.rows[2:]] == derive.cited_titles(RUN)
    used = {m for _, b in _blocks(doc) for k, v in b.attributes if k == "Regulatory basis"
            for m in re.findall(r"\[R\d+\]", v)}
    assert used and used <= set(tags), "every tag used in a block is in the references"


# ── Derivations ────────────────────────────────────────────────────────

@pytest.mark.parametrize("statement, cats, section", [
    ("The system shall display a loan decision within 30 seconds", ["Functional", "Performance"], "performance"),
    ("The system shall pull credit scores from CIBIL in real time", ["Functional", "Integration"], "software_interfaces"),
    ("The system shall encrypt customer data at rest", ["Security", "Data-management"], "security"),
    ("The system shall store video KYC sessions for 8 years", ["Data-management", "Regulatory"], "database"),
    ("The system shall escalate high-risk customers", ["Functional", "Regulatory"], "functional"),
    ("The system shall comply with IRDAI regulations", ["Regulatory"], "compliance"),
    ("The system shall do something", [], "functional"),
])
def test_primary_section(statement, cats, section):
    assert derive.primary_section(statement, cats) == section


def test_proposed_requirements_dedupe_and_only_weak_findings_amend():
    run = {"compliance": {"rule_findings": [
        {"rule_id": "A", "status": "gap", "proposed_requirement": "The system shall do X.",
         "family": "record_retention", "severity": "high", "satisfied_by": []},
        {"rule_id": "B", "status": "weak", "proposed_requirement": "The system shall do Y.",
         "family": "consent", "severity": "medium", "satisfied_by": ["REQ-002"]},
        {"rule_id": "C", "status": "gap", "proposed_requirement": "the system shall do  x.",
         "family": "consent", "severity": "low"},
        {"rule_id": "D", "status": "satisfied", "proposed_requirement": "The system shall do Z."},
    ]}}
    out = derive.proposed_requirements(run, start=8)
    assert [p["id"] for p in out] == ["REQ-008", "REQ-009"]
    assert out[0]["amends"] == [] and out[0]["section_hint"] == "database"
    assert out[1]["amends"] == ["REQ-002"] and out[1]["dependencies"] == ["REQ-002"]


def test_glossary_includes_only_terms_present_and_flags_unknown_acronyms():
    found, unknown = derive.glossary_for("Verify the PAN via NSDL for KYC. Also XYZQ and REQ-001.")
    assert {e["term"] for e in found} >= {"PAN", "NSDL", "KYC"}
    assert "RBI" not in {e["term"] for e in found}
    assert unknown == ["XYZQ"]


def test_tracing_marks_implicit_requirements():
    out = derive.trace_sources("Customers upload PAN cards. Decisions are fast.",
                               {"REQ-1": "The system shall let customers upload PAN cards",
                                "REQ-2": "The system shall rotate encryption keys quarterly"}, SIM)
    assert out["REQ-1"][0] == "Customers upload PAN cards."
    assert out["REQ-2"] is None


@pytest.mark.parametrize("statement, goal", [
    ("The system shall validate PAN details against the NSDL database in real-time",
     "Validate PAN details"),
    ("The system shall assign a risk score to each customer based on their profile",
     "Assign a risk score to each customer"),
    ("The system shall allow customers to upload documents", "Upload documents"),
    ("The system shall let the customer resume an onboarding session", "Resume an onboarding session"),
])
def test_use_case_names_come_from_the_verb_phrase(statement, goal):
    assert _goal(statement) == goal


def test_named_user_classes_finds_people_not_systems():
    names = [u[0] for u in named_user_classes(
        "Customers apply online; a loan officer reviews; the NSDL database is queried.", [])]
    assert names == ["Customer", "Loan Officer"]


# ── LLM gates ──────────────────────────────────────────────────────────

def test_narrative_drops_unquoted_items_and_one_word_prose(monkeypatch):
    scenario = ("A neo-bank needs a digital onboarding platform for customers. It must go live in "
                "six months. Card issuance is out of scope.")
    monkeypatch.setattr("src.utils.llm.call_structured", lambda **kw: SRSNarrative(
        product_name="Digital Onboarding Platform",
        purpose="This SRS specifies the digital onboarding platform for a neo-bank, for its business, engineering and compliance readers.",
        scope="The platform onboards customers digitally for the neo-bank, verifying identity and recording consent before accounts open.",
        product_perspective="new",
        operational_concept="Customers apply.",
        objectives=[QuotedItem(text="Faster onboarding", quote="the board wants growth")],
        constraints=[QuotedItem(text="Six-month deadline", quote="go live in six months")],
        out_of_scope=[QuotedItem(text="Card issuance", quote="Card issuance is out of scope"),
                      QuotedItem(text="Branch onboarding", quote=""),
                      QuotedItem(text="Crypto trading", quote="crypto is excluded"),
                      QuotedItem(text="To be confirmed", quote=""),
                      QuotedItem(text="Assumption: The system will not handle customer support.", quote="")],
        user_classes=[UserClassDraft(name="Customer", description="Low, as they only upload documents online.",
                                     technical_expertise="Low", frequency_of_use="Once"),
                      UserClassDraft(name="Space Pilot", description="x", technical_expertise="High",
                                     frequency_of_use="Daily")]))
    n = write_narrative(scenario, [])
    assert n.source == "llm" and n.dropped == 2
    assert n.objectives == [] and n.constraints == [("Six-month deadline", "go live in six months")]
    assert len(n.perspective.split()) > 5, "a one-word section falls back"
    assert len(n.operational_concept.split()) > 5, "a two-word section falls back"
    assert n.out_of_scope == [("Card issuance", "Card issuance is out of scope"), ("Branch onboarding", ""),
                              ("Handle customer support", "")], "placeholders dropped, prefixes stripped"
    assert [u[0] for u in n.user_classes] == ["Customer"], "a user class the text never names is dropped"
    _, desc, skill, _ = n.user_classes[0]
    assert (desc, skill) == ("Upload documents online.", "Low"), \
        "an expertise reason in the description becomes the level plus what they do"


def test_use_cases_accept_only_real_ids_once_and_lose_nothing(monkeypatch):
    functional = [{"id": f"REQ-00{i}", "statement": f"The system shall let the customer apply for product {i}",
                   "source_stakeholder": "Business Analyst"} for i in range(1, 5)]
    scenario = "Customers apply online and a credit bureau is consulted."

    def fake(**kw):
        if kw["response_model"] is UseCaseList:
            return UseCaseList(use_cases=[
                UseCaseHead(name="Apply for product", primary_actor="Customer",
                            supporting_actors=["Credit Bureau", "Mars Rover"],
                            description="d", preconditions=["p"], postconditions=["q"],
                            requirement_ids=["REQ-001", "REQ-999", "REQ-002"]),
                UseCaseHead(name="Submit loan application", primary_actor="Business Analyst",
                            supporting_actors=[], description="invented", preconditions=[],
                            postconditions=[], requirement_ids=["REQ-002", "REQ-003"]),
            ])
        return UseCaseFlow(main_flow=["1. Step 1: The customer applies", "Step 2: The system checks",
                                      "3 documents are stored"],
                           alternative_flows=["Step 2: bureau down -> retry"])

    monkeypatch.setattr("src.utils.llm.call_structured", fake)
    cases, how = write_use_cases(functional, scenario, functional)
    assert how == "llm"
    assert [(c.id, c.requirement_ids) for c in cases] == [
        ("UC-01", ["REQ-001", "REQ-002"]), ("UC-02", ["REQ-003"]), ("UC-03", ["REQ-004"])]
    assert cases[0].name == "Apply for product"
    assert cases[0].supporting_actors == ["Credit Bureau"], "ungrounded actor dropped"
    assert cases[1].name == "Apply for product 3", "a goal its requirements never describe is renamed"
    assert cases[1].description != "invented"
    assert cases[1].primary_actor == "Customer", "a requirement source is not an actor"
    assert cases[2].name == "Apply for product 4", "a requirement the model left out gets its own use case"
    assert cases[2].primary_actor == "Customer"
    assert cases[0].main_flow == ["The customer applies", "The system checks", "3 documents are stored"]


def test_grounding_needs_every_distinctive_word():
    from src.generators.srs.analysis_models import grounded, name_supported

    corpus = "the customer's pan is validated against the nsdl database via internet banking"
    assert grounded("NSDL Database", corpus) and grounded("Customers", corpus)
    assert not grounded("Internet Service Provider", corpus)
    assert name_supported("Validate customer PAN", corpus)
    assert not name_supported("Submit loan application", corpus)


def test_data_model_keeps_only_grounded_entities_and_their_relationships(monkeypatch):
    reqs = [{"id": "REQ-001", "statement": "The system shall store each customer and their documents"}]
    monkeypatch.setattr("src.utils.llm.call_structured", lambda **kw: DataModelDraft(
        entities=[EntityDraft(name="Customer", description="d", attributes=["id", "name"], identifier="id"),
                  EntityDraft(name="Document", description="d", attributes=["id"], identifier="id"),
                  EntityDraft(name="Spaceship", description="d", attributes=["id"], identifier="id")],
        relationships=[RelationshipDraft(source="Customer", verb="uploads", target="Document",
                                         targets_per_source="many", sources_per_target="one"),
                       RelationshipDraft(source="Customer", verb="flies", target="Spaceship",
                                         targets_per_source="one", sources_per_target="one"),
                       RelationshipDraft(source="Document", verb="belongs to", target="Customer",
                                         targets_per_source="one", sources_per_target="many")]))
    dm = write_data_model("", reqs)
    assert [e[0] for e in dm.entities] == ["Customer", "Document"]
    assert dm.relationships == [
        ("Customer", "uploads", "Document", "1:N (one Customer to many Document)"),
        ("Document", "belongs to", "Customer", "N:1 (many Document to one Customer)")]


def test_a_foreign_key_settles_the_side_it_is_on(monkeypatch):
    # The 8B labelled "a customer can have multiple KYC documents" N:1. The
    # document row carries customer_id, so each document has ONE customer.
    reqs = [{"id": "REQ-001", "statement": "The system shall store each customer and their documents"}]
    monkeypatch.setattr("src.utils.llm.call_structured", lambda **kw: DataModelDraft(
        entities=[EntityDraft(name="Customer", description="d", attributes=["customer_id", "name"],
                              identifier="customer_id"),
                  EntityDraft(name="Document", description="d", attributes=["document_id", "customer_id"],
                              identifier="document_id")],
        relationships=[RelationshipDraft(source="Customer", verb="has", target="Document",
                                         targets_per_source="many", sources_per_target="many")]))
    assert write_data_model("", reqs).relationships[0][3].startswith("1:N")


def test_interfaces_drop_counterparts_the_text_never_names(monkeypatch):
    reqs = [{"id": "REQ-001", "statement": "The system shall validate PAN against NSDL"}]
    monkeypatch.setattr("src.utils.llm.call_structured", lambda **kw: InterfaceSet(interfaces=[
        InterfaceDraft(name="PAN check", kind="software", counterpart="NSDL", inputs="PAN",
                       outputs="status", data_format=""),
        InterfaceDraft(name="Satellite", kind="communications", counterpart="Starlink", inputs="x",
                       outputs="y", data_format="z")]))
    spec = write_interfaces("", reqs)
    assert spec.rows == [("PAN check", "software", "NSDL", "PAN", "status", "To be confirmed")]


def test_llm_failure_falls_back_without_losing_requirements(monkeypatch):
    def down(**kw):
        raise ConnectionError("endpoint gone")

    monkeypatch.setattr("src.utils.llm.call_structured", down)
    functional = [{"id": "REQ-001", "statement": "The system shall verify identity"}]
    cases, how = write_use_cases(functional, "", functional)
    assert how == "fallback" and cases[0].requirement_ids == ["REQ-001"]
    assert write_data_model("", functional).entities == []
    assert write_interfaces("", functional).rows == []


# ── Rendering ──────────────────────────────────────────────────────────

def test_docx_and_markdown_render(tmp_path):
    from docx import Document

    from src.generators.srs.docx_render import render_docx
    from src.generators.srs.md_render import render_markdown

    doc = _doc()
    path = tmp_path / "srs.docx"
    render_docx(doc, path)
    d = Document(str(path))
    headings = [p.text for p in d.paragraphs if p.style.name.startswith("Heading")]
    assert headings[0].split() == ["1", "Introduction"]
    assert any(h.startswith("Appendix A") for h in headings)
    assert headings[-1] == "Index"
    xml = d.element.xml
    assert 'TOC \\o "1-3"' in xml, "table of contents field"
    assert "updateFields" in d.settings.element.xml
    n_req_tables = sum(1 for t in d.tables if t.rows[0].cells[0].text.startswith("REQ"))
    assert n_req_tables == len(ALL_IDS)
    md = render_markdown(doc)
    for rid in ALL_IDS:
        assert f"**{rid}**" in md
    assert "## Index" in md


def test_markdown_keeps_line_breaks_inside_cells():
    from src.generators.srs.md_render import _cell

    assert _cell("1. a\n2. b|c") == "1. a<br>2. b\\|c"


def test_generate_srs_offline_writes_both_layouts(tmp_path):
    from src.generators.srs import generate_srs

    for layout in ("ieee830", "iso29148"):
        paths = generate_srs(RUN, tmp_path, layout=layout, use_llm=False, sim=SIM, verbose=False,
                             formats=("docx", "md"))
        assert {p.suffix for p in paths.values()} == {".docx", ".md"}
        assert all(p.exists() and p.stat().st_size > 1000 for p in paths.values())


def test_markdown_is_the_default_format(tmp_path):
    from src.generators.srs import generate_srs

    paths = generate_srs(RUN, tmp_path, use_llm=False, sim=SIM, verbose=False)
    assert [p.name for p in paths.values()] == ["srs.md"]
    assert not (tmp_path / "srs.docx").exists()


def test_both_layouts_share_one_analysis_pass(tmp_path, monkeypatch):
    from src.generators.srs import generate as gen

    calls = []
    monkeypatch.setattr(gen, "fallback_narrative",
                        lambda *a: calls.append("narrative") or fallback_narrative(*a))
    real = gen.write_use_cases
    monkeypatch.setattr(gen, "write_use_cases",
                        lambda *a, **k: calls.append("use_cases") or real(*a, **k))
    paths = gen.generate_srs(RUN, tmp_path, layout="both", use_llm=False, sim=SIM, verbose=False)
    assert calls == ["narrative", "use_cases"]
    assert {p.name for p in paths.values()} == {"srs.md", "srs_iso29148.md"}


def test_camelcase_entity_names_are_split_before_grounding(monkeypatch):
    # qwen3:14b wrote "KYCDocument"; read as one word it matched nothing in the
    # scenario and 5 of 7 entities were dropped.
    from src.generators.srs.analysis_models import spaced

    assert spaced("VideoKYCSession") == "Video KYC Session"
    assert spaced("KYC_Document") == "KYC Document"
    assert spaced("Customer") == "Customer"
    reqs = [{"id": "REQ-001", "statement": "The system shall store each customer's KYC document"}]
    monkeypatch.setattr("src.utils.llm.call_structured", lambda **kw: DataModelDraft(
        entities=[EntityDraft(name="Customer", description="d", attributes=["customer_id"],
                              identifier="customer_id"),
                  EntityDraft(name="KYCDocument", description="d", attributes=["document_id", "customer_id"],
                              identifier="document_id")],
        relationships=[RelationshipDraft(source="Customer", verb="has", target="KYCDocument",
                                         targets_per_source="many", sources_per_target="one")]))
    dm = write_data_model("", reqs)
    assert [e[0] for e in dm.entities] == ["Customer", "KYC Document"]
    assert dm.relationships[0][:3] == ("Customer", "has", "KYC Document")
