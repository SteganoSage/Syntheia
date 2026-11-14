"""The structural chunker, against the synthetic RBI-shaped fixture.

Clause-level chunking is what lets a citation read "§9.2(b)" instead of
"somewhere in this PDF". These tests hold the properties problem log §4 had to
fix one at a time: clause ids carry their parent, tables of contents are not
indexed, oversized sections are not cut mid-word.
"""

from __future__ import annotations

import pytest

from src.rag.chunker import (
    MAX_CHARS,
    _split_oversized,
    chunk_document,
    detect_heading,
    looks_like_toc,
)
from src.rag.loader import MissingManifest, load_document, load_manifest, load_supplied
from src.rag.schema import SourceDoc

from tests.conftest import FIXTURES

FIXTURE = FIXTURES / "knowledge_base/synthetic_kyc_direction.md"


# ── detect_heading ─────────────────────────────────────────────────────

@pytest.mark.parametrize("line, expected", [
    ("9. Customer Identification Procedure", ("numbered", "9")),
    ("9.2 Where the customer is an individual", ("numbered", "9.2")),
    ("(b) an officially valid document", ("alpha", "b")),
    ("(iv) the fourth item", ("alpha", "iv")),
    ("CHAPTER VI", ("chapter", "VI")),
    ("Section 5 Definitions", ("named", "5")),
    ("FR-12: The system shall", ("req", "FR-12")),
    ("REQ-001 The system shall", ("req", "REQ-001")),
])
def test_headings_are_recognised(line, expected):
    assert detect_heading(line) == expected


@pytest.mark.parametrize("line", [
    "",
    "the customer shall provide a photograph",
    "2026 was the year the direction came into force",   # number, no capital after
    "9. " + "Body text that merely starts with a number " * 6,  # > 200 chars
])
def test_body_text_is_not_a_heading(line):
    assert detect_heading(line) is None


# ── _split_oversized ───────────────────────────────────────────────────

def test_small_sections_are_left_whole():
    assert _split_oversized("short text") == ["short text"]


def test_oversized_sections_never_cut_mid_word():
    # Problem log 4.4: a hard character cut split "essential" into "esse" and
    # "ntial" on a real RBI direction, making both halves unretrievable.
    words = "essential documentation requirement " * 200    # one huge paragraph
    pieces = _split_oversized(words)
    assert len(pieces) > 1
    vocabulary = {"essential", "documentation", "requirement"}
    for p in pieces:
        assert len(p) <= MAX_CHARS
        assert set(p.split()) <= vocabulary, "a word was cut in half"


# ── looks_like_toc ─────────────────────────────────────────────────────

def test_a_table_of_contents_is_detected():
    toc = "\n".join([
        "1. Short title and commencement",
        "2. Applicability",
        "3. Definitions",
        "4. Customer Acceptance Policy",
        "5. Risk Management",
        "6. Customer Identification Procedure",
    ])
    assert looks_like_toc(toc)


def test_prose_is_not_a_table_of_contents():
    prose = "\n".join([
        "9.1 Every regulated entity shall undertake identification of the customer.",
        "It shall record the officially valid documents relied upon.",
        "The records shall be retained for five years after the relationship ends.",
        "Verification shall use the issuing authority's records where available.",
        "Exceptions shall be approved by a senior officer and documented.",
    ])
    assert not looks_like_toc(prose)


# ── chunk_document on the fixture ──────────────────────────────────────

@pytest.fixture(scope="module")
def chunks():
    text, doc = load_document(FIXTURE)
    return chunk_document(text, doc)


def test_fixture_produces_clause_level_chunks(chunks):
    assert chunks
    ids = [c.clause_id for c in chunks]
    assert all(ids), "every chunk of a structured document has a clause id"


def test_sub_clauses_carry_their_numbered_parent(chunks):
    # A bare "(b)" cites nothing; there is a (b) in every section.
    ids = {c.clause_id for c in chunks}
    assert any(i.startswith("9.2(") or i == "9.2" for i in ids)
    assert not any(i.startswith("(") for i in ids)


def test_chunk_ids_are_stable_and_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "synthetic_kyc_direction::0000"


def test_citations_name_the_document_clause_and_version(chunks):
    cite = chunks[0].citation
    assert "SYNTHETIC Master Direction" in cite
    assert "§" in cite
    assert "(FIXTURE/001)" in cite
    assert "[fixture v1]" in cite


def test_unstructured_text_gets_no_clause_id():
    doc = SourceDoc(path="x/notes.md", title="Notes", issuer="me", jurisdiction="IN")
    text = ("these notes have no numbering at all and run on as plain prose "
            "about onboarding customers and verifying documents. ") * 5
    out = chunk_document(text, doc)
    assert out and all(c.clause_id == "" for c in out)


# ── Provenance ─────────────────────────────────────────────────────────

def test_a_document_without_a_manifest_is_refused(tmp_path):
    orphan = tmp_path / "orphan.md"
    orphan.write_text("9. Something\n\nBody.", encoding="utf-8")
    with pytest.raises(MissingManifest):
        load_manifest(orphan)


def test_manifest_provenance_is_carried(chunks):
    doc = chunks[0].doc
    assert doc.jurisdiction == "IN"
    assert doc.effective_date.isoformat() == "2025-08-14"
    assert doc.provenance == "curated"


def test_a_supplied_document_is_marked_in_its_citation(tmp_path):
    path = tmp_path / "irdai_note.md"
    path.write_text("4.2 Claims shall be settled within 30 days.", encoding="utf-8")
    _, doc = load_supplied(path)
    assert doc.provenance == "supplied"
    assert doc.citation("4.2").startswith("[supplied] irdai_note.md §4.2")
