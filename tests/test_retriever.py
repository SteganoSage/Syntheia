"""Citation logic in the retriever, with the vector search stubbed out.

`ground_requirement` decides which retrieved passages may be called
"applicable regulations". That decision is code, not similarity, and it is the
part that has to be right: tenders are never regulations, and a named
instrument the corpus does not hold is never "supported" by something merely
on-topic.
"""

from __future__ import annotations

import pytest

from src.rag import retriever
from src.rag.retriever import _dedupe, _named_instruments, _supports_named


# ── Named-instrument detection ─────────────────────────────────────────

def test_a_named_regulation_yields_its_distinctive_words():
    named = _named_instruments("The system shall comply with RBI's digital lending guidelines.")
    assert named == [{"digital", "lending"}]


def test_account_activation_is_not_a_named_act():
    # Problem log 4.2: without a trailing word boundary "act" matched inside
    # "account activation" and demoted a correct KYC citation.
    assert _named_instruments("Escalate high-risk customers before account activation.") == []


def test_issuer_names_alone_are_not_distinctive():
    # Every RBI document matches "RBI"; it must not satisfy a named-rule check.
    assert _named_instruments("Comply with RBI rules.") == []


def test_a_title_overlap_supports_the_named_instrument():
    cites = [{"title": "Digital Lending Directions, 2025"}]
    assert _supports_named(cites, "Comply with RBI's digital lending guidelines.")


def test_an_on_topic_but_different_document_does_not():
    cites = [{"title": "Master Direction on IT Governance"}]
    assert not _supports_named(cites, "Comply with RBI's digital lending guidelines.")


def test_nothing_named_means_nothing_to_check():
    assert _supports_named([], "The system shall encrypt data at rest.")


# ── Deduplication ──────────────────────────────────────────────────────

def _hit(doc, clause, score=0.7, doc_type="regulation", title="T"):
    return {"score": score, "text": f"{doc} {clause} text",
            "metadata": {"doc_path": doc, "clause_id": clause, "doc_type": doc_type,
                         "title": title, "citation": f"{title} §{clause}"}}


def test_repeated_clauses_are_dropped():
    out = _dedupe([_hit("a", "9"), _hit("a", "9"), _hit("a", "10")])
    assert [h["metadata"]["clause_id"] for h in out] == ["9", "10"]


def test_per_document_cap():
    out = _dedupe([_hit("a", "1"), _hit("a", "2"), _hit("a", "3"), _hit("b", "1")],
                  max_per_doc=2)
    assert [(h["metadata"]["doc_path"], h["metadata"]["clause_id"]) for h in out] == [
        ("a", "1"), ("a", "2"), ("b", "1")]


# ── is_available: missing vs broken ────────────────────────────────────

def _failing_stats(*a, **k):
    raise RuntimeError("Error in compaction")


def test_a_missing_index_is_quietly_unavailable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(retriever.store, "stats", _failing_stats)
    assert retriever.is_available(tmp_path) is False
    assert capsys.readouterr().out == ""


def test_a_broken_index_says_so(tmp_path, monkeypatch, capsys):
    (tmp_path / "chroma.sqlite3").write_bytes(b"")
    monkeypatch.setattr(retriever.store, "stats", _failing_stats)
    monkeypatch.setattr(retriever, "_REPORTED_BROKEN", set())
    assert retriever.is_available(tmp_path) is False
    out = capsys.readouterr().out
    assert "exists but cannot be read" in out and "Error in compaction" in out
    retriever.is_available(tmp_path)
    assert capsys.readouterr().out == "", "said once, not on every call"


# ── ground_requirement, search stubbed ─────────────────────────────────

@pytest.fixture
def search(monkeypatch):
    """Replace the vector search with a fixed hit list."""
    def install(hits):
        monkeypatch.setattr(retriever, "_query_all", lambda *a, **k: list(hits))
    return install


def test_a_tender_is_precedent_never_a_regulation(search):
    search([_hit("rfp.pdf", "4.1", 0.80, doc_type="tender", title="SBI Loan RFP")])
    g = retriever.ground_requirement("The system shall process loan applications.")
    assert g["applicable_regulations"] == []
    assert [p["title"] for p in g["precedents"]] == ["SBI Loan RFP"]
    assert g["needs_review"] is True


def test_a_regulation_above_the_floor_is_cited(search):
    search([_hit("kyc.pdf", "16(b)", 0.70, title="KYC Master Direction")])
    g = retriever.ground_requirement("Customers shall be risk-categorised.")
    assert g["applicable_regulations"] == ["KYC Master Direction §16(b)"]
    assert g["needs_review"] is False
    assert g["confidence"] == 0.70


def test_below_the_floor_escalates(search):
    search([_hit("kyc.pdf", "16", retriever.MIN_CONFIDENCE - 0.01)])
    g = retriever.ground_requirement("Customers shall be risk-categorised.")
    assert g["citations"] == []
    assert g["needs_review"] is True


def test_a_named_instrument_the_corpus_lacks_is_unsupported(search):
    # The first grounded run cited the IT Governance direction at 0.67 for a
    # requirement naming the digital lending guidelines.
    search([_hit("itgov.html", "3", 0.67, title="Master Direction on IT Governance")])
    g = retriever.ground_requirement(
        "The system shall comply with RBI's digital lending guidelines.")
    assert g["named_instrument_unmatched"] is True
    assert g["applicable_regulations"] == []
    assert g["precedents"], "the near-miss is kept for a reviewer to see"
    assert g["needs_review"] is True
