"""Sector filtering: keep a bank's citations out of the insurance regulator, and back."""

from __future__ import annotations

import pytest

from src.rag import retriever
from src.rag.sectors import allowed, doc_sector, scenario_sectors

IRDAI = {"applicability": "insurance", "title": "IRDAI Cyber Guidelines"}
RBI = {"applicability": "kyc_onboarding,lending,payments", "title": "RBI KYC"}
DPDP = {"applicability": "kyc_onboarding,general", "title": "DPDP Act"}
FUNCTION_ONLY = {"applicability": "fraud_detection,regulatory_reporting", "title": "x"}


@pytest.mark.parametrize("text, sectors", [
    ("A neo-bank wants digital KYC onboarding.", {"banking"}),
    ("An NBFC wants a credit risk engine.", {"banking"}),
    ("An insurance company needs a claims processing portal under IRDAI.", {"insurance"}),
    ("A bank's insurance arm sells policies to account holders.", {"banking", "insurance"}),
    ("A company is rewriting its payroll system.", set()),
])
def test_scenario_sectors(text, sectors):
    assert scenario_sectors(text) == sectors


def test_document_sectors():
    assert doc_sector(IRDAI["applicability"]) == "insurance"
    assert doc_sector(RBI["applicability"]) == "banking"
    assert doc_sector(DPDP["applicability"]) == "cross", "general binds everyone"
    assert doc_sector(FUNCTION_ONLY["applicability"]) == "cross", "never exclude on a guess"
    assert doc_sector("supplied") == "cross"


def test_a_bank_does_not_cite_the_insurance_regulator():
    assert not allowed(IRDAI, {"banking"})
    assert allowed(RBI, {"banking"}) and allowed(DPDP, {"banking"})


def test_an_insurer_does_not_cite_the_banking_regulator():
    assert not allowed(RBI, {"insurance"})
    assert allowed(IRDAI, {"insurance"}) and allowed(DPDP, {"insurance"})


def test_unknown_or_mixed_sectors_filter_nothing():
    for meta in (IRDAI, RBI, DPDP):
        assert allowed(meta, set()) and allowed(meta, None)
        assert allowed(meta, {"banking", "insurance"})


def test_grounding_drops_out_of_sector_citations(monkeypatch):
    hit = lambda path, app, score: {"score": score, "text": "t", "metadata": {
        "doc_path": path, "clause_id": "1", "doc_type": "regulation",
        "title": path, "citation": f"{path} §1", "applicability": app}}
    monkeypatch.setattr(retriever, "_query_all", lambda *a, **k: [
        hit("irdai", "insurance", 0.80), hit("rbi", "kyc_onboarding", 0.70)])
    g = retriever.ground_requirement("The system shall assign user IDs.", sectors={"banking"})
    assert g["applicable_regulations"] == ["rbi §1"]
    g = retriever.ground_requirement("The system shall assign user IDs.")
    assert g["applicable_regulations"][0] == "irdai §1", "no sector, no filter"
