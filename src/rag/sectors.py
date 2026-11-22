"""Keep retrieval inside the regulatory sector the scenario belongs to.

`SourceDoc.applicability` was designed for exactly this ("a lending scenario
should not be grounded in insurance solvency rules", schema.py) and nothing
used it. It mattered once the IRDAI guidelines went in on 14 Nov: 1,262 chunks
of generic information-security text (user IDs, passwords, "integrity") that
match banking questions well -- and an insurance regulator's clause is not an
applicable regulation for a bank, however similar the wording.

Sector, not business function, is the line that matters. Within banking, a KYC
requirement may legitimately cite the outsourcing direction; across sectors the
regulator itself is wrong. So:

  document sector  from its applicability tags --
                     "insurance"                       -> insurance
                     "general" (or a supplied doc)     -> cross-sector, always allowed
                     any banking function tag          -> banking
  scenario sector  from its words (`SECTOR_TERMS`), possibly both, possibly none

  allowed when the scenario's sectors are unknown, when the document is
  cross-sector, or when the document's sector is one the scenario is in.

Deliberately conservative: a scenario that mentions both sectors, or neither,
filters nothing. Wrongly excluding the right regulator loses a citation
silently; wrongly including a neighbour is visible and reviewable.
"""

from __future__ import annotations

from src.utils.evidence import find_terms

SECTOR_TERMS: dict[str, list[str]] = {
    "insurance": ["insur*", "policyholder*", "policy holder*", "irdai", "reinsur*",
                  "actuar*", "claims processing", "insurance claim*", "claim settlement",
                  "motor policy", "health policy", "life policy"],
    "banking": ["bank*", "neo bank*", "nbfc*", "loan*", "lend*", "borrower*",
                "credit card*", "debit card*", "upi", "wallet*", "prepaid*", "kyc",
                "deposit*", "rbi", "fund transfer*", "emi", "fintech", "payment gateway*",
                "merchant*", "core banking"],
}

BANKING_TAGS = {"kyc_onboarding", "lending", "payments", "digital_banking"}


def scenario_sectors(text: str) -> set[str]:
    return {s for s, terms in SECTOR_TERMS.items() if find_terms(text or "", terms)}


def doc_sector(applicability) -> str:
    tags = (set(t.strip() for t in applicability.split(",") if t.strip())
            if isinstance(applicability, str) else set(applicability or []))
    if "general" in tags or "supplied" in tags:
        return "cross"
    if "insurance" in tags:
        return "insurance"
    if tags & BANKING_TAGS:
        return "banking"
    return "cross"          # untagged or function-only: never excluded on a guess


def allowed(metadata: dict, sectors: set[str] | None) -> bool:
    if not sectors:
        return True
    s = doc_sector(metadata.get("applicability", ""))
    return s == "cross" or s in sectors


def in_sector(hits: list[dict], sectors: set[str] | None) -> list[dict]:
    return [h for h in hits if allowed(h["metadata"], sectors)]
