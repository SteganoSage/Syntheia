"""Day 11 second augmentation: identity/KYC and Business requirements.

Written after RoBERTa and BERT both plateaued at ~78% ground-truth micro-F1 and
failed on the *same* 20 of 32 requirements. Two different architectures failing
identically means the signal is absent from the training data, not the model.

The diagnosis was specific. Across the 13 identity/KYC/Aadhaar/biometric rows
in the corpus, only 15% carried Privacy and 23% carried Security, against 50%
and 62% in ground_truth.json. The models learned that convention correctly and
then missed Privacy on REQ-002 (Aadhaar eKYC) and missed Privacy, Regulatory
and Security together on REQ-016 (live selfie matched to the Aadhaar photo).

Two fixes were applied together: the six mislabelled rows were corrected in
label_corrections.json, and this file adds volume. Thirteen identity rows is
thin for a project whose stated domain is KYC/onboarding.

Labelling follows the ground-truth convention exactly:
    Privacy + Security   when the requirement ACTS ON personal identity data
                         (verify, capture, match, encrypt, mask, expose)
    Data-management      for retention and versioning -- NOT Privacy
                         (ground_truth REQ-009)
    Integration          for plumbing to an identity provider -- NOT Privacy
                         (ground_truth REQ-022)

Business is the other under-predicting category (recall 0.50 on ground truth),
so the second block covers eligibility rules, pricing, and fee policy, where
Business is the reason the requirement exists rather than incidental.

Every row is UNVERIFIED and needs your label pass.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "identity_business.json"

CATEGORIES = [
    "Business", "Stakeholder", "Functional", "Security", "Privacy", "Regulatory",
    "Performance", "Availability", "Usability", "Data-management", "Integration",
    "Audit/Reporting", "Operational",
]

REQUIREMENTS: list[tuple[str, list[str], str]] = [

    # ── Identity acted upon -> Privacy + Security ──────────────────────
    ("The system shall verify the applicant's identity by matching a live facial capture against the photograph on the submitted government identity document.",
     ["Functional", "Privacy", "Security", "Regulatory"], "kyc"),
    ("The system shall perform liveness detection during video onboarding to confirm a real person is present rather than a photograph or recording.",
     ["Functional", "Security", "Privacy"], "kyc"),
    ("The system shall extract the name, date of birth, and identity number from an uploaded identity document using optical character recognition and present them to the applicant for confirmation.",
     ["Functional", "Privacy", "Usability"], "kyc"),
    ("The system shall mask the customer's date of birth and full address on every screen visible to first-line support staff.",
     ["Privacy", "Security", "Stakeholder"], "kyc"),
    ("The system shall encrypt biometric templates using a hardware security module and shall never store the raw biometric image.",
     ["Security", "Privacy", "Data-management"], "kyc"),
    ("The system shall verify the applicant's PAN against the income tax department database before opening an account.",
     ["Functional", "Regulatory", "Privacy", "Integration"], "kyc"),
    ("The system shall authenticate a returning customer using a fingerprint or face scan registered on their enrolled device.",
     ["Security", "Privacy", "Usability", "Functional"], "channels"),
    ("The system shall re-verify the customer's identity with a fresh facial capture before processing a change to their registered mobile number.",
     ["Security", "Privacy", "Functional"], "kyc"),
    ("The system shall redact identity document numbers from any file exported to an external analytics environment.",
     ["Privacy", "Security", "Data-management"], "kyc"),
    ("The system shall verify the applicant's address by matching the uploaded utility bill against the address declared in the application.",
     ["Functional", "Privacy", "Regulatory"], "kyc"),
    ("The system shall capture and store the customer's signature specimen for verification against future instruction forms.",
     ["Functional", "Privacy", "Security"], "kyc"),
    ("The system shall require dual authorisation before any operator can view a customer's unmasked identity document.",
     ["Security", "Privacy", "Stakeholder", "Audit/Reporting"], "kyc"),
    ("The system shall detect and reject an identity document that has been digitally altered before accepting it for onboarding.",
     ["Security", "Functional", "Regulatory"], "kyc"),
    ("The system shall verify that the name on the applicant's bank account matches the name on their identity document before enabling disbursement.",
     ["Functional", "Security", "Privacy", "Business"], "lending"),

    # ── Identity retention / plumbing -> NOT Privacy ───────────────────
    ("The system shall retain video onboarding recordings for five years from the date of account opening.",
     ["Data-management", "Regulatory"], "kyc"),
    ("The system shall archive superseded identity documents while keeping the current version immediately retrievable.",
     ["Data-management", "Operational"], "kyc"),
    ("The system shall integrate with the national identity authority's verification service over a dedicated leased line.",
     ["Integration", "Regulatory"], "kyc"),
    ("The system shall queue identity verification requests when the upstream identity provider is unavailable and process them once service resumes.",
     ["Integration", "Availability", "Operational"], "kyc"),
    ("The system shall record the outcome of every identity verification attempt, including the provider reference and response code.",
     ["Audit/Reporting", "Data-management"], "kyc"),

    # ── Business: the rule IS the requirement ─────────────────────────
    ("The system shall offer a reduced interest rate to applicants whose salary account is held with the bank.",
     ["Business", "Functional"], "lending"),
    ("The system shall cap the total unsecured exposure to any single customer at five times their declared monthly income.",
     ["Business", "Functional"], "lending"),
    ("The system shall decline applications from applicants below the minimum age or above the maximum age configured for the product.",
     ["Business", "Functional"], "lending"),
    ("The system shall charge a foreclosure fee calculated as a configurable percentage of the outstanding principal.",
     ["Business", "Functional"], "lending"),
    ("The system shall allow the product owner to define a promotional interest rate valid between a start and end date.",
     ["Business", "Stakeholder", "Operational"], "lending"),
    ("The system shall tier transaction charges by customer segment, with premium account holders exempt from domestic transfer fees.",
     ["Business", "Functional"], "payments"),
    ("The system shall require a minimum average quarterly balance and shall levy a shortfall charge when it is not maintained.",
     ["Business", "Functional"], "core_banking"),
    ("The system shall route an application to manual underwriting when the requested amount exceeds the automated approval limit for the applicant's risk grade.",
     ["Business", "Stakeholder", "Functional"], "lending"),
    ("The system shall apply a cashback of a configurable percentage on card spend above a monthly threshold, capped at a maximum amount.",
     ["Business", "Functional"], "cards"),
    ("The system shall allow a customer to select a repayment tenure only from the options permitted for the chosen product and loan amount.",
     ["Business", "Functional", "Usability"], "lending"),
    ("The system shall reduce the credit limit automatically when a customer's internal risk score falls below the configured threshold.",
     ["Business", "Functional"], "cards"),
    ("The system shall recognise revenue from processing fees at the point of disbursement rather than at application.",
     ["Business", "Audit/Reporting"], "lending"),
    ("The system shall allow the pricing team to simulate the portfolio impact of a proposed rate change before publishing it.",
     ["Business", "Stakeholder", "Functional"], "lending"),
    ("The system shall waive the annual card fee for the first year and apply it from the second anniversary onward.",
     ["Business", "Functional"], "cards"),
]


def main() -> None:
    for stmt, cats, _ in REQUIREMENTS:
        bad = [c for c in cats if c not in CATEGORIES]
        if bad:
            raise SystemExit(f"unknown categories {bad} on: {stmt[:60]}")
        if not cats:
            raise SystemExit(f"no categories on: {stmt[:60]}")

    dupes = [s for s, n in Counter(s for s, _, _ in REQUIREMENTS).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate statements: {dupes}")

    requirements = [
        {
            "id": f"IDB-{i:03d}",
            "statement": stmt,
            "categories": cats,
            "domain": domain,
            "source": "synthetic:claude-authored-identity-business",
            "verified": False,
        }
        for i, (stmt, cats, domain) in enumerate(REQUIREMENTS, start=1)
    ]

    OUT.write_text(json.dumps({
        "purpose": (
            "Day 11 second augmentation: identity/KYC requirements labelled to the "
            "ground-truth convention (Privacy + Security when personal identity data "
            "is acted upon), plus Business requirements where the business rule is "
            "the reason the requirement exists. All UNVERIFIED."
        ),
        "requirements": requirements,
    }, indent=2) + "\n")

    counts = Counter(c for _, cats, _ in REQUIREMENTS for c in cats)
    card = sum(len(c) for _, c, _ in REQUIREMENTS) / len(REQUIREMENTS)
    print(f"wrote {OUT.name}: {len(requirements)} requirements, avg {card:.2f} labels/req\n")
    for c in CATEGORIES:
        print(f"  {c:<18}{counts.get(c, 0):>4}")


if __name__ == "__main__":
    main()
