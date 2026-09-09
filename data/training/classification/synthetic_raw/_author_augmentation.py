"""Day 11 augmentation: hand-authored finance requirements for the scarce categories.

Written after the 45-epoch BERT run plateaued at 74.4% validation micro-F1 with
eval loss flat from epoch ~24 while train loss kept falling -- a training-data
ceiling, not a schedule problem.

Two changes were made together:
  1. the 60 PROMISE-NFR rows were dropped from the corpus (single-label, mean
     1.00 labels/req against a ground truth averaging 2.50 -- they taught the
     wrong output shape, and included dead examples like "The website should
     use Flash to provide animation on the home page")
  2. this file adds finance-domain requirements weighted toward the categories
     left thinnest by that drop

Category targeting (count in corpus AFTER the PROMISE drop -> priority here):
    Performance      23  <- hit hardest by the drop, heavily weighted below
    Usability        30
    Availability     31
    Data-management  36
    Privacy          37
    Integration      40
    Audit/Reporting  46
    Operational      50
    Stakeholder      58
    Security         60
    Business         62
    Regulatory       96  <- deliberately NOT boosted; its 0.61 F1 is a label
                            boundary problem (96 examples, co-occurs with all
                            12 other labels, never appears alone), so more rows
                            would add noise rather than signal
    Functional      121  <- already plentiful, appears only where it belongs

Every statement is hand-written, not model-generated. `verified` is False on
all of them -- they still need your label pass before they are trusted.

Running this file writes finance_augmentation.json into this directory, which
build_training_set.py picks up automatically via its synthetic_raw/*.json glob.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "finance_augmentation.json"

CATEGORIES = [
    "Business", "Stakeholder", "Functional", "Security", "Privacy", "Regulatory",
    "Performance", "Availability", "Usability", "Data-management", "Integration",
    "Audit/Reporting", "Operational",
]

# (statement, [categories], domain)
REQUIREMENTS: list[tuple[str, list[str], str]] = [

    # ── Performance ────────────────────────────────────────────────────
    ("The system shall return a credit bureau score to the loan officer within 2 seconds at the 95th percentile of requests.",
     ["Performance", "Integration"], "lending"),
    ("The system shall process at least 2,000 UPI payment authorisations per second during peak festival season traffic.",
     ["Performance"], "payments"),
    ("The system shall complete end-of-day interest accrual for all savings accounts within a 90 minute batch window.",
     ["Performance", "Operational"], "core_banking"),
    ("The loan application form shall render its first meaningful paint within 1.5 seconds over a 3G mobile connection.",
     ["Performance", "Usability"], "lending"),
    ("The system shall support 10,000 concurrent authenticated customer sessions without degradation of transaction response times.",
     ["Performance"], "core_banking"),
    ("The system shall complete a full text search across five years of transaction history and return the first page of results within 3 seconds.",
     ["Performance", "Data-management"], "core_banking"),
    ("The fraud scoring engine shall evaluate each card transaction and return an approve or decline decision within 200 milliseconds.",
     ["Performance", "Security"], "cards"),
    ("The system shall generate a customer account statement covering a twelve month period within 5 seconds of the request.",
     ["Performance", "Functional"], "core_banking"),
    ("The system shall sustain a throughput of 500 NACH mandate registrations per minute during collection cycles.",
     ["Performance", "Integration"], "payments"),
    ("The nightly reconciliation job shall process up to 5 million settlement records within a 4 hour window.",
     ["Performance", "Operational", "Data-management"], "payments"),
    ("The system shall complete onboarding identity verification within 60 seconds of document submission at the 90th percentile.",
     ["Performance", "Functional"], "kyc"),
    ("The system shall ingest 50,000 bulk salary credit instructions within a 10 minute processing window.",
     ["Performance", "Integration"], "payments"),
    ("The system shall return account balance queries within 500 milliseconds under peak load.",
     ["Performance"], "core_banking"),
    ("The system shall re-price the entire retail loan portfolio within 30 minutes of a change in the benchmark lending rate.",
     ["Performance", "Business"], "lending"),

    # ── Availability ───────────────────────────────────────────────────
    ("The payment gateway shall maintain 99.95 percent monthly uptime measured excluding scheduled maintenance windows.",
     ["Availability"], "payments"),
    ("The system shall recover from a primary database failure with a recovery time objective of 15 minutes and a recovery point objective of 1 minute.",
     ["Availability", "Operational"], "core_banking"),
    ("The customer mobile banking application shall remain functional in read-only mode when the core banking system is unavailable.",
     ["Availability", "Usability"], "core_banking"),
    ("The system shall replicate all committed transactions synchronously to a geographically separate data centre.",
     ["Availability", "Data-management"], "core_banking"),
    ("The system shall perform rolling deployments without terminating active customer sessions.",
     ["Availability", "Operational"], "core_banking"),
    ("The UPI payment service shall degrade gracefully by queueing collect requests when the NPCI switch is unreachable.",
     ["Availability", "Integration"], "payments"),
    ("The system shall schedule all planned maintenance outside the hours of 06:00 to 23:00 local time.",
     ["Availability", "Operational"], "core_banking"),
    ("The system shall fail over to a standby authentication service within 30 seconds of detecting primary service failure.",
     ["Availability", "Security"], "core_banking"),
    ("The system shall retain full payment functionality during the failure of any single availability zone.",
     ["Availability", "Operational"], "payments"),
    ("The ATM network monitoring service shall detect and report a terminal outage within 3 minutes of occurrence.",
     ["Availability", "Operational", "Integration"], "channels"),

    # ── Usability ──────────────────────────────────────────────────────
    ("The loan application shall allow an applicant to save a partially completed form and resume it later from any device.",
     ["Usability", "Functional"], "lending"),
    ("The system shall present all fee and interest disclosures in the applicant's selected regional language before requiring consent.",
     ["Usability", "Regulatory"], "lending"),
    ("The system shall meet WCAG 2.1 Level AA accessibility criteria on all customer facing screens.",
     ["Usability", "Regulatory"], "channels"),
    ("The system shall display a progress indicator showing the applicant's current step and the remaining steps in the onboarding journey.",
     ["Usability"], "kyc"),
    ("The system shall provide inline validation messages that identify the specific field in error and the corrective action required.",
     ["Usability", "Functional"], "channels"),
    ("The system shall allow a customer to complete video KYC using a mobile device without installing an additional application.",
     ["Usability", "Functional", "Regulatory"], "kyc"),
    ("The transaction history screen shall allow customers to filter by date range, amount range, and counterparty without a page reload.",
     ["Usability", "Functional"], "core_banking"),
    ("The system shall support keyboard-only navigation through the entire loan disbursement workflow for assistive technology users.",
     ["Usability"], "lending"),
    ("The system shall allow a customer to raise a dispute on a transaction directly from the transaction detail screen in no more than three taps.",
     ["Usability", "Functional"], "cards"),
    ("The system shall display the total cost of credit, including all charges, on a single screen before the applicant accepts the offer.",
     ["Usability", "Regulatory", "Business"], "lending"),

    # ── Data-management ────────────────────────────────────────────────
    ("The system shall retain KYC records for eight years following the closure of a customer relationship.",
     ["Data-management", "Regulatory"], "kyc"),
    ("The system shall archive transaction records older than 24 months to cold storage while keeping them retrievable within 24 hours.",
     ["Data-management", "Operational"], "core_banking"),
    ("The system shall maintain a golden record for each customer, deduplicating profiles matched on PAN and registered mobile number.",
     ["Data-management", "Functional"], "kyc"),
    ("The system shall perform a full backup nightly and incremental backups every four hours, with restores tested quarterly.",
     ["Data-management", "Operational"], "core_banking"),
    ("The system shall record the lineage of every derived credit risk attribute back to its source system and extraction timestamp.",
     ["Data-management", "Audit/Reporting"], "lending"),
    ("The system shall purge the personal data of rejected loan applicants once the statutory retention period expires.",
     ["Data-management", "Privacy", "Regulatory"], "lending"),
    ("The system shall validate that every incoming account record conforms to the defined schema and reject non-conforming records with a reason code.",
     ["Data-management", "Integration"], "core_banking"),
    ("The system shall maintain referential integrity between disbursement records and their originating loan applications.",
     ["Data-management"], "lending"),
    ("The system shall version every change to a customer's KYC document set and retain prior versions for the statutory period.",
     ["Data-management", "Regulatory", "Audit/Reporting"], "kyc"),
    ("The system shall reconcile the customer master between the core banking system and the CRM daily and report discrepancies.",
     ["Data-management", "Integration", "Audit/Reporting"], "core_banking"),

    # ── Privacy ────────────────────────────────────────────────────────
    ("The system shall obtain and record the customer's explicit consent before sharing their financial data with an account aggregator.",
     ["Privacy", "Regulatory", "Functional"], "open_banking"),
    ("The system shall mask all but the last four digits of the account number on every screen accessible to contact centre agents.",
     ["Privacy", "Security", "Stakeholder"], "channels"),
    ("The system shall allow a customer to withdraw previously granted data sharing consent and shall cease sharing within 24 hours.",
     ["Privacy", "Regulatory", "Functional"], "open_banking"),
    ("The system shall collect only the personal attributes necessary for the requested product and shall not require optional fields for approval.",
     ["Privacy", "Regulatory"], "kyc"),
    ("The system shall provide a customer with a machine readable export of their personal data within 30 days of a verified request.",
     ["Privacy", "Regulatory", "Functional"], "kyc"),
    ("The system shall log every access to customer personal data with the accessing user, timestamp, and stated purpose.",
     ["Privacy", "Audit/Reporting", "Security"], "core_banking"),
    ("The system shall pseudonymise customer identifiers in all non-production environments used for testing.",
     ["Privacy", "Security", "Operational"], "core_banking"),
    ("The system shall notify affected customers and the regulator within 72 hours of confirming a personal data breach.",
     ["Privacy", "Regulatory", "Operational"], "core_banking"),
    ("The system shall restrict access to customer health information collected for insurance underwriting to authorised underwriters only.",
     ["Privacy", "Security", "Stakeholder"], "insurance"),
    ("The system shall obtain separate unbundled consent for marketing communications and shall not condition product approval on it.",
     ["Privacy", "Regulatory", "Business"], "kyc"),

    # ── Integration ────────────────────────────────────────────────────
    ("The system shall query the CIBIL, Experian, and Equifax bureaus and shall proceed with the first successful response.",
     ["Integration", "Functional"], "lending"),
    ("The system shall consume the NPCI UPI switch API using mutual TLS authentication for all payment requests.",
     ["Integration", "Security"], "payments"),
    ("The system shall publish a disbursement event to the downstream accounting system within 60 seconds of funds release.",
     ["Integration", "Functional"], "lending"),
    ("The system shall retry failed calls to the Aadhaar authentication service using exponential backoff for up to three attempts.",
     ["Integration", "Operational"], "kyc"),
    ("The system shall expose a REST API allowing partner fintechs to originate loan applications on behalf of their customers.",
     ["Integration", "Business"], "lending"),
    ("The system shall reconcile settlement files received from the card network against internal transaction records daily.",
     ["Integration", "Data-management", "Audit/Reporting"], "cards"),
    ("The system shall retrieve business turnover data from the GSTN portal for MSME loan underwriting.",
     ["Integration", "Business", "Functional"], "lending"),
    ("The system shall open a circuit breaker that stops calls to an external credit bureau after five consecutive timeouts.",
     ["Integration", "Availability", "Operational"], "lending"),
    ("The system shall accept webhook callbacks from the payment aggregator and shall verify the signature of every callback.",
     ["Integration", "Security"], "payments"),
    ("The system shall synchronise the branch and ATM location master from the enterprise reference data service every 24 hours.",
     ["Integration", "Data-management"], "channels"),

    # ── Security ───────────────────────────────────────────────────────
    ("The system shall enforce multi-factor authentication for all users initiating a fund transfer above ten thousand rupees.",
     ["Security", "Functional"], "payments"),
    ("The system shall encrypt all cardholder data at rest using AES-256 and in transit using TLS 1.3.",
     ["Security", "Regulatory", "Data-management"], "cards"),
    ("The system shall lock a customer account after five consecutive failed authentication attempts within fifteen minutes.",
     ["Security", "Functional"], "channels"),
    ("The system shall enforce role based access control so that a loan officer cannot approve an application they originated.",
     ["Security", "Business", "Stakeholder"], "lending"),
    ("The system shall rotate all service account credentials every ninety days without service interruption.",
     ["Security", "Operational"], "core_banking"),
    ("The system shall scan all uploaded documents for malware before storing them in the document repository.",
     ["Security", "Functional"], "kyc"),
    ("The system shall record the geolocation and device fingerprint of every mobile banking login attempt.",
     ["Security", "Privacy", "Audit/Reporting"], "channels"),

    # ── Business / Stakeholder ─────────────────────────────────────────
    ("The system shall decline any loan application where the applicant's debt to income ratio exceeds fifty percent.",
     ["Business", "Functional"], "lending"),
    ("The system shall apply a risk based interest rate derived from the applicant's credit score band and the requested loan tenure.",
     ["Business", "Functional"], "lending"),
    ("The system shall allow a relationship manager to view the consolidated portfolio of all customers assigned to them.",
     ["Stakeholder", "Functional"], "core_banking"),
    ("The system shall escalate any loan application pending underwriter action for more than 48 hours to the branch manager.",
     ["Stakeholder", "Business", "Operational"], "lending"),
    ("The system shall enable a compliance officer to place a hold on any account flagged by the transaction monitoring engine.",
     ["Stakeholder", "Security", "Regulatory"], "compliance"),
    ("The system shall waive the processing fee for applicants who are existing customers with a relationship older than three years.",
     ["Business", "Functional"], "lending"),
    ("The system shall allow a credit committee member to record an approval decision with a mandatory written justification.",
     ["Stakeholder", "Audit/Reporting", "Functional"], "lending"),
    ("The system shall support a maker checker workflow in which no single operator can both create and authorise a beneficiary.",
     ["Business", "Security", "Stakeholder"], "payments"),
    ("The system shall apply a late payment penalty automatically once the configured grace period lapses.",
     ["Business", "Functional"], "lending"),

    # ── Audit / Reporting ──────────────────────────────────────────────
    ("The system shall generate a suspicious transaction report in the format prescribed by the financial intelligence unit.",
     ["Audit/Reporting", "Regulatory"], "compliance"),
    ("The system shall maintain an immutable audit trail of every change to a customer's risk category, including the actor and the reason.",
     ["Audit/Reporting", "Security", "Data-management"], "compliance"),
    ("The system shall produce a daily exception report listing all transactions that bypassed automated approval rules.",
     ["Audit/Reporting", "Operational"], "compliance"),
    ("The system shall allow an internal auditor to reconstruct the complete decision path of any rejected loan application.",
     ["Audit/Reporting", "Stakeholder"], "lending"),

    # ── Operational ────────────────────────────────────────────────────
    ("The system shall emit structured application logs to the central observability platform with a correlation identifier per request.",
     ["Operational", "Integration"], "core_banking"),
    ("The system shall raise an alert to the on-call engineer when the payment authorisation error rate exceeds two percent over five minutes.",
     ["Operational", "Availability"], "payments"),
    ("The system shall allow an authorised administrator to configure business rules without a code deployment.",
     ["Operational", "Business"], "core_banking"),
    ("The system shall expose a health check endpoint reporting the status of every downstream dependency.",
     ["Operational", "Integration", "Availability"], "core_banking"),

    # ── Functional / mixed ─────────────────────────────────────────────
    ("The system shall allow a customer to schedule a standing instruction for recurring loan repayment on a chosen date each month.",
     ["Functional", "Usability"], "lending"),
    ("The system shall recalculate and display the revised equated monthly instalment when a customer requests a change in loan tenure.",
     ["Functional", "Business", "Usability"], "lending"),
    ("The system shall block outward remittances to jurisdictions appearing on the sanctions list maintained by the compliance team.",
     ["Functional", "Security", "Regulatory"], "compliance"),
    ("The system shall allow partial prepayment of a loan and shall recompute the amortisation schedule immediately.",
     ["Functional", "Business"], "lending"),
    ("The system shall generate a unique reference number for every payment instruction and return it to the initiating channel.",
     ["Functional", "Integration"], "payments"),
    ("The system shall notify the customer by SMS and email within 30 seconds of any debit above the customer configured threshold.",
     ["Functional", "Usability", "Security"], "channels"),
    ("The system shall allow a customer to freeze and unfreeze their debit card instantly from the mobile application.",
     ["Functional", "Security", "Usability"], "cards"),
    ("The system shall support cancellation of a loan application by the applicant at any stage before disbursement.",
     ["Functional", "Usability", "Business"], "lending"),
    ("The system shall present a consolidated dashboard of outstanding loans, due dates, and payment history to the customer.",
     ["Functional", "Usability", "Stakeholder"], "lending"),
]


def main() -> None:
    for stmt, cats, _ in REQUIREMENTS:
        bad = [c for c in cats if c not in CATEGORIES]
        if bad:
            raise SystemExit(f"unknown categories {bad} on: {stmt[:60]}")
        if not cats:
            raise SystemExit(f"no categories on: {stmt[:60]}")

    statements = [s for s, _, _ in REQUIREMENTS]
    dupes = [s for s, n in Counter(statements).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate statements: {dupes}")

    requirements = [
        {
            "id": f"AUG-{i:03d}",
            "statement": stmt,
            "categories": cats,
            "domain": domain,
            "source": "synthetic:claude-authored-finance-augmentation",
            "verified": False,
        }
        for i, (stmt, cats, domain) in enumerate(REQUIREMENTS, start=1)
    ]

    OUT.write_text(json.dumps({
        "purpose": (
            "Day 11 augmentation. Hand-authored finance requirements weighted "
            "toward the categories left thinnest after the PROMISE-NFR rows were "
            "dropped. Every row is UNVERIFIED and needs a human label pass."
        ),
        "requirements": requirements,
    }, indent=2) + "\n")

    counts = Counter(c for _, cats, _ in REQUIREMENTS for c in cats)
    card = sum(len(c) for _, c, _ in REQUIREMENTS) / len(REQUIREMENTS)
    print(f"wrote {OUT.name}: {len(requirements)} requirements, "
          f"avg {card:.2f} labels/req\n")
    print(f"  {'category':<18}{'added':>7}")
    print(f"  {'-' * 25}")
    for c in CATEGORIES:
        print(f"  {c:<18}{counts.get(c, 0):>7}")


if __name__ == "__main__":
    main()
