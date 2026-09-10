"""Day 11 third augmentation: requirements grounded in the RBI KYC Master Direction.

Source: RBI/DBR/2015-16/18, Master Direction DBR.AML.BC.No.81/14.01.001/2015-16,
"Know Your Customer (KYC) Direction, 2016", updated as on August 14, 2025.
107 pages, retrieved from rbi.org.in.

Why this source, over more synthetic data:
    Both fine-tuned models plateaued at ~78% ground-truth micro-F1, failing on
    the same requirements, with the losses concentrated in Privacy, Regulatory,
    Business and Data-management. Those are precisely the duties a real KYC
    regulation spells out. Every statement below is derived from an actual
    obligation in the Direction rather than invented, so the Regulatory label
    is grounded in a real mandate instead of my guess about one -- which was
    the weakness of the earlier synthetic batches.

    It also front-loads work for Month 2: this same document is the RAG corpus
    the compliance agent will need.

Each entry cites the paragraph it derives from. These are *software*
requirements derived from regulatory obligations, not the regulation's own
wording -- the Direction binds "REs" (regulated entities), while a requirement
has to bind the system.

Labelling follows the ground_truth.json convention (see label_corrections.json):
    Privacy + Security   when personal identity data is acted upon
    Data-management      for retention/records -- NOT Privacy
    Integration          for plumbing to an external registry -- NOT Privacy
    Regulatory           because a named external mandate is the reason each of
                         these exists -- which is the whole point of the source

UNVERIFIED: my mapping from clause to requirement, and the labels, both need
your pass. The clause citations let you check each one against the source.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "rbi_grounded.json"

CATEGORIES = [
    "Business", "Stakeholder", "Functional", "Security", "Privacy", "Regulatory",
    "Performance", "Availability", "Usability", "Data-management", "Integration",
    "Audit/Reporting", "Operational",
]

# (statement, [categories], clause)
REQUIREMENTS: list[tuple[str, list[str], str]] = [

    # ── V-CIP / liveness / biometric identity (para 18) ────────────────
    ("The system shall perform face liveness and spoof detection during video-based customer identification before accepting the session as valid.",
     ["Security", "Privacy", "Functional", "Regulatory"], "para 18(a)(v)"),
    ("The system shall match the customer's face captured during the video identification session against the photograph on their identity document.",
     ["Security", "Privacy", "Functional", "Regulatory"], "para 18(a)(v)"),
    ("The system shall record the live GPS coordinates and a date-time stamp for every video-based customer identification session.",
     ["Privacy", "Audit/Reporting", "Functional", "Regulatory"], "para 18(a)(iv)"),
    ("The system shall capture video of sufficient quality to allow identification of the customer beyond doubt and shall reject sessions falling below that threshold.",
     ["Functional", "Regulatory", "Security"], "para 18(a)(iv)"),
    ("The system shall abort and require restart of a video identification session if the video is paused or the call is disconnected at any point.",
     ["Functional", "Security", "Regulatory"], "para 18(a)(ii)"),
    ("The system shall restrict operation of the video identification workflow to officials specifically trained and authorised for that purpose.",
     ["Stakeholder", "Security", "Operational", "Regulatory"], "para 18(a)(i)"),
    ("The system shall provide an assisted video identification path so that the liveness check does not exclude persons with special needs.",
     ["Usability", "Regulatory", "Stakeholder"], "para 18(a)(i)"),

    # ── Aadhaar redaction / masking (para 16) ──────────────────────────
    ("The system shall redact the Aadhaar number from any submitted proof of possession of Aadhaar where Aadhaar authentication is not required.",
     ["Privacy", "Security", "Regulatory"], "para 16, Explanation 1"),
    ("The system shall blackout the Aadhaar number in stored copies of officially valid documents, including those retrieved from DigiLocker.",
     ["Privacy", "Security", "Data-management", "Regulatory"], "para 16"),
    ("The system shall accept offline Aadhaar verification through an XML file or a secure QR code without storing the full Aadhaar number.",
     ["Privacy", "Security", "Functional", "Regulatory"], "para 16"),

    # ── Consent and confidentiality (paras 11, 55) ─────────────────────
    ("The system shall obtain and record the customer's explicit consent before collecting any personal information beyond that specified in the KYC policy.",
     ["Privacy", "Regulatory", "Functional"], "para 11(e)"),
    ("The system shall not disclose customer information to third parties except under compulsion of law, a public duty to disclose, or the customer's express consent.",
     ["Privacy", "Security", "Regulatory"], "para 55"),
    ("The system shall keep a customer's risk categorisation and the reasons for it confidential and shall not expose them on any customer-facing screen.",
     ["Privacy", "Security", "Business", "Regulatory"], "para 12(d)"),

    # ── Risk categorisation (para 12) ──────────────────────────────────
    ("The system shall categorise every customer as low, medium, or high risk based on the institution's documented risk assessment parameters.",
     ["Business", "Functional", "Regulatory"], "para 12(a)"),
    ("The system shall derive a customer's risk category from parameters including identity, social and financial status, and nature of business activity.",
     ["Business", "Functional", "Regulatory"], "para 12(c)"),
    ("The system shall apply enhanced due diligence steps automatically when a customer is classified as high risk.",
     ["Business", "Functional", "Security", "Regulatory"], "para 12"),

    # ── Periodic updation and account limits (paras 17, 38) ────────────
    ("The system shall trigger periodic KYC updation for each customer according to the cadence defined for their risk category.",
     ["Business", "Functional", "Regulatory"], "para 38"),
    ("The system shall restrict an account opened through OTP-based electronic KYC to a maximum of one year unless full identification is completed.",
     ["Business", "Functional", "Regulatory"], "para 17(vi)"),
    ("The system shall cap the aggregate of term loans sanctioned through OTP-based electronic KYC at the prescribed annual limit per customer.",
     ["Business", "Functional", "Regulatory"], "para 17(v)"),
    ("The system shall require fresh Aadhaar OTP authentication when converting an account opened in non-face-to-face mode to full KYC status.",
     ["Security", "Privacy", "Functional", "Regulatory"], "para 17(vi)"),

    # ── Onboarding fairness and recorded decisions (para 11A) ──────────
    ("The system shall require the reviewing officer to record a written reason before any onboarding or periodic updation request can be rejected.",
     ["Stakeholder", "Audit/Reporting", "Regulatory"], "para 11A"),
    ("The system shall not deny account opening on grounds that would disadvantage persons with disabilities or financially excluded applicants.",
     ["Usability", "Business", "Regulatory"], "para 11A"),

    # ── Records and retention (para 46) ────────────────────────────────
    ("The system shall retain customer identification records and transaction history for five years from the end of the business relationship.",
     ["Data-management", "Regulatory"], "para 46"),
    ("The system shall preserve the results of any due diligence analysis alongside the related business correspondence for the statutory period.",
     ["Data-management", "Audit/Reporting", "Regulatory"], "para 46"),
    ("The system shall maintain records of transactions in third-party products separately and for the prescribed retention period.",
     ["Data-management", "Regulatory"], "para 46"),

    # ── Monitoring, alerts, reporting (paras 17, 46, Chapter IX) ───────
    ("The system shall generate an alert to the compliance team on any breach of the limits applicable to non-face-to-face onboarded accounts.",
     ["Operational", "Security", "Stakeholder", "Regulatory"], "para 17(ix)"),
    ("The system shall capture, generate, and analyse alerts for transactions that meet suspicious activity criteria.",
     ["Security", "Audit/Reporting", "Operational", "Regulatory"], "Chapter IX"),
    ("The system shall verify the identity and address of a walk-in customer for any transaction above the prescribed cash threshold.",
     ["Functional", "Privacy", "Security", "Regulatory"], "para 13(e)"),

    # ── External registries -> Integration, not Privacy ────────────────
    ("The system shall retrieve a customer's KYC records from the Central KYC Records Registry using the KYC identifier supplied by the customer.",
     ["Integration", "Functional", "Regulatory"], "para 56"),
    ("The system shall register non-profit organisation customers on the government's designated portal and retain the registration records.",
     ["Integration", "Data-management", "Regulatory"], "para 46A"),
    ("The system shall accept equivalent electronic documents bearing a valid digital signature as officially valid documents.",
     ["Integration", "Functional", "Regulatory"], "para 3(x)"),
    ("The system shall apply customer due diligence at the unique customer identification code level rather than per account.",
     ["Data-management", "Functional", "Regulatory"], "para 11(f)"),
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
            "id": f"RBI-{i:03d}",
            "statement": stmt,
            "categories": cats,
            "domain": "kyc_onboarding",
            "source": "synthetic:grounded-rbi-kyc-master-direction-2016-aug2025",
            "source_clause": clause,
            "source_document": (
                "RBI/DBR/2015-16/18 Master Direction DBR.AML.BC.No.81/14.01.001/2015-16, "
                "Know Your Customer (KYC) Direction 2016, updated as on August 14, 2025"
            ),
            "verified": False,
        }
        for i, (stmt, cats, clause) in enumerate(REQUIREMENTS, start=1)
    ]

    OUT.write_text(json.dumps({
        "purpose": (
            "Requirements derived from actual obligations in the RBI KYC Master "
            "Direction (updated August 14, 2025). Each cites the paragraph it comes "
            "from so the mapping can be checked against the source. UNVERIFIED."
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
