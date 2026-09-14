# Financial knowledge base — sources and ingestion

Deliverable 1's corpus. The supervisor's part 1 is *"feed a lightweight LLM
existing requirement documents from a specific domain"*, and this is what gets
fed in. Problem statement §5 requires it be versioned and attributed; §6 requires
generation to be grounded in it.

**Every URL below was checked with `curl` on 14 Sep 2026.** The sourcing research
marked several links "Verified: Yes" that do not resolve — so status here is what
actually returned content, not what was claimed.

## Category A — regulatory

| # | Document | Format | Status |
|---|----------|--------|--------|
| 1 | RBI Master Direction — KYC, 2016 (`RBI/DBR/2015-16/18`) | PDF 906 KB | ✅ |
| 2 | RBI MD — IT Governance, Risk, Controls (`RBI/2023-24/107`) | HTML 212 KB | ✅ |
| 3 | RBI MD — Outsourcing of IT Services (`RBI/2023-24/102`) | HTML 229 KB | ✅ **URL corrected** |
| 4 | RBI — Cyber Security Framework in Banks (2016) | PDF 1.3 MB | ✅ *superseded in part* |
| 5 | RBI MD — Fraud Risk Management (2024) | HTML 233 KB | ✅ |
| 6 | RBI MD — Credit/Debit Card Issuance (2025) | HTML 239 KB | ✅ |
| 7 | RBI MD — Prepaid Payment Instruments | HTML 337 KB | ✅ |
| 8 | Digital Personal Data Protection Act, 2023 | PDF 382 KB | ✅ |
| 9 | UIDAI — Aadhaar Authentication Regulations 2021 | — | ❌ 404 |
| 10 | SEBI — CSCRF | HTML 8 KB | ⚠️ landing page only |
| 11 | IRDAI — Information and Cyber Security Guidelines 2023 | HTML 241 KB | ✅ |
| 12 | EU 2018/389 — SCA / RTS | — | ⚠️ 202, empty |

Document 3's published URL was mangled (`systemhealth.rbi.org.in/…aspx_id%3D12486.html`).
The canonical form works:
`https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12486`

## Category B — requirement / specification documents

The half that satisfies "existing requirement documents". Harder to source, and
the reason a corpus of pure regulation would only answer half the brief.

| # | Document | Format | Status |
|---|----------|--------|--------|
| 1 | **SBI (UK) — RFP for Loan Origination System**, V3.0, 56 pp | PDF 754 KB | ✅ **best source** |
| 2 | Central Bank of India — Digital Banking Platform RFP, 205 pp | — | ❌ 404 |
| 3 | UCO Bank — Official Language in CBS (Annexure-VI specs) | PDF 1.4 MB | ✅ |
| 4 | GIC Housing Finance — Fraud Detection RFP, 75 pp | PDF 465 KB | ✅ |
| 5 | Open Banking UK — Read/Write API Specification | repo | ✅ *needs file-level fetch* |
| 6 | Berlin Group — NextGenPSD2 | HTML 1.5 MB | ✅ |
| 7 | UK Payment Services Regulations 2017 | — | ⚠️ 202, empty |
| 8 | NPCI — UPI Product Booklet | — | ❌ 403 bot-blocked |

**Working today: 14 of 20.** Three real bank requirement documents resolve
(SBI UK, UCO, GIC Housing), which covers the hard half.

## Adding a document

1. Save it under `data/knowledge_base/regulations/` or `requirement_docs/`.
   Accepted: `.pdf`, `.docx`, `.html`, `.htm`, `.md`, `.txt`.
   For an HTML source, save the page itself — do not print it to PDF. The
   heading hierarchy survives in markup and has to be reconstructed from layout
   in a PDF.
2. Write a sidecar manifest beside it, `<filename>.manifest.yaml`:

```yaml
title: "Master Direction on Outsourcing of Information Technology Services"
issuer: "Reserve Bank of India"
doc_id: "RBI/2023-24/102"
jurisdiction: IN                # IN | UK | EU | US | GLOBAL
effective_date: 2023-10-01
version: "issued 2023-04-10"
applicability: [payments, lending, regulatory_reporting]
doc_type: regulation            # regulation | srs | api_spec | tender | standard
url: "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12486"
```

   Ingestion **fails** on a document without one. That is deliberate: a passage
   with no provenance lets the pipeline emit confident references nobody can
   check, which is the failure §6 exists to prevent.

3. Dry run first and read the clause-coverage figure:

```bash
python -m notebooks.09_build_knowledge_base --dry-run
python -m notebooks.09_build_knowledge_base            # then index
```

   Coverage below 50% means the document's numbering did not survive extraction,
   and its citations will degrade to document level. Check the source before
   indexing it.

## What the loaders already guard against

- **Scanned PDFs** — extraction yield per page is checked; an image-only PDF is
  refused rather than indexed as empty chunks.
- **JavaScript-rendered or error pages** — an HTML file yielding under 500
  characters is refused.
- **ASP.NET WebForms** — RBI wraps entire pages in `<form runat="server">`.
  Stripping form elements took a 54,192-character direction down to 41
  characters. `form` is therefore never stripped; only its controls are.
- **Tables of contents** — a published regulation opens with a contents list
  whose every line looks like a heading. Indexed, it matches any query about
  section titles while containing no rule. Detected and skipped.

## Known gaps

- `MIN_CONFIDENCE = 0.45` in `src/rag/retriever.py` is **uncalibrated**. Sweep it
  against the real corpus and record the number, as the classifier threshold was.
- Superseded documents are flagged in prose only. RBI's 2016 Cyber Security
  Framework is partly superseded; citing a withdrawn rule is a real error, so
  `status` should become a manifest field once there is a second superseded doc.
- Nothing yet checks for near-duplicate chunks across documents.
