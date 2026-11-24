# Roadmap — what remains, and what is deferred

Written 14 Nov 2026, after reconciling the repo against `implementation_plan3.md`
and the full `Problem Statement.docx`. Companion to `bottlenecks.md` (open
problems) and `problem_log.md` (closed ones).

---

## Decisions taken 14 Nov

1. **All functional phases first, web app last.** The Streamlit stub at
   `app/streamlit_app.py` stays parked until everything below is done.
2. **Phase 2 (the SDLC engine) is accepted as it stands for now.** Deepening it
   is deferred, not dropped. See [Deferred: SDLC deepening](#deferred-sdlc-deepening).
3. **Branches stay as they are.** Work continues on `extra-fucntionalities`.

---

## Where the system stands against the problem statement

The supervisor's written brief marks two parts, and both ship. The problem
statement asks for considerably more. Mapped section by section:

| PS § | Asks for | State |
|---|---|---|
| 5 | Versioned knowledge base, source/jurisdiction/date/version per item | **done** — 13 docs, manifests, `sources.yaml` |
| 6 | Retrieve → generate → cite + confidence → escalate | **done** — retrieval, grounding node, `needs_review` |
| 7 | Adaptive requirement-gathering interview, follow-ups on vague answers | **partial** — interview exists, but only fills SDLC framework gaps |
| 8 | 13-field requirement structure | **done** |
| 9 | Multi-label classification | **done** — RoBERTa, 80.2% micro-F1 |
| 10 | Quality: ambiguity, incompleteness, inconsistency, **duplication**, infeasibility, testability, missing source, undefined terms, **conflicting expectations**, **missing controls**; failures go back to clarification | **partial** — single-requirement checks only; no cross-requirement checks, no loop back |
| 11 | Compliance agent: jurisdiction, control, evidence, **gaps**, approval checkpoints, retention/reporting obligations | **not started** — grounding cites, but nothing maps controls or finds gaps |
| 12 | Artefacts: SRS, user stories, use cases, workflows, data/interface reqs, risk register, compliance matrix, RTM, assumptions register, open issues | **not started** — `src/generators/` is empty |
| 13–14 | SDLC decision factors; hybrid rules + MCDA + LLM; ranked output | **partial** — 19(+1) question rubric, ranked, justified |
| 15 | Project-specific SDLC workflow: phases, roles, gates, security and compliance checkpoints | **not started** |
| 16 | Human approval: accept / reject / modify / regenerate | **not started** — `approval_status` exists, nothing sets it |
| 17 | Security: masking, prompt-injection protection, source allowlisting, audit logging | **partial** — manifests act as an allowlist; nothing else |
| 19 | Evaluation: extraction P/R/F1, hallucination rate, citation correctness, SDLC accuracy… | **partial** — classification ladder, retrieval recall, SDLC framework check |
| 20 | Record prompts, evidence, decisions, user changes, approvals | **not started** |

Agents from the PS table that do not exist yet: clarification, conflict
detection, compliance, security & privacy, risk analysis, documentation,
validation (beyond the quality agent), human approval.

---

## Remaining functional work, in order

LLM-dependent results are not accuracy evidence until the cloud 8B tier is
running again (Colab not set up as of 14 Nov). Harnesses get built now; numbers
get recorded from the cloud tier.

### Phase A — finish Month 1 (Days 20–30 of the plan)

- **A1. Test suite** for the deterministic core: framework loader and scoring,
  evidence gate, chunker, citation regex, `needs_review`, manifests. No LLM.
  **Done 14 Nov** — 105 tests, ~5 s, offline (`python -m pytest tests -q`).
  Includes a pinned table of what the framework concludes on all seven
  scenarios, so a framework edit that changes a result has to say so. Spot-
  checked by putting back two old bugs (problem log 4.2, 6.8); both were caught.
  **It found a live bug:** the quality filter dropped every finding that named
  a second requirement, because "REQ-004" matched the citation pattern — so
  Inconsistency, Duplication and Conflicting Expectations could never reach a
  report. Fixed in `src/agents/quality.py` `_keep`.
- **A2. Grounded vs ungrounded harness** (Day 20–21): citation coverage,
  hallucinated-regulation rate, latency cost of retrieval. Run on the cloud tier.
- **A3. Edge cases** (Day 25–26): short, long, non-financial, contradictory
  input; a query the knowledge base cannot answer must escalate, not invent.
- **A4. Prompts** to `src/utils/prompts.py`; thresholds and paths to one config.
- **A5. README** (currently empty) and a CLAUDE.md refresh. Known stale in
  CLAUDE.md: "Day 18 is next"; no mention of the CLI app, interview, `--docs`,
  the `prof_*` scenarios or the test suite; and it (with problem log 6.6) still
  says PTR-5 is `evidence_from: scenario`, which 897d8b7 relaxed.

### Phase B — analysis agents (PS §7, §10, §11)

**Built 14 Nov on branch `phase-b-analysis-agents`.** Five nodes between
quality and SDLC (`src/pipeline_analysis.py`), off with `--no-analysis`:
conflicts → compliance → security → risk → clarification. Deterministic first
(rule packs in `data/compliance/`, `data/security/`, `data/risk/`,
`data/clarification/`; similarity candidates), LLM only to judge, map and
revise, and every LLM claim quote-checked in code (`src/utils/evidence.py`).
188 offline tests; five gates spot-checked by putting their bug back.

Measured so far (no LLM, so tier-independent): every regulatory rule resolves
to a clause (`notebooks/13_check_rule_packs.py`); conflict candidates reach
16/16 planted conflicts+duplicates with 0/5 unrelated pairs
(`notebooks/14_eval_conflicts.py`) — on Claude's own unverified pairs, so
agreement with a draft, not a result. Similarity cannot separate a conflict
from a duplicate (both 0.88), so the judge is necessary; measure it on the
cloud tier with `--judge`.

First real run (14 Nov, `kyc_onboarding`, local 3B, `--no-sdlc`): runs end to
end, exit 1 as designed. Not accuracy — the 3B — but four behaviours to check
on the 8B before trusting the LLM half:
- **All five clause mappings came back `not_addressed`.** A uniform answer is
  the "constant, not an inference" pattern (problem log 6.5). If the 8B does
  the same, the mapping prompt needs work.
- **One "confirmed" conflict looks like an overlap**: two requirements feeding
  different inputs into one risk score. The quotes were real, so the gate
  passed it — the gate catches invented quotes, not wrong verdicts.
- **The three LLM-proposed risks were requirements restated as risks**, again
  on real quotes. `SYNTHEIA_MAX_LLM_RISKS=0` turns them off.
- **CMP-KYC-01 reported a gap** although a requirement stores video-KYC
  sessions for 8 years: the rule does not count a session recording as an
  identification record. Borderline — decide when verifying the pack.
- The register had 31 risks, 22 of them one-per-rule-gap. Grouping by family
  may read better; a design call.
- Timing: the run's own clock said 143 s for the pipeline (after 100 s
  warm-up), but file timestamps span ~22 minutes. Not reconciled — time it with
  `time python main.py ...`.

**8B run, 14 Nov** (cloud tier; all outputs in
`evaluation/results/runs_8b_20261114/`, run on the 13-document corpus, before
the six new documents were indexed):
- Planted defects: both contradictions confirmed with real quotes; 9/9
  compliance and 5/5 security outcomes as expected; the planted duplicate was
  lost at extraction (the 8B never extracted the second login sentence).
- Conflict judge on the 26 pairs: conflicts 10/10, zero false conflicts, quotes
  100% traceable — but **duplicates 1/6**: the 8B calls them "overlap", which
  is dropped. One systematic error; a prompt definition or a "similar overlap
  → possible duplicate" rule should fix it. Measure with `--judge`.
- Clause mapping still uniformly `not_addressed` on real scenarios. Cause
  found by reading what it was sent: grounding hands it near-miss clauses
  (video-KYC storage judged against the KYC *updation-notice* clause), and it
  never answers `not_applicable`; it also reads "does not say how" as
  `not_addressed`. Fix: ask applicability as its own step, define "addressed"
  as committing to the obligation's substance.
- LLM-proposed risks are real risks on the 8B (the 3B restated requirements).
- SDLC: 4 pass, 1 tie of two acceptable models, 2 fail (fintech_startup;
  kyc_onboarding, disputed). Both V-Shaped passes come from the 8B answering
  PTR-2 (and, on payroll, PTR-3) YES against our draft answers — the same
  reading the instructor's reasoning supports. Input for the SDLC deepening.
- Each full run is 165–302 s wall clock including Phase B (KYC 302 s).

**Corpus grown to 19 documents, 14 Nov.** The user supplied Digital Lending
Directions 2025, DPSC 2021, CERT-In 2022, IRDAI 2023, PML Records Rules 2005
and the 2018 payment-data circular; manifests read from each document. 11 new
compliance rules and 2 security controls upgraded to regulation-backed; all 43
rules resolve to the clause they target (two wrong landings caught by reading
the excerpts and fixed).
- **The incremental index build corrupted ChromaDB** ("Error in compaction:
  Failed to apply logs to the hnsw segment writer"). Rebuilt with `--reset`
  (4,554 chunks, ~6 min). Broken store backed up first. The retriever used to
  report a broken store as "no knowledge base indexed" and run ungrounded
  silently — now it says the store exists but cannot be read.
- **Retrieval eval dropped: recall@1 91.7% → 75%, recall@5 100% → 96%.** Of
  the six regressed queries, three are new documents legitimately answering
  better (PML Rules on suspicious transactions and identity documents; DPSC
  §33 on card-not-present authentication) — the gold set allows one document
  per query. Two are real intrusions: IRDAI's generic infosec text outranking
  banking sources. **Open decision:** filter retrieval by the manifest
  `applicability` tags (the schema already says it should), and/or let the
  gold set accept more than one correct source per query, each justified.

**Fixes applied 14 Nov (afternoon), each re-measured:**
- **Sector filter** (`src/rag/sectors.py`): grounding and scenario context stay
  inside the scenario's regulatory sector, using the manifest `applicability`
  tags the schema always intended for this. Conservative — a scenario in both
  sectors or neither filters nothing. PML Rules and the Aadhaar regulations
  tagged `general` (they bind insurers too); index rebuilt. Retrieval with the
  banking filter: recall@1 79% strict / 88% with three justified alternates,
  recall@5 100%; both IRDAI intrusions gone. The eval now reports strict and
  alternate scoring side by side (`--sector`, `--out`).
- **Conflict judge**: duplicate/overlap definitions rewritten (examples checked
  for zero overlap with the eval set), and a rule-detected clash can no longer
  be erased by the judge — it is kept as "possible" with the judge's view.
  Reported to the user: conflicts 10/10, duplicates 4/6 (was 1/6), one
  unconfirmed false report. The eval now scores the judge AND what is shown.
- **Clause mapping**: two steps — "does this clause govern the requirement?"
  first (schema field order forces it), then coverage with "addressed" defined
  as committing to the obligation's substance.

**Re-measured on the 8B, 14 Nov (`runs_8b_20261114/v2`, `v3`):**
- Sector filter works end to end: insurance cites IRDAI, never RBI.
- Two-step mapping first over-corrected to "not_applicable" (15 of 16) — but
  read clause by clause, 5 of 6 spot-checked verdicts were right: grounding's
  top citation usually does not govern its requirement. The one wrong verdict
  (staff seeing full Aadhaar vs the Aadhaar confidentiality clause) is fixed:
  a requirement that does what a clause forbids is "governed, not addressed".
- Reading up to 3 citations per requirement found one more governing clause
  for ~20 extra LLM calls, and planted_defects grew 241 s → 332 s. Low yield:
  **grounding retrieval is now the bottleneck for citation quality**, not the
  checker. Unsupported citations now reach the review list.
- Extraction varies run to run (insurance: 5, 2, 5 requirements) and on the
  8B again emitted project facts as requirements ("developed within 18
  months") — problem log 6.1 recurring. Extraction is the only agent still at
  temperature 0.2, and the scope rule is prompt-only.

**Then fixed (14 Nov, evening):**
- Extraction now runs at temperature 0, like every other agent.
- `src/agents/project_facts.py` drops "requirements" that are project facts
  (delivery timeline, team, budget, methodology, the requirements process) and
  records each in the run's `dropped` list and a "removed" panel. Narrow by
  design: over all 62 distinct requirements extracted in today's runs it drops
  exactly the two project facts and nothing else.
- Clause-mapping budget 16 → 10 calls, spent breadth-first: every
  requirement's first clause before anyone's second.
- Measured on the 8B: insurance extracted identically on two runs; the project
  fact was dropped and recorded; planted_defects 332 s → 252 s with the same
  9/9 and 5/5. KYC breakdown: warm-up 27 s, extraction 46 s, quality 26 s,
  compliance 55 s, everything else ~30 s — 187 s total.
- **The unexplained timing gaps were system suspend**, not the code: a later
  run's wall clock spanned 8,037 s while the laptop slept. The pipeline's clock
  is monotonic and excludes suspend. Keep the machine awake during runs.
- Every run now saves a timing breakdown (`timing` in the JSON) and prints one
  line naming the slowest stages.
- The embedder loads at warm-up and from disk first: retrieval 13 s → 4 s, and
  no Hugging Face request on every run (a network dependency the offline claim
  did not allow).
- A dead LLM endpoint — at start or mid-run — exits 2 ("could not run") with
  the fallback command, instead of a traceback with exit 1 (which means "needs
  review"). Other exceptions still raise.

Found while building, not yet fixed:
- **Chunker mislabels numeric sub-sections.** DPDP s.8(6) cites as "§8(b)":
  "(6)" is not a recognised heading, so the last lettered label sticks. Every
  DPDP citation with a numbered sub-section is affected. Fix + re-index.
- **Card Directions para 65 is unreachable** — unrelated clauses outrank it.
  Rule CMP-CARD-01 dropped until re-chunking.

Original plan for reference:

- **B1. Conflict & duplicate detection** — embedding similarity proposes pairs,
  the LLM judges only those pairs.
- **B2. Compliance agent** — per requirement: jurisdiction, control, evidence,
  retention/reporting obligation, approval checkpoint. Plus **coverage gaps**:
  clauses that apply to the scenario that no requirement covers. Rule-based
  validation alongside the LLM, as §11 and the PS preamble ask.
- **B3. Security & privacy agent** — missing security/privacy controls (§10's
  last check), driven by the same retrieval.
- **B4. Risk-analysis agent** — business, technical, compliance risk; feeds the
  risk register.
- **B5. Clarification loop** — requirements that fail quality go back as
  questions (§7, §10). Extends the existing interview from SDLC gaps to
  requirement gaps.

### Phase C — artefacts (PS §12, §15)

**SRS built 22 Nov on branch `phase-c-artefacts`** (`src/generators/srs/`).
The default layout follows **the course's IEEE 830-1998 slides exactly**: 1.1
Purpose (with intended audience), 1.2 Scope (what the system will *and will
not* do), 1.3–1.5; 2.1 Product perspective (business case, operational concept,
the five interface kinds, memory / operations / site adaptation), 2.2–2.5; 3.1
External interfaces (an inputs/outputs table), 3.2 Functions (use-case
specifications), 3.3 Performance, 3.4 Logical database (entities and
relationships), 3.5 Design constraints, 3.6 Software system quality
attributes; appendices; index. **3.7 Object-oriented models is omitted** — the
user's decision, 22 Nov; no UML is generated. ISO/IEC/IEEE 29148:2018 remains
available with `--srs-layout iso29148`.

    python main.py scenario.txt --export out/                  # run + SRS
    python -m src.generators.export run.json --out out/ --layout both
    /export out/ [iso29148|both]                               # interactive app

**More requirements, from three origins**, each labelled on its block:

| origin | where from | typical count |
|---|---|---|
| Stated | extraction from the scenario | 5–7 |
| Implied | the elaboration node (after extraction): what such a system needs that the scenario leaves unsaid; gated against project facts and near-duplicates, confidence capped at 0.75, max 12 | 5–12 |
| Proposed | every compliance/security gap or weak finding becomes a requirement, citing its clause and marked for sign-off | 10–25 |

The KYC fixture (which predates elaboration) gives 7 stated + 21 proposed = 28.
SDLC scoring sees **stated requirements only** — the implied ones are our
inference, and letting them vote would make the recommendation depend on how
much the elaboration agent wrote. Quality analysis runs in batches of 8, since
20+ requirements overran the 2,048-token output cap.

- Every requirement stated once, in the section its **type** puts it —
  functional (3.2, under its use case), external interface (3.1),
  non-functional (3.3, 3.4, 3.6), design constraint (3.5) — with 14
  attributes: type, origin, priority, rationale, source, acceptance criteria,
  verification method, dependencies, assumptions, risk, regulatory basis,
  classification, quality findings, status. A type × origin count table opens
  section 3.
- Every section the outline names is present; an empty one says so.
- A compliance gap appears once: as a Proposed requirement in section 3, as
  "Proposed as REQ-0nn" in the compliance matrix, and as "confirm REQ-0nn" in
  the open issues — not re-asked as a question.
- LLM writes the narrative, use cases (two calls: list, then flows per use
  case), data model and interface table. All gated in code: quotes must be in
  the scenario; actors, entities and counterparts must be named in the
  scenario or requirements (every distinctive word, not one); a use-case name
  must rest on its own requirements or it is renamed from them; requirements
  the model leaves out get use cases of their own. `--no-llm` gives a
  complete deterministic document.
- Checked on the local 3B (22 Nov): the gates caught an invented "Submit loan
  application" use case on a KYC platform, requirement sources offered as
  actors, a "Business Owner" user class grounded on one shared word, and
  alternative flows copied from the prompt's examples. Each is now a test.

Still to do in Phase C: a definitive sample from a fresh 8B run with
elaboration and SDLC (blocked 22 Nov: the network was resetting
`*.trycloudflare.com` tunnels; switching networks fixed it), user stories with
acceptance criteria (§12), the project-specific SDLC workflow (§15), and
standalone spreadsheets for the RTM / risk register / compliance matrix.

Original plan:

- **C1.** SRS (ISO/IEC/IEEE 29148:2018 structure, which supersedes IEEE 830),
  user stories with acceptance criteria, use cases.
- **C2.** RTM, compliance-control matrix, threat & risk register, assumptions
  and dependency register, open-issues list.
- **C3.** Project-specific SDLC workflow: phases, activities, roles,
  deliverables, security activities, compliance checkpoints, approval gates,
  entry/exit criteria.
- Every artefact links back to the source statement and supporting document.

### Phase D — human-in-the-loop and audit trail (PS §16, §20)

- **D1.** Accept / reject / modify / regenerate per requirement; versioned
  baselines; edits re-classified and re-grounded.
- **D2.** Audit trail per run: prompts, retrieved evidence, agent decisions,
  user changes, approvals.

### Phase E — security (PS §3, §17)

- Sensitive-data masking before anything reaches the LLM; prompt-injection
  screening on scenarios and supplied documents; output filtering.

### Phase F — evaluation (PS §19, plan Week 11)

- The E1–E7 ablation on the cloud tier. Extraction P/R/F1 needs per-scenario
  reference requirements, which do not exist yet.

### Then: SDLC deepening, then the web app.

---

## Deferred: SDLC deepening

**The direction, in the user's words (14 Nov):** the engine currently answers
19 fixed questions; it also needs to *capture the essence and the context of
the project* alongside them — more techniques, possibly a better RAG system,
better context capture and essence-finding, with reasoning.

**The problem statement backs this directly.** §13 lists the decision factors
the engine should derive, and they are broader than the rubric: requirement
stability, regulatory criticality, security risk, project complexity, system
size, legacy-system dependence, frequency of change, need for continuous
delivery, stakeholder availability, testing and documentation needs, budget and
schedule, need for formal verification, consequences of failure. §14 asks for
"a hybrid approach combining deterministic rules, multi-criteria decision
analysis, and LLM-generated explanations", ranked with scores. Its condition
table names **DevSecOps** and **hybrids** (Agile–V-Model) that the rubric cannot
produce.

### Measured findings to carry in (no LLM involved, so they hold on any tier)

`--framework` mode, scoring the expected answers through the tables:

- **Baseline 2/7.** The instructor's own payroll example (marked V-Model) gets
  Waterfall even with perfect answers.
- **PTR-3 gloss.** Our gloss forces "enhancement to an existing system?" to NO
  for an as-is conversion. The instructor's reasoning — "a conversion of an
  existing system rather than development of a new system" — reads as YES.
  Re-reading it as YES for the two conversions takes the check to **4/7** with
  no weights, breaking nothing. **Not applied, by decision** — it belongs in
  the deepening work, not as a one-off patch.
- **The Agile PTR-3 cell is not the fix it looks like.** Flipping it (CLAUDE.md
  calls it our defect) turns `fintech_startup` into a tie but **breaks the
  instructor's wristwatch example** (Prototype → Agile). The instructor-verified
  scenarios are the positive controls; any change must keep them passing.
- Still failing after the PTR-3 re-reading: `fintech_startup` (Prototype over
  Agile/RAD), `kyc_onboarding` (disputed), `prof_wristwatch` (tie, caused by
  our derived Agile column).

### Candidate techniques

- **A project-profile step before the rubric.** Extract §13's factors as a
  structured profile, each with a quoted reason and a confidence. The 19
  questions are then answered *from* the profile plus the scenario, and the
  profile carries what the rubric cannot ask.
- **An MCDA layer per §14** (weighted factors, e.g. TOPSIS or AHP-derived
  weights, argued in writing rather than fitted) alongside the supervisor's
  tables. Report both, and where they disagree.
- **Retrieval for SDLC selection.** §5 lists "SDLC selection rules" as
  knowledge-base content. Index SE references and solved cases; retrieve
  analogous projects (including the instructor's examples) as reasoning
  precedent.
- **DevSecOps and hybrid models** from the PS condition table.
- **Self-consistency.** Several inference passes; agreement becomes the
  confidence score §6 asks for.
- **One call per question** (problem log 6.10, still open).
- **The instructor's examples as a gate.** Nothing ships that breaks them.

---

## Enhancement ideas for what already works

Not scheduled; pick up when a phase above touches the same code.

- **Quote check on extraction.** A required `source_quote` per requirement,
  verified in code against the scenario and supplied documents — the SDLC
  evidence gate applied to Phase 1. Catches the 3B's invented requirements and
  makes "with its reasoning" checkable.
- **Citation reranking.** A cross-encoder over the retrieved top-k, targeting
  the limit `src/graph.py` admits: similarity is not applicability. Measure on
  ~30 hand-labelled citations.
- **Deterministic quality pre-checks.** ISO 29148 vague-term list, unmeasurable
  NFRs, missing actor — code first, LLM for the rest.
- **Sensitivity view.** Which answers would flip the SDLC recommendation,
  shown after the decision (never during the interview — that leads the
  witness).
- **Hybrid recommendation** when the top two are within one question.

---

## Web app (last)

When it comes: drive the compiled graph through `stream()` so stages report as
they finish (the stub re-implements the pipeline by hand and has already
drifted — no retrieval, grounding or SDLC), and replace the blocking
`ask_user` callback with LangGraph `interrupt()` plus a checkpointer so an
interview can pause in a browser. The same mechanism serves the approval
workflow. Stack (Streamlit vs FastAPI + React) still to choose.
