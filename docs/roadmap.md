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
