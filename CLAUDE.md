# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Syntheia is an agentic AI system for **automated requirement gathering, analysis, and SDLC methodology identification in financial services** (loan origination, KYC/onboarding, payments). It takes a natural-language problem scenario and produces structured, classified, quality-checked software requirements (Phase 1), then recommends a ranked SDLC methodology given project conditions (Phase 2).

The project follows a day-by-day 4-month implementation roadmap at `implementation-plans/implementation_plan3.md` (the current/superseding plan). **This directory is gitignored** — it's a local planning reference only.

### ⚠️ Architecture change — plan revised 11 Sep 2026

The plan's Day 12 section carries an **ARCHITECTURE CHANGE** that supersedes the earlier design:

> The core pipeline must use a **local lightweight LLM via Ollama**, **NOT** the Gemini API. Gemini is kept only as a baseline for comparison. QLoRA fine-tuning is pulled into Month 1.

Three consequences:

1. **Gemini is demoted to a baseline.** The shipping pipeline must run offline with no API calls. `src/utils/llm_gemini.py` stays, but only for comparison runs and as a data-generation tool (output manually verified).
2. **Two fine-tuning targets:** a 3–4B local model (chosen Day 12 by benchmark) and Llama-3.1-8B on cloud. Same script, same dataset, two adapters.
3. **QLoRA moves from Month 2 into Month 1** (Days 13–16), as does the whole SDLC engine.

### ⚠️ SCOPE CORRECTION — 14 Sep 2026: RAG is deliverable 1; fine-tuning is not required

The supervisor stated the individual SE project in writing. Both parts are marked
after midsems:

> 1) Feed a lightweight LLM **existing requirement documents** from a specific domain — here, finance. Then prompt it to generate the specific requirements for a particular functionality, **along with its reasoning**.
> 2) Then feed those generated requirements back into an LLM and ask it to identify the **most appropriate SDLC**, along with a proper software engineering **justification**.

Checked against `Problem Statement.docx`:

| Term | Occurrences |
|---|---|
| `fine-tun` / `finetun` / `fine tun` | **0** |
| `BERT`, `embedding`, `vector` | **0** |
| `train` | 3 — all inside "cons**train**ts" |
| Retrieval-Augmented Generation | **its own section (§6)** |

**"Feed a lightweight LLM existing requirement documents" IS retrieval-grounded
generation** — it is part 1's input mechanism, not a preamble to it. §6 gives the
workflow: retrieve → provide evidence to the LLM → generate → attach citations and
confidence → escalate low-confidence output. §5 requires the knowledge base behind
it, versioned, with source, jurisdiction, effective date and applicability per item.

The extraction agent currently receives a bare scenario string and generates from
parametric memory. **No documents are fed in, so by the supervisor's framing part 1
is not started** — even though extraction works.

**QLoRA is self-imposed scope.** It appears nowhere in the problem statement or the
brief. Worth doing — it is the strongest ML content for the report and the Week 11
ablation needs it — but it is optional, expensive (hand-verifying ~150 pairs is the
bottleneck), and was scheduled *ahead of* the thing being marked.

**Multi-label classification stays justified** — §9: *"Multi-label classification
should be allowed because one requirement may belong to several categories."* The
statement mandates the capability, not the technique, so fine-tuned RoBERTa is a
legitimate way to deliver it.

**Revised Days 13–30** (plan updated):

| Days | Was | Now |
|---|---|---|
| 13–15 | Extraction QLoRA | **Knowledge base + retrieval-grounded extraction** — deliverable 1 |
| 16–17 | SDLC + QLoRA | **SDLC engine, prompted** — deliverable 2 |
| 18–19 | End-to-end | End-to-end + CLI, both parts |
| 20–21 | Comparison | **Grounded vs ungrounded** + Week 3 review |
| 22–24 | Weakest link | **QLoRA — optional, time-boxed, gated on 13–21 being done** |
| 25–30 | Hardening → checkpoint | Day 30 gates on *both* deliverables |

The Month 1 headline result is no longer "fine-tuned beats prompted" but
**"grounded beats ungrounded"** — which is what §6 exists to establish.

### Two-tier architecture (cloud API permitted — confirmed 11 Sep)

The supervisor confirmed a cloud-hosted LLM reached by API is **permitted**. So the system has two tiers, not one:

| Tier | Model | Where | Role |
|------|-------|-------|------|
| **Cloud (primary)** | Llama-3.1-8B, QLoRA fine-tuned | Colab/Kaggle T4 + Ollama + tunnel | Demo day, best quality |
| **Local (fallback)** | **`llama3.2:3b`** — pinned Day 12 by benchmark | This laptop, Ollama | Dev loop, offline capability, demo insurance |

**One wrapper serves both.** Ollama's API is OpenAI-compatible and behaves identically on localhost and behind an ngrok/Cloudflare tunnel, so `src/utils/llm_ollama.py` takes a configurable base URL (`OLLAMA_HOST`). Switching tiers is an environment variable, never a second implementation. Write it that way from the start.

Why keep the local tier at all, now that cloud is allowed:
- the offline claim survives — the local tier needs no network
- **graceful degradation becomes a real engineering story**, not a workaround: a dead endpoint on demo day downgrades the demo instead of ending it
- the dev loop stays fast and free

### Why the laptop still cannot serve 8B

This is why the local tier stays at 3–4B. A 4GB RTX 3050 with 7GB system RAM (~5GB free) cannot hold an 8B model at 4-bit (~4.7–4.9GB) — it spills into RAM already shared by Python, RoBERTa and the OS, giving a few tokens/second against long JSON extraction outputs.

### "Local tier" means served locally, not trained locally

| | Local 3–4B | Cloud 8B |
|---|---|---|
| Trained | Colab T4 (probably) | Colab T4 |
| Served | this laptop | Colab/Kaggle + tunnel |

QLoRA *training* needs optimiser states, gradients and activations on top of the weights — several times the inference footprint — so even a 3B may exceed 4GB VRAM to train. Both fine-tunes likely run on Colab; **only serving differs.** Write `notebooks/07_finetune_extraction.py` to run on Colab and be parameterised by base model, so one script produces both adapters.

### The dataset is the gate

Neither fine-tune can start until ~100–150 verified instruction/output pairs exist, and hand-verification is the slow part. Days 9–11 proved the point: model and hyperparameter work moved micro-F1 2.9 points across three experiments; one targeted data fix moved it 2.2. Budget Day 13 for the dataset and expect it to be the bottleneck.

### Free-tier realities (verified 11 Sep)

Colab free: T4 16GB, **~12h max session, ~90min idle timeout** (unpublished, varies), 12–13GB RAM, 2 concurrent notebooks. Kaggle free: ~30h/week GPU.

- **Nothing persists** — each session re-pulls ~5GB unless cached to mounted Drive. Budget 10–15 min before any demo.
- **Free ngrok URLs change on restart** — the endpoint is configuration, never hard-coded.
- Start the notebook before the demo, keep the tab alive, and rehearse the fallback.

The plan has been reconciled with this change (banner at the top, §2/§3 tables, §8 structure, Week 5, and the E1–E7 matrix). Older prose elsewhere in it may still read as pre-revision; **the banner wins.**

### Current status

**Days 9–11 are complete.** The 3-node LangGraph pipeline is operational and the classification node runs a fine-tuned RoBERTa classifier (80.2% micro-F1 on held-out ground truth, 21 ms/requirement) instead of the Gemini few-shot baseline.

One Definition-of-Done item was **not** met: Day 11 required the fine-tuned model to beat the LLM baselines. It does not — 80.2% against few-shot Gemini's 84.0%. It wins on latency by ~1,450x. See the baseline table below.

**Days 12–13 are complete.** Both LLM tiers work (`llama3.2:3b` local, `llama3.1:8b` on Colab behind a tunnel), and the cold start is down from 611 s to 89 s.

**Days 14–15 are complete — deliverable 1 ships.** `src/rag/` holds loader, structural
chunker, embedder, ChromaDB store and retriever; `data/knowledge_base/` has 13
documents (9 regulations, 4 tender/RFP) registered in a tracked `sources.yaml`
with 7 explicit `unavailable:` entries. Retrieval measured at recall@1 92%,
recall@5 100% before being wired in. The pipeline is six nodes:
`retrieval → extraction → grounding → classification → quality → sdlc`.

**Days 16–17 are complete — deliverable 2 ships.** The SDLC engine infers the
framework answers, scores them deterministically against the supervisor's
tables, and has an LLM justify the result it did not choose. Five scenarios
produce five different recommendations, so it discriminates.

**Current accuracy: 3 of 5 scenarios correct (8B, 15 Sep).** Do not read that
as the state of the engine — every fix from 17 Sep postdates it and the set has
not been re-run. See `docs/problem_log.md` §6.

**Day 18 is next:** settle the two open disagreements (below), re-run the five
scenarios, then decide on weights.

### The four layers, and which one is failing

Diagnosing an SDLC failure means naming the layer first. They are separable and
`notebooks/11_eval_sdlc.py` separates them:

| layer | what it does | state |
|---|---|---|
| 1. Extraction | scenario → requirements | 3B invents on thin scenarios (2 of 4 on `fintech_startup` are not in the text); 8B does not |
| 2. Inference | requirements + scenario → 20 answers | **38/40 on the decision-carrying questions**, measured on the 8B pre-fix. Not the weak part. |
| 3. Scoring | answers → ranking | deterministic, correct, ~8 lines |
| 4. Framework | the tables themselves | **3 of 5 expected answers unreachable with perfect inference** |

Layer 4 is the binding constraint, and no amount of prompt work or fine-tuning
reaches it — fine-tuning targets layer 2, which is already at ~95%.

**Two of those three layer-4 failures are disagreements, not defects**, and both
need a human decision rather than a commit:

- **`bank_migration`** — expected V-Shaped, framework gives Waterfall 10/11.
  V-Shaped and Waterfall agree on 8 of 11 questions here; the rubric picks
  Waterfall on four signals against V-Shaped's one. **The framework may be right
  and the expectation wrong.** Settle before weighting anything to force it.
- **`kyc_onboarding`** — expected Agile/Spiral. Its requirements come from an
  RBI master direction, which makes "well known and stable" defensible, and that
  points plan-driven. Marked `DISPUTED` in the expected-answers file.
- **`fintech_startup`** — this one *is* a framework defect, and it is ours: the
  derived Agile column claims Agile expects "enhancement to an existing
  system = yes". Agile is not an enhancement methodology. One cell decides the
  scenario. Left untouched pending the supervisor's view of the column.

### Ground truth for the SDLC engine

`data/test_cases/<scenario>/expected.yaml` holds the expected SDLC **and** all 20
framework answers. Two independent claims, and the file says which is which:

- `expected_models` — the user's, trustworthy
- `answers` — Claude's draft, `verified: false`

Until the answers are verified, answer-level accuracy measures agreement with an
opinion. **This is `ground_truth.json`'s problem (B1) repeating**, so it is
labelled rather than assumed. Verifying them is also the seed of the
fine-tuning set: 5 scenarios × 20 questions = 100 labelled inference pairs.

`not_stated` is used wherever a scenario is silent — the same standard the
evidence gate holds the pipeline to. Ground truth that answers from thin air
would reward the pipeline for the thing the gate just stopped.

## Progress Summary (Days 1–11)

| Day | What Was Built | Key Files |
|-----|---------------|-----------|
| **1–2** | Environment setup, HF basics, venv, dependencies | `requirements.txt`, `notebooks/01_hf_basics.py` |
| **3** | LangGraph pipeline skeleton, LLM gateway (Gemini + OpenAI), Requirement model | `src/graph.py`, `src/utils/llm.py`, `src/models/requirement.py` |
| **4** | Extraction agent with few-shot prompting, baseline on 3 scenarios | `src/agents/extraction.py`, `evaluation/results/day4_extraction_baseline.json` |
| **5** | Zero-shot & few-shot classification baselines, 32-requirement ground truth | `src/agents/classification.py`, `data/training/classification/ground_truth.json`, `evaluation/results/day5_classification_baseline.json` |
| **6** | Quality analysis agent (10 checks), 3-node pipeline complete | `src/agents/quality.py`, `src/models/quality.py` |
| **7–8** | 343 labelled training requirements (56% finance), HF Dataset conversion | `data/training/classification/training_set.json`, `notebooks/04_prepare_bert_dataset.py` |
| **9–11** | Fine-tuned BERT + RoBERTa multi-label classifiers, per-architecture registry, corpus grown to 445 with RBI-grounded requirements, classifier wired into the pipeline | `src/models/multilabel.py`, `src/models/{bert,roberta}_classifier.py`, `notebooks/05_train_bert.py`, `notebooks/06_evaluate_bert.py` |

### Month 1 remaining

Superseded by the 14 Sep scope correction — the pre-correction table (QLoRA on
Days 13–16) is gone because it contradicted the plan that replaced it.

| Day | What to build | State |
|-----|--------------|-------|
| **12** | Local tier benchmarked and pinned; cloud tier behind a tunnel; one `OLLAMA_HOST`-configurable wrapper for both | done |
| **13–15** | Knowledge base + retrieval-grounded extraction — **deliverable 1** | done |
| **16–17** | SDLC engine, prompted — **deliverable 2** | done |
| **18** | Settle `bank_migration` and `kyc_onboarding` (expectation vs framework), re-run the five scenarios post-fix, then decide weights | **next** |
| **18–19** | End-to-end + CLI (`main.py`). Verify by disabling networking and re-running | |
| **20–21** | **Grounded vs ungrounded** (`--no-rag` is the control arm) + Week 3 review | |
| **22–24** | QLoRA — optional, time-boxed, gated on 13–21 being done | |
| **25–30** | Hardening; Day 30 gates on *both* deliverables | |

**On weights and fine-tuning — the current open question.** Equal weighting is
not the absence of a modelling choice; `framework.yaml`'s own comment ("no
weights are invented anywhere") presents it as neutrality, but it asserts that
all 20 questions are equally diagnostic, and nobody has defended that. Two
things to keep straight before acting on it:

1. **Fine-tuning targets layer 2, which is already at ~95%.** It cannot reach
   the layer-4 failures. Learning the weights is not an LLM fine-tune either —
   it is a 20-parameter linear model, blocked on data, not compute.
2. **5 scenarios cannot fit 20 free parameters.** Use a small number of discrete
   tiers, assign questions to them *by argument* with a written rationale (the
   treatment PTR-6 and the Agile column already get), and use the scenarios to
   falsify the assignment. `--framework` mode runs in a second with no LLM, so
   this is cheap to iterate.

Sourcing more scenarios: the 4 RFPs already in the knowledge base state most of
the 20 answers outright (timeline, funding, integration, greenfield vs
enhancement) and several state their own delivery model. But RFPs are
procurement documents — fixed scope, milestone payments — so they skew almost
entirely to Waterfall/V-Shaped/Incremental. A dataset built only from tenders
teaches "say Waterfall" and scores ~70% doing it. The iterative end has to come
from case studies or be hand-authored.

### Classification results (Days 9–11)

| Classifier | Micro F1 | Macro F1 | Exact Match | Latency |
|-----------|----------|----------|-------------|---------|
| Zero-Shot (Gemini) | 80.7% | 81.9% | 40.6% | 23.2s |
| Few-Shot (Gemini) | **84.0%** | **85.7%** | **53.1%** | 30.5s |
| Fine-tuned BERT | 78.0% | 78.5% | 34.4% | 19 ms |
| **Fine-tuned RoBERTa** | **80.2%** | **79.1%** | 31.2% | **21 ms** (3.6 ms batched) |

The fine-tuned models do **not** beat the few-shot baseline on F1 — RoBERTa is 3.8 points short, roughly 3 label decisions out of 80 on a 32-requirement eval set. They win on latency by ~1,450x, which is the trade the pipeline takes: classification runs per requirement, and 30 s each does not scale. Treat 1-point differences between runs as noise at this eval size.

## Architecture

The system is a **three-layer ML architecture** orchestrated as a LangGraph multi-agent pipeline:

- **Layer 1 — Fine-tuned specialist models (this team trains these):**
  - `bert-base-uncased` and `roberta-base` fine-tuned as **multi-label classifiers** over 13 requirement categories (Month 1). Multi-label via `BCEWithLogitsLoss` — each category is an independent binary decision — decoded with sigmoid plus a threshold calibrated on validation (not a fixed 0.5). One class per architecture, dispatched by `config.model_type` through `src/models/multilabel.py`.
  - A **3–4B local model** fine-tuned via **QLoRA** as a requirement **extractor** and **SDLC recommender** (Month 1, Days 13–16). The specific model is chosen on Day 12 by benchmark, not assumed — candidates must fit ~3GB of real headroom. Train on Colab's free T4 if 4GB is too tight, then deploy the adapter locally.
- **Layer 2 — RAG:** ChromaDB + `sentence-transformers` grounding agents in financial **requirement documents and regulations**. **Built Days 13–15 — this is deliverable 1**, not a Month 2 enhancement (see the scope correction above). The supervisor said *requirement documents*, not only regulations, so the corpus carries both: 9 regulations and 4 tender/RFP documents. Chunking is **structural (clause-level), not fixed-size** — that is what makes a citation auditable down to `§147(d)`. Two floors that matter: `MIN_CONFIDENCE = 0.55` for a citation and `MIN_CONTEXT_RELEVANCE = 0.60` for scenario-level passages, the second added because a COBOL migration was extracting "a mobile-responsive online loan application portal" out of retrieved loan RFPs. Retrieved tender text is **precedent, not material to extract from** — the prompt frames it as "reference only".
- **Layer 3 — LLM via Ollama, two tiers:** the core pipeline's general-purpose reasoner — extraction, quality checking, SDLC justification. Cloud Llama-3.1-8B for the demo, local 3–4B for offline/fallback, both behind one base-URL-configurable wrapper. Local model TBD on Day 12.
- **Baseline only — Gemini API:** the yardstick every local/fine-tuned model is measured against. Kept and maintained, never in the shipping path. Also legitimate as a *data-generation tool* for building training sets, provided the output is manually verified before use.

### Pipeline (LangGraph `StateGraph`, one node per agent)

**Phase 1 — Problem Scenario → Requirements**

Currently (extraction and quality still on the API — to be replaced Days 12–18):
`Extraction (Gemini) → Classification (fine-tuned RoBERTa) → Quality (Gemini) → Structured Requirements`

Target after Day 18 — no *third-party* API anywhere in this path; the LLM is our own fine-tuned model on either tier:
`Extraction (fine-tuned local/cloud) → Classification (fine-tuned RoBERTa) → Quality (local/cloud) → SDLC (fine-tuned local/cloud)`

The classification agent resolves its model from `models/roberta_classifier` then `models/bert_classifier`, overridable with `SYNTHEIA_CLASSIFIER_DIR`, and falls back to the few-shot LLM when nothing is trained. Retraining changes what the pipeline uses with no code edit. Compare against the baselines with `python -m src.graph --classifier few_shot`.

**Phase 2 — Requirements → SDLC Recommendation (Days 15–16):**
`Requirements → SDLC Agent (fine-tuned local/cloud) → Ranked SDLC + justification`

Grounded in the supervisor's selection framework: **4 tables** — Requirements,
Project Type & Risk, Project Team, User Community — over **7 SDLC models**:
Waterfall, V-Shaped, Prototype, RAD, Incremental, Spiral, Agile.

His tables are **19 questions × 6 models**. What the engine scores is **20 × 7**
(v1.1): the Agile column and PTR-6 are ours, both marked in `framework.yaml`.
Stripping the two additions gives his framework exactly — which is the point of
marking them, since a recommendation claiming to be scored against his rubric
has to be able to show which parts are.

**Superseded 15 Sep: all four tables are now inferred.** The original decision
(13 Sep) scored only Requirements and Project Type & Risk, on the grounds that a
problem scenario says nothing about team size. That was true of the original
scenarios and false of the current ones — *"the project team consists of 2
developers"*, *"the team has 15 developers, funding is approved for 18 months"*,
*"the product team is small, 3 people"*. The facts were there; nothing was
reading them. `infer_project_context()` now answers Project Team and User
Community from the scenario, and anything it cannot support comes back
`not_stated` and is left unscored.

The framework is **v1.1: 20 questions × 7 models**, not the transcribed 19 × 6.
Three additions, each marked in `framework.yaml` so a reader can strip them back
to the supervisor's original:

| addition | marker | why |
|---|---|---|
| `agile` column | `source: derived` | his tables have six columns and no Agile; the problem statement and the DoD both name it |
| **PTR-6** *"Does the project require formal verification and testing at multiple stages?"* | `origin: derived` | his tables ask nothing about verification rigour, which leaves V-Shaped indistinguishable from Waterfall on a testing-heavy project |
| `gloss` on all 20 | — | **not** a change to the framework: his wording is untouched, the gloss only says how to read it |

The glosses exist because terse phrasing was being answered as a different
question. `PTR-2 "Is the project a system integration project?"` came back **NO**
for a platform doing real-time NSDL lookups, Aadhaar verification and video KYC —
read as *"is a systems-integrator vendor doing the work"*. Adding the gloss
flipped it to YES citing NSDL, and moved `kyc_onboarding` from
`Incremental = Waterfall` to `Agile = Prototype = Spiral`.

**Ties are reported as ties.** `score()` breaks them by name so a rerun cannot
reorder equal results, but a reproducible tie-break is not a finding: two of five
recommendations were decided alphabetically and presented as conclusions, and one
of those coin-flips happened to be right. `tied_with` now surfaces it.

**The evidence gate.** A yes/no whose evidence is not traceable to the problem
statement is demoted to `not_stated`. Needed because across five scenarios
*every* supplied-table answer that got through was **yes** and five of six cited
nothing — `USER-2` came back yes in four of five runs, a constant rather than an
inference, and `USER-2 = yes` scores for Prototype, Spiral and Incremental and
for nobody else. Questions marked `evidence_from: scenario` (PTR-4 funding, PTR-5
lifespan) additionally reject requirement ids: **requirements do not testify
about budgets**, however real the cited id is.

Those two tables are the ones that can be inferred: requirement clarity,
stability and complexity are properties of the requirements themselves, and
criticality, regulatory exposure and technology novelty are already carried in
the `risk_level` and `applicable_regulations` fields. Project Team (size,
experience) and User Community (availability, involvement) are *not* derivable
from a problem scenario — a 118-word KYC brief says nothing about team size —
so inventing answers for them would be fabrication dressed as inference. They
move to a later phase, where a supplied or interactive input makes sense.

Consequence for the training pairs: the agent emits the inferred criteria
*alongside* the recommendation in one structured output —
`{inferred_criteria: [{question, answer, evidence: REQ-id}], ranked_models,
recommendation, justification}`. One call, one pair per scenario. Going straight
from requirements to a bare SDLC name would make the justification unfalsifiable
and fail the Definition of Done, which requires it to reference actual
requirements *and* selection criteria.

Definition of Done worth noting: a loan scenario should recommend Agile or Spiral (**not** Waterfall), a payment system should differ, and the justification must reference actual requirements and selection criteria.

### Requirement Model (13 fields)

**The extractor produces 12 of them.** `category` is deliberately absent from
`ExtractedRequirement` / `ExtractionDraft`, the schema sent to the LLM: the
classification node overwrites it with the fine-tuned RoBERTa output, so
generating it spent tokens on an answer discarded one node later, and would
have meant hand-verifying categories across every extraction training pair to
teach a field nothing reads. `Requirement` adds it back, so the rest of the
system is unchanged — it simply arrives empty and the classifier fills it.

Note also that Pydantic turns a class docstring into the JSON-schema
`description`, which is sent to the model on every call. Keep rationale in
comments on these classes, not docstrings, or it is billed as prompt tokens and
read as instruction.

Each extracted requirement has: `id`, `statement`, `category` (list), `source_stakeholder`, `business_justification`, `priority`, `dependencies`, `assumptions`, `acceptance_criteria`, `applicable_regulations`, `risk_level`, `confidence_score`, `approval_status`.

### The 13 requirement categories

Business, Stakeholder, Functional, Security, Privacy, Regulatory, Performance, Availability, Usability, Data-management, Integration, Audit/Reporting, Operational — every requirement can carry multiple labels.

### Project structure

```
src/agents/       # One module per pipeline agent (extraction.py, classification.py, quality.py)
src/models/       # Pydantic schemas (requirement.py, quality.py) plus the classifier
                  #   multilabel.py -- registry, build/load, shared head + BCE loss,
                  #                    decode helpers, MultiLabelClassifier wrapper
                  #   bert_classifier.py / roberta_classifier.py -- one per architecture
src/utils/        # LLM gateway (llm.py → llm_gemini.py/llm_openai.py) with round-robin key rotation
                  #   llm_ollama.py -- Day 12. Same interface as llm_gemini, with a
                  #                    configurable base URL so ONE wrapper drives both
                  #                    the local model and the cloud endpoint
src/rag/          # Deliverable 1 -- BUILT. loader.py (PDF/DOCX/HTML/MD, mandatory
                  #   sidecar manifest -- MissingManifest is fatal), chunker.py
                  #   (clause-level, TOC detection, word-boundary splits),
                  #   embedder.py, store.py (ChromaDB), retriever.py
src/sdlc/         # framework.py -- loads data/sdlc/framework.yaml and scores.
                  #   NO LLM here: the tables ARE the scoring function, so
                  #   scoring is counting matches. Strict loader -- a model
                  #   missing an answer would win on a smaller denominator.
src/training/     # QLoRA fine-tuning scripts/configs -- Days 22-24, optional
src/generators/   # Document generation (SRS) — Month 3
app/              # Web application layer — Month 3
data/sdlc/        # framework.yaml -- v1.1, 20 questions x 7 models
data/test_cases/  # <scenario>/scenario_01.txt plus expected.yaml (ground truth:
                  #   expected_models is the user's, `answers` is a draft with
                  #   verified: false -- the file says which is which)
data/training/    # classification/ 445 requirements, 13 categories (done)
                  # extraction/     instruction-tuning pairs for QLoRA (Day 13-14)
                  # sdlc/           requirements -> SDLC + justification (Day 15-16)
models/           # Trained models. Everything here is gitignored except MODEL_CARD.md
                  # in each directory -- weights, config.json, tokenizer.json and
                  # threshold.json are all regenerated by training
notebooks/        # Sequential scripts. 01_hf_basics .. 06_evaluate_bert (classifier),
                  #   08_benchmark_local_llm, 09_build_knowledge_base + 09a_fetch_sources,
                  #   10_eval_retrieval, 11_eval_sdlc, 12_calibrate_evidence_relevance
main.py           # CLI entry point -- `python main.py --scenario x.txt`, Day 17-18
evaluation/       # Baseline results (day4_extraction, day5_classification)
tests/
```

## Commands

```bash
# rebuild corpus -> HF dataset (run after any data or label change)
python -m data.training.classification.build_training_set   # or run the file directly
python -m notebooks.04_prepare_bert_dataset

# train (output paths are keyed off config.model_type, so runs cannot clobber each other)
python -m notebooks.05_train_bert --epochs 45 --patience 8                      # bert
python -m notebooks.05_train_bert --model roberta-base --epochs 45 --patience 8 # roberta

# evaluate against the held-out ground truth and the Day 5 baselines
python -m notebooks.06_evaluate_bert --model-dir models/roberta_classifier --compare

# run the pipeline
python -m src.graph                          # fine-tuned classifier
python -m src.graph --classifier few_shot    # LLM baseline, for comparison

# local LLM (Day 12 onward)
ollama serve                                 # daemon
ollama run llama3.2:3b                       # smoke-test the model

# cloud tier -- one env var, same wrapper, no second code path
export OLLAMA_HOST=https://<tunnel>.trycloudflare.com
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_NUM_CTX=8192
# ... and `unset OLLAMA_HOST OLLAMA_MODEL OLLAMA_NUM_CTX` afterwards, or the
# next local run silently goes to a dead tunnel

# ── SDLC evaluation ──
# Mode 1 first, ALWAYS. Scores the EXPECTED answers through the tables with no
# LLM: "if inference were perfect, would the framework pick the right SDLC?"
# If it cannot, prompt work and fine-tuning are both wasted -- the problem is
# the tables or the expectations.
python -m notebooks.11_eval_sdlc --framework

# Mode 2. Compare saved runs against the expected answers, per question.
for s in bank_migration insurance_claims credit_risk fintech_startup kyc_onboarding; do
  python -u -m src.graph --scenario data/test_cases/$s/scenario_01.txt \
    --json evaluation/results/sdlc_runs/$s.json
done
python -m notebooks.11_eval_sdlc --runs evaluation/results/sdlc_runs

# Why the split matters: errors come back as wrong / missed / unsupported, and
# the three want different fixes -- inference quality, an over-strict abstention
# prompt, and the evidence gate respectively. Before this they all looked
# identical ("the recommendation changed").
```

## Documentation

| file | what it is |
|---|---|
| `docs/problem_log.md` | **Every defect and wrong turn from the first commit, with what it cost.** Read §8 (the recurring patterns) before debugging anything. Also records what we chose *not* to fix, and why. |
| `docs/bottlenecks.md` | Problems still **open**, ordered by damage if ignored. B1 (unverified ground truth) qualifies every classification number reported. |
| `docs/knowledge_base.md` | Corpus provenance, per §5 of the problem statement |
| `docs/demo_runbook.md` | Rehearsed demo-day sequence, including the fallback |

Never run two trainings at once — they share `models/experiments/<model_type>_run/`
and will corrupt each other's checkpoints.

Training writes the model **twice**: to `models/<type>_classifier` in the repo
(canonical, survives a WSL reset) and to `~/.local/share/syntheia/models/` (what
the pipeline actually loads, via `SYNTHEIA_CLASSIFIER_DIR`). Both every run, so
the fast copy cannot go stale and silently serve old weights. `SYNTHEIA_MODEL_HOME=off`
disables the mirror.

`--pos-weight` is off by default and should stay off: micro-F1 is monotonic in
the weighting (10x -> 75.6%, 3x -> 76.2%, off -> 77.3%).

## Environment & Configuration

- **Python 3.12**. The virtualenv lives at **`~/venvs/syntheia`**, NOT in the repo.
  Activate it (`source ~/venvs/syntheia/bin/activate`) or call
  `~/venvs/syntheia/bin/python` directly.
  - **Do not symlink `venv` in the repo to it.** Python derives `sys.prefix`
    from the path you invoke, so `sys.path` keeps the `/mnt/f` spelling and
    every import stats through v9fs anyway: measured 2.4 s direct vs 54 s
    through the symlink, a 22x penalty for the convenience.
  - `venv.v9fs-slow/` is the old in-repo venv, kept until the new one has
    proved itself. Delete it to reclaim 6.6 GB.
  - `requirements.txt` is unpinned (`>=` throughout); `requirements.lock.txt`
    pins the 219 packages actually installed. Install from the lock file —
    a fresh resolve can move off `transformers` 5.15.1, whose v5 API the
    code depends on.
- Secrets in `.env` (gitignored); `.env.example` is the template.
  - `GEMINI_API_KEY` — supports **comma-separated multiple keys** for round-robin rotation (bypasses free-tier rate limits). Keys from different GCP projects.
  - `OPENAI_API_KEY`, `WANDB_API_KEY` — not yet active.
  - `HF_TOKEN` — **currently a placeholder**. Less urgent since the revision: Phi-3.5 is not gated, and Ollama pulls its own weights independently of the Hub. Still worth setting — anonymous Hub downloads are rate-limited, which is how `roberta-base` first failed. `notebooks/05_train_bert.py` and `06_evaluate_bert.py` call `load_dotenv()` so it is picked up once set.
- **Gemini model:** `gemini-3.6-flash` via `google-genai` SDK (structured output with `response_schema`). **Baseline only** after the Sep 11 revision — not part of the shipping pipeline.
- **Local LLM: `llama3.2:3b`**, pinned Day 12 by benchmark (`evaluation/results/day12_local_llm_benchmark.json`). 36 tok/s, 2.3GB VRAM, 100% valid JSON, 1.8 categories/requirement.
  - **`qwen2.5:3b` was faster (47 tok/s) and rejected anyway.** It scored 100% on every automated metric while assigning 5.5 of the 13 categories to every requirement — every label legal, the output useless. `phi3.5` was slowest at 14 tok/s and the only one to invent an out-of-taxonomy label. The speed ranking was the inverse of the quality ranking.
  - **Correction (13 Sep):** that category evidence was measuring a field the pipeline *discards* — the classification node overwrites `category`, so the extractor's labels never survive. The rejection still stands, but the reasoning is indirect rather than direct: a model that assigns 8–10 of 13 categories while the prompt says *"only assign a category if the requirement clearly relates to it"* is demonstrating poor instruction-following, and that bears on the fields we do keep. The statement-quality margin was thinner than first recorded — qwen missed the portal requirement, llama missed the RBI compliance one. `category` has since been removed from the extraction schema, and the benchmark now measures `shall_rate` and `field_completeness` instead: same instruction-following signal, against fields that survive.
  - Set `OLLAMA_MODEL` to override.
- **Ollama is installed** (11 Sep) as a systemd service, `active` and `enabled`, with models on **`/mnt/f/ollama-models`** via a drop-in (`sudo systemctl edit ollama` → `Environment="OLLAMA_MODELS=/mnt/f/ollama-models"`). Not the default `~/.ollama`: that lives on WSL's ext4 vhdx whose `df` "937G free" is the virtual disk's *maximum*, not real space — the vhdx sits on Windows `C:` which has only **45GB free**, while `F:` has 72GB. NTFS slowness does not bite here, since weights are read once at load and then cached.
- **Cloud LLM:** Llama-3.1-8B via Ollama on a Colab/Kaggle T4 behind an ngrok/Cloudflare tunnel. Reached through the same wrapper by setting `OLLAMA_HOST`.
- **LLM gateway** (`src/utils/llm.py`): Auto-selects provider based on available `.env` keys. `llm_ollama.py` joins it at Day 12 and should become the default, with Gemini reachable explicitly for comparison runs.
- **Hardware:** RTX 3050 **4GB VRAM**, **7GB system RAM** (~5GB free), WSL2. This is the binding constraint on every model choice — BERT/RoBERTa fine-tune comfortably, but anything above ~3GB at 4-bit cannot be served. QLoRA may need Colab's free T4 for training, with the adapter deployed locally. Writes to `/mnt/f` are slow (NTFS through WSL): a 440 MB checkpoint save costs ~6 s.

## Working Conventions

- **Day-wise git commits** — each day's work is committed with `"Day N: description"` message.
- **Baseline-first approach** — every fine-tuned model must be benchmarked against zero-shot and few-shot LLM baselines, with results saved to `evaluation/results/`. After the Sep 11 revision the comparison ladder is: Gemini (prompted) vs local (prompted) vs local (fine-tuned).
- **Model choices are made by benchmark, not assumption.** Day 12 measured candidates on this hardware before pinning one, and the result inverted the expected ranking. Record the rejects and why — that reasoning is the Week 11 E1–E7 ablation story.
- **Enforce in code what is enforceable; the prompt only asks.** A 3B does not
  follow negative constraints, and neither does the 8B reliably. Three separate
  instructions failed to stop the model arguing from priority to requirement
  stability — withholding the field (`_format_statements`) worked immediately.
  Two instructions failed to produce citations — removing the schema default,
  which made the field *required* in the generated grammar, worked immediately.
  Same pattern in the evidence gate and in `filter_issues`. When a rule can be
  checked in Python, check it in Python.
- **When a model reports something absurd, check what it was sent before
  blaming its reasoning.** The quality agent reported requirements as missing
  acceptance criteria because it was sent a truncated requirement that excluded
  them. The justifier stated the opposite of the data because it was shown which
  questions matched but not *what the project answered*. Neither was a reasoning
  failure; both were answering the question actually asked.
- **A fix that silently changes a second behaviour is a new bug.** The
  scenario-relevance floor also disabled per-requirement citations. The
  abstention change broke the justifier's yes/no rendering (all three `Literal`
  values are truthy strings, so everything read as "yes"). The question glosses
  made the model answer where it used to abstain — sometimes from noise. Each
  was a good fix with an unexamined blast radius.
- **Write down what you expect before changing anything.** Everything found on
  17 Sep was found in one day, after the expected answers went into files. The
  same defects had been in the terminal output for a week and were invisible in
  scrollback. A recommendation that merely *differs* tells you nothing about
  whether it improved.
- **Do not weight a question you have not exercised.** PTR-6 would need a weight
  of ≥2.5 ordinary questions to flip `bank_migration` to V-Shaped — and PTR-6 is
  answered by exactly **one** of the five scenarios. Weighting it would build the
  result on a single data point. Rule: nothing goes in a high-importance tier
  until more than one scenario tests it. (And fitting 20 free weights to 5
  scenarios is memorisation, not learning — argue the weights and use the
  scenarios to *falsify* them.)
- **A metric that cannot fail is not a metric.** `in_taxonomy_rate` read 100% for a model spraying 5.5 of 13 categories at every requirement: it validated label *names* and never label *counts*. It later turned out to be worse than that — it was scoring a field the pipeline throws away. Ask what reads a number before trusting it. Both the classifier work (Days 9–11) and the extractor work hit this — automated checks confirm format, not sense. **Print the actual output and read it before pinning anything.**
- **No third-party LLM API in the core path.** Gemini is a baseline and a data-generation tool, never a dependency. Our own fine-tuned model on either tier is fine — cloud-hosted was confirmed permitted on 11 Sep.
- **Write for both tiers from the start.** Anything touching the LLM goes through the base-URL-configurable wrapper, so local and cloud never diverge into two code paths.
- **Ground truth is sacred** — `ground_truth.json` (32 verified requirements) is the held-out eval set. NEVER included in training data.
- **Training data quality** — synthetic examples are manually reviewed before use. Don't treat LLM-generated labels as ground truth. **The Day 9–11 additions are not yet reviewed**: 65 rows and 13 label corrections are all `verified: false`.
- **Label conventions are inferred, not documented.** The Regulatory and identity-data rules in `data/training/classification/label_corrections.json` were derived by reading what `ground_truth.json` actually does, because nothing states them. They carry the evidence (which ground-truth requirements support each rule) and the known tensions — REQ-008 vs REQ-029 on encryption, REQ-020 vs REQ-028 on audit logging. Read that file before labelling anything new.
- **Corpus edits go through `label_corrections.json`**, not by editing source files. `build_training_set.py` applies them after merging and fails loudly if a correction targets a label that is no longer present, so a correction cannot drift silently.
- **Diagnose before adding data.** The Day 9–11 gain came from measuring where the training convention diverged from ground truth (identity rows carried Privacy 15% of the time in training vs 50% in ground truth), not from adding rows blindly. Model-side tuning gave 2.9 points across three experiments; one targeted data fix gave 2.2.
- Prompts are currently inline in agent files. Once `src/utils/prompts.py` exists, externalize them there.

## Gitignored Items

- `venv/`, `.env`, `data/training/classification/bert_dataset/` (generated HF Dataset)
- `models/**` entirely, except `MODEL_CARD.md` in each model directory. Weights, `config.json`, `tokenizer.json` and `threshold.json` are all regenerated by training; tracking them meant a branch switch could pair one run's config with another run's weights, which loads silently and predicts garbage. Hyperparameters live in `evaluation/results/day9_*_training.json`.
- `/implementation-plans`, `/learning-pytorch`
