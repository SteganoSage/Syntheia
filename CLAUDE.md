# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Syntheia is an agentic AI system for **automated requirement gathering, analysis, and SDLC methodology identification in financial services** (loan origination, KYC/onboarding, payments). It takes a natural-language problem scenario and produces structured, classified, quality-checked software requirements (Phase 1), then recommends a ranked SDLC methodology given project conditions (Phase 2).

The project follows a day-by-day 4-month implementation roadmap at `implementation-plans/implementation_plan3.md` (the current/superseding plan). **This directory is gitignored** — it's a local planning reference only.

### ⚠️ Architecture change — plan revised 11 Sep 2026

The plan's Day 12 section carries an **ARCHITECTURE CHANGE** that supersedes the earlier design:

> The core pipeline must use a **local lightweight LLM** (Phi-3.5 3.8B via Ollama), **NOT** the Gemini API. Gemini is kept only as a baseline for comparison. QLoRA fine-tuning is pulled into Month 1.

Three consequences:

1. **Gemini is demoted to a baseline.** The shipping pipeline must run offline with no API calls. `src/utils/llm_gemini.py` stays, but only for comparison runs.
2. **Phi-3.5 3.8B replaces Llama-3.1-8B** as the fine-tuning target — it fits the 4GB RTX 3050 in this environment, where an 8B model does not.
3. **QLoRA moves from Month 2 into Month 1** (Days 13–16), as does the whole SDLC engine.

**The plan is internally inconsistent about this.** Only Days 12–20 were revised; the rest of the document still names Llama-3.1-8B (11 mentions) and GPT-4o (25 mentions). When those conflict with the Day 12 note, **the Day 12 note wins** — it is dated later and explicitly labelled an architecture change.

### Current status

**Days 9–11 are complete.** The 3-node LangGraph pipeline is operational and the classification node runs a fine-tuned RoBERTa classifier (80.2% micro-F1 on held-out ground truth, 21 ms/requirement) instead of the Gemini few-shot baseline.

One Definition-of-Done item was **not** met: Day 11 required the fine-tuned model to beat the LLM baselines. It does not — 80.2% against few-shot Gemini's 84.0%. It wins on latency by ~1,450x. See the baseline table below.

**Day 12 is next:** install Ollama on WSL2, pull `phi3.5`, and write `src/utils/llm_ollama.py` with the same interface as `llm_gemini.py`.

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

### Month 1 remaining (revised plan)

| Day | What to build | Key files |
|-----|--------------|-----------|
| **12** | Ollama on WSL2, pull `phi3.5` (~2.5 GB), local LLM wrapper matching the `llm_gemini.py` interface. Prompted extraction + SDLC baselines. | `src/utils/llm_ollama.py` |
| **13–14** | Extraction dataset (~100–150 instruction/output pairs, Gemini as a *data-generation tool* then manually verified). QLoRA fine-tune Phi-3.5. Colab free T4 if 4GB VRAM is too tight — deploy the adapter locally. | `notebooks/07_finetune_phi_extraction.py`, `data/training/extraction/` |
| **15–16** | SDLC recommendation engine + QLoRA fine-tune on the SDLC task (multi-task or a separate adapter). ~100–150 examples covering all 7 models. | `src/models/sdlc.py`, `src/agents/sdlc.py`, `data/training/sdlc/` |
| **17–18** | End-to-end pipeline with **no API calls**: Phi-3.5 extraction → RoBERTa classification → Phi-3.5 SDLC. CLI entry point. | `main.py`, `src/graph.py` |
| **19–20** | Comparison: Gemini (prompted) vs Phi-3.5 (prompted) vs Phi-3.5 (fine-tuned), on 3+ scenarios, with latency. | comparison notebook |
| **21** | Metrics, Week 3 review. | |
| **22–30** | Weakest-link improvement, edge cases, cleanup, demo notebook, docs, Month 1 checkpoint. | |

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
  - `Phi-3.5-mini` (3.8B) fine-tuned via **QLoRA** as a requirement **extractor** and **SDLC recommender** (Month 1, Days 13–16). Replaces the originally planned Llama-3.1-8B: 8B does not fit the 4GB RTX 3050 here, Phi-3.5 does. Train on Colab's free T4 if 4GB is too tight for training, then deploy the adapter locally.
- **Layer 2 — RAG:** ChromaDB + `sentence-transformers` grounding agents in financial regulation documents (Month 2).
- **Layer 3 — Local LLM (Phi-3.5 via Ollama):** the core pipeline's general-purpose reasoner — extraction, quality checking, SDLC justification. Must run **offline, with no API calls**.
- **Baseline only — Gemini API:** the yardstick every local/fine-tuned model is measured against. Kept and maintained, never in the shipping path. Also legitimate as a *data-generation tool* for building training sets, provided the output is manually verified before use.

### Pipeline (LangGraph `StateGraph`, one node per agent)

**Phase 1 — Problem Scenario → Requirements**

Currently (extraction and quality still on the API — to be replaced Days 12–18):
`Extraction (Gemini) → Classification (fine-tuned RoBERTa) → Quality (Gemini) → Structured Requirements`

Target after Day 18 — no API calls anywhere in this path:
`Extraction (fine-tuned Phi-3.5) → Classification (fine-tuned RoBERTa) → Quality (Phi-3.5) → SDLC (fine-tuned Phi-3.5)`

The classification agent resolves its model from `models/roberta_classifier` then `models/bert_classifier`, overridable with `SYNTHEIA_CLASSIFIER_DIR`, and falls back to the few-shot LLM when nothing is trained. Retraining changes what the pipeline uses with no code edit. Compare against the baselines with `python -m src.graph --classifier few_shot`.

**Phase 2 — Requirements → SDLC Recommendation (Days 15–16):**
`Requirements → SDLC Agent (fine-tuned Phi-3.5) → Ranked SDLC + justification`

Grounded in the supervisor's selection framework: **4 tables × 19 questions** (Requirements, Project Team, Project Type & Risk, User Community) over **7 SDLC models** — Waterfall, V-Shape, Prototyping, RAD, Incremental & Iterative, Spiral, Agile.

Definition of Done worth noting: a loan scenario should recommend Agile or Spiral (**not** Waterfall), a payment system should differ, and the justification must reference actual requirements and selection criteria.

### Requirement Model (13 fields)

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
                  #   llm_ollama.py -- local Phi-3.5, Day 12, same interface as llm_gemini
src/rag/          # Document loader, chunker, embedder, ChromaDB — Month 2
src/training/     # QLoRA fine-tuning scripts/configs — Days 13-16
src/generators/   # Document generation (SRS) — Month 3
app/              # Web application layer — Month 3
data/training/    # classification/ 445 requirements, 13 categories (done)
                  # extraction/     instruction-tuning pairs for QLoRA (Day 13-14)
                  # sdlc/           requirements -> SDLC + justification (Day 15-16)
models/           # Trained models. Everything here is gitignored except MODEL_CARD.md
                  # in each directory -- weights, config.json, tokenizer.json and
                  # threshold.json are all regenerated by training
notebooks/        # Sequential scripts (01_hf_basics → 06_evaluate_bert;
                  #   07_finetune_phi_extraction next)
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
ollama run phi3.5                            # smoke-test the model
```

Never run two trainings at once — they share `models/experiments/<model_type>_run/`
and will corrupt each other's checkpoints.

`--pos-weight` is off by default and should stay off: micro-F1 is monotonic in
the weighting (10x -> 75.6%, 3x -> 76.2%, off -> 77.3%).

## Environment & Configuration

- **Python 3.12**, dependencies in `requirements.txt`. Virtualenv lives in `venv/` (gitignored).
- Secrets in `.env` (gitignored); `.env.example` is the template.
  - `GEMINI_API_KEY` — supports **comma-separated multiple keys** for round-robin rotation (bypasses free-tier rate limits). Keys from different GCP projects.
  - `OPENAI_API_KEY`, `WANDB_API_KEY` — not yet active.
  - `HF_TOKEN` — **currently a placeholder**. Less urgent since the revision: Phi-3.5 is not gated, and Ollama pulls its own weights independently of the Hub. Still worth setting — anonymous Hub downloads are rate-limited, which is how `roberta-base` first failed. `notebooks/05_train_bert.py` and `06_evaluate_bert.py` call `load_dotenv()` so it is picked up once set.
- **Gemini model:** `gemini-3.6-flash` via `google-genai` SDK (structured output with `response_schema`). **Baseline only** after the Sep 11 revision — not part of the shipping pipeline.
- **Local LLM:** Phi-3.5-mini 3.8B via Ollama on WSL2 (~2.5 GB pull). The core pipeline's reasoner from Day 12.
- **LLM gateway** (`src/utils/llm.py`): Auto-selects provider based on available `.env` keys. `llm_ollama.py` joins it at Day 12 and should become the default, with Gemini reachable explicitly for comparison runs.
- **GPU:** RTX 3050 4GB (WSL2). Drives the model choices — BERT/RoBERTa fine-tune comfortably; Phi-3.5 QLoRA may need Colab's free T4 for training, with the adapter deployed locally afterwards. Writes to `/mnt/f` are slow (NTFS through WSL), so a 440 MB checkpoint save costs ~6 s.

## Working Conventions

- **Day-wise git commits** — each day's work is committed with `"Day N: description"` message.
- **Baseline-first approach** — every fine-tuned model must be benchmarked against zero-shot and few-shot LLM baselines, with results saved to `evaluation/results/`. After the Sep 11 revision the comparison ladder is: Gemini (prompted) vs local Phi-3.5 (prompted) vs Phi-3.5 (fine-tuned).
- **The shipping pipeline runs offline.** No API calls in the core path. Gemini is a baseline and a data-generation tool, never a dependency.
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
