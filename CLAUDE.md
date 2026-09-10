# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Syntheia is an agentic AI system for **automated requirement gathering, analysis, and SDLC methodology identification in financial services** (loan origination, KYC/onboarding, payments). It takes a natural-language problem scenario and produces structured, classified, quality-checked software requirements (Phase 1), then recommends a ranked SDLC methodology given project conditions (Phase 2).

The project follows a day-by-day 4-month implementation roadmap at `implementation-plans/implementation_plan3.md` (the current/superseding plan). **This directory is gitignored** — it's a local planning reference only.

The project is currently at the end of **Month 1, Day 9–11** — the 3-node LangGraph pipeline is operational and the classification node now runs a fine-tuned RoBERTa classifier (80.2% micro-F1 on held-out ground truth, 21 ms/requirement) instead of the Gemini few-shot baseline. RAG and the QLoRA Llama extractor are next (Month 2).

## Progress Summary (Day 1–8)

| Day | What Was Built | Key Files |
|-----|---------------|-----------|
| **1–2** | Environment setup, HF basics, venv, dependencies | `requirements.txt`, `notebooks/01_hf_basics.py` |
| **3** | LangGraph pipeline skeleton, LLM gateway (Gemini + OpenAI), Requirement model | `src/graph.py`, `src/utils/llm.py`, `src/models/requirement.py` |
| **4** | Extraction agent with few-shot prompting, baseline on 3 scenarios | `src/agents/extraction.py`, `evaluation/results/day4_extraction_baseline.json` |
| **5** | Zero-shot & few-shot classification baselines, 32-requirement ground truth | `src/agents/classification.py`, `data/training/classification/ground_truth.json`, `evaluation/results/day5_classification_baseline.json` |
| **6** | Quality analysis agent (10 checks), 3-node pipeline complete | `src/agents/quality.py`, `src/models/quality.py` |
| **7–8** | 343 labelled training requirements (56% finance), HF Dataset conversion | `data/training/classification/training_set.json`, `notebooks/04_prepare_bert_dataset.py` |
| **9–11** | Fine-tuned BERT + RoBERTa multi-label classifiers, per-architecture registry, corpus grown to 445 with RBI-grounded requirements, classifier wired into the pipeline | `src/models/multilabel.py`, `src/models/{bert,roberta}_classifier.py`, `notebooks/05_train_bert.py`, `notebooks/06_evaluate_bert.py` |

### Baseline Numbers (to beat with fine-tuned BERT)

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
  - `Llama-3.1-8B-Instruct` fine-tuned via **QLoRA** as a requirement **extractor** (Month 2).
- **Layer 2 — RAG:** ChromaDB + `sentence-transformers` grounding agents in financial regulation documents (Month 2).
- **Layer 3 — General LLM API (Gemini):** used for extraction, quality checking, SDLC justification. Also serves as the **zero-shot/few-shot baseline** that fine-tuned models are benchmarked against.

### Pipeline (LangGraph `StateGraph`, one node per agent)

**Phase 1 — Problem Scenario → Requirements (OPERATIONAL):**
`Extraction Agent (Gemini) → Classification Agent (fine-tuned RoBERTa) → Quality Agent (Gemini) → Structured Requirements`

The classification agent resolves its model from `models/roberta_classifier` then `models/bert_classifier`, overridable with `SYNTHEIA_CLASSIFIER_DIR`, and falls back to the few-shot LLM when nothing is trained. Retraining changes what the pipeline uses with no code edit. Compare against the baselines with `python -m src.graph --classifier few_shot`.

**Phase 2 — Requirements → SDLC Recommendation (Day 15-20):**
`Feature Extractor → Scoring Matrix → LLM Justification → Ranked SDLC`

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
src/rag/          # Document loader, chunker, embedder, ChromaDB — Month 2
src/training/     # Fine-tuning scripts/configs — Month 2
src/generators/   # Document generation (SRS) — Month 3
app/              # Web application layer — Month 3
data/training/    # Hand-labelled training datasets (343 requirements, 13 categories)
models/           # Trained models. Everything here is gitignored except MODEL_CARD.md
                  # in each directory -- weights, config.json, tokenizer.json and
                  # threshold.json are all regenerated by training
notebooks/        # Sequential scripts (01_hf_basics → 04_prepare_bert_dataset)
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
  - `HF_TOKEN` — **currently a placeholder**. Public models download anonymously (rate-limited), but Month 2's gated Llama-3.1-8B will need a real token. `notebooks/05_train_bert.py` and `06_evaluate_bert.py` call `load_dotenv()` so it is picked up once set.
- **Gemini model:** `gemini-3.6-flash` via `google-genai` SDK (structured output with `response_schema`).
- **LLM gateway** (`src/utils/llm.py`): Auto-selects provider based on available `.env` keys. Currently only Gemini is active.

## Working Conventions

- **Day-wise git commits** — each day's work is committed with `"Day N: description"` message.
- **Baseline-first approach** — every fine-tuned model must be benchmarked against zero-shot and few-shot LLM baselines, with results saved to `evaluation/results/`.
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
