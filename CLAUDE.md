# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Syntheia is an agentic AI system for **automated requirement gathering, analysis, and SDLC methodology identification in financial services** (loan origination, KYC/onboarding, payments). It takes a natural-language problem scenario and produces structured, classified, quality-checked software requirements (Phase 1), then recommends a ranked SDLC methodology given project conditions (Phase 2).

The project follows a day-by-day 4-month implementation roadmap at `implementation-plans/implementation_plan3.md` (the current/superseding plan). **This directory is gitignored** — it's a local planning reference only.

The project is currently in **Month 1, Day 9** — the 3-node LangGraph pipeline is operational (extraction → classification → quality), LLM baselines are established, and BERT fine-tuning is next.

## Progress Summary (Day 1–8)

| Day | What Was Built | Key Files |
|-----|---------------|-----------|
| **1–2** | Environment setup, HF basics, venv, dependencies | `requirements.txt`, `notebooks/01_hf_basics.py` |
| **3** | LangGraph pipeline skeleton, LLM gateway (Gemini + OpenAI), Requirement model | `src/graph.py`, `src/utils/llm.py`, `src/models/requirement.py` |
| **4** | Extraction agent with few-shot prompting, baseline on 3 scenarios | `src/agents/extraction.py`, `evaluation/results/day4_extraction_baseline.json` |
| **5** | Zero-shot & few-shot classification baselines, 32-requirement ground truth | `src/agents/classification.py`, `data/training/classification/ground_truth.json`, `evaluation/results/day5_classification_baseline.json` |
| **6** | Quality analysis agent (10 checks), 3-node pipeline complete | `src/agents/quality.py`, `src/models/quality.py` |
| **7–8** | 343 labelled training requirements (56% finance), HF Dataset conversion | `data/training/classification/training_set.json`, `notebooks/04_prepare_bert_dataset.py` |

### Baseline Numbers (to beat with fine-tuned BERT)

| Classifier | Micro F1 | Macro F1 | Exact Match | Latency |
|-----------|----------|----------|-------------|---------|
| Zero-Shot (Gemini) | 80.7% | 81.9% | 40.6% | 23.2s |
| Few-Shot (Gemini) | **84.0%** | **85.7%** | **53.1%** | 30.5s |
| Fine-tuned BERT | TBD | TBD | TBD | <0.1s |

## Architecture

The system is a **three-layer ML architecture** orchestrated as a LangGraph multi-agent pipeline:

- **Layer 1 — Fine-tuned specialist models (this team trains these):**
  - `bert-base-uncased` fine-tuned as a **multi-label classifier** over 13 requirement categories (Month 1). Multi-label via `BCEWithLogitsLoss`, sigmoid + 0.5 threshold.
  - `Llama-3.1-8B-Instruct` fine-tuned via **QLoRA** as a requirement **extractor** (Month 2).
- **Layer 2 — RAG:** ChromaDB + `sentence-transformers` grounding agents in financial regulation documents (Month 2).
- **Layer 3 — General LLM API (Gemini):** used for extraction, quality checking, SDLC justification. Also serves as the **zero-shot/few-shot baseline** that fine-tuned models are benchmarked against.

### Pipeline (LangGraph `StateGraph`, one node per agent)

**Phase 1 — Problem Scenario → Requirements (OPERATIONAL):**
`Extraction Agent (Gemini) → Classification Agent (Gemini few-shot, swapping to BERT Day 9-11) → Quality Agent (Gemini) → Structured Requirements`

**Phase 2 — Requirements → SDLC Recommendation (Day 15-20):**
`Feature Extractor → Scoring Matrix → LLM Justification → Ranked SDLC`

### Requirement Model (13 fields)

Each extracted requirement has: `id`, `statement`, `category` (list), `source_stakeholder`, `business_justification`, `priority`, `dependencies`, `assumptions`, `acceptance_criteria`, `applicable_regulations`, `risk_level`, `confidence_score`, `approval_status`.

### The 13 requirement categories

Business, Stakeholder, Functional, Security, Privacy, Regulatory, Performance, Availability, Usability, Data-management, Integration, Audit/Reporting, Operational — every requirement can carry multiple labels.

### Project structure

```
src/agents/       # One module per pipeline agent (extraction.py, classification.py, quality.py)
src/models/       # Pydantic schemas (requirement.py, quality.py) and ML model wrappers
src/utils/        # LLM gateway (llm.py → llm_gemini.py/llm_openai.py) with round-robin key rotation
src/rag/          # Document loader, chunker, embedder, ChromaDB — Month 2
src/training/     # Fine-tuning scripts/configs — Month 2
src/generators/   # Document generation (SRS) — Month 3
app/              # Web application layer — Month 3
data/training/    # Hand-labelled training datasets (343 requirements, 13 categories)
models/           # Saved checkpoints; large weight files are gitignored
notebooks/        # Sequential scripts (01_hf_basics → 04_prepare_bert_dataset)
evaluation/       # Baseline results (day4_extraction, day5_classification)
tests/
```

## Environment & Configuration

- **Python 3.12**, dependencies in `requirements.txt`. Virtualenv lives in `venv/` (gitignored).
- Secrets in `.env` (gitignored); `.env.example` is the template.
  - `GEMINI_API_KEY` — supports **comma-separated multiple keys** for round-robin rotation (bypasses free-tier rate limits). Keys from different GCP projects.
  - `OPENAI_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN` — not yet active.
- **Gemini model:** `gemini-3.6-flash` via `google-genai` SDK (structured output with `response_schema`).
- **LLM gateway** (`src/utils/llm.py`): Auto-selects provider based on available `.env` keys. Currently only Gemini is active.

## Working Conventions

- **Day-wise git commits** — each day's work is committed with `"Day N: description"` message.
- **Baseline-first approach** — every fine-tuned model must be benchmarked against zero-shot and few-shot LLM baselines, with results saved to `evaluation/results/`.
- **Ground truth is sacred** — `ground_truth.json` (32 verified requirements) is the held-out eval set. NEVER included in training data.
- **Training data quality** — synthetic examples are manually reviewed before use. Don't treat LLM-generated labels as ground truth.
- Prompts are currently inline in agent files. Once `src/utils/prompts.py` exists, externalize them there.

## Gitignored Items

- `venv/`, `.env`, `data/training/classification/bert_dataset/` (generated HF Dataset)
- Model weights: `*.safetensors`, `*.bin`, `*.pt`, `*.pth`
- `/implementation-plans`, `/learning-pytorch`
