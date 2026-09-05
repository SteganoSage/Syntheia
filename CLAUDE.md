# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Syntheia is an agentic AI system for **automated requirement gathering, analysis, and SDLC methodology identification in financial services** (loan origination, KYC/onboarding, payments). It takes a natural-language problem scenario and produces structured, classified, quality-checked software requirements (Phase 1), then recommends a ranked SDLC methodology given project conditions (Phase 2).

The project follows a day-by-day 4-month implementation roadmap at `implementation-plans/implementation_plan3.md` (the current/superseding plan — `implementation_plan.md` and `implementation_plan2.md` are earlier iterations). **This directory is gitignored** (`/implementation-plans` in `.gitignore`) — it's a local planning reference only, not part of the shipped repo. Consult it for the current day's goals, deliverables, and Definition-of-Done before starting work; update task status by moving to the next day's section rather than rewriting history.

The project is currently in **Month 1 (Core Engine + Fine-Tuned Classifier)**, early Week 1 — foundational setup and Hugging Face basics are in place (`notebooks/01_hf_basics.py`); LangGraph + LLM API integration is next.

## Architecture

The system is a **three-layer ML architecture** orchestrated as a LangGraph multi-agent pipeline:

- **Layer 1 — Fine-tuned specialist models (this team trains these):**
  - `bert-base-uncased`/`roberta-base` fine-tuned as a **multi-label classifier** over 13 requirement categories (Month 1). Multi-label via `problem_type="multi_label_classification"`, `BCEWithLogitsLoss`, sigmoid + 0.5 threshold.
  - `Llama-3.1-8B-Instruct` (or Mistral-7B) fine-tuned via **QLoRA** (`peft` + `trl` SFTTrainer) as a requirement **extractor**, raw text → structured JSON (Month 2).
- **Layer 2 — RAG:** ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) grounding agents in financial regulation documents (Month 2).
- **Layer 3 — General LLM API (GPT-4o-mini / Gemini):** used for tasks needing broad reasoning rather than task-specific training — quality checking, compliance mapping, SDLC justification, interview/conversation. Also serves as the **zero-shot/few-shot baseline** that fine-tuned models are benchmarked against throughout the project.

Fine-tuned models are not a replacement for the LLM API layer — both coexist. Month 1 uses the LLM API for extraction (baseline) while classification moves to the fine-tuned BERT; Month 2 swaps extraction to the fine-tuned Llama and adds RAG.

### Pipeline (LangGraph `StateGraph`, one node per agent)

**Phase 1 — Problem Scenario → Requirements:**
`Extraction Agent (LLM API, later fine-tuned Llama) → Classification Agent (fine-tuned BERT) → Quality Agent (LLM API) → Structured Requirements`

**Phase 2 — Requirements + Conditions → SDLC Recommendation:**
`Feature Extractor (derives project characteristics via LLM) → Scoring Matrix (deterministic, weighted) → LLM Justification → Ranked SDLC`

Later months add Interview, Conflict Detection, and Compliance agents ahead of Extraction, and route flagged requirements conditionally through the graph.

### The 13 requirement categories

Business, Stakeholder, Functional, Security, Privacy, Regulatory, Performance, Availability, Usability, Data-management, Integration, Audit/Reporting, Operational — every requirement can carry multiple labels.

### Project structure

```
src/agents/       # One module per pipeline agent (extraction.py, classification.py, quality.py, sdlc.py, ...)
src/models/       # Pydantic schemas (e.g. Requirement) and ML model wrappers
src/utils/        # LLM API wrapper (llm.py), externalized prompts (prompts.py)
src/rag/          # Document loader, chunker, embedder, ChromaDB vector store
src/training/     # Fine-tuning scripts/configs (BERT, QLoRA/Llama)
src/generators/   # Document generation (SRS/Jinja2, python-docx) — Month 3
app/              # Web application layer (Streamlit → FastAPI/Next.js) — Month 3
data/training/    # Hand-labelled + reviewed training datasets (classification/, extraction/)
models/           # Saved checkpoints; large weight files (*.safetensors, *.bin, *.pt) are gitignored, keep configs/model_cards in git
notebooks/        # Numbered, sequential exploration notebooks (01_hf_basics, 02_bert_finetuning, ...)
evaluation/       # Model comparison / evaluation scripts
tests/
```

Most of these directories currently contain only `__init__.py` — the codebase is being built out day-by-day per the implementation plan.

## Environment & Configuration

- Python 3.12, dependencies in `requirements.txt` (spans the full 4-month stack: `torch`/`transformers`/`peft`/`trl`/`bitsandbytes` for fine-tuning, `langchain`/`langgraph`/`openai` for agents, `chromadb` for RAG, `streamlit`/`fastapi` for the app layer, `wandb` for experiment tracking). Virtualenv lives in `venv/` (gitignored).
- Secrets live in `.env` (gitignored); `.env.example` is the template. Keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `WANDB_API_KEY`/`WANDB_PROJECT`, `HF_TOKEN`, `APP_ENV`, `CHROMA_PERSIST_DIRECTORY`.
- No build/lint/test tooling is configured yet (no `pyproject.toml`, `pytest.ini`, or `Makefile`) — `pytest` is a dependency but there's no test suite or CLI entry point yet. As these are added, prefer updating this file with the actual commands rather than guessing at conventions.

## Working Conventions

- Model comparisons are a running theme: every fine-tuned model is expected to be benchmarked against a zero-shot and few-shot LLM-API baseline, with results tracked in Weights & Biases and surfaced in comparison notebooks (`notebooks/03_model_comparison.ipynb`, etc.). When adding a new model/agent, keep this comparison path in mind rather than replacing baselines outright.
- Training data quality is a stated priority in the plan: synthetic examples (GPT-4o generated) must be manually reviewed before use — don't treat generated labels as ground truth.
- Prompts belong in `src/utils/prompts.py` (externalized), not inline in agent code, once that file exists.
