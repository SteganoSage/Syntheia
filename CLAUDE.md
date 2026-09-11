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

**Day 12 is next:** install Ollama on WSL2, benchmark the candidate models that fit VRAM, write `src/utils/llm_ollama.py` with the same interface as `llm_gemini.py`, and pin the winner here and in the plan.

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
| **12** | **Local tier:** Ollama on WSL2 (models on `/mnt/f/ollama-models`). Benchmark candidates that fit VRAM (`phi3.5` ~2.2 GB, `qwen2.5:3b` ~1.9 GB, `llama3.2:3b` ~2.0 GB) on tokens/sec, peak VRAM, valid-JSON rate, extraction quality; record oversized rejects too. Wrapper with configurable `OLLAMA_HOST`. **Cloud tier:** Colab/Kaggle notebook running Ollama + `llama3.1:8b` behind a tunnel; same wrapper, same benchmark. Demo-day runbook, rehearsed. | `src/utils/llm_ollama.py`, `notebooks/cloud_llm_server.ipynb`, `evaluation/results/day12_local_llm_benchmark.json` |
| **13–14** | Extraction dataset (~100–150 instruction/output pairs, Gemini as a *data-generation tool* then manually verified). QLoRA fine-tune **both** the local 3–4B and Llama-3.1-8B — same script parameterised by base model, same dataset, two adapters. Report the 8B-vs-local gap either way. | `notebooks/07_finetune_extraction.py`, `data/training/extraction/`, `models/local_extractor/`, `models/cloud_extractor/` |
| **15–16** | SDLC recommendation engine + QLoRA on the SDLC task. Separate adapters are easier to evaluate and swap than multi-task. ~100–150 examples covering all 7 models. | `src/models/sdlc.py`, `src/agents/sdlc.py`, `data/training/sdlc/`, `models/local_sdlc/` |
| **17–18** | End-to-end pipeline with **no API calls**: local extraction → RoBERTa classification → local SDLC. CLI entry point. **Verify by disabling networking and re-running.** | `main.py`, `src/graph.py` |
| **19–20** | Comparison: Gemini (prompted) vs local (prompted) vs local (fine-tuned), on 3+ scenarios, with latency. | comparison notebook |
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
  - A **3–4B local model** fine-tuned via **QLoRA** as a requirement **extractor** and **SDLC recommender** (Month 1, Days 13–16). The specific model is chosen on Day 12 by benchmark, not assumed — candidates must fit ~3GB of real headroom. Train on Colab's free T4 if 4GB is too tight, then deploy the adapter locally.
- **Layer 2 — RAG:** ChromaDB + `sentence-transformers` grounding agents in financial regulation documents (Month 2).
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
                  #   llm_ollama.py -- Day 12. Same interface as llm_gemini, with a
                  #                    configurable base URL so ONE wrapper drives both
                  #                    the local model and the cloud endpoint
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
- **Local LLM: `llama3.2:3b`**, pinned Day 12 by benchmark (`evaluation/results/day12_local_llm_benchmark.json`). 36 tok/s, 2.3GB VRAM, 100% valid JSON, 1.8 categories/requirement.
  - **`qwen2.5:3b` was faster (47 tok/s) and rejected anyway.** It scored 100% on every automated metric while assigning 5.5 of the 13 categories to every requirement — every label legal, the output useless. `phi3.5` was slowest at 14 tok/s and the only one to invent an out-of-taxonomy label. The speed ranking was the inverse of the quality ranking.
  - Set `OLLAMA_MODEL` to override.
- **Ollama is installed** (11 Sep) as a systemd service, `active` and `enabled`, with models on **`/mnt/f/ollama-models`** via a drop-in (`sudo systemctl edit ollama` → `Environment="OLLAMA_MODELS=/mnt/f/ollama-models"`). Not the default `~/.ollama`: that lives on WSL's ext4 vhdx whose `df` "937G free" is the virtual disk's *maximum*, not real space — the vhdx sits on Windows `C:` which has only **45GB free**, while `F:` has 72GB. NTFS slowness does not bite here, since weights are read once at load and then cached.
- **Cloud LLM:** Llama-3.1-8B via Ollama on a Colab/Kaggle T4 behind an ngrok/Cloudflare tunnel. Reached through the same wrapper by setting `OLLAMA_HOST`.
- **LLM gateway** (`src/utils/llm.py`): Auto-selects provider based on available `.env` keys. `llm_ollama.py` joins it at Day 12 and should become the default, with Gemini reachable explicitly for comparison runs.
- **Hardware:** RTX 3050 **4GB VRAM**, **7GB system RAM** (~5GB free), WSL2. This is the binding constraint on every model choice — BERT/RoBERTa fine-tune comfortably, but anything above ~3GB at 4-bit cannot be served. QLoRA may need Colab's free T4 for training, with the adapter deployed locally. Writes to `/mnt/f` are slow (NTFS through WSL): a 440 MB checkpoint save costs ~6 s.

## Working Conventions

- **Day-wise git commits** — each day's work is committed with `"Day N: description"` message.
- **Baseline-first approach** — every fine-tuned model must be benchmarked against zero-shot and few-shot LLM baselines, with results saved to `evaluation/results/`. After the Sep 11 revision the comparison ladder is: Gemini (prompted) vs local (prompted) vs local (fine-tuned).
- **Model choices are made by benchmark, not assumption.** Day 12 measured candidates on this hardware before pinning one, and the result inverted the expected ranking. Record the rejects and why — that reasoning is the Week 11 E1–E7 ablation story.
- **A metric that cannot fail is not a metric.** `in_taxonomy_rate` read 100% for a model spraying 5.5 of 13 categories at every requirement: it validated label *names* and never label *counts*. Both the classifier work (Days 9–11) and the extractor work hit this — automated checks confirm format, not sense. **Print the actual output and read it before pinning anything.**
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
