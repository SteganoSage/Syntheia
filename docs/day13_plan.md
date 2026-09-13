# Day 13 — plan

Written end of Day 12. Two things changed what Day 13 should contain: the
cold-start measurement below, and the decision to narrow the SDLC framework
for Phase 1.

---

## 0. Do this first — it has lead time (15 min)

**Request access to Llama-3.1-8B on Hugging Face and set a real `HF_TOKEN`.**

`HF_TOKEN` in `.env` is still a 27-character placeholder. Meta's Llama
repositories are *gated*: you submit a form and wait for approval, which can
take hours. Day 14 QLoRA-fine-tunes Llama-3.1-8B and **cannot start without
it**. Ollama pulls its own weights and never needed the Hub, which is why this
has stayed invisible until now.

1. huggingface.co/meta-llama/Llama-3.1-8B → "Request access"
2. huggingface.co/settings/tokens → new read token
3. Put it in `.env` as `HF_TOKEN=hf_...`

Start the clock on this before anything else on this list.

---

## 1. Fix the cold start (1-2 h)

Measured on the KYC scenario, Day 12:

| Stage | Cold | Warm | What the cold time actually was |
|---|---|---|---|
| extraction | 114.3s | ~35s | 78.5s Ollama loading the model |
| classification | 406.3s | **0.2s** | 234s loading RoBERTa + 42s `import torch` |
| quality | 90.7s | ~19s | 71.5s Ollama **re**loading the model |
| **total** | **611s** | **~55s** | |

Inference is 30 ms/requirement, exactly as documented. Every one of those
minutes is reading files off `/mnt/f` — NTFS through WSL.

There is also a cascade: classification took longer than Ollama's 5-minute
default keep-alive, so Ollama evicted `llama3.2:3b` *during* classification and
paid a second 71.5s load for the quality stage. Fixing the classifier load
removes that second load for free.

Three fixes, cheapest first:

- **`OLLAMA_KEEP_ALIVE=-1`** (or `30m`) so the model is never evicted mid-run.
  One environment variable in the systemd drop-in.
- **Move the venv off `/mnt/f`.** `import torch` at 42s is the venv being read
  over the NTFS bridge; on ext4 it is a few seconds. `models/` is the other
  candidate but it is large, so measure before moving.
- **Warm on startup.** Whatever process serves the demo should load the
  classifier and ping Ollama once at boot, so the first user-visible request is
  already warm. This is what makes a long-lived process (a UI, a CLI that
  stays open) worth more than repeated `python -m src.graph` invocations.

**Definition of done:** second consecutive KYC run completes in under 90s, and
the number is recorded. Do not claim the fix without a warm *and* a cold
measurement.

## 2. Verify the cloud 524 (30 min, needs Colab up)

Day 12 left this as a hypothesis, not a finding.

Streaming is now on by default (`OLLAMA_STREAM=0` restores the old blocking
path). Measured locally: 76 chunks with a maximum 0.03s gap once generating —
so a tunnel has nothing to time out on *during generation*. But the first chunk
arrived at 70.6s because the model was loading, and **streaming cannot cover a
cold load**. KYC was the first scenario the cloud benchmark ran, so it almost
certainly hit a cold 4.9 GB `llama3.1:8b`.

Test, in this order:
1. Restart `notebooks/cloud_llm_server.ipynb`, run cells 1-6 — **including the
   smoke test**, which is what warms the model.
2. `OLLAMA_HOST=<url> python -m notebooks.08_benchmark_local_llm --tier cloud --models llama3.1:8b`
3. If KYC now passes: the cause was cold-load silence. Record it and add
   "always run the smoke-test cell" to the runbook as a *required* step.
4. If it still 524s: the cause is elsewhere — try ngrok, and split extraction
   into smaller calls.

## 3. Extraction dataset — the day's real work (4-6 h)

This is the Day 13-14 deliverable and the gate on everything after it.
~100-150 instruction/output pairs in `data/training/extraction/`.

Per the working conventions: **hand-authored, or Gemini-generated and then
manually verified.** Not generated and trusted. The Day 9-11 lesson was that a
targeted data fix beat three rounds of model tuning, and it only worked because
the divergence was diagnosed first.

Shape of a pair: problem scenario in, the 13-field requirement JSON out. Reuse
the extraction system prompt so the training distribution matches serving.

Two things to settle before authoring at volume:
- **How many requirements per scenario?** Ground truth averages 2.5 categories
  per requirement; the local model produces 1.8 and the cloud 8B 1.5. Both
  under-label. The training pairs should reflect ground-truth convention, not
  what the current model does.
- **Scenario sources.** The 5 existing test cases are not enough and
  `ground_truth.json` is off limits — it is the held-out set.

## 4. Carried over from Day 12

- **Rehearse the demo-day fallback once.** `docs/demo_runbook.md` describes a
  procedure nobody has executed. Unset the vars mid-run, watch the tier line
  flip, confirm output keeps coming. 15 minutes.
- **Prompted SDLC baseline probe.** Deliberately small: 3 scenarios, 3 tiers
  (Gemini / local 3B / cloud 8B), print the ranked recommendations and
  justifications side by side and *read them*. No metrics file — there is no
  eval set yet, and a number out of 3 is not a baseline.

  **Phase 1 scope is now narrowed** (decided Day 12): only the *Requirements*
  and *Project Type & Risk* tables of the supervisor's 4-table, 19-question
  framework. Project Team and User Community are Phase 2. This makes the Day 15
  dataset materially smaller and removes the awkward problem of inventing team-size
  answers from a 52-word scenario.

  Still undecided, and it shapes the Day 15 dataset: are the framework
  questions **inferred** from the requirements, **supplied** by the user, or
  **asked** interactively? Decide before authoring SDLC training pairs, or they
  get authored twice.

- **65 rows + 13 label corrections are `verified: false`** and still need a
  human label pass.

## Not doing

- `qwen2.5:7b` oversized-reject benchmark. The VRAM cliff is already documented
  better by the `num_ctx` 8192 measurement — 2.3 GB → 3.08 GB with 0.78 GB
  spilling to RAM shows the cliff is about total footprint, not parameter count.
- Any UI work. Deferred by decision; terminal output with good debug messages
  is the agreed interim.

## Uncommitted at end of Day 12

- `src/utils/llm_ollama.py` — streaming support
- `docs/demo_runbook.md`, `docs/day13_plan.md`
- `app/streamlit_app.py` — parked, untracked
- cloud-tier results in `evaluation/results/day12_local_llm_benchmark.json`
