# Project bottlenecks — register

Written Day 13 (13 Sep 2026). A working list, not a post-mortem: cross items
off as they are resolved. Ordered by *how much damage they do if ignored*, not
by how hard they are to fix.

---

## Tier 1 — these threaten the validity of results already reported

### B1. `ground_truth.json` has never been verified by a human

The file says so itself:

```
status:  "UNVERIFIED_CANDIDATE_SET"
purpose: "... NOT verified truth yet -- every entry needs a human pass before use."
```

31 of its 32 entries carry `verified: false`. Only `REQ-001` is marked checked.
27 entries are sourced `synthetic-needs-review`, and the reviewer notes on
individual rows openly ask for confirmation — *"Added Performance — the '3
seconds' target was not in the original statement, I inferred it. Adjust or
remove if not accurate."*

**Every classification number this project has reported is measured against
this file**: zero-shot 80.7%, few-shot 84.0%, BERT 78.0%, RoBERTa 80.2%. So is
the conclusion that the fine-tuned model fell 3.8 points short of the few-shot
baseline, and the Day 9-11 story that a targeted data fix beat model tuning.

CLAUDE.md describes it as "32 **verified** requirements" and "sacred". The first
word is wrong, and it has been wrong since Day 5.

This is not a claim that the numbers are *false* — the labels may well be
mostly right. It is that **nobody knows**, and every downstream conclusion
inherits that uncertainty. It is also the cheapest fix on this page: one careful
human pass over 32 requirements, a few hours.

**Fix:** review all 32, flip `verified` to true, re-run
`notebooks/06_evaluate_bert.py --compare`, and report whether the numbers moved.
Whatever happens, the result is worth more than any model change queued behind
it. Do this before Day 14.

### B2. All 445 training rows are `verified: false` too

Same defect, lower stakes — training tolerates label noise far better than
evaluation does. But 65 of these rows plus 13 label corrections were added
during Days 9-11 specifically to fix a labelling divergence, and none of that
work has been checked by a person.

**Fix:** sample-audit ~50 rows rather than all 445. If the error rate is low,
note it and move on. If it is high, that changes the plan.

### B3. The eval set is too small for the claims being made on it

32 requirements. One percentage point of micro-F1 is under one label decision.
The headline finding — "fine-tuned RoBERTa is 3.8 points behind few-shot
Gemini" — is roughly 3 label decisions. That is not nothing, but it is not a
robust ranking either, and CLAUDE.md already warns that 1-point differences are
noise.

**Fix:** after B1, grow the held-out set toward ~100. Expensive, because it is
hand labour, and it competes directly with the extraction dataset for the same
hours.

### B4. Labelling conventions are inferred, not written down

The Regulatory and identity-data rules in `label_corrections.json` were derived
by reading what `ground_truth.json` does, because no document states them. Two
known contradictions remain unresolved: REQ-008 vs REQ-029 on encryption, and
REQ-020 vs REQ-028 on audit logging.

Every new row labelled before these are settled risks being labelled to a rule
that later changes.

**Fix:** write the conventions down as part of the B1 pass — they are the same
piece of work, and the pass is what surfaces the rules anyway.

---

## Tier 2 — these block work that is already scheduled

### B5. `HF_TOKEN` is a placeholder and Llama-3.1-8B is gated

Meta's Llama repositories require an approved access request, which can take
hours. Day 14 QLoRA-fine-tunes Llama-3.1-8B and cannot begin without it.

This stayed invisible because Ollama pulls its own weights and never touches the
Hub, so nothing has failed yet.

**Fix:** 15 minutes of clicking, today, regardless of what else happens.
Lead time is the whole problem.

### B6. The extraction dataset does not exist

~100-150 verified instruction/output pairs gate Days 13-14 (extraction QLoRA)
and, by extension, Days 15-16. CLAUDE.md is explicit: *"the dataset is the
gate."*

### B7. Phase 2 has no eval set at all

The SDLC recommender has no ground truth — nothing analogous to
`ground_truth.json`, unverified or otherwise. The only stated success criteria
are the plan's Definition of Done: a loan scenario should yield Agile or Spiral
rather than Waterfall, and a payment system should differ. That is 3 scenarios
and a sanity check, not a metric.

Compounding it: **the input shape is still undecided** — are the framework
questions inferred from the requirements, supplied by the user, or asked
interactively? Authoring SDLC training pairs before deciding means authoring
them twice.

*Decided Day 12:* Phase 1 uses only the Requirements and Project Type & Risk
tables, not all 4. That materially shrinks the problem.

### B8. The quality agent has never been checked for correctness

It runs 10 checks and flagged 5 of 7 requirements on the KYC scenario. Nobody
has read those 5 flags to see whether they are right.

This is the exact trap that `in_taxonomy_rate` set on Day 12 — a metric that
read 100% while the model sprayed 5.5 of 13 categories at everything. Automated
checks confirm format, not sense. CLAUDE.md already records the lesson;
the quality agent has not yet had it applied.

**Fix:** read one report properly. An hour.

---

## Tier 3 — engineering and infrastructure

### B9. Cold start dominates every run (diagnosed, not yet fixed)

Measured on KYC, Day 12:

| Stage | Cold | Warm | The cold time was |
|---|---|---|---|
| extraction | 114.3s | ~35s | 78.5s Ollama loading the model |
| classification | 406.3s | **0.2s** | 234s loading RoBERTa, 42s `import torch` |
| quality | 90.7s | ~19s | 71.5s Ollama **re**loading the model |
| **total** | **611s** | **~55s** | |

Inference is 30 ms/requirement, exactly as documented. All of it is reading
files off `/mnt/f` — NTFS through WSL.

There is a cascade: classification outlasts Ollama's 5-minute keep-alive, so the
model is evicted *during* classification and reloaded for the quality stage.
Fixing the classifier load removes that second load for free.

**Fix:** `OLLAMA_KEEP_ALIVE=-1`; move the venv off `/mnt/f` to ext4; warm both
models at process start. Detail in `docs/day13_plan.md`.

### B10. 4 GB VRAM is a permanent ceiling

RTX 3050 4 GB, 7 GB system RAM (~5 GB free). It has already forced: a 3B local
model instead of 8B, `num_ctx` pinned at 4096 (8192 spills 0.78 GB to RAM and
the KYC scenario stopped finishing), and QLoRA training pushed to Colab.

Not fixable, only designed around. It is worth stating plainly in the report —
the constraint drove real architecture decisions and that is a legitimate
engineering narrative.

### B11. The cloud tier is fragile in four separate ways

- The tunnel URL changes on every restart — configuration, never hard-coded
- Free Colab: ~90 min idle timeout, ~12 h session cap, nothing persists
- ~10-15 min of setup before any demo
- **Cold-load 524:** the KYC scenario failed both cloud runs. Streaming (added
  Day 12) keeps the connection alive *during generation* — measured 76 chunks,
  max gap 0.03 s — but cannot cover the ~70 s of silence while the model loads.
  Warming the model first is the likely fix. **Untested.**

### B12. There are no tests

`tests/` contains `__init__.py` and nothing else. For a software-engineering
project this is both a technical risk and an awkward thing to present.

The Day 9-11 history argues the case: a hardcoded `BertModel` made RoBERTa load
a random encoder and predict garbage at 33.8% with no error raised, and two
concurrent training runs silently corrupted each other's checkpoints. Both are
exactly what a small test suite catches.

### B13. Prompts are inline in agent files

`src/utils/prompts.py` does not exist. CLAUDE.md already flags this. Minor, but
it grows more expensive once the same prompts have to be versioned against
fine-tuned adapters.

### B14. Several conclusions rest on one or two runs

The benchmark defaults to 2 repeats; some findings come from a single
observation. The `num_ctx` 8192 result — "did not finish in 6.5 min" — is n=1.
Cheap to strengthen where it matters, and worth doing before any of it goes in
the report.

---

## Tier 4 — delivery risk

### B15. Fine-tuned models do not beat the prompted baseline

Day 11's Definition of Done required it; RoBERTa's 80.2% against few-shot
Gemini's 84.0% does not meet it. The honest framing is the latency trade —
~1,450x faster, which is what makes per-requirement classification viable at
all. But B1 means even this comparison is measured against unverified labels.

### B16. There is no user interface

Deferred by decision, and the supervisor has since emphasised an agentic
application. `app/streamlit_app.py` exists but is parked and untracked. The
agreed interim is terminal output with good debug messages.

### B17. Days 13-16 are carrying too much

Two QLoRA fine-tunes (local 3B and cloud 8B), the SDLC engine, and two training
datasets — compressed into four days, on top of a dataset that does not exist
yet and a token that has not been requested. This is the schedule risk, and B1
is about to take hours out of it.

---

## Suggested order

1. **B5** — request Llama access now, it has lead time and costs 15 minutes
2. **B1 + B4** — verify the 32 ground-truth rows and write the conventions down
   while doing it; re-run the evaluation and report whether anything moved
3. **B9** — cold start; makes every later iteration cheaper, so it pays back
4. **B8** — read one quality report; an hour
5. **B6** — extraction dataset, the bulk of Days 13-14
6. **B11** — verify the 524 theory when Colab is next up
7. **B7** — decide the SDLC input shape before authoring any SDLC pairs

B2, B3, B12, B13, B14 are real but can wait. B10 is permanent. B15 and B16 are
reporting and demo decisions rather than engineering ones.
