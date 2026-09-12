# Demo-day runbook

Day 12 deliverable. The system has two tiers and this document exists so that
switching between them on demo day is a decision already made, not a decision
made under pressure.

| Tier | Model | Served from | When |
|---|---|---|---|
| **Cloud** (primary) | `llama3.1:8b` | Colab T4 + Ollama + Cloudflare tunnel | the demo |
| **Local** (fallback) | `llama3.2:3b` | this laptop, Ollama systemd service | dev loop, and when the cloud dies |

Classification is **always** local — the fine-tuned RoBERTa runs on the laptop
at 21 ms/requirement in both tiers. Only extraction and quality analysis move.

---

## T-60 min — bring the cloud tier up

Do this **before** the audience is in the room. The model pull alone is 5–10
minutes and it is not something to do with people watching.

1. Open `notebooks/cloud_llm_server.ipynb` at
   [colab.research.google.com](https://colab.research.google.com)
   (`File → Upload notebook`).
2. `Runtime → Change runtime type → T4 GPU → Save`. **Before running any
   cell** — switching afterwards restarts the runtime and discards everything.
3. Run cells 1–6 in order. Expected: `Tesla T4`, `Ollama is up`, the pull, then
   a `CLOUD ENDPOINT: https://....trycloudflare.com` line, then a smoke-test
   tok/s figure.
4. Copy that URL. It is regenerated on every restart — never hard-code it.

If cell 1 shows no GPU, step 2 did not take. If cell 5 raises
`No tunnel URL`, uncomment the ngrok cell below it (needs a free authtoken
from dashboard.ngrok.com).

## T-30 min — point the laptop at it and verify

```bash
export OLLAMA_HOST=https://<the-url>.trycloudflare.com
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_NUM_CTX=8192          # the T4 has VRAM to spare; the laptop does not

curl $OLLAMA_HOST/api/tags          # must return JSON naming llama3.1:8b
```

If that curl returns JSON, the tunnel works. If it hangs or 502s the problem is
the tunnel, not the laptop — re-run cell 5 for a fresh URL.

Then run the pipeline once, end to end, on the scenario you plan to show:

```bash
python -m src.graph
```

The header prints which tier served the run. **Read it.** A demo that silently
ran on the local 3B while you narrated the 8B is the failure this line prevents.

## T-10 min — keep the session alive

Run cell 7 in the notebook and leave the tab open. Free Colab reclaims an idle
runtime after roughly 90 minutes; the cell pings every 5 minutes so the session
counts as active. It is an infinite loop — do not run it before you are done
with the other cells.

---

## If the cloud tier dies mid-demo

In the laptop terminal:

```bash
unset OLLAMA_HOST
unset OLLAMA_MODEL
unset OLLAMA_NUM_CTX
```

That is the whole fallback. The next call goes to the local `llama3.2:3b` at
`http://localhost:11434` and the pipeline keeps working — slower, and with
noticeably plainer requirement statements, but working.

This is worth saying out loud rather than hiding: graceful degradation across
two tiers behind one wrapper is an engineering result, not an accident. The
reason it works is that `src/utils/llm_ollama.py` reads `OLLAMA_HOST` at call
time and there is no second code path to drift.

**Rehearse this once.** Unset the variables mid-run, watch the tier line
change, and confirm output still appears. A fallback that has never been
exercised is a hypothesis.

## Preconditions on the laptop

```bash
systemctl is-active ollama          # expect: active
ollama list                         # expect: llama3.2:3b present
ls models/roberta_classifier        # expect: weights + threshold.json
```

If `models/roberta_classifier` is missing, classification silently falls back
to the few-shot LLM baseline — correct output, but ~30 s per requirement and
not the thing being demonstrated. Retrain with:

```bash
python -m notebooks.05_train_bert --model roberta-base --epochs 45 --patience 8
```

## Known timing

Measured 11 Sep, local tier, `llama3.2:3b` at `num_ctx` 4096:

| Scenario | Words in | Requirements out | Wall clock |
|---|---|---|---|
| loan_processing | 52 | 5 | ~2–3 min |
| kyc_onboarding | 118 | 7 | **614 s (10 min 14 s)** |

Ten minutes is not a demo. This is the reason the cloud tier is the primary
one rather than a nice-to-have — and the reason `num_ctx` is pinned at 4096
locally: at 8192 the model grows 2.3 → 3.08 GB, spills 0.78 GB into system RAM,
and the KYC scenario did not finish extraction at all in 6.5 minutes.

Cloud-tier timings live in `evaluation/results/day12_local_llm_benchmark.json`
under the `cloud` key, alongside the local numbers for direct comparison.

**Pick the short scenario for the live run** and show the long one from saved
output. Generation speed is not what is being assessed.
