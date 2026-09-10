# bert multi-label requirement classifier

Fine-tuned `bert-base-uncased` over the 13-category requirement taxonomy. Multi-label:
each category is an independent binary decision, trained with
`BCEWithLogitsLoss`, decoded with sigmoid + a calibrated threshold.

Class: `src/models/bert_classifier.py`. Loaded through
`src.models.multilabel.MultiLabelClassifier`, which dispatches on
`config.model_type`.

## Reproduce

```bash
python -m notebooks.04_prepare_bert_dataset     # rebuild the HF dataset
python -m notebooks.05_train_bert --epochs 45 --patience 8
python -m notebooks.06_evaluate_bert --model-dir models/bert_classifier
```

The weights are not in git (too large) and neither are `config.json`,
`tokenizer.json` or `threshold.json` -- training regenerates all of them, and
tracking them meant a branch switch could pair one run's config with another
run's weights, which loads silently and predicts garbage. This card and
`evaluation/results/day9_bert_*.json` are the durable record.

## Training

| | |
|---|---|
| base model | `bert-base-uncased` |
| examples | 304 train / 76 validation |
| epochs | 45 requested, 43 run (early stopping) |
| learning rate | 2e-05 |
| batch size | 16 |
| max length | 128 |
| classifier dropout | 0.2 |
| pos_weight | False |
| decision threshold | 0.3 (calibrated on validation) |
| training time | 15.5 min |

## Results on held-out ground truth (32 requirements)

`data/training/classification/ground_truth.json` has been held out since Day 5
and never appears in training.

| classifier | micro-F1 | macro-F1 | exact match | latency/req |
|---|---|---|---|---|
| zero-shot (Gemini) | 80.7% | 81.9% | 40.6% | 23.20 s |
| few-shot (Gemini) | **84.0%** | **85.7%** | **53.1%** | 30.55 s |
| this model | 78.0% | 78.5% | 34.4% | 19.1 ms |

Batched: 3.4 ms/req at batch 32.

Micro-F1 is **-6.0%** against the few-shot baseline at
roughly 1601x lower latency. The Definition of Done asked for
>84.0% and <100 ms; latency passes, F1 does not.

### Per category (worst first)

| category | support | precision | recall | F1 |
|---|---|---|---|---|
| Privacy | 6 | 0.60 | 0.50 | 0.55 |
| Business | 6 | 0.75 | 0.50 | 0.60 |
| Regulatory | 11 | 0.62 | 0.73 | 0.67 |
| Data-management | 6 | 0.80 | 0.67 | 0.73 |
| Usability | 4 | 0.75 | 0.75 | 0.75 |
| Security | 6 | 0.71 | 0.83 | 0.77 |
| Stakeholder | 5 | 0.80 | 0.80 | 0.80 |
| Integration | 5 | 0.80 | 0.80 | 0.80 |
| Availability | 3 | 0.75 | 1.00 | 0.86 |
| Functional | 14 | 0.81 | 0.93 | 0.87 |
| Performance | 5 | 0.83 | 1.00 | 0.91 |
| Operational | 5 | 0.83 | 1.00 | 0.91 |
| Audit/Reporting | 4 | 1.00 | 1.00 | 1.00 |

## Caveats

- The training corpus is synthetic, hand-authored, plus requirements derived
  from the RBI KYC Master Direction. Every row is `verified: false` -- nobody
  has done a human label pass yet.
- Category conventions were **inferred** from `ground_truth.json` rather than
  documented independently; see `data/training/classification/label_corrections.json`.
  If that reading is wrong, this model has learned the error confidently.
- 32 requirements is a small eval set. A 1-point difference is well under half
  a label decision, so treat small gaps between runs as noise.
