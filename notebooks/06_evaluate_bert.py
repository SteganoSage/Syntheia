"""Day 9-11 Task 4: score the fine-tuned BERT against the held-out ground truth.

    python -m notebooks.06_evaluate_bert
    python -m notebooks.06_evaluate_bert --threshold 0.5 --show-errors 10

This is the number that counts. `05_train_bert.py` reports validation micro-F1,
but the validation split moves whenever the corpus is rebuilt, so it is not
comparable across runs and not comparable to the Day 5 LLM baselines. The 32
requirements in ground_truth.json have been held out from training since Day 5
and are scored here against exactly the metrics the baselines were scored on.

Definition of Done for the day:
    - micro-F1 > 84.0%  (beats the Day 5 few-shot Gemini baseline)
    - inference latency < 100 ms per requirement

Latency is reported two ways because they answer different questions:
    per-requirement, sequential -- what the LangGraph pipeline actually pays,
                                   classifying one requirement at a time
    batched                     -- throughput when scoring a whole document
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dotenv import load_dotenv

# Loads HF_TOKEN so model downloads are authenticated. Without it every pull
# from the Hub is anonymous and subject to a much lower rate limit -- which is
# how `--model roberta-base` failed with a network error while cached models
# kept working.
load_dotenv()

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.models.multilabel import MultiLabelClassifier

DEFAULT_MODEL_DIR = REPO / "models/bert_classifier"
GT_PATH = REPO / "data/training/classification/ground_truth.json"
BASELINE_PATH = REPO / "evaluation/results/day5_classification_baseline.json"
RESULTS_DIR = REPO / "evaluation/results"


def out_path_for(model_type: str) -> Path:
    """One results file per architecture, so runs do not overwrite each other."""
    return RESULTS_DIR / f"day9_{model_type}_evaluation.json"

TARGET_MICRO_F1 = 0.8402   # Day 5 few-shot Gemini
TARGET_LATENCY_MS = 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help="a trained model directory, e.g. models/roberta_classifier")
    p.add_argument("--compare", action="store_true",
                   help="also table every other day9_*_evaluation.json on disk")
    p.add_argument("--threshold", type=float, default=None,
                   help="override the threshold saved by training")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--show-errors", type=int, default=8,
                   help="how many mismatched requirements to print")
    p.add_argument("--latency-runs", type=int, default=3,
                   help="passes over the eval set when timing sequential inference")
    p.add_argument("--cpu", action="store_true",
                   help="time on CPU instead of GPU (the honest number if you deploy CPU-only)")
    return p.parse_args()


# ── Metrics ────────────────────────────────────────────────────────────

def multilabel_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    kw = {"zero_division": 0}
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", **kw),
        "macro_f1": f1_score(y_true, y_pred, average="macro", **kw),
        "precision_micro": precision_score(y_true, y_pred, average="micro", **kw),
        "recall_micro": recall_score(y_true, y_pred, average="micro", **kw),
        "exact_match_rate": float((y_true == y_pred).all(axis=1).mean()),
    }


def per_category_report(y_true, y_pred, categories) -> list[dict]:
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    precs = precision_score(y_true, y_pred, average=None, zero_division=0)
    recs = recall_score(y_true, y_pred, average=None, zero_division=0)
    return [
        {
            "category": c,
            "support": int(y_true[:, i].sum()),
            "precision": round(float(precs[i]), 4),
            "recall": round(float(recs[i]), 4),
            "f1": round(float(f1s[i]), 4),
        }
        for i, c in enumerate(categories)
    ]


# ── Latency ────────────────────────────────────────────────────────────

def time_sequential(clf: MultiLabelClassifier, statements: list[str],
                    runs: int) -> dict[str, float]:
    """One requirement at a time -- what the pipeline actually pays per call."""
    clf.predict_one(statements[0])            # warm up kernels / autograd caches
    if clf.device.type == "cuda":
        torch.cuda.synchronize()

    timings_ms = []
    for _ in range(runs):
        for s in statements:
            start = time.perf_counter()
            clf.predict_one(s)
            if clf.device.type == "cuda":
                torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(timings_ms)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(arr.max()), 2),
        "samples": len(arr),
    }


def time_batched(clf: MultiLabelClassifier, statements: list[str],
                 batch_size: int) -> dict[str, float]:
    clf.predict(statements[:2], batch_size=batch_size)
    if clf.device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    clf.predict(statements, batch_size=batch_size)
    if clf.device.type == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000
    return {
        "total_ms": round(total_ms, 2),
        "per_requirement_ms": round(total_ms / len(statements), 2),
        "batch_size": batch_size,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not (args.model_dir / "config.json").exists():
        raise SystemExit(
            f"No trained model at {args.model_dir} -- "
            "run `python -m notebooks.05_train_bert` first."
        )

    gt = json.loads(GT_PATH.read_text())["requirements"]
    statements = [r["statement"] for r in gt]
    true_cats = [r["categories"] for r in gt]

    clf = MultiLabelClassifier(
        args.model_dir,
        device="cpu" if args.cpu else None,
        threshold=args.threshold,
    )
    categories = clf.categories

    model_type = clf.model.config.model_type
    out_path = out_path_for(model_type)

    print(f"Model      {args.model_dir.name}  [{model_type}]  ({clf.device})")
    print(f"Threshold  {clf.threshold}"
          f"{' (overridden)' if args.threshold is not None else ' (from training)'}")
    print(f"Eval set   {len(gt)} held-out requirements from {GT_PATH.name}\n")

    # ── Predict ────────────────────────────────────────────────────────
    pred_cats = clf.predict(statements, batch_size=args.batch_size)

    mlb = MultiLabelBinarizer(classes=categories)
    y_true = mlb.fit_transform(true_cats)
    y_pred = mlb.transform(pred_cats)
    assert list(mlb.classes_) == categories, mlb.classes_

    scores = multilabel_scores(y_true, y_pred)

    # ── Latency ────────────────────────────────────────────────────────
    seq = time_sequential(clf, statements, args.latency_runs)
    batched = time_batched(clf, statements, args.batch_size)

    # ── Baselines ──────────────────────────────────────────────────────
    baselines = {}
    if BASELINE_PATH.exists():
        baselines = json.loads(BASELINE_PATH.read_text())["baselines"]

    # ── Comparison table ───────────────────────────────────────────────
    print(f"{'=' * 74}")
    print("  CLASSIFIER COMPARISON  (32 held-out requirements)")
    print(f"{'=' * 74}")
    print(f"  {'Classifier':<22}{'Micro F1':>10}{'Macro F1':>10}"
          f"{'Exact Match':>13}{'Latency/req':>14}")
    print(f"  {'-' * 69}")

    for key, name in (("zero_shot", "Zero-Shot (Gemini)"), ("few_shot", "Few-Shot (Gemini)")):
        b = baselines.get(key)
        if not b:
            continue
        print(f"  {name:<22}{b['micro_f1']:>9.1%}{b['macro_f1']:>10.1%}"
              f"{b['exact_match_rate']:>13.1%}{b['avg_latency_seconds']:>12.2f}s")

    print(f"  {model_type.upper() + ' (fine-tuned)':<22}{scores['micro_f1']:>9.1%}{scores['macro_f1']:>10.1%}"
          f"{scores['exact_match_rate']:>13.1%}{seq['mean_ms']:>11.1f}ms")

    fs = baselines.get("few_shot")
    if fs:
        d_f1 = scores["micro_f1"] - fs["micro_f1"]
        speedup = (fs["avg_latency_seconds"] * 1000) / seq["mean_ms"]
        print(f"  {'-' * 69}")
        print(f"  {'vs few-shot':<22}{d_f1:>+9.1%}"
              f"{scores['macro_f1'] - fs['macro_f1']:>+10.1%}"
              f"{scores['exact_match_rate'] - fs['exact_match_rate']:>+13.1%}"
              f"{speedup:>11.0f}x faster")

    # ── Per-category ───────────────────────────────────────────────────
    cat_report = per_category_report(y_true, y_pred, categories)
    print(f"\n  {'category':<18}{'support':>8}{'prec':>8}{'recall':>8}{'f1':>8}")
    print(f"  {'-' * 50}")
    for row in sorted(cat_report, key=lambda r: r["f1"]):
        print(f"  {row['category']:<18}{row['support']:>8}"
              f"{row['precision']:>8.2f}{row['recall']:>8.2f}{row['f1']:>8.2f}")

    # ── Latency detail ─────────────────────────────────────────────────
    print(f"\n  Latency ({clf.device})")
    print(f"    sequential   mean {seq['mean_ms']:.1f} ms | median {seq['median_ms']:.1f} ms "
          f"| p95 {seq['p95_ms']:.1f} ms  (n={seq['samples']})")
    print(f"    batched      {batched['per_requirement_ms']:.1f} ms/req "
          f"at batch {batched['batch_size']}")

    # ── Error analysis ─────────────────────────────────────────────────
    mismatches = [
        {
            "id": gt[i]["id"],
            "statement": statements[i],
            "true": sorted(true_cats[i]),
            "predicted": sorted(pred_cats[i]),
            "missed": sorted(set(true_cats[i]) - set(pred_cats[i])),
            "spurious": sorted(set(pred_cats[i]) - set(true_cats[i])),
        }
        for i in range(len(gt))
        if set(true_cats[i]) != set(pred_cats[i])
    ]

    if args.show_errors and mismatches:
        print(f"\n{'=' * 74}")
        print(f"  MISCLASSIFIED  ({len(mismatches)} of {len(gt)} not an exact match)")
        print(f"{'=' * 74}")
        for m in mismatches[:args.show_errors]:
            print(f"\n  {m['id']}  {m['statement'][:76]}")
            if m["missed"]:
                print(f"      missed:   {', '.join(m['missed'])}")
            if m["spurious"]:
                print(f"      spurious: {', '.join(m['spurious'])}")

        missed = [c for m in mismatches for c in m["missed"]]
        spurious = [c for m in mismatches for c in m["spurious"]]
        if missed or spurious:
            from collections import Counter
            print(f"\n  Most missed:   {dict(Counter(missed).most_common(4))}")
            print(f"  Most spurious: {dict(Counter(spurious).most_common(4))}")

    # ── Save ───────────────────────────────────────────────────────────
    payload = {
        "model_type": model_type,
        "model_dir": str(args.model_dir.relative_to(REPO))
        if args.model_dir.is_relative_to(REPO) else str(args.model_dir),
        "ground_truth_file": str(GT_PATH.relative_to(REPO)),
        "total_requirements": len(gt),
        "threshold": clf.threshold,
        "device": str(clf.device),
        "bert": {k: round(v, 4) for k, v in scores.items()},
        "latency": {"sequential": seq, "batched": batched},
        "baselines": baselines,
        "per_category": cat_report,
        "misclassified": mismatches,
        "per_requirement": [
            {
                "id": gt[i]["id"],
                "statement": statements[i],
                "true_categories": true_cats[i],
                "predicted_categories": pred_cats[i],
                "exact_match": set(true_cats[i]) == set(pred_cats[i]),
            }
            for i in range(len(gt))
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  Saved to {out_path.relative_to(REPO)}")

    # ── Cross-model comparison ─────────────────────────────────────────
    if args.compare:
        others = sorted(RESULTS_DIR.glob("day9_*_evaluation.json"))
        if len(others) > 1:
            print(f"\n{'=' * 74}")
            print("  FINE-TUNED MODELS SIDE BY SIDE")
            print(f"{'=' * 74}")
            print(f"  {'model_type':<16}{'Micro F1':>10}{'Macro F1':>10}"
                  f"{'Exact Match':>13}{'Latency/req':>14}")
            print(f"  {'-' * 63}")
            for path in others:
                doc = json.loads(path.read_text())
                b = doc.get("bert", {})
                lat = doc.get("latency", {}).get("sequential", {}).get("mean_ms")
                print(f"  {doc.get('model_type', path.stem):<16}"
                      f"{b.get('micro_f1', 0):>9.1%}{b.get('macro_f1', 0):>10.1%}"
                      f"{b.get('exact_match_rate', 0):>13.1%}"
                      f"{(lat if lat else 0):>11.1f}ms")

    # ── Definition of Done ─────────────────────────────────────────────
    f1_ok = scores["micro_f1"] > TARGET_MICRO_F1
    lat_ok = seq["mean_ms"] < TARGET_LATENCY_MS
    print(f"\n{'=' * 74}")
    print(f"  DEFINITION OF DONE")
    print(f"{'=' * 74}")
    print(f"  [{'PASS' if f1_ok else 'FAIL'}]  micro-F1 {scores['micro_f1']:.1%} "
          f"{'>' if f1_ok else '<='} {TARGET_MICRO_F1:.1%} few-shot baseline")
    print(f"  [{'PASS' if lat_ok else 'FAIL'}]  latency {seq['mean_ms']:.1f} ms "
          f"{'<' if lat_ok else '>='} {TARGET_LATENCY_MS:.0f} ms per requirement")

    if not f1_ok:
        gap = TARGET_MICRO_F1 - scores["micro_f1"]
        print(f"\n  {gap:.1%} short. The per-category table above shows where -- "
              f"sorted worst first.")
        print("  Weak categories with low support are a data problem; weak "
              "categories with\n  high support are a label-consistency problem.")


if __name__ == "__main__":
    main()
