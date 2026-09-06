"""Day 5: Classification Baseline Evaluation

Runs zero-shot and few-shot classifiers on ground truth data,
computes multi-label F1 scores, and saves results for comparison
against the fine-tuned BERT classifier (Day 9-11).
"""

import json
import time
from pathlib import Path

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from src.agents.classification import (
    ALL_CATEGORIES,
    classify_few_shot,
    classify_zero_shot,
)

# ── Load ground truth ──────────────────────────────────────────────────

GT_PATH = Path("data/training/classification/ground_truth.json")


def load_ground_truth() -> list[dict]:
    data = json.loads(GT_PATH.read_text())
    return data["requirements"]


# ── Run evaluation ─────────────────────────────────────────────────────

def evaluate_classifier(classifier_fn, classifier_name: str, requirements: list[dict]):
    """Run a classifier on all requirements and compute F1 metrics."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {classifier_name}")
    print(f"{'=' * 60}\n")

    mlb = MultiLabelBinarizer(classes=ALL_CATEGORIES)

    true_labels = []
    pred_labels = []
    per_req_results = []

    total_start = time.perf_counter()

    for i, req in enumerate(requirements):
        statement = req["statement"]
        true_cats = req["categories"]

        start = time.perf_counter()
        try:
            pred_cats = classifier_fn(statement)
        except Exception as e:
            print(f"  ❌ [{req['id']}] Error: {e}")
            pred_cats = []
        elapsed = time.perf_counter() - start

        true_labels.append(true_cats)
        pred_labels.append(pred_cats)

        # Check if prediction matches
        match = "✅" if set(pred_cats) == set(true_cats) else "❌"
        print(f"  {match} [{req['id']}] ({elapsed:.1f}s)")
        print(f"       True: {true_cats}")
        print(f"       Pred: {pred_cats}")

        per_req_results.append({
            "id": req["id"],
            "statement": statement,
            "true_categories": true_cats,
            "predicted_categories": pred_cats,
            "exact_match": set(pred_cats) == set(true_cats),
            "latency_seconds": round(elapsed, 3),
        })

    total_elapsed = time.perf_counter() - total_start

    # Compute F1 scores
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(pred_labels)

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    exact_match_count = sum(1 for r in per_req_results if r["exact_match"])

    print(f"\n{'─' * 40}")
    print(f"  RESULTS: {classifier_name}")
    print(f"{'─' * 40}")
    print(f"  Micro F1:       {micro_f1:.4f}  ({micro_f1 * 100:.1f}%)")
    print(f"  Macro F1:       {macro_f1:.4f}  ({macro_f1 * 100:.1f}%)")
    print(f"  Exact Match:    {exact_match_count}/{len(requirements)}  "
          f"({exact_match_count / len(requirements) * 100:.1f}%)")
    print(f"  Total Time:     {total_elapsed:.1f}s")
    print(f"  Avg Latency:    {total_elapsed / len(requirements):.2f}s/req")

    # Per-category report
    print(f"\n  Per-Category Report:")
    report = classification_report(
        y_true, y_pred,
        target_names=ALL_CATEGORIES,
        zero_division=0,
        output_dict=True,
    )
    for cat in ALL_CATEGORIES:
        r = report[cat]
        if r["support"] > 0:
            print(f"    {cat:24s}  P={r['precision']:.2f}  R={r['recall']:.2f}  "
                  f"F1={r['f1-score']:.2f}  (n={int(r['support'])})")

    return {
        "classifier": classifier_name,
        "total_requirements": len(requirements),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match_rate": round(exact_match_count / len(requirements), 4),
        "total_latency_seconds": round(total_elapsed, 3),
        "avg_latency_seconds": round(total_elapsed / len(requirements), 3),
        "per_requirement": per_req_results,
    }


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    requirements = load_ground_truth()
    print(f"Loaded {len(requirements)} requirements from {GT_PATH}\n")

    # Run zero-shot baseline
    zs_results = evaluate_classifier(classify_zero_shot, "Zero-Shot (Gemini)", requirements)

    # Run few-shot baseline
    fs_results = evaluate_classifier(classify_few_shot, "Few-Shot (Gemini)", requirements)

    # Save combined results
    output = {
        "ground_truth_file": str(GT_PATH),
        "total_requirements": len(requirements),
        "baselines": {
            "zero_shot": {
                "micro_f1": zs_results["micro_f1"],
                "macro_f1": zs_results["macro_f1"],
                "exact_match_rate": zs_results["exact_match_rate"],
                "avg_latency_seconds": zs_results["avg_latency_seconds"],
            },
            "few_shot": {
                "micro_f1": fs_results["micro_f1"],
                "macro_f1": fs_results["macro_f1"],
                "exact_match_rate": fs_results["exact_match_rate"],
                "avg_latency_seconds": fs_results["avg_latency_seconds"],
            },
        },
        "zero_shot_details": zs_results,
        "few_shot_details": fs_results,
    }

    out_path = Path("evaluation/results/day5_classification_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'Zero-Shot':>12} {'Few-Shot':>12}")
    print(f"  {'─' * 44}")
    print(f"  {'Micro F1':<20} {zs_results['micro_f1']:>11.1%} {fs_results['micro_f1']:>11.1%}")
    print(f"  {'Macro F1':<20} {zs_results['macro_f1']:>11.1%} {fs_results['macro_f1']:>11.1%}")
    print(f"  {'Exact Match':<20} {zs_results['exact_match_rate']:>11.1%} {fs_results['exact_match_rate']:>11.1%}")
    print(f"  {'Avg Latency':<20} {zs_results['avg_latency_seconds']:>10.2f}s {fs_results['avg_latency_seconds']:>10.2f}s")
    print(f"\n  Results saved to {out_path}")
    print(f"\n  💡 Your fine-tuned BERT (Day 9-11) needs to beat the Few-Shot F1!")
