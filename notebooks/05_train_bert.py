"""Day 9-11: Fine-tune BERT as a multi-label classifier over the 13-category taxonomy.

    python -m notebooks.05_train_bert                      # defaults
    python -m notebooks.05_train_bert --epochs 15 --pos-weight
    python -m notebooks.05_train_bert --model roberta-base --wandb

What happens, in order:
    1. load the DatasetDict written by notebooks/04_prepare_bert_dataset.py
    2. tokenize the statements (WordPiece, truncated to --max-length)
    3. fine-tune every BERT weight plus a fresh Linear(768 -> 13) head,
       supervised by BCEWithLogitsLoss (13 independent binary decisions)
    4. keep the checkpoint with the best validation micro-F1
    5. calibrate the decision threshold on validation and save it beside the
       weights, so evaluation and the pipeline agent use the same cut-off
    6. write a run summary to evaluation/results/

The number that matters: validation micro-F1 has to clear the Day 5 few-shot
LLM baseline of 84.0% -- with the whole 32-requirement ground_truth.json still
held out for notebooks/06_evaluate_bert.py.
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

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.models.bert_classifier import (
    BertForMultiLabel,
    compute_pos_weight,
    logits_to_multihot,
    save_threshold,
)

DATASET_DIR = REPO / "data/training/classification/bert_dataset"
DEFAULT_OUTPUT_DIR = REPO / "models/bert_classifier"
CHECKPOINT_DIR = REPO / "models/experiments/bert_run"
SUMMARY_PATH = REPO / "evaluation/results/day9_bert_training.json"

FEW_SHOT_BASELINE_MICRO_F1 = 0.8402  # evaluation/results/day5_classification_baseline.json

# Thresholds swept after training to pick the decision cut-off. 0.50 is in the
# grid, so calibration can never do worse than the default.
THRESHOLD_GRID = np.round(np.arange(0.20, 0.71, 0.05), 2)


def show(path: Path) -> str:
    """Repo-relative path for printing, falling back to absolute for --output-dir
    pointed somewhere outside the repo."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="bert-base-uncased",
                   help="base checkpoint to fine-tune")
    p.add_argument("--epochs", type=float, default=12,
                   help="343 examples is small -- more passes than the usual 3")
    p.add_argument("--batch-size", type=int, default=16,
                   help="drop to 8 if the GPU runs out of memory")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="standard BERT fine-tuning LR; higher risks wrecking pretrained weights")
    p.add_argument("--max-length", type=int, default=128,
                   help="requirement statements are short sentences")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.2,
                   help="classifier-head dropout; above BERT's 0.1 default given the small dataset")
    p.add_argument("--patience", type=int, default=4,
                   help="early-stopping patience in evaluations; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos-weight", action="store_true",
                   help="weight positive examples by category rarity (helps recall on rare labels)")
    p.add_argument("--pos-weight-max", type=float, default=3.0,
                   help="clamp on the rarity weighting; higher favours recall, lower favours "
                        "precision. 10.0 reproduces the over-predicting Day 11 run, 1.0 is "
                        "equivalent to no weighting at all. Only used with --pos-weight")
    p.add_argument("--threshold", type=float, default=None,
                   help="fix the decision threshold instead of calibrating it on validation")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--wandb", action="store_true", help="log the run to Weights & Biases")
    p.add_argument("--cpu", action="store_true", help="force CPU training")
    return p.parse_args()


# ── Data ───────────────────────────────────────────────────────────────

def load_splits():
    if not DATASET_DIR.exists():
        raise SystemExit(
            f"{show(DATASET_DIR)} not found -- "
            "run `python -m notebooks.04_prepare_bert_dataset` first."
        )
    ds = load_from_disk(str(DATASET_DIR))
    categories = json.loads((DATASET_DIR / "label_names.json").read_text())["categories"]
    return ds, categories


def tokenize_splits(ds, tokenizer, max_length: int):
    """Statements -> input_ids / attention_mask, keeping only what the model eats.

    Truncation only; padding is left to the collator so each batch pads to its
    own longest sequence rather than to a fixed 128.
    """
    def encode(batch):
        return tokenizer(batch["statement"], truncation=True, max_length=max_length)

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    tokenized = ds.map(encode, batched=True)
    for split in tokenized:
        drop = [c for c in tokenized[split].column_names if c not in keep]
        tokenized[split] = tokenized[split].remove_columns(drop)
    return tokenized


def make_collator(tokenizer):
    """Pad input_ids/attention_mask per batch and stack labels as a float matrix.

    Written out rather than reusing DataCollatorWithPadding because the
    multi-hot `labels` column is a list per example, not a scalar class id --
    it needs stacking into a (batch, 13) float tensor for BCE.
    """
    def collate(features: list[dict]) -> dict:
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.float32)
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


# ── Metrics ────────────────────────────────────────────────────────────

def multilabel_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    kw = {"zero_division": 0}
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", **kw),
        "f1_macro": f1_score(y_true, y_pred, average="macro", **kw),
        "precision_micro": precision_score(y_true, y_pred, average="micro", **kw),
        "recall_micro": recall_score(y_true, y_pred, average="micro", **kw),
        # subset accuracy: the whole label set has to match, the strictest metric
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
    }


def build_compute_metrics(threshold: float = 0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        y_pred = logits_to_multihot(logits, threshold).numpy()
        y_true = np.asarray(labels).astype(int)
        return {k: round(v, 4) for k, v in multilabel_scores(y_true, y_pred).items()}

    return compute_metrics


def calibrate_threshold(logits, y_true: np.ndarray) -> tuple[float, dict]:
    """Pick the sigmoid cut-off that maximises validation micro-F1.

    0.5 is only the natural threshold when positives and negatives are
    balanced. With 13 sparse labels most categories are mostly-absent, so the
    model tends to under-predict and a lower cut-off usually wins. The sweep is
    over a single global threshold -- per-category thresholds would overfit a
    validation split this small.
    """
    rows = []
    for t in THRESHOLD_GRID:
        y_pred = logits_to_multihot(logits, float(t)).numpy()
        scores = multilabel_scores(y_true, y_pred)
        rows.append((float(t), scores))

    # Ties break toward 0.5: when two cut-offs score the same, prefer the less
    # extreme one rather than whichever the grid happened to reach first.
    best_t, best_scores = max(
        rows, key=lambda r: (round(r[1]["f1_micro"], 4), -abs(r[0] - 0.5))
    )
    return best_t, {
        "best_threshold": best_t,
        "best_scores": {k: round(v, 4) for k, v in best_scores.items()},
        "sweep": [
            {"threshold": t, "f1_micro": round(s["f1_micro"], 4),
             "f1_macro": round(s["f1_macro"], 4)}
            for t, s in rows
        ],
    }


def per_category_report(y_true: np.ndarray, y_pred: np.ndarray,
                        categories: list[str]) -> list[dict]:
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


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    use_gpu = torch.cuda.is_available() and not args.cpu
    print(f"Device: {'cuda — ' + torch.cuda.get_device_name(0) if use_gpu else 'cpu'}")

    ds, categories = load_splits()
    n_labels = len(categories)
    print(f"Train {len(ds['train'])} | Validation {len(ds['validation'])} "
          f"| {n_labels} categories")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = tokenize_splits(ds, tokenizer, args.max_length)

    # Warn if truncation is actually biting -- at 128 it should not.
    longest = max(len(x) for x in tokenized["train"]["input_ids"])
    print(f"Longest tokenized statement: {longest} tokens (max_length={args.max_length})")

    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=n_labels,
        problem_type="multi_label_classification",
        id2label={i: c for i, c in enumerate(categories)},
        label2id={c: i for i, c in enumerate(categories)},
        classifier_dropout=args.dropout,
    )

    pos_weight = None
    if args.pos_weight:
        pos_weight = compute_pos_weight(ds["train"]["labels"], max_weight=args.pos_weight_max)
        print(f"pos_weight (clamped at {args.pos_weight_max:g}x): "
              + ", ".join(f"{c}={w:.1f}" for c, w in zip(categories, pos_weight.tolist())))

    model = BertForMultiLabel.from_pretrained(args.model, config=config)
    model.set_pos_weight(pos_weight)  # after load -- see set_pos_weight() docstring

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        # transformers>=5 folded warmup_ratio into warmup_steps: a float below
        # 1.0 is read as a fraction of total training steps.
        warmup_steps=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=use_gpu,
        use_cpu=not use_gpu,
        report_to=["wandb"] if args.wandb else [],
        run_name=f"bert-multilabel-{args.model.split('/')[-1]}",
    )

    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=make_collator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(0.5),
        callbacks=callbacks,
    )

    print(f"\n{'=' * 62}\n  TRAINING\n{'=' * 62}")
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    print(f"\nTrained in {train_seconds / 60:.1f} min "
          f"(best checkpoint reloaded: load_best_model_at_end=True)")

    # ── Threshold calibration on the best checkpoint ───────────────────
    prediction = trainer.predict(tokenized["validation"])
    val_logits = prediction.predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    y_true = np.asarray(prediction.label_ids).astype(int)

    scores_at_half = multilabel_scores(
        y_true, logits_to_multihot(val_logits, 0.5).numpy()
    )

    if args.threshold is not None:
        threshold = args.threshold
        sweep_info = {"best_threshold": threshold, "note": "fixed via --threshold"}
    else:
        threshold, sweep_info = calibrate_threshold(val_logits, y_true)

    y_pred = logits_to_multihot(val_logits, threshold).numpy()
    final_scores = multilabel_scores(y_true, y_pred)

    print(f"\n{'=' * 62}\n  VALIDATION RESULTS\n{'=' * 62}")
    print(f"  {'metric':<18}{'@0.50':>10}{f'@{threshold:.2f}':>10}")
    print(f"  {'-' * 38}")
    for k in ("f1_micro", "f1_macro", "precision_micro", "recall_micro", "exact_match"):
        print(f"  {k:<18}{scores_at_half[k]:>9.1%}{final_scores[k]:>10.1%}")

    print(f"\n  {'category':<18}{'support':>8}{'prec':>8}{'recall':>8}{'f1':>8}")
    print(f"  {'-' * 50}")
    categories_report = per_category_report(y_true, y_pred, categories)
    for row in categories_report:
        print(f"  {row['category']:<18}{row['support']:>8}"
              f"{row['precision']:>8.2f}{row['recall']:>8.2f}{row['f1']:>8.2f}")

    # ── Save ───────────────────────────────────────────────────────────
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "label_names.json").write_text(
        json.dumps({"categories": categories}, indent=2) + "\n"
    )
    save_threshold(
        out_dir,
        threshold,
        calibrated_on="validation",
        val_f1_micro=round(final_scores["f1_micro"], 4),
    )
    print(f"\n  Model + tokenizer + threshold saved to {show(out_dir)}")

    summary = {
        "base_model": args.model,
        "dataset": show(DATASET_DIR),
        "train_examples": len(ds["train"]),
        "validation_examples": len(ds["validation"]),
        "categories": categories,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "classifier_dropout": args.dropout,
            "pos_weight": args.pos_weight,
            "pos_weight_max": args.pos_weight_max if args.pos_weight else None,
            "seed": args.seed,
        },
        "train_minutes": round(train_seconds / 60, 2),
        "threshold": threshold,
        "threshold_calibration": sweep_info,
        "validation_at_0.5": {k: round(v, 4) for k, v in scores_at_half.items()},
        "validation_at_threshold": {k: round(v, 4) for k, v in final_scores.items()},
        "per_category": categories_report,
        "log_history": trainer.state.log_history,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  Run summary saved to {show(SUMMARY_PATH)}")

    # ── Verdict ────────────────────────────────────────────────────────
    delta = final_scores["f1_micro"] - FEW_SHOT_BASELINE_MICRO_F1
    print(f"\n{'=' * 62}")
    print(f"  Validation micro-F1  {final_scores['f1_micro']:.1%}")
    print(f"  Day 5 few-shot LLM   {FEW_SHOT_BASELINE_MICRO_F1:.1%}  "
          f"({delta:+.1%})")
    print(f"{'=' * 62}")
    if delta <= 0:
        print("  Below the baseline on validation. Things to try:")
        print("    --epochs 20        more passes over a small dataset")
        print("    --pos-weight --pos-weight-max 3   lift recall on rare categories")
        print("    --dropout 0.3      if train loss << val loss (overfitting)")
        print("    --model roberta-base")
    print("\n  Next: python -m notebooks.06_evaluate_bert  "
          "(scores the held-out ground_truth.json vs the Day 5 baselines)")


if __name__ == "__main__":
    main()
