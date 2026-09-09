"""Day 7-8: Convert training_set.json into a HuggingFace Dataset for BERT fine-tuning.

Pipeline:
    1. Load data/training/classification/training_set.json
    2. Turn each requirement's category names into a multi-hot vector (length 13)
    3. Split 80/20 into train / validation, stratified so every category is
       represented in both splits in roughly its overall proportion
    4. Save as a HuggingFace DatasetDict via save_to_disk()
    5. Print label-distribution stats and run a reload sanity check

Multi-hot example:
    categories: ["Security", "Privacy"]
    labels:     [0,0,0,1,1,0,0,0,0,0,0,0,0]
                 B St F  Se Pr Re Pe Av Us Dm In Au Op

The label order below is identical to src/agents/classification.ALL_CATEGORIES,
so index i means the same category to the baseline classifier and to BERT.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.preprocessing import MultiLabelBinarizer

# ── Paths & constants ─────────────────────────────────────────────────

REPO = Path(__file__).resolve().parents[1]
TRAIN_JSON = REPO / "data/training/classification/training_set.json"
OUT_DIR = REPO / "data/training/classification/bert_dataset"

CATEGORIES = [
    "Business", "Stakeholder", "Functional", "Security", "Privacy", "Regulatory",
    "Performance", "Availability", "Usability", "Data-management", "Integration",
    "Audit/Reporting", "Operational",
]

VAL_FRACTION = 0.20
SEED = 42


# ── Load ──────────────────────────────────────────────────────────────

def load_training_records() -> list[dict]:
    data = json.loads(TRAIN_JSON.read_text())
    records = data["requirements"]
    if not records:
        raise SystemExit(f"{TRAIN_JSON} has no requirements")
    for r in records:
        unknown = [c for c in r["categories"] if c not in CATEGORIES]
        if unknown:
            raise SystemExit(f"{r['id']}: categories not in taxonomy: {unknown}")
        if not r["categories"]:
            raise SystemExit(f"{r['id']}: empty categories")
    return records


# ── Stratified multi-label split ──────────────────────────────────────

def stratified_multilabel_split(
    y: np.ndarray, val_fraction: float, seed: int, min_val_per_label: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx).

    No iterative-stratification library is available, so this does a bounded
    random search: sample many candidate splits and keep the one whose per-label
    frequencies in the training split best match the overall frequencies, subject
    to every label appearing at least `min_val_per_label` times in validation.
    """
    n, n_labels = y.shape
    n_val = max(int(round(n * val_fraction)), n_labels)
    overall = y.mean(axis=0)

    rng = np.random.default_rng(seed)
    best_idx: np.ndarray | None = None
    best_score = np.inf

    for _ in range(4000):
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        val_label_counts = y[val_idx].sum(axis=0)
        if (val_label_counts < min_val_per_label).any():
            continue
        train_label_counts = y[train_idx].sum(axis=0)
        if (train_label_counts < min_val_per_label).any():
            continue
        train_freq = train_label_counts / len(train_idx)
        score = np.abs(train_freq - overall).sum()
        if score < best_score:
            best_score, best_idx = score, (np.sort(train_idx), np.sort(val_idx))

    if best_idx is None:
        raise SystemExit(
            "Could not find a split with every category present in both parts. "
            "Add more examples for the rarest categories."
        )
    return best_idx


# ── Stats ─────────────────────────────────────────────────────────────

def print_distribution(name: str, y: np.ndarray) -> None:
    counts = y.sum(axis=0).astype(int)
    covered = int((counts > 0).sum())
    print(f"\n  {name}: {y.shape[0]} examples | {covered}/{len(CATEGORIES)} categories covered")
    print(f"  {'category':<18}{'count':>7}{'share':>9}")
    print(f"  {'-' * 34}")
    for c, n in zip(CATEGORIES, counts):
        share = n / y.shape[0] if y.shape[0] else 0
        print(f"  {c:<18}{n:>7}{share:>8.0%}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    records = load_training_records()
    print(f"Loaded {len(records)} requirements from {TRAIN_JSON.relative_to(REPO)}")

    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    y = mlb.fit_transform([r["categories"] for r in records]).astype(np.float32)
    statements = [r["statement"] for r in records]
    ids = [r["id"] for r in records]

    # sanity: MultiLabelBinarizer must not have invented/re-ordered classes
    assert list(mlb.classes_) == CATEGORIES, mlb.classes_

    card = y.sum(axis=1)
    print(f"Label matrix: {y.shape}  |  labels/req: "
          f"min {int(card.min())}, mean {card.mean():.2f}, max {int(card.max())}")

    counts = Counter(c for r in records for c in r["categories"])
    scarce = [c for c in CATEGORIES if counts[c] < 10]
    if scarce:
        print(f"  WARNING: below 10 examples: {scarce}")

    train_idx, val_idx = stratified_multilabel_split(y, VAL_FRACTION, SEED)

    def build(split_idx: np.ndarray) -> Dataset:
        return Dataset.from_dict({
            "id": [ids[i] for i in split_idx],
            "text": [statements[i] for i in split_idx],
            "statement": [statements[i] for i in split_idx],
            "labels": [y[i].tolist() for i in split_idx],
            "categories": [records[i]["categories"] for i in split_idx],
        })

    ds = DatasetDict({"train": build(train_idx), "validation": build(val_idx)})
    ds = ds.with_format(None)

    y_train, y_val = y[train_idx], y[val_idx]
    print_distribution("TRAIN", y_train)
    print_distribution("VALIDATION", y_val)

    train_covered = int((y_train.sum(axis=0) > 0).sum())
    val_covered = int((y_val.sum(axis=0) > 0).sum())

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUT_DIR))
    # id2label / label2id for the Day 9-11 model config
    (OUT_DIR / "label_names.json").write_text(json.dumps({
        "categories": CATEGORIES,
        "id2label": {i: c for i, c in enumerate(CATEGORIES)},
        "label2id": {c: i for i, c in enumerate(CATEGORIES)},
    }, indent=2) + "\n")

    # ── reload sanity check ──────────────────────────────────────────
    reloaded = load_from_disk(str(OUT_DIR))
    r_train, r_val = reloaded["train"], reloaded["validation"]
    assert len(r_train) == len(train_idx), (len(r_train), len(train_idx))
    assert len(r_val) == len(val_idx), (len(r_val), len(val_idx))
    assert len(r_train[0]["labels"]) == len(CATEGORIES), len(r_train[0]["labels"])
    assert len(r_train) + len(r_val) == len(records)
    assert set(r_train["id"]).isdisjoint(set(r_val["id"])), "train/val id overlap"

    print(f"\n{'=' * 48}")
    print(f"  Total examples:      {len(records)}")
    print(f"  Train set:           {len(r_train)}")
    print(f"  Validation set:      {len(r_val)}")
    print(f"  Label dimensions:    {len(CATEGORIES)}")
    print(f"  Categories covered:  {min(train_covered, val_covered)}/{len(CATEGORIES)} "
          f"(train {train_covered}/{len(CATEGORIES)}, val {val_covered}/{len(CATEGORIES)})")
    print(f"  Saved to:            {OUT_DIR.relative_to(REPO)}")
    print(f"  Reload check:        OK")
    print(f"{'=' * 48}")

    if min(train_covered, val_covered) < len(CATEGORIES):
        raise SystemExit("A category is missing from a split -- fix before fine-tuning.")


if __name__ == "__main__":
    main()
