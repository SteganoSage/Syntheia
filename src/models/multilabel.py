"""Shared machinery for the multi-label requirement classifiers.

One architecture per module -- `bert_classifier.py`, `roberta_classifier.py` --
each contributing only what is genuinely architecture-specific. Everything the
architectures share lives here:

    build_head()          the Linear(hidden -> 13) head and dropout
    multilabel_loss()     the BCEWithLogitsLoss that makes this multi-label
    build_model()         base checkpoint  -> the right model class
    load_model()          trained directory -> the right model class
    MultiLabelClassifier  inference wrapper used by evaluation and the agent
    decode helpers        logits -> sigmoid -> threshold -> category names

Why a registry instead of AutoModelForSequenceClassification:
    The Auto class would work, but it hides the three things this project is
    meant to make explicit -- the pooled representation, the 13-way head, and
    the independent-sigmoid loss. Keeping one small class per architecture
    keeps those visible while still supporting more than BERT.

Why this exists at all:
    `BertForMultiLabel` hardcoded `BertModel`, so pointing --model at
    roberta-base built a BERT encoder from a RoBERTa config. Nothing raised:
    the checkpoint keys (roberta.*) simply did not match what the class
    expected (bert.*), the encoder stayed randomly initialised, and training
    ran for 9 epochs to a degenerate 33.8% before early stopping. `build_model`
    now refuses a config it has no class for, so that failure is impossible to
    repeat silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn

DEFAULT_THRESHOLD = 0.5

# Written next to the weights by notebooks/05_train_bert.py so evaluation and
# the pipeline agent use the threshold the training run calibrated.
THRESHOLD_FILE = "threshold.json"

# config.model_type -> "module:class". Imported lazily inside the lookup so the
# architecture modules can import from here without a circular import.
ARCHITECTURES: dict[str, str] = {
    "bert": "src.models.bert_classifier:BertForMultiLabel",
    "roberta": "src.models.roberta_classifier:RobertaForMultiLabel",
}


# ── The multi-label head ───────────────────────────────────────────────

def build_head(config) -> tuple[nn.Dropout, nn.Linear]:
    """Return the (dropout, classifier) pair every architecture puts on top.

    Returned as loose modules for the caller to assign directly as
    `self.dropout` / `self.classifier`, rather than wrapped in a submodule.
    That keeps the state_dict keys flat -- "classifier.weight", not
    "head.classifier.weight" -- so checkpoints stay interchangeable with
    HF's own *ForSequenceClassification layout and with every checkpoint this
    project has already written.

    The loss, not this head, is what makes the task multi-label: see
    `multilabel_loss`.
    """
    dropout_prob = getattr(config, "classifier_dropout", None)
    if dropout_prob is None:
        dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
    return nn.Dropout(dropout_prob), nn.Linear(config.hidden_size, config.num_labels)


def multilabel_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE over independent sigmoids. Labels are a float multi-hot matrix."""
    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels.to(logits.dtype),
        pos_weight=None if pos_weight is None else pos_weight.to(logits.dtype),
    )


class PosWeightMixin:
    """`set_pos_weight()` shared by every architecture.

    Deliberately not an `__init__` argument. `from_pretrained()` introspects
    `__init__` to build the model on the meta device before streaming the
    checkpoint into it, and an extra parameter there makes it skip
    initialisation entirely -- you get uninitialised memory as weights and a
    loss around 1e30 instead of 0.69, with nothing raised.

    Registered non-persistently: the weighting describes the training run, not
    the trained model, so it stays out of the checkpoint.
    """

    def set_pos_weight(self, pos_weight: torch.Tensor | None) -> None:
        # A plain attribute set in __init__ blocks register_buffer() on the
        # same name, so drop it before (re)registering.
        if hasattr(self, "pos_weight"):
            del self.pos_weight
        if pos_weight is None:
            self.pos_weight = None
            return
        self.register_buffer(
            "pos_weight", pos_weight.to(torch.float32), persistent=False
        )


# ── Architecture dispatch ──────────────────────────────────────────────

def _class_for(config):
    """Resolve config.model_type to its model class, or fail loudly."""
    model_type = getattr(config, "model_type", None)
    target = ARCHITECTURES.get(model_type)
    if target is None:
        raise SystemExit(
            f"No multi-label class registered for model_type {model_type!r}. "
            f"Supported: {sorted(ARCHITECTURES)}. Add one in src/models/ and "
            f"register it in ARCHITECTURES -- do not force it through another "
            f"architecture's class, the weights will not load and training "
            f"will silently produce a random encoder."
        )
    module_path, _, class_name = target.partition(":")
    import importlib
    return getattr(importlib.import_module(module_path), class_name)


def build_model(model_name: str, config, pos_weight: torch.Tensor | None = None):
    """Load a *base* checkpoint (bert-base-uncased, roberta-base, ...).

    Verifies the class matches the config so an architecture mismatch raises
    here rather than after a wasted training run.
    """
    cls = _class_for(config)
    model = cls.from_pretrained(model_name, config=config)
    if pos_weight is not None:
        model.set_pos_weight(pos_weight)   # after load -- see PosWeightMixin
    return model


def load_model(model_dir: str | Path):
    """Load a *trained* directory, dispatching on its saved config."""
    from transformers import AutoConfig

    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(str(model_dir))
    return _class_for(config).from_pretrained(str(model_dir))


# ── Logits -> categories ───────────────────────────────────────────────

def logits_to_multihot(
    logits,
    threshold: float = DEFAULT_THRESHOLD,
    force_top1: bool = True,
) -> torch.Tensor:
    """Sigmoid each logit independently, then threshold. Returns int 0/1.

    `force_top1` guards against the empty prediction: every requirement in the
    corpus carries at least one label, so when no logit clears the threshold we
    fall back to the single highest-scoring category. Training metrics and the
    inference wrapper both come through here, so a calibrated threshold means
    the same thing in the pipeline as it did during training.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    logits = logits.float()

    multihot = (torch.sigmoid(logits) > threshold).int()

    if force_top1:
        empty = multihot.sum(dim=1) == 0
        if empty.any():
            top1 = logits.argmax(dim=1)
            multihot[empty, top1[empty]] = 1

    return multihot


def multihot_to_categories(row: Sequence[int], categories: Sequence[str]) -> list[str]:
    return [categories[i] for i, on in enumerate(row) if on]


def decode_predictions(
    logits,
    categories: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    force_top1: bool = True,
) -> list[list[str]]:
    """Decode a (batch, num_labels) logit matrix into per-example category lists."""
    multihot = logits_to_multihot(logits, threshold, force_top1=force_top1)
    return [multihot_to_categories(row.tolist(), categories) for row in multihot]


# ── Checkpoint metadata ────────────────────────────────────────────────

def categories_from_config(config) -> list[str]:
    """Recover label order from config.id2label (keys may be str or int)."""
    id2label = config.id2label
    return [id2label[k] for k in sorted(id2label, key=lambda k: int(k))]


def saved_threshold(model_dir: str | Path) -> float:
    path = Path(model_dir) / THRESHOLD_FILE
    if not path.exists():
        return DEFAULT_THRESHOLD
    return float(json.loads(path.read_text()).get("threshold", DEFAULT_THRESHOLD))


def save_threshold(model_dir: str | Path, threshold: float, **extra) -> None:
    payload = {"threshold": round(float(threshold), 4), **extra}
    (Path(model_dir) / THRESHOLD_FILE).write_text(json.dumps(payload, indent=2) + "\n")


def compute_pos_weight(
    label_matrix: Iterable[Sequence[float]],
    max_weight: float = 3.0,
    min_weight: float = 0.5,
) -> torch.Tensor:
    """BCE `pos_weight` per category: (#negatives / #positives), clamped.

    Measured on the held-out ground truth, more weighting is monotonically
    worse for micro-F1: 10x -> 75.6%, 3x -> 76.2%, off -> 77.3%. Privacy alone
    improves with it (F1 0.40 -> 0.55 -> 0.77) but it is one category with
    support 6, and the common categories that dominate micro-F1 all prefer no
    weighting. Left in because it is the right lever if the label distribution
    shifts, but it is off by default.

    Note that at max_weight=3 the clamp saturates -- 11 of 13 categories pin to
    exactly 3.0 -- so the rarity signal flattens into a near-uniform scaling.
    """
    if max_weight < min_weight:
        raise ValueError(f"max_weight {max_weight} below min_weight {min_weight}")
    y = torch.as_tensor(list(label_matrix), dtype=torch.float32)
    positives = y.sum(dim=0).clamp(min=1.0)
    negatives = y.shape[0] - positives
    return (negatives / positives).clamp(min=min_weight, max=max_weight)


# ── Inference wrapper ──────────────────────────────────────────────────

class MultiLabelClassifier:
    """Load a trained checkpoint and classify requirement statements.

    Architecture-agnostic: dispatches on the saved config, so the same wrapper
    serves a fine-tuned BERT or RoBERTa. Used by notebooks/06_evaluate_bert.py
    and by the pipeline's classification agent, so both take the same decode
    path.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        threshold: float | None = None,
        max_length: int = 128,
    ):
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = load_model(self.model_dir)
        self.model.to(self.device).eval()

        self.categories = categories_from_config(self.model.config)
        self.threshold = (
            threshold if threshold is not None else saved_threshold(self.model_dir)
        )

    @torch.inference_mode()
    def predict_logits(self, statements: Sequence[str], batch_size: int = 32):
        chunks = []
        for start in range(0, len(statements), batch_size):
            batch = list(statements[start:start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            chunks.append(self.model(**encoded).logits.cpu())
        return torch.cat(chunks) if chunks else torch.empty(0, len(self.categories))

    def predict(self, statements: Sequence[str], batch_size: int = 32) -> list[list[str]]:
        logits = self.predict_logits(statements, batch_size=batch_size)
        return decode_predictions(logits, self.categories, self.threshold)

    def predict_one(self, statement: str) -> list[str]:
        return self.predict([statement])[0]

    def predict_scores(
        self, statements: Sequence[str], batch_size: int = 32
    ) -> list[dict[str, float]]:
        """Per-category sigmoid probabilities -- useful for inspecting near-misses."""
        probs = torch.sigmoid(self.predict_logits(statements, batch_size).float())
        return [
            dict(zip(self.categories, (round(p, 4) for p in row.tolist())))
            for row in probs
        ]
