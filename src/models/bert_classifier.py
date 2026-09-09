"""Fine-tuned BERT multi-label classifier for the 13-category requirement taxonomy.

Day 9-11. This replaces the LLM-API classification baselines from Day 5
(`src.agents.classification.classify_zero_shot` / `classify_few_shot`).

Why a custom model class instead of `BertForSequenceClassification`?
    HF's built-in head *can* do multi-label via `problem_type=
    "multi_label_classification"`, but writing it out makes the three pieces
    that matter explicit and gives us room for two things the built-in head
    does not offer: a `pos_weight` term for the rare categories, and a
    calibrated decision threshold stored alongside the weights.

The three pieces:
    1. pooled [CLS] representation                 (768-dim)
    2. dropout -> Linear(768 -> 13)                (13 raw logits)
    3. BCEWithLogitsLoss                           (13 *independent* binary
                                                    decisions, NOT a softmax)

Sigmoid + threshold happens at *inference* time, not inside the loss --
`BCEWithLogitsLoss` fuses the sigmoid into the loss for numerical stability,
so `forward()` must return raw logits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

DEFAULT_THRESHOLD = 0.5

# Written next to the weights by notebooks/05_train_bert.py so that evaluation
# and the pipeline agent pick up the same decision threshold the training run
# calibrated on the validation split.
THRESHOLD_FILE = "threshold.json"


# ── Model ──────────────────────────────────────────────────────────────

class BertForMultiLabel(BertPreTrainedModel):
    """BERT encoder + dropout + linear head, trained with BCEWithLogitsLoss.

    `config.num_labels` sets the head width and `config.id2label` carries the
    category names, so a checkpoint saved by `save_pretrained()` is fully
    self-describing -- `from_pretrained()` needs no extra arguments.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        # `classifier_dropout` is the standard HF override knob; fall back to
        # the encoder's own dropout rate when it is not set.
        dropout_prob = getattr(config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Set later via set_pos_weight(); see that method for why it is not an
        # __init__ argument.
        self.pos_weight = None

        self.post_init()

    def set_pos_weight(self, pos_weight: torch.Tensor | None) -> None:
        """Attach the BCE class-imbalance weighting, after loading the weights.

        This is deliberately *not* an `__init__` argument. `from_pretrained()`
        introspects `__init__` to build the model on the meta device before
        streaming the checkpoint into it, and an extra parameter there makes it
        skip initialisation entirely -- you get uninitialised memory as weights
        and a loss around 1e30 instead of 0.69, with no error raised.

        Registered non-persistently: the weighting describes the training run,
        not the trained model, so it stays out of the checkpoint and inference
        never has to know about it.
        """
        # A plain attribute set in __init__ blocks register_buffer() on the same
        # name, so drop it before (re)registering.
        if hasattr(self, "pos_weight"):
            del self.pos_weight
        if pos_weight is None:
            self.pos_weight = None
            return
        self.register_buffer(
            "pos_weight", pos_weight.to(torch.float32), persistent=False
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # pooler_output is tanh(W . [CLS]) -- the sentence-level summary BERT
        # was pre-trained to produce. Fall back to the bare [CLS] hidden state
        # for configs built without a pooler.
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            # Labels arrive as a float multi-hot matrix (batch, 13); BCE needs
            # float targets in the same dtype as the logits.
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.to(logits.dtype),
                pos_weight=(
                    None if self.pos_weight is None
                    else self.pos_weight.to(logits.dtype)
                ),
            )

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ── Logits -> categories ───────────────────────────────────────────────

def logits_to_multihot(
    logits,
    threshold: float = DEFAULT_THRESHOLD,
    force_top1: bool = True,
) -> torch.Tensor:
    """Sigmoid each logit independently, then threshold. Returns int 0/1.

    `force_top1` guards against the empty prediction: every requirement in the
    corpus carries at least one label, so when no logit clears the threshold we
    fall back to the single highest-scoring category rather than predicting
    nothing. Both the training-time metrics and the inference wrapper go
    through here, so a threshold calibrated during training means the same
    thing in the pipeline.
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


def multihot_to_categories(
    row: Sequence[int], categories: Sequence[str]
) -> list[str]:
    """Turn one 0/1 row into the category names it selects."""
    return [categories[i] for i, on in enumerate(row) if on]


def decode_predictions(
    logits,
    categories: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    force_top1: bool = True,
) -> list[list[str]]:
    """Decode a (batch, 13) logit matrix into per-example category lists."""
    multihot = logits_to_multihot(logits, threshold, force_top1=force_top1)
    return [multihot_to_categories(row.tolist(), categories) for row in multihot]


# ── Inference wrapper ──────────────────────────────────────────────────

class MultiLabelClassifier:
    """Load a trained checkpoint and classify requirement statements.

    Used by `notebooks/06_evaluate_bert.py` and by the pipeline's
    classification agent, so both hit exactly the same decode path.
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
        self.model = BertForMultiLabel.from_pretrained(str(self.model_dir))
        self.model.to(self.device).eval()

        self.categories = _categories_from_config(self.model.config)
        self.threshold = (
            threshold if threshold is not None else _saved_threshold(self.model_dir)
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

    def predict(
        self, statements: Sequence[str], batch_size: int = 32
    ) -> list[list[str]]:
        """statements -> one category list per statement."""
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


# ── Checkpoint metadata helpers ────────────────────────────────────────

def _categories_from_config(config) -> list[str]:
    """Recover the label order from `config.id2label` (keys may be str or int)."""
    id2label = config.id2label
    return [id2label[k] for k in sorted(id2label, key=lambda k: int(k))]


def _saved_threshold(model_dir: Path) -> float:
    path = Path(model_dir) / THRESHOLD_FILE
    if not path.exists():
        return DEFAULT_THRESHOLD
    return float(json.loads(path.read_text()).get("threshold", DEFAULT_THRESHOLD))


def save_threshold(model_dir: str | Path, threshold: float, **extra) -> None:
    """Persist the validation-calibrated threshold next to the weights."""
    payload = {"threshold": round(float(threshold), 4), **extra}
    (Path(model_dir) / THRESHOLD_FILE).write_text(json.dumps(payload, indent=2) + "\n")


def compute_pos_weight(label_matrix: Iterable[Sequence[float]]) -> torch.Tensor:
    """BCE `pos_weight` per category: (#negatives / #positives).

    Scales up the loss contribution of positive examples for rare categories so
    the model is not rewarded for always predicting 0 on them. Clamped to avoid
    a single-example category dominating the gradient.
    """
    y = torch.as_tensor(list(label_matrix), dtype=torch.float32)
    positives = y.sum(dim=0).clamp(min=1.0)
    negatives = y.shape[0] - positives
    return (negatives / positives).clamp(min=0.5, max=10.0)
