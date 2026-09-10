"""BERT multi-label classifier over the 13-category requirement taxonomy.

Everything shared with the other architectures lives in `multilabel.py`; this
module holds only what is specific to BERT:

    - the encoder is `BertModel`, stored as `self.bert` so the checkpoint keys
      (bert.embeddings.*, bert.encoder.*) line up on load
    - the pooled representation is `pooler_output`, i.e. tanh(W . [CLS]) --
      BERT pre-trains that pooler on next-sentence prediction, so it is a
      genuine sentence-level summary rather than an arbitrary token

The names re-exported at the bottom keep older imports
(`from src.models.bert_classifier import MultiLabelClassifier`) working.
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

from src.models.multilabel import (
    DEFAULT_THRESHOLD,
    THRESHOLD_FILE,
    MultiLabelClassifier,
    build_head,
    PosWeightMixin,
    compute_pos_weight,
    decode_predictions,
    logits_to_multihot,
    multihot_to_categories,
    multilabel_loss,
    save_threshold,
)


class BertForMultiLabel(PosWeightMixin, BertPreTrainedModel):
    """BERT encoder + dropout + Linear(768 -> 13), trained with BCE.

    `config.num_labels` sets the head width and `config.id2label` carries the
    category names, so a checkpoint saved by `save_pretrained()` is entirely
    self-describing -- `from_pretrained()` needs no extra arguments.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout, self.classifier = build_head(config)

        self.pos_weight = None   # set later via set_pos_weight()
        self.post_init()

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

        pooled = outputs.pooler_output
        if pooled is None:                       # config built without a pooler
            pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = multilabel_loss(logits, labels, self.pos_weight)

        return SequenceClassifierOutput(loss=loss, logits=logits)


__all__ = [
    "BertForMultiLabel",
    "MultiLabelClassifier",
    "compute_pos_weight",
    "decode_predictions",
    "logits_to_multihot",
    "multihot_to_categories",
    "save_threshold",
    "DEFAULT_THRESHOLD",
    "THRESHOLD_FILE",
]
