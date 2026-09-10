"""RoBERTa multi-label classifier over the 13-category requirement taxonomy.

Mirrors `bert_classifier.py`. Three things differ from BERT, and all three are
silent failures rather than errors if you get them wrong:

    1. the encoder attribute must be `self.roberta`, because the checkpoint
       keys are roberta.* -- storing it as `self.bert` is exactly the bug that
       made --model roberta-base train a randomly initialised encoder to a
       degenerate 33.8%
    2. RoBERTa has no next-sentence-prediction objective, so its pooler is not
       pre-trained on anything meaningful. HF's own RobertaForSequenceClassification
       ignores it and reads the <s> token from last_hidden_state[:, 0]; this
       does the same
    3. RoBERTa's tokenizer emits no token_type_ids, and RobertaModel rejects
       them, so forward() accepts the argument and drops it -- that keeps the
       training collator and the inference wrapper architecture-agnostic
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel,
)

from src.models.multilabel import (
    build_head,
    PosWeightMixin,
    multilabel_loss,
)


class RobertaForMultiLabel(PosWeightMixin, RobertaPreTrainedModel):
    """RoBERTa encoder + dropout + Linear(768 -> 13), trained with BCE."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # add_pooling_layer=False: the <s> token is read directly, so the
        # untrained pooler would just be dead weight in the checkpoint.
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout, self.classifier = build_head(config)

        self.pos_weight = None   # set later via set_pos_weight()
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,   # accepted and ignored; see module docstring
        labels=None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # <s> is RoBERTa's sentence-level slot, the counterpart of BERT's [CLS]
        pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = multilabel_loss(logits, labels, self.pos_weight)

        return SequenceClassifierOutput(loss=loss, logits=logits)


__all__ = ["RobertaForMultiLabel"]
