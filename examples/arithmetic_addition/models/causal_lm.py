"""Shared causal-LM collation and loss utilities.

Both experiments (RoPE baseline and Abacus) must use the exact same batching
and loss code so that any difference in results comes from the positional
encoding, not from a duplicated implementation drifting apart.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class CausalLMCollateFn:
    """Collate sequences for decoder-only autoregressive training.

    Pads a batch of variable-length sequences to the same length using
    *left*-padding.  Left-padding keeps all real tokens right-aligned, which is
    compatible with a single causal mask for the whole batch.

    The implementation flips each sequence, right-pads with ``pad_sequence``,
    then flips the result back -- equivalent to left-padding.

    ``ignore_index`` (-100 by default) is used for padded label positions so
    that PyTorch's ``CrossEntropyLoss`` ignores them during training.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = pad_id if pad_id is not None else 0
        self.ignore_index = -100

    def __call__(self, batch: list[Any]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for item in batch:
            input_ids_list.append(item.train_input_ids.flip(0))
            labels_list.append(item.train_label_ids.flip(0))
            attention_mask_list.append(item.attention_mask.flip(0))

        padded_input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id,
        ).flip(1)
        padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).flip(1)
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index).flip(1)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }


class CausalLMCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-entropy for logits with vocabulary in the last dimension."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return super().forward(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
