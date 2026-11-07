"""Local causal-LM prompt/completion transform for arithmetic addition.

This replaces the previous runtime dependency on
``trainite.datasets.string_reverse.PromptCompletionTransform``.  The transform
is task-generic: it takes a sample with ``source`` and ``target`` strings and
produces the tensors needed for teacher-forced training plus the prompt used
for autoregressive evaluation.

Token layout (``combined = [BOS] source [SEP] target [EOS]``)::

    train_input_ids : combined[:-1]   # includes the prompt and all target digits
    train_label_ids : combined[1:]    # prompt positions masked with ignore_index
    eval_input_ids  : [BOS] source [SEP]

Training therefore sees exactly the same token sequence that generation will
produce, which is what keeps Abacus column ids consistent between the two
modes.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class DatapointModel(BaseModel):
    """Contract for a causal-LM transformed item: training tensors + eval prompt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor

    # Optional metadata used by evaluation and readable logging.  They are not
    # needed by the model or the training loop.
    original_source: str | None = None
    original_target: str | None = None
    operand_lengths: tuple[int, int] | None = None
    digit_order: str = "lsd_first"


class PromptCompletionTransform:
    """Convert a raw arithmetic sample into training tensors.

    The model is trained with next-token prediction on the target string and is
    evaluated by generating the target from the prompt alone.  All prompt
    positions (including the trailing SEP) are masked out of the loss, so only
    target digits and EOS contribute.
    """

    def __init__(self, tokenizer: Any, ignore_index: int = -100) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def build_prompt(self, sample: dict[str, Any]) -> list[int]:
        """Token ids for the generation prompt: [BOS] + source + [SEP]."""
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        source_tokens = self.tokenizer(sample["source"], add_special_tokens=False)["input_ids"]
        return [bos] + source_tokens + [sep]

    def __call__(self, sample: dict[str, Any]) -> DatapointModel:
        source = sample["source"]
        target = sample["target"]

        eos = self.tokenizer.eos_token_id
        prompt_ids = self.build_prompt(sample)
        target_tokens = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # Full sequence: prompt + target tokens + EOS.  input_ids is the sequence
        # shifted right by one and labels are shifted left by one, so position t
        # predicts token t+1.
        combined_input_ids = prompt_ids + target_tokens + [eos]
        input_ids = torch.tensor(combined_input_ids[:-1], dtype=torch.long)
        labels = torch.tensor(combined_input_ids[1:], dtype=torch.long)

        # labels = combined[1:]; the prompt spans len(prompt_ids) - 1 positions.
        labels[: len(prompt_ids) - 1] = self.ignore_index
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        operand_lengths = sample.get("operand_lengths")
        if operand_lengths is not None:
            operand_lengths = tuple(operand_lengths)
            if len(operand_lengths) != 2:
                operand_lengths = None

        return DatapointModel(
            source=source,
            target=target,
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=attention_mask,
            eval_input_ids=torch.tensor(prompt_ids, dtype=torch.long),
            original_source=sample.get("original_source"),
            original_target=sample.get("original_target"),
            operand_lengths=operand_lengths,
            digit_order=sample.get("digit_order", "lsd_first"),
        )
