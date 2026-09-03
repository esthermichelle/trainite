from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class DatapointModel(BaseModel):
    """Tokenized WikiText sample used by Trainite's causal LM pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor


class WikiTextTransform:
    """Convert a WikiText sample into a causal language-modeling datapoint.

    WikiText samples contain a single ``text`` field. The complete text is
    used as the autoregressive training sequence.

    Layout:

        input_ids:  [BOS] text
        labels:     text [EOS]
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample: dict[str, object]) -> DatapointModel:
        text = str(sample["text"])

        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,
        )["input_ids"]

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id

        if bos is None:
            raise ValueError("Tokenizer must define bos_token_id")

        if eos is None:
            raise ValueError("Tokenizer must define eos_token_id")

        combined = [bos] + token_ids + [eos]

        input_ids = torch.tensor(combined[:-1], dtype=torch.long)
        labels = torch.tensor(combined[1:], dtype=torch.long)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return DatapointModel(
            source=text,
            target=text,
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=attention_mask,
            eval_input_ids=torch.tensor(
                [bos] + token_ids,
                dtype=torch.long,
            ),
        )
