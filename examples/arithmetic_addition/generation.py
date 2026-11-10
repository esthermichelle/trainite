"""Model-generated labels for the self-improvement curriculum.

Round ``r`` of the difficulty ladder needs problems one digit longer than the
model has been trained on, answered by the model itself.  This module is the
only new generation code: it reuses ``evaluation.left_pad``,
``evaluation.generate_batch`` and ``evaluation.extract_completion`` so that
self-generated labels are decoded exactly like evaluation predictions.

Keeping this separate from the driver makes it easy to inspect a round's raw
outputs without touching the training loop.
"""

from __future__ import annotations

from typing import Any

import torch

from evaluation import extract_completion, generate_batch, left_pad


@torch.no_grad()
def generate_answers(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Any,
    transform: Any,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    """Greedily answer every problem in ``dataset``.

    Args:
        model: checkpoint that labels this round (``M_{r-1}``).
        tokenizer: character tokenizer used for decoding.
        dataset: raw problem set; items are plain dicts with ``source``.
        transform: ``PromptCompletionTransform`` building the prompt tensors.
        device: device the model lives on.
        max_new_tokens: upper bound on generated answer length (digits + EOS).
        batch_size: generation batch size.

    Returns:
        One ``{"sample": <raw item>, "prediction": <decoded LSB-first string>}``
        per problem, in dataset order.
    """
    model.eval()
    results: list[dict[str, Any]] = []

    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        items = [transform(sample) for sample in samples]
        input_ids, attention_mask = left_pad(items, tokenizer, device)
        generated = generate_batch(
            model,
            input_ids,
            attention_mask,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        for sample, row in zip(samples, generated.tolist()):
            prediction = extract_completion(
                row,
                tokenizer.sep_token_id,
                tokenizer.eos_token_id,
                tokenizer.decode,
            )
            results.append({"sample": sample, "prediction": prediction})

    return results


__all__ = ["generate_answers"]
