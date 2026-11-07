"""Shared evaluation utilities for the arithmetic-addition experiments.

Exact-match accuracy is the primary metric: a prediction is correct only if the
entire generated answer string matches the target exactly.  Teacher-forced token
accuracy is kept as a secondary diagnostic.

The same functions are used by the training loop's final evaluation and by
``evaluate.py``, so both report identical metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from utils import build_dataset, create_dataloader


@dataclass
class SplitEvaluation:
    exact_match: float
    num_samples: int
    per_length: dict[str, dict[str, float]] = field(default_factory=dict)
    wrong: list[tuple[str, str, str]] = field(default_factory=list)


@torch.no_grad()
def generate_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_id: int | None,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy autoregressive generation for a left-padded batch."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated, attention_mask=attention_mask)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if eos_id is not None:
            already_ended = generated[:, -1:].eq(eos_id)
            next_token = torch.where(already_ended, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token], dim=-1)

        next_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        if eos_id is not None:
            already_ended = generated[:, -2:-1].eq(eos_id)
            next_mask = torch.where(already_ended, torch.zeros_like(next_mask), next_mask)
        attention_mask = torch.cat([attention_mask, next_mask], dim=-1)
        if eos_id is not None and generated[:, -1].eq(eos_id).all():
            break
    return generated


def extract_completion(ids: list[int], sep_id: int, eos_id: int | None, decode) -> str:
    """Return the decoded text after the first SEP, stopping at EOS."""
    if sep_id in ids:
        completion = ids[ids.index(sep_id) + 1 :]
    else:
        completion = ids
    if eos_id is not None and eos_id in completion:
        completion = completion[: completion.index(eos_id)]
    return decode(completion, skip_special_tokens=True)


def left_pad(items: list[Any], tokenizer: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad eval prompts so every sequence ends in the same column."""
    seqs = [item.eval_input_ids.tolist() for item in items]
    max_len = max(len(seq) for seq in seqs)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    input_ids = torch.tensor([[pad_id] * (max_len - len(seq)) + seq for seq in seqs], device=device)
    attention_mask = torch.tensor(
        [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs],
        device=device,
    )
    return input_ids, attention_mask


def _length_key(item: Any) -> str:
    lengths = getattr(item, "operand_lengths", None)
    if lengths is None:
        return "unknown"
    return f"len_{lengths[0]}x{lengths[1]}"


def _display_source(item: Any) -> str:
    original = getattr(item, "original_source", None)
    if original is not None:
        return original
    return item.source[::-1] if getattr(item, "digit_order", None) == "lsd_first" else item.source


def _display_target(item: Any) -> str:
    original = getattr(item, "original_target", None)
    if original is not None:
        return original
    return item.target[::-1] if getattr(item, "digit_order", None) == "lsd_first" else item.target


def _display_prediction(prediction: str, item: Any) -> str:
    return prediction[::-1] if getattr(item, "digit_order", None) == "lsd_first" else prediction


def evaluate_exact_match(
    model: torch.nn.Module,
    tokenizer: Any,
    split_config: Any,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int = 128,
    max_examples: int | None = None,
) -> SplitEvaluation:
    """Generate answers for a split and compute exact-match accuracy."""
    dataset = build_dataset(split_config.dataset, split_config.transform, tokenizer)
    n = len(dataset) if max_examples is None else min(len(dataset), max_examples)

    correct = 0
    per_length_counts: dict[str, list[int]] = {}
    wrong: list[tuple[str, str, str]] = []
    model.eval()

    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, n))]
        input_ids, attention_mask = left_pad(items, tokenizer, device)
        generated = generate_batch(
            model,
            input_ids,
            attention_mask,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        for item, gen in zip(items, generated.tolist()):
            prediction = extract_completion(
                gen,
                tokenizer.sep_token_id,
                tokenizer.eos_token_id,
                tokenizer.decode,
            )
            key = _length_key(item)
            counts = per_length_counts.setdefault(key, [0, 0])
            counts[1] += 1
            if prediction == item.target:
                correct += 1
                counts[0] += 1
            elif len(wrong) < 10:
                wrong.append(
                    (
                        _display_source(item),
                        _display_target(item),
                        _display_prediction(prediction, item),
                    )
                )

    per_length = {
        key: {"correct": counts[0], "total": counts[1], "accuracy": counts[0] / counts[1]}
        for key, counts in sorted(per_length_counts.items())
    }
    return SplitEvaluation(
        exact_match=correct / n if n else 0.0,
        num_samples=n,
        per_length=per_length,
        wrong=wrong,
    )


@torch.no_grad()
def teacher_forced_token_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    split_config: Any,
    device: torch.device,
    collate_fn_target: str,
    batch_size: int = 128,
    ignore_index: int = -100,
    max_examples: int | None = None,
) -> float:
    """Token-level next-token accuracy on non-ignored (answer) positions."""
    dataset = build_dataset(split_config.dataset, split_config.transform, tokenizer)
    if max_examples is not None and max_examples < len(dataset):
        dataset = Subset(dataset, range(max_examples))
    loader = create_dataloader(
        dataset,
        split_config.dataloader,
        tokenizer,
        shuffle=False,
        collate_fn_target=collate_fn_target,
    )
    # create_dataloader uses the split's batch size; rebuild if an override was
    # requested so training and standalone evaluation can share this helper.
    if loader.batch_size != batch_size:
        collate_fn = loader.collate_fn
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=split_config.dataloader.num_workers,
            collate_fn=collate_fn,
        )

    correct = 0
    total = 0
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask=attention_mask)
        mask = labels != ignore_index
        predictions = logits.argmax(dim=-1)
        correct += int((predictions[mask] == labels[mask]).sum().item())
        total += int(mask.sum().item())
    return correct / total if total else 0.0


def evaluate_split(
    model: torch.nn.Module,
    tokenizer: Any,
    split_config: Any,
    device: torch.device,
    max_new_tokens: int,
    collate_fn_target: str,
    batch_size: int = 128,
    max_examples: int | None = None,
) -> tuple[SplitEvaluation, float]:
    """Return exact-match evaluation plus teacher-forced token accuracy."""
    exact = evaluate_exact_match(
        model,
        tokenizer,
        split_config,
        device,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_examples=max_examples,
    )
    token_accuracy = teacher_forced_token_accuracy(
        model,
        tokenizer,
        split_config,
        device,
        collate_fn_target=collate_fn_target,
        batch_size=batch_size,
        max_examples=max_examples,
    )
    return exact, token_accuracy
