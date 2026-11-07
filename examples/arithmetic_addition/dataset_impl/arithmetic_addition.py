"""Synthetic dataset for character-level arithmetic addition.

The samples follow the data format used by the Abacus paper
(McLeish et al., 2024, https://arxiv.org/abs/2405.17399): numbers are written
least-significant digit first (LSB-first / reversed).  Reversal is what makes
digit columns line up across operands and, more importantly, what makes the
Abacus column id of an already-generated token independent of how many more
digits will be generated.

The same dataset implementation and the same deterministic splits are used by
both experiments (RoPE + AdamW baseline and Abacus + AdamW).  Set
``reverse_digits: true`` in both configs to keep the comparison fair.
"""

from __future__ import annotations

import random

from torch.utils.data import Dataset


class ArithmeticAdditionDataset(Dataset):
    """Generate addition problems with disjoint train/validation splits.

    For ``split="train"`` and ``split="val"``, samples are drawn from a single
    pool of ``train_size + val_size`` unique problems and sliced, so the two
    splits are disjoint by construction.  Both splits must be constructed with
    the same ``seed``, digit range, and pool sizes.

    For ``split="test"``, ``size`` samples are generated independently.  Use a
    distinct seed and/or digit range (the OOD configs use 5-digit operands and
    seed 200) to guarantee disjointness from train/val.

    Each item is a dictionary with:

    * ``source`` / ``target``: the strings fed to the tokenizer (LSB-first when
      ``reverse_digits=True``).
    * ``original_source`` / ``original_target``: the conventional MSB-first
      strings, kept for readable logs and reports.
    * ``operand_lengths``: ``(digits_a, digits_b)``, used for per-length
      exact-match reporting.
    * ``digit_order``: ``"lsd_first"`` or ``"msd_first"``.
    """

    def __init__(
        self,
        size: int | None = None,
        min_digits: int = 2,
        max_digits: int = 4,
        seed: int = 42,
        split: str = "train",
        train_size: int = 10000,
        val_size: int = 1000,
        reverse_digits: bool = False,
    ):
        if min_digits <= 0 or max_digits < min_digits:
            raise ValueError("Invalid operand digit range")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if train_size <= 0 or val_size <= 0:
            raise ValueError("train_size and val_size must be positive")

        expected_size = {
            "train": train_size,
            "val": val_size,
            "test": size if size is not None else 1000,
        }[split]
        if size is None:
            size = expected_size
        elif split != "test" and size != expected_size:
            raise ValueError(f"size must be {expected_size} for split={split!r}")
        if size <= 0:
            raise ValueError("size must be positive")

        self.size = size
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.seed = seed
        self.split = split
        self.reverse_digits = reverse_digits

        if split == "test":
            self.samples = self._generate_samples(size, seed)
        else:
            all_samples = self._generate_samples(train_size + val_size, seed)
            self.samples = all_samples[:train_size] if split == "train" else all_samples[train_size:]

    def _generate_samples(self, count: int, seed: int) -> list[dict[str, object]]:
        rng = random.Random(seed)
        samples: list[dict[str, object]] = []
        seen_sources: set[str] = set()

        while len(samples) < count:
            digits_a = rng.randint(self.min_digits, self.max_digits)
            digits_b = rng.randint(self.min_digits, self.max_digits)

            a = rng.randint(10 ** (digits_a - 1), 10**digits_a - 1)
            b = rng.randint(10 ** (digits_b - 1), 10**digits_b - 1)

            original_source = f"{a}+{b}"
            original_target = str(a + b)
            if self.reverse_digits:
                source = f"{str(a)[::-1]}+{str(b)[::-1]}"
                target = original_target[::-1]
                digit_order = "lsd_first"
            else:
                source = original_source
                target = original_target
                digit_order = "msd_first"

            if source in seen_sources:
                continue

            seen_sources.add(source)
            samples.append(
                {
                    "source": source,
                    "target": target,
                    "original_source": original_source,
                    "original_target": original_target,
                    "operand_lengths": (digits_a, digits_b),
                    "digit_order": digit_order,
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]
