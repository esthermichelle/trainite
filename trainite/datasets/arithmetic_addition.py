
"""Synthetic dataset for character-level arithmetic addition."""

import random

from torch.utils.data import Dataset


class ArithmeticAdditionDataset(Dataset):
    """Generate unique addition problems with disjoint train/validation splits."""

    def __init__(
        self,
        size: int = 1000,
        min_digits: int = 2,
        max_digits: int = 4,
        seed: int = 42,
        split: str = "train",
    ):
        if size <= 0:
            raise ValueError("size must be positive")
        if min_digits <= 0 or max_digits < min_digits:
            raise ValueError("Invalid operand digit range")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.size = size
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.seed = seed
        self.split = split

        rng = random.Random(seed)
        self.samples = []
        seen_sources = set()

        while len(self.samples) < size:
            digits_a = rng.randint(min_digits, max_digits)
            digits_b = rng.randint(min_digits, max_digits)

            a = rng.randint(10 ** (digits_a - 1), 10**digits_a - 1)
            b = rng.randint(10 ** (digits_b - 1), 10**digits_b - 1)

            # Deterministic partition: each ordered pair belongs to one split.
            belongs_to_train = (a + 31 * b) % 2 == 0
            if belongs_to_train != (split == "train"):
                continue

            source = f"{a}+{b}"
            if source in seen_sources:
                continue

            seen_sources.add(source)
            self.samples.append(
                {
                    "source": source,
                    "target": str(a + b),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
