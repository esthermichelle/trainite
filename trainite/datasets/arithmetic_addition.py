"""Synthetic dataset for character-level arithmetic addition."""

import random

from torch.utils.data import Dataset


class ArithmeticAdditionDataset(Dataset):
    """Generate unique addition problems with configurable operand lengths.

    Each sample has:
        source: "123+456"
        target: "579"
    """

    def __init__(
        self,
        size: int = 1000,
        min_digits: int = 2,
        max_digits: int = 4,
        seed: int = 42,
    ):
        if size <= 0:
            raise ValueError("size must be positive")
        if min_digits <= 0 or max_digits < min_digits:
            raise ValueError("Invalid operand digit range")

        possible_values = sum(
            9 * 10 ** (digits - 1)
            for digits in range(min_digits, max_digits + 1)
        )
        if size > possible_values**2:
            raise ValueError(
                "Requested size exceeds the number of unique ordered pairs"
            )

        self.size = size
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.seed = seed

        rng = random.Random(seed)
        self.samples = []
        seen_sources = set()

        while len(self.samples) < size:
            digits_a = rng.randint(min_digits, max_digits)
            digits_b = rng.randint(min_digits, max_digits)

            a = rng.randint(10 ** (digits_a - 1), 10**digits_a - 1)
            b = rng.randint(10 ** (digits_b - 1), 10**digits_b - 1)
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
