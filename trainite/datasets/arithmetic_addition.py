"""Synthetic dataset for character-level arithmetic addition."""

import random

from torch.utils.data import Dataset


class ArithmeticAdditionDataset(Dataset):
    """Generate addition problems with disjoint train/validation splits.

    For ``split="train"`` and ``split="val"``, the samples are drawn from a
    single pool of ``train_size + val_size`` unique problems and sliced, so the
    two splits are disjoint by construction (they must share the same seed,
    digit range, and pool sizes).

    For ``split="test"``, ``size`` samples are generated independently and
    disjointness from train/val is *not* enforced; pass a distinct seed and/or
    digit range to guarantee it.
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
            "test": size or 1000,
        }[split]
        if size is None:
            size = expected_size
        elif size != expected_size and split != "test":
            raise ValueError(f"size must be {expected_size} for split={split!r}")
        if size <= 0:
            raise ValueError("size must be positive")

        self.size = size
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.seed = seed
        self.split = split

        if split == "test":
            self.samples = self._generate_samples(size, seed)
        else:
            expected_size = train_size if split == "train" else val_size
            if size != expected_size:
                raise ValueError(f"size must be {expected_size} for split={split!r}")

            all_samples = self._generate_samples(train_size + val_size, seed)

            if split == "train":
                self.samples = all_samples[:train_size]
            else:
                self.samples = all_samples[train_size:]

    def _generate_samples(self, count, seed):
        rng = random.Random(seed)
        samples = []
        seen_sources = set()

        while len(samples) < count:
            digits_a = rng.randint(self.min_digits, self.max_digits)
            digits_b = rng.randint(self.min_digits, self.max_digits)

            a = rng.randint(10 ** (digits_a - 1), 10**digits_a - 1)
            b = rng.randint(10 ** (digits_b - 1), 10**digits_b - 1)

            source = f"{a}+{b}"

            if source in seen_sources:
                continue

            seen_sources.add(source)
            samples.append(
                {
                    "source": source,
                    "target": str(a + b),
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
