"""Datasets for the self-improvement (difficulty-curriculum) experiment.

Reference: Lee et al. (2025), "Self-Improving Transformers Overcome
Easy-to-Hard and Length Generalization Challenges". https://arxiv.org/abs/2502.01612

The loop trains on ``D0 + D1 + ... + Dr``, where ``D0`` is the labelled base
split and each ``Dr`` is a round of *model-generated* labels at one higher
difficulty level.  The newest round is up-sampled so that the model keeps
seeing the frontier it is currently learning, while the labelled base split is
never dropped (this is what prevents the model-collapse / error-avalanche
failure mode described in the paper).

Two datasets live here:

* :class:`JsonlArithmeticDataset` -- replays one round of generated samples.
* :class:`SelfImproveDataset` -- mixes the base split with the generated
  rounds according to fixed weights.

Both return the same plain-dict shape as
:class:`dataset_impl.arithmetic_addition.ArithmeticAdditionDataset`, so they
can be wrapped by the existing ``PromptCompletionTransform`` and collated by
the existing ``CausalLMCollateFn``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


def _normalise(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce a JSONL record into the dataset sample contract.

    Only ``source`` and ``target`` are required.  ``operand_lengths`` is what
    the evaluator keys per-length accuracy on, so it is restored to a tuple of
    ints when present.
    """
    if "source" not in record or "target" not in record:
        raise ValueError(f"Sample is missing 'source'/'target': {record!r}")

    lengths = record.get("operand_lengths")
    if lengths is not None:
        lengths = tuple(int(value) for value in lengths)
        if len(lengths) != 2:
            lengths = None

    sample: dict[str, Any] = {
        "source": str(record["source"]),
        "target": str(record["target"]),
        "operand_lengths": lengths,
        "digit_order": record.get("digit_order", "lsd_first"),
    }
    for optional in ("original_source", "original_target"):
        if record.get(optional) is not None:
            sample[optional] = str(record[optional])
    return sample


class JsonlArithmeticDataset(Dataset):
    """Replay a round of generated samples from a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Generated round not found: {self.path}")

        samples: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    samples.append(_normalise(json.loads(line)))

        if not samples:
            raise ValueError(f"Generated round is empty: {self.path}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class SelfImproveDataset(Dataset):
    """Mix the labelled base split with self-generated rounds.

    Sampling is **deterministic**: the index map is built once from a seeded
    RNG, so re-running a round with the same seed produces the same mixture and
    a resumed run is indistinguishable from an uninterrupted one.

    Weighting follows the paper: the newest round receives ``latest_weight``
    of the samples and the remaining probability mass is split evenly across
    the labelled base split and all older rounds.

    Args:
        base: the labelled ``D0`` split.
        older: generated rounds ``D1 .. D_{r-1}`` (may be empty).
        latest: the most recently generated round ``Dr`` (may be ``None``).
        latest_weight: fraction of samples drawn from ``latest``.
        total_size: number of index slots in one epoch.
        seed: RNG seed for the index map.
    """

    def __init__(
        self,
        base: Dataset,
        older: list[Dataset] | None = None,
        latest: Dataset | None = None,
        latest_weight: float = 0.5,
        total_size: int = 150_000,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= latest_weight <= 1.0:
            raise ValueError("latest_weight must be in [0, 1]")
        if total_size <= 0:
            raise ValueError("total_size must be positive")

        older = list(older or [])

        groups: list[tuple[Dataset, float]] = []
        if latest is not None and latest_weight > 0.0:
            groups.append((latest, latest_weight))

        remaining = 1.0 - (latest_weight if latest is not None else 0.0)
        pool = [base, *older]
        if remaining > 0.0 and pool:
            share = remaining / len(pool)
            groups.extend((dataset, share) for dataset in pool)

        if not groups:
            raise ValueError("SelfImproveDataset has no sources to sample from")

        rng = random.Random(seed)
        index: list[tuple[int, int]] = []
        sizes = [len(dataset) for dataset, _ in groups]
        for group_id, ((_, weight), size) in enumerate(zip(groups, sizes)):
            if size <= 0:
                raise ValueError(f"Source {group_id} is empty")
            count = max(1, round(weight * total_size))
            for _ in range(count):
                index.append((group_id, rng.randrange(size)))
        rng.shuffle(index)

        self._groups = [dataset for dataset, _ in groups]
        self._index = index
        self.weights = [weight for _, weight in groups]
        self.sizes = sizes

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group_id, sample_index = self._index[index]
        return self._groups[group_id][sample_index]
