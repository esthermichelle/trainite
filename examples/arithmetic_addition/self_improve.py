"""Resumable self-improvement (difficulty-curriculum) driver.

Reference: Lee et al. (2025), "Self-Improving Transformers Overcome Easy-to-Hard
and Length Generalization Challenges". https://arxiv.org/abs/2502.01612

The ladder starts from a checkpoint trained on ``d0``-digit addition and walks
the frontier one digit at a time.  Round ``r`` (1-indexed):

1. generate ``samples_per_round`` problems with ``d0 + r`` digits,
2. label them with the *current* model (the checkpoint from round ``r - 1``),
3. fine-tune on the labelled base split ``D0`` plus every generated round
   ``D1 .. Dr``, with the newest round up-sampled,
4. save ``rounds/round_XX/model.pt`` and advance ``state.json``.

Resume is **round-granular** by design: a round is ~1500 steps (a couple of
minutes), so redoing a crashed round is cheaper and far more robust than
checkpointing optimizer/scheduler/dataloader state mid-round.

Safety rules:

* every artefact is written to a ``.tmp`` file, ``fsync``-ed, then
  ``os.replace``-d -- a half-written round never looks complete;
* ``state.json`` advances **last**, after ``model.pt`` exists;
* per-round seeds are ``base_seed + r`` for both generation and training, so a
  resumed run produces the same data as an uninterrupted one;
* the run directory is content-addressed by a hash of the experiment config, so
  mutating the config creates a new run instead of silently resuming into it.

Usage::

    # smoke test: rounds 1-2 only, then stop cleanly
    python self_improve.py config_si_a2_base.yaml --rounds 2

    # continue from the last completed round (nothing is recomputed)
    python self_improve.py config_si_a2_base.yaml

    # changed the config? start a new run directory
    python self_improve.py config_si_a2_base.yaml --fresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ignite.engine import Engine, Events

from config import ProjectConfig
from dataset_impl.arithmetic_addition import ArithmeticAdditionDataset
from dataset_impl.prompt_completion import PromptCompletionTransform
from dataset_impl.self_improve import JsonlArithmeticDataset, SelfImproveDataset
from dataset_impl.transformed import TransformedDataset
from evaluation import evaluate_split
from generation import generate_answers
from utils import (
    attach_lr_scheduler,
    build_model,
    create_accelerator,
    create_dataloader,
    dump_config,
    instantiate,
    load_config,
    resolve_wandb_mode,
)


# ---------------------------------------------------------------------------
# Filesystem helpers: atomic writes are what make resume safe
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _atomic_write_text(path, "".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_commit() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - git is optional at runtime
        return "unknown"


# ---------------------------------------------------------------------------
# Run identity: config hash, run directory, state.json
# ---------------------------------------------------------------------------


def _config_hash(config: ProjectConfig) -> str:
    """Hash everything that would make a resume a *different* experiment.

    ``rounds`` and ``eval_every_rounds`` are intentionally excluded: shortening
    the smoke test or turning evaluation off must not invalidate progress.
    """
    si = config.self_improve
    assert si is not None
    si_payload = si.model_dump()
    for key in ("rounds", "eval_every_rounds"):
        si_payload.pop(key, None)

    payload = {
        "model": config.model.model_dump(by_alias=True, polymorphic_serialization=True),
        "optimizer": config.optimizer.model_dump(by_alias=True, polymorphic_serialization=True),
        "loss": config.loss.model_dump(by_alias=True, polymorphic_serialization=True),
        "base_dataset": config.data.train.dataset.model_dump(by_alias=True, polymorphic_serialization=True),
        "self_improve": si_payload,
        "seed": config.seed,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _resolve_run_dir(
    config: ProjectConfig,
    args: argparse.Namespace,
    digest: str,
) -> tuple[Path, bool]:
    """Return ``(run_dir, resumed)``.

    Without ``--run-dir``/``--fresh`` the newest directory whose ``state.json``
    carries the same config hash is reused, so a plain invocation is idempotent.
    """
    if args.run_dir is not None:
        path = Path(args.run_dir)
        return path, (path / "state.json").exists()

    root = Path(config.output.root) / config.output.run_name
    if not args.fresh and root.exists():
        candidates = sorted(
            (path.parent for path in root.glob("*/state.json")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            try:
                if _read_json(candidate / "state.json").get("config_hash") == digest:
                    return candidate, True
            except (OSError, ValueError):
                continue

    name = f"run_{digest.split(':')[-1][:8]}"
    path = root / name
    if args.fresh and path.exists():
        path = root / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return path, False


def _load_state(run_dir: Path, config: ProjectConfig, digest: str) -> tuple[dict[str, Any], bool]:
    path = run_dir / "state.json"
    if not path.exists():
        si = config.self_improve
        assert si is not None
        return (
            {
                "config_hash": digest,
                "git_commit": _git_commit(),
                "d0": si.d0,
                "d_max": si.d_max,
                "rounds": si.rounds,
                "completed_rounds": 0,
                "last_checkpoint": None,
                "base_seed": si.seed,
                "created": _now(),
                "updated": _now(),
            },
            True,
        )

    state = _read_json(path)
    if state.get("config_hash") != digest:
        raise SystemExit(
            "Refusing to resume: the config hash differs from state.json.\n"
            f"  state:  {state.get('config_hash')}\n"
            f"  config: {digest}\n"
            "This would be a different experiment. Pass --fresh (new run directory) "
            "or point --run-dir at the intended run."
        )
    return state, False


def _round_dir(run_dir: Path, round_index: int) -> Path:
    return run_dir / "rounds" / f"round_{round_index:02d}"


def _previous_checkpoint(config: ProjectConfig, run_dir: Path, round_index: int) -> Path:
    si = config.self_improve
    assert si is not None
    if round_index == 1:
        path = Path(si.start_checkpoint)
    else:
        path = _round_dir(run_dir, round_index - 1) / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing starting checkpoint for round {round_index}: {path}")
    return path


# ---------------------------------------------------------------------------
# Generation and dataset assembly
# ---------------------------------------------------------------------------


def _generate_round(
    config: ProjectConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    round_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label one round of harder problems with the current model."""
    si = config.self_improve
    assert si is not None
    difficulty = si.d0 + round_index
    seed = si.seed + round_index

    dataset = ArithmeticAdditionDataset(
        size=si.samples_per_round,
        min_digits=difficulty,
        max_digits=difficulty,
        seed=seed,
        split="test",
        reverse_digits=True,
    )
    transform = PromptCompletionTransform(tokenizer, ignore_index=config.loss.ignore_index)

    answers = generate_answers(
        model,
        tokenizer,
        dataset,
        transform,
        device,
        max_new_tokens=difficulty + 2,
        batch_size=si.gen_batch_size,
    )

    tau = si.filter.length
    records: list[dict[str, Any]] = []
    dropped_invalid = 0
    dropped_length = 0
    correct = 0

    for answer in answers:
        sample = answer["sample"]
        prediction = answer["prediction"]
        gold = str(sample["original_target"])
        is_correct = prediction == sample["target"]
        correct += int(is_correct)

        if not prediction or not prediction.isdigit():
            dropped_invalid += 1
            continue
        if tau is not None and abs(len(prediction) - len(gold)) > tau:
            dropped_length += 1
            continue

        records.append(
            {
                "source": sample["source"],
                "target": prediction,
                "original_source": sample["original_source"],
                "original_target": gold,
                "operand_lengths": [difficulty, difficulty],
                "digit_order": "lsd_first",
                "round": round_index,
                "generated_by_seed": seed,
                "correct": is_correct,
            }
        )

    if not records:
        raise RuntimeError(
            f"Round {round_index} produced no usable labels "
            f"(invalid={dropped_invalid}, length={dropped_length}). "
            "The model has collapsed on the new difficulty; lower steps/lr or "
            "enable filter.length before continuing."
        )

    kept_correct = sum(int(record["correct"]) for record in records)
    stats = {
        "round": round_index,
        "difficulty": difficulty,
        "seed": seed,
        "n_samples": len(answers),
        "kept": len(records),
        "dropped_invalid": dropped_invalid,
        "dropped_length": dropped_length,
        "length_filter": tau,
        "gen_accuracy": correct / len(answers),
        "kept_accuracy": kept_correct / len(records),
        "mean_prediction_length": sum(len(record["target"]) for record in records) / len(records),
    }
    return records, stats


def _build_training_dataset(
    config: ProjectConfig,
    run_dir: Path,
    tokenizer: Any,
    round_index: int,
) -> TransformedDataset:
    """``D0 + D1 + ... + Dr`` with the newest round up-sampled."""
    si = config.self_improve
    assert si is not None

    base = instantiate(config.data.train.dataset)
    older = [JsonlArithmeticDataset(_round_dir(run_dir, index) / "data.jsonl") for index in range(1, round_index)]
    latest = JsonlArithmeticDataset(_round_dir(run_dir, round_index) / "data.jsonl")

    batch_size = config.data.train.dataloader.batch_size
    total_size = si.steps_per_round * batch_size

    mixed = SelfImproveDataset(
        base,
        older=older,
        latest=latest,
        latest_weight=si.latest_upsample,
        total_size=total_size,
        seed=si.seed + round_index,
    )
    transform = instantiate(config.data.train.transform, tokenizer=tokenizer)
    return TransformedDataset(mixed, transform)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _load_model(
    config: ProjectConfig,
    tokenizer: Any,
    device: torch.device,
    checkpoint_path: Path,
) -> torch.nn.Module:
    model = build_model(
        config.model,
        device,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def _train_round(
    config: ProjectConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    accelerator: Any,
    dataset: Any,
    round_index: int,
    logger: logging.Logger,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Fine-tune on the mixed dataset for ``steps_per_round`` optimizer steps."""
    si = config.self_improve
    assert si is not None

    seed = si.seed + round_index
    torch.manual_seed(seed)
    random.seed(seed)

    optimizer = instantiate(config.optimizer, params=model.parameters())
    criterion = instantiate(config.loss)
    loader = create_dataloader(
        dataset,
        config.data.train.dataloader,
        tokenizer,
        shuffle=True,
        collate_fn_target=config.model.collate_fn_target,
    )

    device = accelerator.device
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    grad_clip = config.trainer.grad_clip_norm
    losses: list[float] = []

    def step(engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model.train()
        inputs = batch["input_ids"].to(device)
        targets = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, attention_mask=attention_mask)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        if grad_clip is not None:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.detach()))
        return {"loss": loss.detach()}

    engine = Engine(step)
    attach_lr_scheduler(engine, optimizer, si.steps_per_round, config.optimizer.lr)

    log_every = max(1, si.steps_per_round // 10)

    @engine.on(Events.ITERATION_COMPLETED(every=log_every))
    def _log_progress(engine: Engine) -> None:
        logger.info(
            "round %d step %d/%d loss=%.4f lr=%.2e",
            round_index,
            engine.state.iteration,
            si.steps_per_round,
            losses[-1],
            optimizer.param_groups[0]["lr"],
        )

    logger.info(
        "round %d: training %d steps (difficulty=%d, dataset=%d, seed=%d)",
        round_index,
        si.steps_per_round,
        si.d0 + round_index,
        len(dataset),
        seed,
    )
    engine.run(loader, max_epochs=1)

    unwrapped = accelerator.unwrap_model(model)
    stats = {
        "round": round_index,
        "steps": int(engine.state.iteration),
        "seed": seed,
        "mean_loss": sum(losses) / len(losses) if losses else float("nan"),
        "final_loss": losses[-1] if losses else float("nan"),
        "last_100_loss": sum(losses[-100:]) / len(losses[-100:]) if losses else float("nan"),
    }
    return unwrapped, stats


# ---------------------------------------------------------------------------
# Optional per-round evaluation and W&B
# ---------------------------------------------------------------------------


def _evaluate_round(
    config: ProjectConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any] | None:
    split = getattr(config.data, "test", None)
    if split is None:
        return None
    result, token_accuracy = evaluate_split(
        model,
        tokenizer,
        split,
        device,
        max_new_tokens=config.trainer.max_inference_new_tokens,
        collate_fn_target=config.model.collate_fn_target or "models.causal_lm.CausalLMCollateFn",
        batch_size=config.trainer.eval_batch_size,
    )
    return {
        "exact_match": result.exact_match,
        "token_accuracy": token_accuracy,
        "num_samples": result.num_samples,
        "per_length": result.per_length,
    }


def _setup_wandb(config: ProjectConfig, run_dir: Path, logger: logging.Logger) -> Any | None:
    if not config.wandb.enabled:
        return None
    mode = resolve_wandb_mode(config.wandb.mode)
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed; continuing without it")
        return None

    run_id_path = run_dir / "wandb_run_id.txt"
    run_id = run_id_path.read_text(encoding="utf-8").strip() if run_id_path.exists() else None

    init_kwargs: dict[str, Any] = {
        "project": config.wandb.project,
        "name": config.wandb.run_name or config.output.run_name,
        "dir": str(run_dir),
        "mode": mode,
    }
    if config.wandb.entity:
        init_kwargs["entity"] = config.wandb.entity
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:  # pragma: no cover - external service
        logger.warning("wandb.init failed (%s); continuing without it", exc)
        return None

    _atomic_write_text(run_id_path, run.id)
    if config.wandb.tags:
        try:
            run.tags = tuple(config.wandb.tags)
        except Exception:  # pragma: no cover - external service
            pass
    return run


def _log_wandb(run: Any, metrics: dict[str, float], step: int, logger: logging.Logger) -> None:
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:  # pragma: no cover - external service
        logger.warning("wandb.log failed (%s); continuing", exc)


def _finish_wandb(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:  # pragma: no cover - external service
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("self_improve")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "driver.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="config_si_a2_base.yaml")
    parser.add_argument("--rounds", type=int, default=None, help="Stop after this total round (default: config)")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing state and start a new run directory")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory to resume or create")
    parser.add_argument("--from", dest="from_round", type=int, default=None, help="Assert the round to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without touching anything")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    project_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    config: ProjectConfig = load_config(config_path, ProjectConfig)

    si = config.self_improve
    if si is None or not si.enabled:
        raise SystemExit(f"{config_path} has no enabled self_improve block")
    if si.filter.majority_k != 1:
        raise SystemExit("filter.majority_k > 1 is not implemented yet; only greedy labels (k=1) are supported")

    max_rounds = si.d_max - si.d0
    target = args.rounds if args.rounds is not None else (si.rounds if si.rounds is not None else max_rounds)
    target = min(target, max_rounds)
    if target <= 0:
        raise SystemExit("Nothing to do: target rounds is 0")

    digest = _config_hash(config)
    run_dir, resumed = _resolve_run_dir(config, args, digest)
    state, created = _load_state(run_dir, config, digest)

    completed = int(state["completed_rounds"])
    if args.from_round is not None and completed + 1 != args.from_round:
        raise SystemExit(
            f"--from {args.from_round} does not match state: {completed} rounds already completed "
            f"(next round is {completed + 1})"
        )

    if args.dry_run:
        print(f"config:    {config_path}")
        print(f"hash:      {digest}")
        print(f"run_dir:   {run_dir} ({'resume' if resumed else 'new'})")
        print(f"completed: {completed}")
        print(f"target:    {target} rounds (d0={si.d0}, d_max={si.d_max})")
        for index in range(completed + 1, target + 1):
            print(
                f"  round {index}: difficulty={si.d0 + index}, "
                f"samples={si.samples_per_round}, steps={si.steps_per_round}, seed={si.seed + index}"
            )
        return

    if completed >= target:
        print(f"{run_dir} already has {completed} completed rounds (target {target}); nothing to do")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir)
    if created:
        dump_config(config, run_dir / "config.yaml")
        _atomic_write_json(run_dir / "state.json", state)
        logger.info("created new run at %s", run_dir)
    else:
        logger.info("resuming %s at round %d", run_dir, completed + 1)
    logger.info("config hash %s (target round %d)", digest, target)

    accelerator = create_accelerator(config)
    device = accelerator.device
    tokenizer = instantiate(config.preprocessor)
    wandb_run = _setup_wandb(config, run_dir, logger) if accelerator.is_main_process else None

    try:
        for round_index in range(completed + 1, target + 1):
            round_path = _round_dir(run_dir, round_index)
            round_path.mkdir(parents=True, exist_ok=True)
            difficulty = si.d0 + round_index

            checkpoint = _previous_checkpoint(config, run_dir, round_index)
            logger.info(
                "round %d/%d: generating %d problems with %d digits from %s",
                round_index,
                target,
                si.samples_per_round,
                difficulty,
                checkpoint,
            )
            model = _load_model(config, tokenizer, device, checkpoint)
            records, gen_stats = _generate_round(config, model, tokenizer, device, round_index)
            _atomic_write_jsonl(round_path / "data.jsonl", records)
            _atomic_write_json(round_path / "gen_stats.json", gen_stats)
            logger.info(
                "round %d generation: kept=%d/%d gen_acc=%.4f kept_acc=%.4f dropped_invalid=%d dropped_length=%d",
                round_index,
                gen_stats["kept"],
                gen_stats["n_samples"],
                gen_stats["gen_accuracy"],
                gen_stats["kept_accuracy"],
                gen_stats["dropped_invalid"],
                gen_stats["dropped_length"],
            )

            dataset = _build_training_dataset(config, run_dir, tokenizer, round_index)
            model, train_stats = _train_round(
                config,
                model,
                tokenizer,
                accelerator,
                dataset,
                round_index,
                logger,
            )
            _atomic_torch_save(
                round_path / "model.pt",
                {
                    "model": model.state_dict(),
                    "round": round_index,
                    "difficulty": difficulty,
                    "steps": train_stats["steps"],
                    "seed": train_stats["seed"],
                },
            )
            _atomic_write_json(round_path / "train_stats.json", train_stats)
            logger.info(
                "round %d training: mean_loss=%.4f final_loss=%.4f",
                round_index,
                train_stats["mean_loss"],
                train_stats["final_loss"],
            )

            metrics = {
                "self_improve/round": float(round_index),
                "self_improve/difficulty": float(difficulty),
                "self_improve/gen_accuracy": gen_stats["gen_accuracy"],
                "self_improve/kept_accuracy": gen_stats["kept_accuracy"],
                "self_improve/kept_fraction": gen_stats["kept"] / gen_stats["n_samples"],
                "self_improve/train_loss": train_stats["mean_loss"],
            }

            if si.eval_every_rounds is not None and round_index % si.eval_every_rounds == 0:
                eval_payload = _evaluate_round(config, model, tokenizer, device)
                if eval_payload is not None:
                    _atomic_write_json(round_path / "eval.json", eval_payload)
                    metrics["self_improve/eval_exact_match"] = eval_payload["exact_match"]
                    logger.info(
                        "round %d eval: exact_match=%.4f token_accuracy=%.4f",
                        round_index,
                        eval_payload["exact_match"],
                        eval_payload["token_accuracy"],
                    )

            _log_wandb(wandb_run, metrics, step=round_index, logger=logger)

            state["completed_rounds"] = round_index
            state["last_checkpoint"] = str((round_path / "model.pt").relative_to(run_dir))
            state["updated"] = _now()
            _atomic_write_json(run_dir / "state.json", state)

            del model, dataset
            accelerator.free_memory()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("round %d complete; state advanced to %d/%d", round_index, round_index, target)
    finally:
        _finish_wandb(wandb_run)

    logger.info("done: %d/%d rounds complete at %s", state["completed_rounds"], target, run_dir)
    print(f"last checkpoint: {run_dir / state['last_checkpoint']}")


if __name__ == "__main__":
    main()
