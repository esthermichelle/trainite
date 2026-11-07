"""Standalone evaluation for the arithmetic-addition experiments.

This script reloads a run's best checkpoint, evaluates the in-distribution
(validation) and out-of-distribution (test) splits with the *same*
``evaluation.evaluate_split`` helper used by the training loop, and writes the
resulting scalars back into that run's existing experiment tracking:

* the run's TensorBoard directory (``<run_dir>/tensorboard``), and
* the run's existing W&B run id (``<run_dir>/wandb_run_id.txt``).

It never creates a new run directory or a new W&B run.  Exact-match is the
primary metric; teacher-forced token accuracy is reported as a secondary
diagnostic, alongside per-operand-length exact match.

Usage::

    ./.venv/bin/python evaluate.py config.yaml
    ./.venv/bin/python evaluate.py config_abacus.yaml
    ./.venv/bin/python evaluate.py config.yaml --checkpoint outputs/baseline-adamw/<timestamp>/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from config import ProjectConfig
from evaluation import evaluate_split
from utils import (
    build_model,
    instantiate,
    load_config,
    read_eval_context,
    resolve_wandb_mode,
)

BATCH_SIZE = 128


def _setup_logger(run_dir: Path) -> logging.Logger:
    """Print to stdout and append to the run's existing ``output.log``."""
    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "output.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _resolve_checkpoint(config: ProjectConfig, args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = Path.cwd() / run_dir
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"No best.pt found in {run_dir}")
        return checkpoint

    # Match the training entry point: output.root is resolved relative to the
    # current working directory (documented as examples/arithmetic_addition).
    root = Path(config.output.root) / config.output.run_name
    checkpoints = sorted(root.glob("*/best.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(
            f"No best.pt found under {root}. Train first with: "
            f"./.venv/bin/python main.py {args.config}"
        )
    return checkpoints[-1]


def _load_state_dict(checkpoint_path: Path, device: torch.device, logger: logging.Logger) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:  # pragma: no cover - depends on checkpoint contents
        logger.warning(
            "Could not load %s with weights_only=True (%s); retrying with weights_only=False.",
            checkpoint_path,
            exc,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def _split_metrics(
    result: Any,
    token_accuracy: float,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        f"{prefix}/exact_match": result.exact_match,
        f"{prefix}/token_accuracy": token_accuracy,
        f"{prefix}/num_samples": float(result.num_samples),
    }
    for key, stats in result.per_length.items():
        metrics[f"{prefix}/exact_match/{key}"] = stats["accuracy"]
    return metrics


def _log_tensorboard(run_dir: Path, metrics: dict[str, float], step: int) -> None:
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    try:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
        writer.flush()
    finally:
        writer.close()


def _log_wandb(
    config: ProjectConfig,
    run_dir: Path,
    metrics: dict[str, float],
    step: int,
    logger: logging.Logger,
) -> None:
    """Append metrics to the W&B run created during training, if it exists."""
    if not config.wandb.enabled:
        return
    mode = resolve_wandb_mode(config.wandb.mode)
    if mode == "disabled":
        return

    run_id_path = run_dir / "wandb_run_id.txt"
    if not run_id_path.exists():
        logger.warning("No wandb_run_id.txt in %s; skipping W&B logging.", run_dir)
        return
    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        logger.warning("Empty wandb_run_id.txt in %s; skipping W&B logging.", run_dir)
        return

    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed; skipping W&B logging.")
        return

    init_kwargs: dict[str, Any] = {
        "project": config.wandb.project,
        "id": run_id,
        "resume": "allow",
        "name": config.wandb.run_name or config.output.run_name,
        "dir": str(run_dir),
        "mode": mode,
    }
    if config.wandb.entity:
        init_kwargs["entity"] = config.wandb.entity

    try:
        run = wandb.init(**init_kwargs)
        try:
            run.log(metrics, step=step)
        except Exception as exc:  # pragma: no cover - external service
            logger.warning("W&B logging with explicit step failed (%s); retrying without step.", exc)
            run.log(metrics)
        run.finish()
    except Exception as exc:  # pragma: no cover - external service
        logger.warning("W&B logging failed (%s); continuing without W&B.", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to best.pt (defaults to newest run)")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing best.pt")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit each split (smoke testing)")
    parser.add_argument("--skip-wandb", action="store_true", help="Do not log to W&B")
    parser.add_argument("--skip-tensorboard", action="store_true", help="Do not log to TensorBoard")
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    config: ProjectConfig = load_config(config_path, ProjectConfig)

    checkpoint_path = _resolve_checkpoint(config, args)
    run_dir = checkpoint_path.parent
    logger = _setup_logger(run_dir)
    logger.info("=" * 70)
    logger.info("Evaluating checkpoint: %s", checkpoint_path)

    if config.device and config.device not in (None, "auto"):
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = instantiate(config.preprocessor)
    model = build_model(
        config.model,
        device,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    checkpoint = _load_state_dict(checkpoint_path, device, logger)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    context = read_eval_context(run_dir)
    step = int(context.get("best_iteration") or 0)
    epoch = context.get("best_epoch")
    logger.info(
        "Checkpoint provenance: best_epoch=%s best_iteration=%s",
        epoch if epoch is not None else "unknown",
        step if step else "unknown",
    )

    metrics: dict[str, float] = {}
    if epoch is not None:
        metrics["progress/epoch"] = float(epoch)
    metrics["progress/global_step"] = float(step)
    best_val_loss = context.get("best_val_loss")
    if best_val_loss is not None:
        metrics["eval/best_val_loss"] = float(best_val_loss)

    max_new_tokens = config.trainer.max_inference_new_tokens
    collate_fn_target = config.model.collate_fn_target or "models.causal_lm.CausalLMCollateFn"

    for prefix, label, split_config in (
        ("eval/id", "validation (ID)", config.data.val),
        ("eval/ood", "test (OOD)", getattr(config.data, "test", None)),
    ):
        if split_config is None:
            logger.warning("No %s split configured; skipping.", label)
            continue
        result, token_accuracy = evaluate_split(
            model,
            tokenizer,
            split_config,
            device,
            max_new_tokens=max_new_tokens,
            collate_fn_target=collate_fn_target,
            batch_size=BATCH_SIZE,
            max_examples=args.max_examples,
        )
        metrics.update(_split_metrics(result, token_accuracy, prefix))
        logger.info(
            "%s: exact_match=%.4f token_accuracy=%.4f (%d samples)",
            label,
            result.exact_match,
            token_accuracy,
            result.num_samples,
        )
        for key, stats in result.per_length.items():
            logger.info("  %s exact_match=%.4f (%d/%d)", key, stats["accuracy"], stats["correct"], stats["total"])
        for source, target, prediction in result.wrong:
            logger.info("  wrong: %s = %s -> %s", source, target, prediction)

    if not args.skip_tensorboard:
        _log_tensorboard(run_dir, metrics, step)
        logger.info("Logged evaluation metrics to TensorBoard: %s", run_dir / "tensorboard")
    if not args.skip_wandb:
        _log_wandb(config, run_dir, metrics, step, logger)
        logger.info("Attempted W&B logging for run id in %s", run_dir / "wandb_run_id.txt")


if __name__ == "__main__":
    main()
