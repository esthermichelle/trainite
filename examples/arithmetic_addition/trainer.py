import html
import logging
import random
from pathlib import Path

from accelerate import Accelerator
import torch
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver
from ignite.handlers.clearml_logger import ClearMLSaver
from ignite.handlers.logger_utils import setup_clearml_logging, setup_tb_logging
from ignite.metrics import Accuracy, Loss, Metric, RunningAverage
from ignite.utils import setup_logger
from torch.utils.data import DataLoader

from config import (
    ProjectConfig,
    TrainerConfig,
)
from evaluation import evaluate_exact_match, evaluate_split

from utils import (
    attach_early_stopping,
    attach_lr_scheduler,
    build_dataloaders,
    build_model,
    create_accelerator,
    create_run_dir,
    instantiate,
    setup_best_model_checkpoint,
    setup_console_logger,
    setup_training_checkpointing,
    setup_wandb_logging_safe,
    write_eval_context,
)


def _flatten(output: dict[str, torch.Tensor], ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape model output from (B, S, vocab) -> (B*S, vocab) and apply the ignore mask.

    PyTorch's ``CrossEntropyLoss`` and Ignite's ``Loss``/``Accuracy`` metrics
    expect a flat prediction tensor of shape (N, C) and a flat target tensor of
    shape (N,).  This helper:

    1. Flattens the logits and targets across the batch and sequence dimensions.
    2. Filters out positions where the target equals ``ignore_index`` (i.e. padding
       and prompt positions that should not contribute to the loss).

    Args:
        output: Dictionary with keys ``"logits"`` (B, S, vocab_size) and
                ``"targets"`` (B, S).
        ignore_index: Label value to exclude from metrics (default: -100, which
                      is PyTorch's default ``CrossEntropyLoss.ignore_index``).

    Returns:
        Tuple of (filtered_logits, filtered_targets) ready for loss/accuracy metrics.
    """
    logits = output["logits"].reshape(-1, output["logits"].size(-1))
    targets = output["targets"].reshape(-1)
    # Build a boolean mask that is True for every non-ignored position
    mask = targets != ignore_index
    return logits[mask], targets[mask]


class Trainer:
    """High-level training orchestrator for decoder-only language models.

    This class wires together PyTorch-Ignite engines with Hugging Face Accelerate
    for scalable, hardware-agnostic training:

    * **Hardware & Distributed (Accelerate)** – handles device placement (CUDA/MPS/CPU),
      multi-GPU Distributed Data Parallel (DDP), and Automatic Mixed Precision (AMP:
      ``float32``, ``fp16``, ``bf16``) without boilerplate.
    * **Engines (Ignite)** – ``self.trainer`` runs the training loop; ``self.train_evaluator``,
      ``self.val_evaluator``, and ``self.test_evaluator`` run evaluation.
    * **Metrics** – loss and token-accuracy are attached to each evaluator.
    * **Learning-rate schedule** – linear warm-up + linear decay.
    * **Checkpointing** – last and best-model checkpoints saved to ``run_dir``.
    * **Early stopping** – halts training when validation loss stagnates.
    * **Experiment logging** – TensorBoard or ClearML depending on ``config.logger``.
    * **Inference logging** – optional qualitative sample generation every N epochs.

    Typical usage::

        config = load_config("config.yaml", ProjectConfig)
        trainer = Trainer(config)
        trainer.run()

    References:
    * PyTorch-Ignite: https://pytorch-ignite.ai/concepts/
    * Hugging Face Accelerate: https://huggingface.co/docs/accelerate/index
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.logger: logging.Logger = setup_logger("trainer", level=logging.INFO)
        # Fix randomness for reproducibility across runs with the same seed
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self.config: ProjectConfig = config
        self.trainer_config: TrainerConfig = config.trainer

        # Setup Accelerate for device placement, mixed precision, and distributed training
        self.accelerator: Accelerator = create_accelerator(config)
        # Strip ordinal index for CPU (e.g. 'cpu:1' on rank 1) to prevent torch.load errors
        self.device: torch.device = (
            torch.device("cpu") if self.accelerator.device.type == "cpu" else self.accelerator.device
        )

        # Build tokenizer from config (e.g. CharTokenizer)
        self.tokenizer = instantiate(config.preprocessor)
        # Build train/val/test DataLoaders from the data config
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            config.data,
            self.tokenizer,
            config.seed,
            collate_fn_target=config.model.collate_fn_target,
        )
        self.model = build_model(
            config.model, self.device, vocab_size=self.tokenizer.vocab_size, pad_token_id=self.tokenizer.pad_token_id
        )
        self.optimizer = instantiate(config.optimizer, params=self.model.parameters())
        self.epochs: int = self.trainer_config.epochs
        self.grad_clip_norm: float | None = getattr(self.trainer_config, "grad_clip_norm", None)
        self.inference_every_epochs = self.trainer_config.inference_every_epochs
        self.inference_num_samples = self.trainer_config.inference_num_samples
        self.max_inference_new_tokens = self.trainer_config.max_inference_new_tokens
        self.criterion = instantiate(config.loss)
        self.ignore_index: int = getattr(self.criterion, "ignore_index", -100)
        self.requires_router_info = getattr(self.criterion, "requires_router_info", False)
        self.train_forward_kwargs = {"return_router_info": True} if self.requires_router_info else {}

        # Prepare components for distributed execution and mixed precision
        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.test_loader,
        )

        self.total_iters: int = len(self.train_loader) * self.epochs
        self.trainer = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.test_evaluator = Engine(self._eval_step)
        self.metrics: dict = self._attach_metrics()

        # Track the validation epoch that produced the best checkpoint, so the
        # final evaluation is logged at the correct epoch/global step.
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_iteration = 0
        self.val_evaluator.add_event_handler(Events.COMPLETED, self._track_best)

        # Run evaluations at the end of each epoch to log training and validation metrics
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)

        # Create run directory on rank 0 and broadcast to all ranks
        self.run_dir = create_run_dir(config, self.accelerator)

        # Attach loggers for console
        self.logger = setup_console_logger(
            self.run_dir, self.trainer, self.trainer_config.log_every_steps, self.optimizer
        )

        # Attach learning rate scheduler
        attach_lr_scheduler(self.trainer, self.optimizer, self.total_iters, config.optimizer.lr)

        # Attach early stopping
        if self.trainer_config.early_stopping_patience is not None:
            attach_early_stopping(self.val_evaluator, self.trainer, self.trainer_config.early_stopping_patience)

        # Define validation scoring function for best model
        def score_function(engine_val):
            loss = engine_val.state.metrics["loss"]
            return -loss

        # Only training/validation metrics are auto-logged per epoch.  The test
        # set is evaluated once against the best checkpoint at the end, so its
        # metrics are logged at the best checkpoint's step rather than the final
        # training step.
        evaluators = {"training": self.train_evaluator, "validation": self.val_evaluator}
        logging_kwargs = {
            "trainer": self.trainer,
            # Learning rate is logged explicitly at epoch end to keep the tag
            # identical in TensorBoard and W&B (training/learning_rate).
            "optimizers": None,
            "evaluators": evaluators,
            "log_every_iters": self.trainer_config.log_every_steps,
            "trainer_metric_names": ["batch_loss"],
            "evaluator_metric_names": ["loss", "token_accuracy"],
        }
        self.exp_logger = None
        self.wandb_logger = None
        if self.accelerator.is_main_process:
            if config.logger == "clearml":
                self.exp_logger = setup_clearml_logging(
                    **logging_kwargs,
                    project_name=config.project_name,
                    task_name=self.run_dir.name,
                )
                self.exp_logger.get_task().upload_artifact(
                    name="config.yaml", artifact_object=str(self.run_dir / "config.yaml")
                )
            else:
                self.exp_logger = setup_tb_logging(
                    output_path=str(self.run_dir / "tensorboard"),
                    **logging_kwargs,
                )
            # W&B is additive: TensorBoard remains the local source of truth.
            self.wandb_logger = setup_wandb_logging_safe(config, self.run_dir, logging_kwargs)

        # Progress scalars, learning rate, and optional periodic exact match.
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch_progress)
        if self.trainer_config.exact_match_every_epochs is not None:
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.trainer_config.exact_match_every_epochs),
                self._run_periodic_id_eval,
            )

        # Setup save handler and checkpointing on main process only
        if self.accelerator.is_main_process:
            if config.logger == "clearml":
                # Keep local checkpoints in run_dir; ClearML uses its configured output_uri for optional uploads.
                save_handler = ClearMLSaver(
                    logger=self.exp_logger,
                    dirname=str(self.run_dir),
                    output_uri=True,
                    require_empty=False,
                )
            else:
                save_handler = DiskSaver(dirname=str(self.run_dir), require_empty=False)

            # Unwrap model before checkpointing to save clean weights without distributed ('module.') prefixes
            unwrapped_model: torch.nn.Module = self.accelerator.unwrap_model(self.model)
            self.best_checkpoint = setup_best_model_checkpoint(
                self.trainer,
                self.val_evaluator,
                {"model": unwrapped_model, "optimizer": self.optimizer},
                save_handler,
                score_function=score_function,
                score_name="val_loss",
            )
            self.last_checkpoint = setup_training_checkpointing(
                self.trainer,
                {"model": unwrapped_model, "optimizer": self.optimizer},
                save_handler,
            )
        else:
            self.best_checkpoint = None
            self.last_checkpoint = None
        # Attach inference logger if inference logging is enabled
        if self.inference_every_epochs is not None:
            self.attach_inference_logger()

    def _track_best(self, engine: Engine) -> None:
        """Remember which validation epoch produced the best checkpoint."""
        loss = float(engine.state.metrics["loss"])
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_epoch = self.trainer.state.epoch
            self.best_iteration = self.trainer.state.iteration

    def _log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Write the same scalar values to TensorBoard (or ClearML) and W&B."""
        clean = {key: float(value) for key, value in metrics.items() if value is not None}
        if not clean:
            return
        if self.exp_logger is not None:
            if self.config.logger == "clearml":
                for key, value in clean.items():
                    self.exp_logger.report_scalar(title=key, series="value", value=value, iteration=step)
            else:
                for key, value in clean.items():
                    self.exp_logger.writer.add_scalar(key, value, step)
        if self.wandb_logger is not None:
            try:
                self.wandb_logger.log(clean, step=step)
            except Exception as exc:  # pragma: no cover - external service
                # Final evaluation is logged at the best checkpoint's step,
                # which can be earlier than the last step W&B has already
                # consumed.  Retry without an explicit step so the metrics are
                # still recorded instead of losing the whole run.
                self.logger.warning(
                    "W&B logging with explicit step failed (%s); retrying without step.", exc
                )
                try:
                    self.wandb_logger.log(clean)
                except Exception as retry_exc:  # pragma: no cover - external service
                    self.logger.warning("W&B logging failed (%s); continuing without W&B.", retry_exc)
                    self.wandb_logger = None

    def _log_epoch_progress(self, engine: Engine) -> None:
        """Log learning rate, epoch, and global step once per epoch."""
        step = engine.state.iteration
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self._log_scalars(
            {
                "training/learning_rate": learning_rate,
                "progress/epoch": float(engine.state.epoch),
                "progress/global_step": float(step),
            },
            step=step,
        )

    def _run_periodic_id_eval(self, engine: Engine) -> None:
        """Optional generation-based ID exact-match during training."""
        if not self.accelerator.is_main_process:
            return
        unwrapped_model: torch.nn.Module = self.accelerator.unwrap_model(self.model)
        result = evaluate_exact_match(
            unwrapped_model,
            self.tokenizer,
            self.config.data.val,
            self.device,
            max_new_tokens=self.max_inference_new_tokens,
            batch_size=self.trainer_config.eval_batch_size,
        )
        step = engine.state.iteration
        metrics = {
            "eval/id/exact_match": result.exact_match,
            "eval/id/num_samples": float(result.num_samples),
            "progress/epoch": float(engine.state.epoch),
            "progress/global_step": float(step),
        }
        val_token_accuracy = self.val_evaluator.state.metrics.get("token_accuracy")
        if val_token_accuracy is not None:
            metrics["eval/id/token_accuracy"] = float(val_token_accuracy)
        for key, stats in result.per_length.items():
            metrics[f"eval/id/exact_match/{key}"] = stats["accuracy"]
        self._log_scalars(metrics, step=step)
        self.logger.info(
            "epoch=%s ID exact_match=%.4f (%d samples)",
            engine.state.epoch,
            result.exact_match,
            result.num_samples,
        )

    def _train_step(self, engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run a single training iteration: forward pass, loss, backward, optimizer step.

        This is the function passed to the Ignite training ``Engine``.  Ignite calls it
        once per batch and stores the returned dict as ``engine.state.output``, which
        the attached metrics and loggers then read.

        ``set_to_none=True`` in ``zero_grad`` frees gradient memory instead of filling
        it with zeros, which is slightly faster and uses less memory.

        ``accelerator.backward(loss)`` and ``accelerator.clip_grad_norm_`` handle loss
        scaling and gradient synchronization across distributed processes.

        The tensors in the returned dict are ``detach()``-ed so that the compute graph
        is released before they are consumed by metrics or loggers.

        See: https://docs.pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
        """
        self.model.train()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(inputs, attention_mask=attention_mask, **self.train_forward_kwargs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        loss = self.criterion(outputs, targets)

        self.accelerator.backward(loss)

        if self.grad_clip_norm is not None:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        return {
            "loss": loss.detach(),
            "logits": logits.detach(),
            "targets": targets.detach(),
        }

    @torch.no_grad()
    def _eval_step(self, engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run a single evaluation iteration: forward pass only, no gradient tracking.

        ``@torch.no_grad()`` disables gradient computation for the duration of this
        call, reducing memory usage and speeding up evaluation.

        ``loss`` is intentionally absent from the returned dict: the Ignite ``Loss``
        metric recomputes it internally via its ``output_transform`` (``_flatten``),
        ensuring the metric accumulates correctly over the full evaluation set rather
        than averaging pre-computed batch losses.

        See: https://docs.pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
        """
        self.model.eval()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits = self.model(inputs, attention_mask=attention_mask)

        # Pad across processes along sequence dimension (dim=1) if variable lengths exist across ranks
        if self.accelerator.use_distributed:
            logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=0)
            targets = self.accelerator.pad_across_processes(targets, dim=1, pad_index=self.ignore_index)

        # Gather across all ranks and drop duplicated padding samples on the last batch
        logits, targets = self.accelerator.gather_for_metrics((logits, targets))
        return {"logits": logits, "targets": targets}

    def _run_evaluations(self, engine: Engine) -> None:
        """Evaluate on the training and validation sets at the end of each epoch.

        Triggered by ``Events.EPOCH_COMPLETED`` on the training engine.

        The train evaluator is run with ``epoch_length=min(train, val)`` so that
        the number of batches evaluated matches the validation set size.  This keeps
        train and validation metrics comparable without running a full pass over the
        (potentially much larger) training set every epoch.
        """
        self.logger.info("Evaluating on training set...")
        self.train_evaluator.run(self.train_loader, epoch_length=min(len(self.train_loader), len(self.val_loader)))
        train_metrics = self.train_evaluator.state.metrics
        epoch = engine.state.epoch

        self.logger.info("Evaluating on validation set...")
        self.val_evaluator.run(self.val_loader)
        val_metrics = self.val_evaluator.state.metrics
        self.logger.info(
            "epoch=%s train_loss=%.4f train_token_acc=%.4f val_loss=%.4f val_token_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["token_accuracy"],
            val_metrics["loss"],
            val_metrics["token_accuracy"],
        )

    def run(self) -> None:
        self.logger.info("starting run in %s", self.run_dir)
        try:
            self.trainer.run(self.train_loader, max_epochs=self.epochs)
            self.test()
        finally:
            if self.exp_logger is not None:
                self.exp_logger.close()
            if self.wandb_logger is not None:
                try:
                    self.wandb_logger.close()
                except Exception:  # pragma: no cover - external service
                    pass

    def test(self, test_loader: DataLoader | None = None) -> None:
        """Evaluate the best checkpoint on ID (validation) and OOD (test) sets.

        Exact-match accuracy is the primary metric; teacher-forced token
        accuracy is the secondary metric.  Results are logged at the best
        validation checkpoint's epoch/global step so they are not confused with
        the final training step.
        """
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        checkpoint_handler = self.best_checkpoint
        checkpoint_path = None
        if checkpoint_handler and checkpoint_handler.last_checkpoint:
            checkpoint_path = Path(checkpoint_handler.last_checkpoint)
        elif (self.run_dir / "best.pt").exists():
            checkpoint_path = self.run_dir / "best.pt"

        unwrapped_model: torch.nn.Module = self.accelerator.unwrap_model(self.model)
        if checkpoint_path is not None:
            self.logger.info("Loading best model for final evaluation from %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            unwrapped_model.load_state_dict(checkpoint["model"])
        else:
            self.logger.warning("No best model checkpoint found. Using current model for final evaluation.")

        step = self.best_iteration or self.trainer.state.iteration
        epoch = self.best_epoch or self.trainer.state.epoch
        collate_fn_target = self.config.model.collate_fn_target or "models.causal_lm.CausalLMCollateFn"

        metrics: dict[str, float] = {
            "progress/epoch": float(epoch),
            "progress/global_step": float(step),
            "eval/best_val_loss": float(self.best_val_loss) if self.best_val_loss != float("inf") else float("nan"),
        }

        id_result, id_token_accuracy = evaluate_split(
            unwrapped_model,
            self.tokenizer,
            self.config.data.val,
            self.device,
            max_new_tokens=self.max_inference_new_tokens,
            collate_fn_target=collate_fn_target,
            batch_size=self.trainer_config.eval_batch_size,
        )
        metrics.update(
            {
                "eval/id/exact_match": id_result.exact_match,
                "eval/id/token_accuracy": id_token_accuracy,
                "eval/id/num_samples": float(id_result.num_samples),
            }
        )
        for key, stats in id_result.per_length.items():
            metrics[f"eval/id/exact_match/{key}"] = stats["accuracy"]
        self.logger.info(
            "ID (validation) exact_match=%.4f token_accuracy=%.4f (%d samples)",
            id_result.exact_match,
            id_token_accuracy,
            id_result.num_samples,
        )
        for source, target, prediction in id_result.wrong:
            self.logger.info("  ID wrong: %s = %s -> %s", source, target, prediction)

        test_split = getattr(self.config.data, "test", None)
        if test_split is not None:
            ood_result, ood_token_accuracy = evaluate_split(
                unwrapped_model,
                self.tokenizer,
                test_split,
                self.device,
                max_new_tokens=self.max_inference_new_tokens,
                collate_fn_target=collate_fn_target,
                batch_size=self.trainer_config.eval_batch_size,
            )
            metrics.update(
                {
                    "eval/ood/exact_match": ood_result.exact_match,
                    "eval/ood/token_accuracy": ood_token_accuracy,
                    "eval/ood/num_samples": float(ood_result.num_samples),
                }
            )
            for key, stats in ood_result.per_length.items():
                metrics[f"eval/ood/exact_match/{key}"] = stats["accuracy"]
            self.logger.info(
                "OOD (test) exact_match=%.4f token_accuracy=%.4f (%d samples)",
                ood_result.exact_match,
                ood_token_accuracy,
                ood_result.num_samples,
            )
            for source, target, prediction in ood_result.wrong:
                self.logger.info("  OOD wrong: %s = %s -> %s", source, target, prediction)
        else:
            self.logger.warning("No OOD test split configured; skipping OOD evaluation.")

        self._log_scalars(metrics, step=step)
        write_eval_context(
            self.run_dir,
            best_epoch=epoch,
            best_iteration=step,
            last_epoch=self.trainer.state.epoch,
            last_iteration=self.trainer.state.iteration,
            best_val_loss=self.best_val_loss,
            metrics=metrics,
        )

    def _attach_metrics(self) -> dict[str, Metric]:
        """Create and attach Ignite metrics to the trainer and evaluator engines.

        Metrics are attached once and updated automatically by Ignite after each
        iteration/epoch.  Three evaluators share the same metric definitions so
        results are stored under separate keys (``train_*``, ``val_*``, ``test_*``).

        ``RunningAverage`` computes a smoothed per-iteration loss for the training
        engine so that fluctuations between batches are dampened in the logs.

        ``Loss`` and ``Accuracy`` are epoch-level metrics computed over the full
        evaluation set.  Both use ``_flatten`` as their ``output_transform`` so they
        see only the non-ignored token positions.

        See: https://docs.pytorch.org/ignite/metrics.html
        """
        # Running average loss tracked per training iteration (logged in console)
        RunningAverage(output_transform=lambda output: output["loss"]).attach(self.trainer, "batch_loss")

        # Shared transform: flatten and filter out ignored positions for both metrics
        def transform_fn(output):
            return _flatten(output, ignore_index=self.ignore_index)

        # Evaluate language modeling with cross-entropy; router balancing is a training regularizer.
        eval_criterion = self.criterion.cross_entropy if self.requires_router_info else self.criterion
        metrics = {}
        for prefix, evaluator in [
            ("train", self.train_evaluator),
            ("val", self.val_evaluator),
            ("test", self.test_evaluator),
        ]:
            loss = Loss(eval_criterion, output_transform=transform_fn, device=self.device)
            token_acc = Accuracy(output_transform=transform_fn, device=self.device)

            loss.attach(evaluator, "loss")
            token_acc.attach(evaluator, "token_accuracy")

            metrics[f"{prefix}_loss"] = loss
            metrics[f"{prefix}_token_accuracy"] = token_acc

        return metrics

    def attach_inference_logger(self) -> None:
        """Register the qualitative inference handler to run every N epochs.

        ``Events.EPOCH_COMPLETED(every=N)`` is Ignite's event-filter syntax: it fires
        the handler only on epochs that are a multiple of N, rather than every epoch.
        The handler runs ``_log_inference`` on both the training and validation loaders
        so you can compare model outputs on seen and unseen samples side by side.
        """
        for loader, name in ((self.train_loader, "Train"), (self.val_loader, "Val")):
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
                self._log_inference,
                loader,
                name,
            )

    def _log_text(self, tag: str, text: str, step: int) -> None:
        if self.exp_logger is None:
            return
        # Both backends escape HTML/text in the caller; clearml uses report_text, TB uses markdown.
        if self.config.logger == "clearml":
            self.exp_logger.report_text(f"[{tag}] Step {step}:\n{text}")
        else:
            self.exp_logger.writer.add_text(tag, text, global_step=step)

    def _log_inference(self, engine: Engine, loader: DataLoader, name: str) -> None:
        if not self.accelerator.is_main_process:
            return
        self.logger.info(f"Epoch {engine.state.epoch}: Running inference on {name} samples...")

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

        dataset = loader.dataset
        total_samples = len(dataset)  # type: ignore
        num_samples = min(self.inference_num_samples, total_samples)

        # Read generation prompts straight off the DatapointModel contract.
        # ``source``/``target`` are the tokenizer-facing strings (LSB-first for
        # this experiment), while ``original_source``/``original_target`` are
        # the conventional MSB-first strings used for human-readable logs.
        prompt_ids_list: list[torch.Tensor] = []
        targets: list[str] = []
        sources: list[str] = []
        reverse_for_display: list[bool] = []
        for i in range(num_samples):
            item = dataset[i]
            lsd_first = getattr(item, "digit_order", None) == "lsd_first"
            source = item.original_source if getattr(item, "original_source", None) is not None else item.source
            target = item.original_target if getattr(item, "original_target", None) is not None else item.target

            input_ids = item.eval_input_ids
            prompt_ids_list.append(input_ids)
            targets.append(target)
            sources.append(source)
            reverse_for_display.append(lsd_first)
        self.model.eval()

        # Left-pad the prompts (reverse -> pad_sequence -> reverse) so every
        # sequence in the batch ends at the same position and the newly
        # generated tokens line up in a single column.
        #
        # NOTE: this is safe for the default model because it uses rotary
        # (relative) positions, where shifting the whole prompt right by the
        # pad width preserves the distances between real tokens and the padding
        # is masked out of attention. If you swap in a model with absolute or
        # sinusoidal position embeddings indexed by slot, left-padding will
        # offset every real token's position and silently degrade generation --
        # you would then need to right-pad, or derive position_ids from the
        # attention mask instead of using the raw slot index.
        batch_input_ids = (
            torch.nn.utils.rnn.pad_sequence(
                [t.flip(0) for t in prompt_ids_list], batch_first=True, padding_value=pad_token_id
            )
            .flip(1)
            .to(self.device)
        )
        batch_attention_mask = (
            torch.nn.utils.rnn.pad_sequence(
                [torch.ones_like(t, dtype=torch.long).flip(0) for t in prompt_ids_list],
                batch_first=True,
                padding_value=0,
            )
            .flip(1)
            .to(self.device)
        )
        sequences = self.generate(
            input_ids=batch_input_ids,
            max_new_tokens=self.max_inference_new_tokens,
            attention_mask=batch_attention_mask,
        )
        new_tokens = sequences[:, batch_input_ids.shape[1] :]
        decoded_strs = [
            self.tokenizer.decode(
                tokens.tolist(),
                skip_special_tokens=True,
            )
            for tokens in new_tokens
        ]
        # Generated answer tokens are LSB-first; reverse them so the logged
        # prediction matches the conventional MSB-first source/target.
        decoded_strs = [
            prediction[::-1] if reverse else prediction
            for prediction, reverse in zip(decoded_strs, reverse_for_display)
        ]

        for idx, (source, target, prediction) in enumerate(zip(sources, targets, decoded_strs), 1):
            self.logger.info(
                "    Sample %d | Prompt: %r | Target: %r | Prediction: %r",
                idx,
                source,
                target,
                prediction,
            )

        lines = []
        for idx, (source, target, prediction) in enumerate(zip(sources, targets, decoded_strs), 1):
            lines.append(f"Sample {idx}")
            for key, val in [("Prompt", source), ("Target", target), ("Prediction", prediction)]:
                # Escape markup so TB markdown / clearml console don't swallow <, >, & in outputs.
                val = html.escape(val.strip() or "(empty)", quote=False)
                lines.append(f"  {key}:     {val}")
            lines.append("")
        name_map = {"Train": "training", "Val": "validation", "Test": "testing"}
        tag = f"inference/{name_map.get(name, name.lower())}"
        self._log_text(tag, "\n".join(lines), engine.state.iteration)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate text token IDs from prompt input_ids.

        Expects a pre-padded tensor (B, S). The caller is responsible for
        left-padding and providing an attention mask before calling this method.

        Args:
            input_ids: Prompt token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            attention_mask: Optional attention mask of shape (batch, seq_len).
        Returns:
            Tensor containing the full token IDs (prompt + newly generated tokens)
            of shape (batch, prompt_len + new_tokens).
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.eval()

        eos_id = self.tokenizer.eos_token_id

        device = self.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = unwrapped_model(generated, attention_mask=attention_mask)
            # Take the logits at the last position — this is the prediction for the next token
            next_token_logits = logits[:, -1, :]
            # Greedy decoding: pick the highest-probability token at each step
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if eos_id is not None:
                # If a sequence already ended (last token is EOS), keep emitting EOS
                # so the sequence stays frozen while other sequences in the batch finish
                eos_mask = generated[:, -1:].eq(eos_id)
                next_token = torch.where(eos_mask, torch.tensor(eos_id, device=device), next_token)
            generated = torch.cat([generated, next_token], dim=-1)

            # Extend the attention mask by one column for the newly appended token
            next_mask = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=device,
            )

            # Once a sequence has emitted EOS we keep re-emitting it (above) and
            # mask the appended token out of attention, so a finished sequence
            # stays frozen while the rest of the batch keeps generating.
            already_ended = generated[:, -2:-1].eq(eos_id)
            next_mask = torch.where(
                already_ended,
                torch.tensor(0, dtype=attention_mask.dtype, device=device),
                next_mask,
            )
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            # Stop early if every sequence in the batch has produced EOS
            if generated[:, -1].eq(eos_id).all():
                break

        return generated
