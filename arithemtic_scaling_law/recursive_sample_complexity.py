"""Simple training helpers for estimating sample complexity.

This module now runs a single vanilla training loop with periodic evaluations.
Training stops once the configured metric crosses a threshold, and the primary
result is the list of steps where the metric met or exceeded that threshold.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _ensure_logging_initialized() -> None:
    """Set up a basic INFO logger if the user hasn't configured logging."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


class CallbackOnlyTrainer(Trainer):
    """Trainer variant that delegates evaluation to a callable."""

    def __init__(
        self,
        *args,
        eval_metrics_fn: Callable[[Any], dict] | None = None,
        eval_repeats: int = 1,
        **kwargs,
    ):
        self.eval_metrics_fn = eval_metrics_fn
        self.eval_repeats = max(1, int(eval_repeats))
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        metrics: Dict[str, float] = {}
        if self.eval_metrics_fn is not None:
            totals: Dict[str, float] = {}
            for _ in range(self.eval_repeats):
                values = self.eval_metrics_fn(self.model) or {}
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        totals[key] = totals.get(key, 0.0) + float(value)
            if totals:
                metrics = {key: total / float(self.eval_repeats) for key, total in totals.items()}
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=0)

    def compute_loss(self, *args, **kwargs):  # pragma: no cover - not used during eval
        return super().compute_loss(*args, **kwargs)


class S99Callback(TrainerCallback):
    """Stops training once a metric crosses the threshold for the requested patience."""

    def __init__(self, metric_name: str, threshold: float = 0.99, patience: int = 1) -> None:
        self.metric_name = metric_name
        self.threshold = threshold
        self.patience = max(1, int(patience))
        self.best_step: int | None = None
        self.passed_steps: List[int] = []
        self._streak = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # noqa: D401
        value = metrics.get(self.metric_name)
        if value is None:
            self._streak = 0
            return
        if value >= self.threshold:
            if self.best_step is None:
                self.best_step = state.global_step
            self.passed_steps.append(state.global_step)
            self._streak += 1
            if self._streak >= self.patience:
                control.should_training_stop = True
        else:
            self._streak = 0


def configure_training_args(
    output_dir: str,
    per_device_batch_size: int,
    eval_batch_size: int,
    grad_accum: int,
    max_steps: int,
    eval_steps: int,
    logging_steps: int,
    warmup_steps: int = 600,
    save_strategy: str = "no",
    save_steps: int | None = None,
    save_total_limit: int = 1,
    scheduler_kwargs: Dict | None = None,
    eval_delay: int = 0,
) -> TrainingArguments:
    warmup_steps = max(0, int(warmup_steps))
    eval_delay = max(0, int(eval_delay))
    lr_scheduler_kwargs = dict(scheduler_kwargs or {})
    if "num_decay_steps" not in lr_scheduler_kwargs:
        lr_scheduler_kwargs["num_decay_steps"] = warmup_steps

    return TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=warmup_steps,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_delay=eval_delay,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_drop_last=False,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        use_liger_kernel=False,
        torch_compile=True
    )


def list_checkpoint_steps(output_dir: str) -> List[int]:
    """Return sorted checkpoint steps stored under the provided directory."""

    if not os.path.isdir(output_dir):
        return []
    steps: List[int] = []
    prefix = "checkpoint-"
    for entry in os.scandir(output_dir):
        if not entry.is_dir() or not entry.name.startswith(prefix):
            continue
        try:
            step = int(entry.name[len(prefix) :])
        except ValueError:
            continue
        steps.append(step)
    return sorted(steps)


def find_checkpoint_at_or_before(output_dir: str, target_step: int) -> str | None:
    """Return checkpoint path for the requested step, or the closest earlier one."""

    desired = os.path.join(output_dir, f"checkpoint-{target_step}")
    if os.path.isdir(desired):
        return desired

    steps = list_checkpoint_steps(output_dir)
    candidate = None
    for step in steps:
        if step <= target_step:
            candidate = step
        else:
            break
    if candidate is None:
        return None
    path = os.path.join(output_dir, f"checkpoint-{candidate}")
    return path if os.path.isdir(path) else None


def train_with_eval_threshold(
    *,
    model_builder: Callable[[], torch.nn.Module],
    initial_model: torch.nn.Module | None,
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizerBase,
    data_collator,
    greedy_eval_fn: Callable[[torch.nn.Module], Dict[str, float]],
    output_dir: str,
    per_device_batch_size: int,
    per_device_eval_batch_size: int,
    grad_accum: int,
    max_steps: int,
    eval_steps: int,
    metric_name: str,
    success_threshold: float = 0.99,
    warmup_steps: int = 600,
    logging_steps: int | None = None,
    eval_repeats: int = 1,
    patience: int = 1,
    scheduler_kwargs: Dict | None = None,
    eval_delay: int = 0,
) -> Tuple[CallbackOnlyTrainer, S99Callback, List[int]]:
    """Train with periodic evals and stop once the metric meets the threshold.

    Returns a tuple of (trainer, callback, threshold_steps) where threshold_steps
    is the list of global steps whose evaluations met or exceeded the threshold.
    """

    _ensure_logging_initialized()
    logging_steps = logging_steps if logging_steps is not None else eval_steps

    model = initial_model if initial_model is not None else model_builder()
    training_args = configure_training_args(
        output_dir=output_dir,
        per_device_batch_size=per_device_batch_size,
        eval_batch_size=per_device_eval_batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        eval_delay=eval_delay,
        save_strategy="no",
        save_steps=None,
        save_total_limit=1,
        scheduler_kwargs=scheduler_kwargs,
    )
    callback = S99Callback(metric_name=metric_name, threshold=success_threshold, patience=patience)
    trainer = CallbackOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[callback],
        eval_metrics_fn=greedy_eval_fn,
        eval_repeats=eval_repeats,
    )
    trainer.train()
    trainer.save_state()

    threshold_steps = list(callback.passed_steps)
    if threshold_steps:
        logger.info(
            "Threshold %.4f reached at steps: %s (best=%s)",
            success_threshold,
            threshold_steps,
            threshold_steps[0],
        )
    else:
        logger.info(
            "Threshold %.4f not reached within %s steps; best=%s",
            success_threshold,
            max_steps,
            callback.best_step,
        )
    return trainer, callback, threshold_steps


def measure_sample_complexity_with_recursive_rollback(
    *,
    model_builder: Callable[[], torch.nn.Module],
    initial_model: torch.nn.Module | None,
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizerBase,
    data_collator,
    greedy_eval_fn: Callable[[torch.nn.Module], Dict[str, float]],
    output_dir: str,
    per_device_batch_size: int,
    per_device_eval_batch_size: int,
    grad_accum: int,
    max_steps: int,
    initial_eval_steps: int,
    eval_refine_rounds: int,
    metric_name: str,
    resume_optimizer_state: bool = True,
    warmup_steps: int = 600,
    rollback_branches: int = 1,
    success_threshold: float = 0.99,
    eval_repeats: int = 1,
    patience: int = 1,
    scheduler_kwargs: Dict | None = None,
    logging_steps: int | None = None,
    eval_delay: int = 0,
    **_: Any,
) -> Tuple[CallbackOnlyTrainer, S99Callback, List[int]]:
    """Backwards-compatible shim that now uses a simple thresholded training loop.

    Legacy parameters related to rollback are ignored but accepted to avoid breaking
    existing callers. Warnings are emitted when non-default values are provided.
    """

    if eval_refine_rounds != 1 or rollback_branches != 1:
        logger.warning(
            "Recursive rollback is no longer supported; ignoring eval_refine_rounds=%s and rollback_branches=%s.",
            eval_refine_rounds,
            rollback_branches,
        )
    if not resume_optimizer_state:
        logger.warning("resume_optimizer_state is ignored in the simplified training loop.")

    return train_with_eval_threshold(
        model_builder=model_builder,
        initial_model=initial_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        greedy_eval_fn=greedy_eval_fn,
        output_dir=output_dir,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        eval_steps=initial_eval_steps,
        metric_name=metric_name,
        success_threshold=success_threshold,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps or initial_eval_steps,
        eval_repeats=eval_repeats,
        patience=patience,
        scheduler_kwargs=scheduler_kwargs,
        eval_delay=eval_delay,
    )


# Backwards compatibility alias for callers that expect the old name.
run_iterative_training_loop = measure_sample_complexity_with_recursive_rollback


__all__ = [
    "CallbackOnlyTrainer",
    "S99Callback",
    "configure_training_args",
    "list_checkpoint_steps",
    "train_with_eval_threshold",
    "measure_sample_complexity_with_recursive_rollback",
    "run_iterative_training_loop",
]
