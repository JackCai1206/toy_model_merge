"""Reusable utilities for measuring sample complexity via recursive rollback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer import PREFIX_CHECKPOINT_DIR
from transformers.trainer_utils import EvalLoopOutput
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class CallbackOnlyTrainer(Trainer):
    """Overrides evaluation hooks and lets callers pin checkpoints."""

    def __init__(self, *args, eval_metrics_fn: Callable[[Any], dict] | None = None, **kwargs):
        self.eval_metrics_fn = eval_metrics_fn
        self._pinned_checkpoints: set[str] = set()
        super().__init__(*args, **kwargs)

    def pin_checkpoint(self, path: str) -> None:
        """Keeps a checkpoint directory from being pruned by save_total_limit."""
        if not path:
            return
        normalized = os.path.abspath(os.path.normpath(path))
        self._pinned_checkpoints.add(normalized)

    def _checkpoints_in_output_dir(self, output_dir: str | None) -> set[str]:
        if not output_dir or not self._pinned_checkpoints:
            return set()
        abs_output = os.path.abspath(os.path.normpath(output_dir))
        pinned: set[str] = set()
        for path in self._pinned_checkpoints:
            try:
                common = os.path.commonpath([abs_output, path])
            except ValueError:
                continue
            if common == abs_output:
                pinned.add(path)
        return pinned

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        # Immediately return an EvalLoopOutput so Trainer logging behaves normally,
        # but delegate metric computation to the provided callable.
        metrics = {}
        if self.eval_metrics_fn is not None:
            metrics = self.eval_metrics_fn(self.model) or {}
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=0)

    def compute_loss(self, *args, **kwargs):  # pragma: no cover - should never be called during eval
        return super().compute_loss(*args, **kwargs)

    def _sorted_checkpoints(
        self,
        output_dir: str | None = None,
        checkpoint_prefix: str = PREFIX_CHECKPOINT_DIR,
        use_mtime: bool = False,
    ) -> list[str]:
        checkpoints = super()._sorted_checkpoints(
            output_dir=output_dir, checkpoint_prefix=checkpoint_prefix, use_mtime=use_mtime
        )
        if not checkpoints or not self._pinned_checkpoints:
            return checkpoints
        pinned = self._checkpoints_in_output_dir(output_dir or self.args.output_dir)
        if not pinned:
            return checkpoints
        filtered: list[str] = []
        for path in checkpoints:
            normalized = os.path.abspath(os.path.normpath(path))
            if normalized in pinned:
                continue
            filtered.append(path)
        return filtered

    def compare_trainer_and_checkpoint_args(self, training_args, trainer_state):
        # When resuming we want to keep the caller-provided intervals instead of reusing the
        # checkpoint metadata, otherwise DefaultFlowCallback will continue to schedule work
        # using stale values (e.g. save/eval every 2k steps even after we halve the interval).
        for attr in ("logging_steps", "eval_steps", "save_steps"):
            new_value = getattr(training_args, attr, None)
            if new_value is not None:
                setattr(trainer_state, attr, new_value)
        return super().compare_trainer_and_checkpoint_args(training_args, trainer_state)


@dataclass
class BranchState:
    branch_id: int
    output_dir: str
    resume_checkpoint: str | None
    eval_interval: int
    rounds_completed: int
    round_history: List[Dict[str, int | None]]
    model: torch.nn.Module | None = None
    pinned_checkpoints: List[str] = field(default_factory=list)


class S99Callback(TrainerCallback):
    """Stops training once accuracy ≥ threshold for patience evaluations."""

    def __init__(self, metric_name: str, threshold: float = 0.99, patience: int = 5) -> None:
        self.metric_name = metric_name
        self.threshold = threshold
        self.patience = patience
        self.best_step: int | None = None
        self._streak = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # noqa: D401
        value = metrics.get(self.metric_name)
        if value is None:
            self._streak = 0
            return
        if value >= self.threshold:
            if self.best_step is None:
                self.best_step = state.global_step
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
    save_strategy: str = "no",
    save_steps: int | None = None,
    save_total_limit: int = 1,
    scheduler_kwargs: Dict | None = None,
) -> TrainingArguments:
    warmup_steps = 2000
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
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_drop_last=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    rollback_branches: int = 1,
    success_threshold: float = 0.99,
) -> Tuple[CallbackOnlyTrainer, S99Callback, List[Dict[str, int | None]]]:
    """Run recursive rollback training to estimate sample complexity."""

    refine_rounds = max(1, eval_refine_rounds)
    branch_eval_steps = max(1, initial_eval_steps)
    branch_count = max(1, rollback_branches)
    branch_queue: List[BranchState] = [
        BranchState(
            branch_id=0,
            output_dir=output_dir,
            resume_checkpoint=None,
            eval_interval=branch_eval_steps,
            rounds_completed=0,
            round_history=[],
            model=initial_model,
        )
    ]
    branched_once = False
    next_branch_id = 0
    best_trainer: CallbackOnlyTrainer | None = None
    best_callback: S99Callback | None = None
    best_round_history: List[Dict[str, int | None]] = []
    best_steps: int | None = None

    while branch_queue:
        state = branch_queue.pop(0)
        model = state.model
        eval_interval = state.eval_interval
        resume_checkpoint = state.resume_checkpoint
        round_idx = state.rounds_completed
        round_history = list(state.round_history)
        trainer: CallbackOnlyTrainer | None = None
        callback: S99Callback | None = None
        split_state = False

        while round_idx < refine_rounds:
            if model is None:
                model = model_builder()
            training_args = configure_training_args(
                output_dir=state.output_dir,
                per_device_batch_size=per_device_batch_size,
                eval_batch_size=per_device_eval_batch_size,
                grad_accum=grad_accum,
                max_steps=max_steps,
                eval_steps=eval_interval,
                logging_steps=eval_interval,
                save_strategy="steps",
                save_steps=eval_interval,
                save_total_limit=2,
            )
            callback = S99Callback(metric_name=metric_name, threshold=success_threshold, patience=1)
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
            )
            if state.pinned_checkpoints:
                for checkpoint_path in state.pinned_checkpoints:
                    trainer.pin_checkpoint(checkpoint_path)
            train_kwargs = {}
            if resume_checkpoint is not None:
                train_kwargs["resume_from_checkpoint"] = resume_checkpoint
            trainer.train(**train_kwargs)
            trainer.save_state()
            best_step = callback.best_step
            round_history.append(
                {
                    "round": round_idx + 1,
                    "eval_steps": eval_interval,
                    "best_step": best_step,
                    "branch": state.branch_id,
                }
            )
            round_idx += 1
            if best_step is None or round_idx >= refine_rounds:
                break

            previous_step = best_step - eval_interval
            if previous_step <= 0:
                break
            resume_checkpoint = find_checkpoint_at_or_before(state.output_dir, previous_step)
            if resume_checkpoint is None:
                raise FileNotFoundError(
                    f"Checkpoint not found at {os.path.join(state.output_dir, f'checkpoint-{previous_step}')}"
                )
            try:
                actual_step = int(os.path.basename(resume_checkpoint).split("-", maxsplit=1)[-1])
            except ValueError:
                actual_step = previous_step
            if actual_step != previous_step:
                logger.warning(
                    "Falling back to checkpoint %s (requested step %s).", resume_checkpoint, previous_step
                )
            if resume_checkpoint not in state.pinned_checkpoints:
                state.pinned_checkpoints.append(resume_checkpoint)
            eval_interval = max(1, eval_interval // 2)
            state.eval_interval = eval_interval
            model = None

            if not branched_once and branch_count > 1 and round_idx < refine_rounds:
                pinned_checkpoints = list(state.pinned_checkpoints)
                if resume_checkpoint not in pinned_checkpoints:
                    pinned_checkpoints.append(resume_checkpoint)
                branched_once = True
                parent_history = list(round_history)
                branch_states: List[BranchState] = [
                    BranchState(
                        branch_id=state.branch_id,
                        output_dir=state.output_dir,
                        resume_checkpoint=resume_checkpoint,
                        eval_interval=eval_interval,
                        rounds_completed=round_idx,
                        round_history=parent_history.copy(),
                        pinned_checkpoints=pinned_checkpoints,
                    )
                ]
                for _ in range(branch_count - 1):
                    next_branch_id += 1
                    branch_output_dir = f"{output_dir}_branch{next_branch_id}"
                    _ensure_dir(branch_output_dir)
                    branch_states.append(
                        BranchState(
                            branch_id=next_branch_id,
                            output_dir=branch_output_dir,
                            resume_checkpoint=resume_checkpoint,
                            eval_interval=eval_interval,
                            rounds_completed=round_idx,
                            round_history=parent_history.copy(),
                            pinned_checkpoints=[],
                        )
                    )
                branch_queue = branch_states + branch_queue
                split_state = True
                break

        if split_state:
            continue
        if trainer is None or callback is None:
            continue

        s99_steps = callback.best_step if callback.best_step is not None else trainer.args.max_steps
        if best_steps is None or s99_steps < best_steps:
            best_steps = s99_steps
            best_trainer = trainer
            best_callback = callback
            best_round_history = list(round_history)

    if best_trainer is None or best_callback is None:
        raise RuntimeError("Iterative training failed to initialize Trainer.")
    return best_trainer, best_callback, best_round_history


# Backwards compatibility alias for callers that expect the old name.
run_iterative_training_loop = measure_sample_complexity_with_recursive_rollback


__all__ = [
    "CallbackOnlyTrainer",
    "S99Callback",
    "BranchState",
    "configure_training_args",
    "list_checkpoint_steps",
    "find_checkpoint_at_or_before",
    "measure_sample_complexity_with_recursive_rollback",
    "run_iterative_training_loop",
]
