"""Reusable utilities for measuring sample complexity via recursive rollback."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer import PREFIX_CHECKPOINT_DIR
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
    """Overrides evaluation hooks and lets callers pin checkpoints."""

    def __init__(
        self, *args, eval_metrics_fn: Callable[[Any], dict] | None = None, eval_repeats: int = 1, **kwargs
    ):
        self.eval_metrics_fn = eval_metrics_fn
        self.eval_repeats = max(1, int(eval_repeats))
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
            totals: Dict[str, float] = {}
            for _ in range(self.eval_repeats):
                values = self.eval_metrics_fn(self.model) or {}
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        totals[key] = totals.get(key, 0.0) + float(value)
            if totals:
                metrics = {key: total / float(self.eval_repeats) for key, total in totals.items()}
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
    warmup_steps: int = 600,
    save_strategy: str = "no",
    save_steps: int | None = None,
    save_total_limit: int = 1,
    scheduler_kwargs: Dict | None = None,
) -> TrainingArguments:
    warmup_steps = max(0, int(warmup_steps))
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
        dataloader_num_workers=8,
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


def _checkpoint_step(path: str | None) -> int:
    """Best-effort parse of the step number from a checkpoint directory."""

    if not path:
        return 0
    base = os.path.basename(os.path.normpath(path))
    if base.startswith("checkpoint-"):
        try:
            return int(base.split("-", maxsplit=1)[-1])
        except ValueError:
            return 0
    return 0


def _load_model_weights(model_builder: Callable[[], torch.nn.Module], checkpoint: str) -> torch.nn.Module:
    """Rebuild a model and load only its weights from a Trainer checkpoint."""

    model = model_builder()
    state_path = os.path.join(checkpoint, "pytorch_model.bin")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Model weights not found at {state_path}")
    state_dict = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            "While loading weights from %s: missing keys=%s unexpected=%s",
            checkpoint,
            missing,
            unexpected,
        )
    return model


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
) -> Tuple[CallbackOnlyTrainer, S99Callback, List[Dict[str, int | None]]]:
    """Run recursive rollback training to estimate sample complexity."""

    _ensure_logging_initialized()
    refine_rounds = max(1, eval_refine_rounds)
    branch_eval_steps = max(1, initial_eval_steps)
    segment_repeats = max(1, rollback_branches)
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
        trainer_for_branch: CallbackOnlyTrainer | None = None
        callback: S99Callback | None = S99Callback(metric_name=metric_name, threshold=success_threshold, patience=1)
        allow_fresh_start = False

        logger.info(
            "Starting branch %s | eval_interval=%s | repeats_per_interval=%s | rounds_completed=%s | resume=%s",
            state.branch_id,
            eval_interval,
            segment_repeats,
            round_idx,
            resume_checkpoint or "fresh",
        )

        branch_best_step: int | None = None

        while round_idx < refine_rounds:
            if round_idx > 0 and resume_checkpoint is None and not allow_fresh_start:
                raise RuntimeError(
                    f"Round {round_idx + 1} requested resume but no checkpoint was provided for branch {state.branch_id}."
                )
            allow_fresh_start = False
            if resume_checkpoint is not None and not os.path.isdir(resume_checkpoint):
                raise FileNotFoundError(
                    f"Resume checkpoint not found for branch {state.branch_id}: {resume_checkpoint}"
                )

            start_step = _checkpoint_step(resume_checkpoint)
            metrics_this_round: List[float | None] = []
            best_repeat_idx: int | None = None
            best_repeat_metric: float | None = None
            best_repeat_trainer: CallbackOnlyTrainer | None = None

            for repeat_idx in range(segment_repeats):
                repeat_output = os.path.join(
                    state.output_dir, f"branch{state.branch_id}_round{round_idx + 1}_rep{repeat_idx}"
                )
                # Ensure a clean slate for this repeat to avoid checkpoint collisions.
                if os.path.isdir(repeat_output):
                    shutil.rmtree(repeat_output)
                _ensure_dir(repeat_output)

                if repeat_idx > 0 or model is None:
                    if resume_checkpoint is not None and not resume_optimizer_state:
                        model = _load_model_weights(model_builder, resume_checkpoint)
                    else:
                        model = model_builder()

                training_args = configure_training_args(
                    output_dir=repeat_output,
                    per_device_batch_size=per_device_batch_size,
                    eval_batch_size=per_device_eval_batch_size,
                    grad_accum=grad_accum,
                    max_steps=start_step + eval_interval,
                    eval_steps=eval_interval,
                    logging_steps=eval_interval,
                    warmup_steps=warmup_steps,
                    save_strategy="no",
                    save_steps=None,
                    save_total_limit=1,
                )
                logger.info(
                    "Round %s/%s (branch %s, repeat %s/%s): train %s steps from %s",
                    round_idx + 1,
                    refine_rounds,
                    state.branch_id,
                    repeat_idx + 1,
                    segment_repeats,
                    eval_interval,
                    resume_checkpoint or "scratch",
                )
                trainer = CallbackOnlyTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=None,
                    callbacks=[],
                    eval_metrics_fn=None,
                    eval_repeats=1,
                )
                train_kwargs = {}
                if resume_checkpoint is not None and resume_optimizer_state:
                    train_kwargs["resume_from_checkpoint"] = resume_checkpoint
                elif resume_checkpoint is not None and not resume_optimizer_state:
                    # Keep step counting consistent with the checkpoint while resetting optimizer/scheduler.
                    trainer.state.global_step = start_step
                trainer.args.max_steps = start_step + eval_interval
                trainer.train(**train_kwargs)
                trainer.save_state()

                metrics = greedy_eval_fn(trainer.model) or {}
                metric_value = metrics.get(metric_name)
                metrics_this_round.append(metric_value if metric_value is None else float(metric_value))

                if best_repeat_metric is None or (
                    metric_value is not None and metric_value > best_repeat_metric
                ):
                    best_repeat_metric = float(metric_value) if metric_value is not None else None
                    best_repeat_idx = repeat_idx
                    best_repeat_trainer = trainer

            # Compute average over repeats (ignoring None values).
            valid_metrics = [m for m in metrics_this_round if m is not None]
            spread = (max(valid_metrics) - min(valid_metrics)) if len(valid_metrics) >= 2 else None
            if spread is not None and spread <= 1e-4:
                logger.warning(
                    "Repeat metrics are nearly identical (spread=%.4g) for branch %s round %s with %s repeats; "
                    "randomness may be insufficient.",
                    spread,
                    state.branch_id,
                    round_idx + 1,
                    segment_repeats,
                )
            avg_metric = sum(valid_metrics) / len(valid_metrics) if valid_metrics else None
            threshold_hit = avg_metric is not None and avg_metric >= success_threshold
            best_step = start_step + eval_interval if threshold_hit else None
            if threshold_hit:
                branch_best_step = best_step

            # Save the best repeat checkpoint under the main output_dir so later rounds can resume.
            target_checkpoint = os.path.join(state.output_dir, f"checkpoint-{start_step + eval_interval}")
            if best_repeat_trainer is None:
                logger.warning(
                    "No successful repeat produced metrics for branch %s round %s; cannot save checkpoint.",
                    state.branch_id,
                    round_idx + 1,
                )
            else:
                if os.path.isdir(target_checkpoint):
                    shutil.rmtree(target_checkpoint)
                # Save full Trainer checkpoint (model + optimizer/scheduler state) for precise rollback.
                best_repeat_trainer._save(target_checkpoint)
                if target_checkpoint not in state.pinned_checkpoints:
                    state.pinned_checkpoints.append(target_checkpoint)
                model = best_repeat_trainer.model
                resume_checkpoint = target_checkpoint
                best_repeat_trainer.args.output_dir = state.output_dir
                trainer_for_branch = best_repeat_trainer

            round_history.append(
                {
                    "round": round_idx + 1,
                    "eval_steps": eval_interval,
                    "best_step": best_step,
                    "branch": state.branch_id,
                    "avg_metric": avg_metric,
                    "segment_repeats": segment_repeats,
                    "best_repeat": best_repeat_idx,
                }
            )
            logger.info(
                "Completed round %s (branch %s): avg_metric=%s | best_step=%s (threshold=%s)",
                round_idx + 1,
                state.branch_id,
                f"{avg_metric:.4f}" if avg_metric is not None else "n/a",
                best_step if best_step is not None else "not reached",
                success_threshold,
            )
            round_idx += 1

            if best_step is None or round_idx >= refine_rounds:
                if best_step is None:
                    logger.info(
                        "Stopping branch %s after round %s: threshold not reached at interval %s.",
                        state.branch_id,
                        round_idx,
                        eval_interval,
                    )
                else:
                    logger.info(
                        "Branch %s finished refinement rounds; keeping best_step=%s.",
                        state.branch_id,
                        best_step,
                    )
                break

            previous_step = best_step - eval_interval
            resume_checkpoint = None
            actual_step: int
            if previous_step <= 0:
                actual_step = 0
                allow_fresh_start = True
                model = model_builder()
                logger.info(
                    "Branch %s: rolling back to step 0 state for next round (best_step=%s, interval=%s).",
                    state.branch_id,
                    best_step,
                    eval_interval,
                )
            else:
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
            eval_interval = max(1, eval_interval // 2)
            state.eval_interval = eval_interval
            model = None
            checkpoint_label = f"checkpoint-{actual_step}" if resume_checkpoint is not None else "step-0"
            logger.info(
                "Branch %s: refining further from %s; next eval_interval=%s",
                state.branch_id,
                checkpoint_label,
                eval_interval,
            )

        if trainer_for_branch is None or callback is None:
            continue

        # Update callback.best_step with the best step observed in this branch (if any).
        callback.best_step = branch_best_step
        s99_steps = callback.best_step if callback.best_step is not None else max_steps
        if best_steps is None or s99_steps < best_steps:
            best_steps = s99_steps
            best_trainer = trainer_for_branch
            best_callback = callback
            best_round_history = list(round_history)
            logger.info(
                "Updated best branch to %s with best_step=%s after %s rounds.",
                state.branch_id,
                s99_steps,
                len(round_history),
            )

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
