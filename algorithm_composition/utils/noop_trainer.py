"""Trainer subclass that skips built-in eval loops but keeps callbacks."""

from __future__ import annotations

import os
from typing import Any, Callable

from transformers import Trainer
from transformers.trainer import PREFIX_CHECKPOINT_DIR


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
        from transformers.trainer_utils import EvalLoopOutput

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
