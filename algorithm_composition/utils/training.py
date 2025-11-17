"""Shared helpers for Hugging Face Trainer workflows."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import TrainerCallback, TrainingArguments

from models.llama_tiny_6L6H384 import build_nano_llama
from utils.noop_trainer import CallbackOnlyTrainer
from utils.tokenizer import SimpleCharTokenizer, build_tokenizer, encode_prompt_with_sep


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    output_dir: str
    learning_rate: float = 5e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    max_steps: int = 200_000
    eval_steps: int = 500
    logging_steps: int = 500


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


def make_compute_metrics(task_ids: Sequence[int], id_to_name: Dict[int, str]) -> callable:
    """Factory that returns a Hugging Face compatible compute_metrics fn."""

    task_ids = list(task_ids)

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred_ids = np.argmax(predictions, axis=-1)
        mask = label_ids != -100

        exact_hits: List[bool] = []
        token_hits = 0
        token_total = 0
        per_task_matches: Dict[int, List[bool]] = {task_id: [] for task_id in set(task_ids)}

        for idx in range(label_ids.shape[0]):
            valid_positions = mask[idx]
            total = int(valid_positions.sum())
            if total == 0:
                continue
            target = label_ids[idx][valid_positions]
            preds = pred_ids[idx][valid_positions]
            match = np.array_equal(target, preds)
            exact_hits.append(match)
            token_hits += int((target == preds).sum())
            token_total += total
            task_id = task_ids[idx]
            per_task_matches.setdefault(task_id, []).append(match)

        metrics = {
            "eval_exact": float(np.mean(exact_hits)) if exact_hits else 0.0,
            "eval_token_accuracy": token_hits / max(token_total, 1),
        }

        per_task_values = []
        for task_id, matches in per_task_matches.items():
            if not matches:
                continue
            name = id_to_name.get(task_id, str(task_id))
            value = float(np.mean(matches))
            metrics[f"eval_exact_{name}"] = value
            per_task_values.append(value)

        if per_task_values:
            metrics["eval_acc_min"] = min(per_task_values)
        else:
            metrics["eval_acc_min"] = metrics["eval_exact"]
        return metrics

    return compute_metrics


def build_model_from_tokenizer(tokenizer: SimpleCharTokenizer, context_length: int):
    return build_nano_llama(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=context_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def build_model_and_tokenizer(context_length: int = 256):
    tokenizer = build_tokenizer()
    model = build_model_from_tokenizer(tokenizer, context_length)
    return model, tokenizer


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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _list_checkpoint_steps(output_dir: str) -> List[int]:
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


def cleanup_checkpoints(output_dir: str, keep: int = 1) -> None:
    """Remove all but the most recent checkpoint directories."""

    if keep < 1:
        keep = 1
    steps = _list_checkpoint_steps(output_dir)
    if len(steps) <= keep:
        return

    # Keep the checkpoints with the highest step counts and delete the rest.
    survivors = set(steps[-keep:])
    for step in steps:
        if step in survivors:
            continue
        path = os.path.join(output_dir, f"checkpoint-{step}")
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def find_checkpoint_at_or_before(output_dir: str, target_step: int) -> str | None:
    """Return checkpoint path for the requested step, or the closest earlier one."""

    desired = os.path.join(output_dir, f"checkpoint-{target_step}")
    if os.path.isdir(desired):
        return desired

    steps = _list_checkpoint_steps(output_dir)
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


def _strip_after_eos(ids: List[int], eos_token_id: int) -> List[int]:
    trimmed: List[int] = []
    for token_id in ids:
        if token_id == eos_token_id:
            break
        trimmed.append(token_id)
    return trimmed


def greedy_autoregressive_eval(
    model,
    tokenizer: SimpleCharTokenizer,
    dataset,
    max_new_tokens: int,
    batch_size: int = 16,
    match_target_length: bool = False,
):
    """Run greedy decoding on the eval dataset and compute exact/token accuracies."""

    device = next(model.parameters()).device
    model.eval()
    exact_hits: List[bool] = []
    per_task_hits = defaultdict(list)
    token_hits = 0
    token_total = 0

    dataset_len = len(dataset)
    with torch.no_grad():
        for start in range(0, dataset_len, batch_size):
            end = min(dataset_len, start + batch_size)
            batch_items = [dataset.get_prompt_and_target(idx) for idx in range(start, end)]
            prompts = [encode_prompt_with_sep(tokenizer, item["prompt"]) for item in batch_items]
            prompt_lens = [len(p) for p in prompts]
            max_prompt = max(prompt_lens)
            batch = len(prompts)
            targets = [
                tokenizer.encode(item["target"], add_special_tokens=False) for item in batch_items
            ]
            batch_max_new_tokens = max_new_tokens
            if match_target_length and targets:
                batch_max_new_tokens = max(len(ids) for ids in targets)
                batch_max_new_tokens = max(1, batch_max_new_tokens)
            input_ids = torch.full(
                (batch, max_prompt), tokenizer.pad_token_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, ids in enumerate(prompts):
                length = len(ids)
                start_col = max_prompt - length
                input_ids[row, start_col:] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[row, start_col:] = 1

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=batch_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for row, item in enumerate(batch_items):
                gen_ids = generated[row].tolist()[max_prompt:]
                gen_ids = _strip_after_eos(gen_ids, tokenizer.eos_token_id)
                target_ids = targets[row]

                for pos, target_id in enumerate(target_ids):
                    token_total += 1
                    if pos < len(gen_ids) and gen_ids[pos] == target_id:
                        token_hits += 1
                is_exact = gen_ids == target_ids
                exact_hits.append(is_exact)
                per_task_hits[item["task"]].append(is_exact)

    metrics = {
        "eval_exact": float(np.mean(exact_hits)) if exact_hits else 0.0,
        "eval_token_accuracy": token_hits / max(token_total, 1),
    }
    per_task_scores = []
    for task, results in per_task_hits.items():
        if not results:
            continue
        score = float(np.mean(results))
        metrics[f"eval_exact_{task}"] = score
        per_task_scores.append(score)
    metrics["eval_acc_min"] = min(per_task_scores) if per_task_scores else metrics["eval_exact"]
    return metrics


class GreedyEvalCallback:
    """Callable wrapper that runs greedy autoregressive evaluation."""

    def __init__(
        self,
        eval_dataset,
        tokenizer: SimpleCharTokenizer,
        max_new_tokens: int,
        batch_size: int,
        match_target_length: bool = False,
    ) -> None:
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.match_target_length = match_target_length

    def __call__(self, model) -> Dict[str, float]:
        return greedy_autoregressive_eval(
            model=model,
            tokenizer=self.tokenizer,
            dataset=self.eval_dataset,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            match_target_length=self.match_target_length,
        )


def run_iterative_training_loop(
    *,
    model_builder: Callable[[], torch.nn.Module],
    initial_model: torch.nn.Module | None = None,
    train_dataset,
    eval_dataset,
    tokenizer: SimpleCharTokenizer,
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
    """Run training with iterative eval intervals, halving near the S99 threshold."""

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

            if (
                not branched_once
                and branch_count > 1
                and round_idx < refine_rounds
            ):
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
                    ensure_dir(branch_output_dir)
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
