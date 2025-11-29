"""Shared helpers for Hugging Face Trainer workflows."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arithemtic_scaling_law import (
    CallbackOnlyTrainer,
    S99Callback,
    configure_training_args,
    find_checkpoint_at_or_before,
    list_checkpoint_steps,
    measure_sample_complexity_with_recursive_rollback,
)
from algorithm_composition.models.llama_tiny_6L6H384 import build_nano_llama
from algorithm_composition.utils.tokenizer import SimpleCharTokenizer, build_tokenizer, encode_prompt_with_sep


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


def cleanup_checkpoints(output_dir: str, keep: int = 1) -> None:
    """Remove checkpoint-* directories, keeping at most `keep` most recent."""

    if keep < 0:
        keep = 0
    steps = list_checkpoint_steps(output_dir)
    if keep == 0:
        # Remove all checkpoints.
        for step in steps:
            path = os.path.join(output_dir, f"checkpoint-{step}")
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        return

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
    resume_optimizer_state: bool = True,
    rollback_branches: int = 1,
    success_threshold: float = 0.99,
) -> Tuple[CallbackOnlyTrainer, S99Callback, List[Dict[str, int | None]]]:
    """Delegate to the shared recursive rollback runner in arithemtic_scaling_law."""

    return measure_sample_complexity_with_recursive_rollback(
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
        initial_eval_steps=initial_eval_steps,
        eval_refine_rounds=eval_refine_rounds,
        metric_name=metric_name,
        resume_optimizer_state=resume_optimizer_state,
        rollback_branches=rollback_branches,
        success_threshold=success_threshold,
    )
