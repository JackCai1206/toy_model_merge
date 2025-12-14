"""Fine-tune composed task C starting from a provided checkpoint."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

from transformers import AutoModelForCausalLM, set_seed

from data.simple_tasks import GeneratorConfig
from utils.cli import add_shared_training_args
from utils.datasets import (
    HeartbeatEvalDataset,
    SimpleDatasetConfig,
    SimpleTaskDataset,
    build_mixed_task_schedule,
)
from utils.collators import CausalLMDataCollator
from utils.training import (
    append_jsonl,
    compute_eval_delay,
    cleanup_checkpoints,
    ensure_dir,
    GreedyEvalCallback,
    run_iterative_training_loop,
    write_json,
)
from utils.tokenizer import build_tokenizer


def build_eval_schedule(samples: int) -> List[str]:
    schedule = []
    while len(schedule) < samples:
        schedule.append("C")
    return schedule[:samples]


def _extract_eval_history(trainer) -> List[Dict]:
    """Collect eval_* metrics with their step from Trainer log history."""

    history: List[Dict] = []
    for entry in getattr(trainer.state, "log_history", []) or []:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step") or entry.get("global_step")
        if step is None:
            continue
        metrics = {
            key: float(val)
            for key, val in entry.items()
            if key.startswith("eval_") and isinstance(val, (int, float))
        }
        if not metrics:
            continue
        metrics["step"] = int(step)
        history.append(metrics)
    history.sort(key=lambda item: item["step"])
    return history


def run_finetune(
    *,
    seed: int,
    init_checkpoint: str,
    output_dir: str,
    results_dir: str,
    context_length: int,
    dataset_size: int,
    eval_samples: int,
    per_device_batch_size: int,
    per_device_eval_batch_size: int,
    grad_accum: int,
    max_steps: int,
    eval_steps: int,
    eval_refine_rounds: int,
    rollback_branches: int,
    success_threshold: float,
    greedy_eval_batch_size: int,
    greedy_eval_max_new_tokens: int | None,
    greedy_eval_match_target_length: bool,
    atomic_mix_fraction: float = 0.0,
    run_label: str = "finetune",
    eval_jitter_fraction: float = 0.0,
    train_full_steps: bool = False,
    merge_scale: float | None = None,
    eval_data_seed: int | None = None,
) -> Dict:
    """Run the composed-task fine-tuning loop and return metrics."""

    if not os.path.isdir(init_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {init_checkpoint}")

    ensure_dir(output_dir)
    tokenizer = build_tokenizer()

    generator_cfg = GeneratorConfig()
    dataset_cfg = SimpleDatasetConfig(
        generator=generator_cfg,
        max_length=context_length,
        dataset_size=dataset_size,
    )

    mix_fraction = max(0.0, min(float(atomic_mix_fraction), 1.0))
    train_task_schedule = None
    train_tasks: List[str] = ["C"]
    if mix_fraction > 0.0:
        train_task_schedule = build_mixed_task_schedule(
            dataset_size=dataset_size,
            primary_task="C",
            auxiliary_tasks=("A", "B"),
            auxiliary_fraction=mix_fraction,
            seed=seed,
        )
        train_tasks = ["C", "A", "B"]

    train_dataset = SimpleTaskDataset(
        tasks=train_tasks,
        tokenizer=tokenizer,
        seed=seed,
        config=dataset_cfg,
        task_schedule=train_task_schedule,
    )

    eval_seed = eval_data_seed if eval_data_seed is not None else (seed + 1)
    greedy_eval_dataset = SimpleTaskDataset(
        tasks=("C",),
        tokenizer=tokenizer,
        seed=eval_seed,
        config=SimpleDatasetConfig(generator=generator_cfg, max_length=context_length),
        task_schedule=build_eval_schedule(eval_samples),
    )

    greedy_eval_max_new_tokens = greedy_eval_max_new_tokens or context_length
    greedy_eval = GreedyEvalCallback(
        eval_dataset=greedy_eval_dataset,
        tokenizer=tokenizer,
        max_new_tokens=greedy_eval_max_new_tokens,
        batch_size=greedy_eval_batch_size,
        match_target_length=greedy_eval_match_target_length,
    )
    data_collator = CausalLMDataCollator(tokenizer=tokenizer)
    heartbeat_eval_dataset = HeartbeatEvalDataset(tokenizer)

    eval_delay = compute_eval_delay(
        eval_steps,
        eval_jitter_fraction,
        seed,
        salt=sum(ord(ch) for ch in run_label),
    )
    success_threshold = 2.0 if train_full_steps else success_threshold

    model_builder = lambda: AutoModelForCausalLM.from_pretrained(init_checkpoint)
    model = model_builder()
    tokenizer.save_pretrained(output_dir)

    trainer, callback, threshold_steps = run_iterative_training_loop(
        model_builder=model_builder,
        initial_model=model,
        train_dataset=train_dataset,
        eval_dataset=heartbeat_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        greedy_eval_fn=greedy_eval,
        output_dir=output_dir,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        initial_eval_steps=eval_steps,
        eval_refine_rounds=eval_refine_rounds,
        metric_name="eval_exact",
        rollback_branches=rollback_branches,
        success_threshold=success_threshold,
        eval_delay=eval_delay,
    )

    trainer.save_model(output_dir)
    cleanup_checkpoints(output_dir)
    eval_history = _extract_eval_history(trainer)

    final_best = callback.best_step if callback is not None else None
    s99_steps = final_best or trainer.args.max_steps
    record = {
        "seed": seed,
        "phase": run_label,
        "s99_steps": s99_steps,
        "threshold_steps": threshold_steps,
        "checkpoint": output_dir,
        "init_checkpoint": init_checkpoint,
        "atomic_mix_fraction": mix_fraction,
        "eval_steps": eval_steps,
        "eval_delay": eval_delay,
        "eval_jitter_fraction": eval_jitter_fraction,
        "max_steps": max_steps,
        "eval_history": eval_history,
        "train_full_steps": train_full_steps,
        "merge_scale": merge_scale,
        "eval_data_seed": eval_seed,
    }
    metrics_path = os.path.join(results_dir, f"{run_label}_seed{seed}_finetune.json")
    write_json(metrics_path, record)
    append_jsonl(os.path.join(results_dir, "runs.jsonl"), record)
    print(f"S99 ({run_label}) reached at step {s99_steps}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune composed task C from a checkpoint.")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--run_label", type=str, default="finetune")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/finetune")
    parser.add_argument("--results_dir", type=str, default="results")
    add_shared_training_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_finetune(
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
        output_dir=os.path.join(args.output_dir, f"{args.run_label}_seed{args.seed}"),
        results_dir=args.results_dir,
        context_length=args.context_length,
        dataset_size=args.dataset_size,
        eval_samples=args.eval_samples,
        per_device_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        eval_refine_rounds=args.eval_refine_rounds,
        rollback_branches=args.rollback_branches,
        success_threshold=args.success_threshold,
        greedy_eval_batch_size=args.greedy_eval_batch_size,
        greedy_eval_max_new_tokens=args.greedy_eval_max_new_tokens,
        greedy_eval_match_target_length=args.greedy_eval_match_target_length,
        atomic_mix_fraction=args.atomic_mix_fraction,
        run_label=args.run_label,
        eval_jitter_fraction=args.eval_jitter_fraction,
        train_full_steps=args.train_full_steps,
        eval_data_seed=args.eval_data_seed,
    )


if __name__ == "__main__":
    main()
