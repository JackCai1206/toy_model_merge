"""Train a single atomic task (A or B) from scratch."""

from __future__ import annotations

import argparse
import os

from transformers import AutoModelForCausalLM, set_seed

from data.simple_tasks import GeneratorConfig
from utils.cli import add_shared_training_args
from utils.datasets import HeartbeatEvalDataset, SimpleDatasetConfig, SimpleTaskDataset
from utils.collators import CausalLMDataCollator
from utils.training import (
    append_jsonl,
    build_model_and_tokenizer,
    compute_eval_delay,
    cleanup_checkpoints,
    ensure_dir,
    GreedyEvalCallback,
    run_iterative_training_loop,
    write_json,
)
from utils.tokenizer import build_tokenizer, SimpleCharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train task A or B from scratch.")
    parser.add_argument("--task", type=str, required=True, choices=["A", "B"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/atomic")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        help="Optional checkpoint to initialize from. Defaults to a shared init per seed.",
    )
    add_shared_training_args(parser)
    return parser.parse_args()


def build_eval_schedule(task: str, samples: int) -> list[str]:
    schedule = []
    while len(schedule) < samples:
        schedule.append(task)
    return schedule[:samples]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_label = args.task.upper()
    output_dir = os.path.join(args.output_dir, f"{run_label}_seed{args.seed}")
    ensure_dir(output_dir)

    # Reuse an explicit shared initialization for both atomic tasks so task deltas are aligned.
    init_checkpoint = (
        args.init_checkpoint
        if args.init_checkpoint
        else os.path.join("artifacts", "init", f"init_seed{args.seed}")
    )
    ensure_dir(os.path.dirname(init_checkpoint))

    if os.path.isdir(init_checkpoint):
        tokenizer = SimpleCharTokenizer.from_pretrained(init_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(init_checkpoint)
    else:
        model, tokenizer = build_model_and_tokenizer(context_length=args.context_length)
        model.save_pretrained(init_checkpoint)
        tokenizer.save_pretrained(init_checkpoint)

    generator_cfg = GeneratorConfig()
    dataset_cfg = SimpleDatasetConfig(
        generator=generator_cfg,
        max_length=args.context_length,
        dataset_size=args.dataset_size,
    )

    train_dataset = SimpleTaskDataset(
        tasks=(run_label,),
        tokenizer=tokenizer,
        seed=args.seed,
        config=dataset_cfg,
    )

    eval_data_seed = args.eval_data_seed if args.eval_data_seed is not None else args.seed + 1

    greedy_eval_dataset = SimpleTaskDataset(
        tasks=(run_label,),
        tokenizer=tokenizer,
        seed=eval_data_seed,
        config=SimpleDatasetConfig(generator=generator_cfg, max_length=args.context_length),
        task_schedule=build_eval_schedule(run_label, args.eval_samples),
    )

    greedy_eval_max_new_tokens = args.greedy_eval_max_new_tokens or args.context_length
    greedy_eval = GreedyEvalCallback(
        eval_dataset=greedy_eval_dataset,
        tokenizer=tokenizer,
        max_new_tokens=greedy_eval_max_new_tokens,
        batch_size=args.greedy_eval_batch_size,
        match_target_length=args.greedy_eval_match_target_length,
    )
    data_collator = CausalLMDataCollator(tokenizer=tokenizer)
    heartbeat_eval_dataset = HeartbeatEvalDataset(tokenizer)

    eval_delay = compute_eval_delay(
        args.eval_steps,
        args.eval_jitter_fraction,
        args.seed,
        salt=sum(ord(ch) for ch in f"atomic_{run_label}"),
    )
    success_threshold = 2.0 if args.train_full_steps else args.success_threshold

    model_builder = lambda: build_model_from_tokenizer(tokenizer, args.context_length)
    trainer, callback, threshold_steps = run_iterative_training_loop(
        model_builder=model_builder,
        initial_model=model,
        train_dataset=train_dataset,
        eval_dataset=heartbeat_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        greedy_eval_fn=greedy_eval,
        output_dir=output_dir,
        per_device_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        initial_eval_steps=args.eval_steps,
        eval_refine_rounds=args.eval_refine_rounds,
        metric_name="eval_exact",
        rollback_branches=args.rollback_branches,
        success_threshold=success_threshold,
        eval_delay=eval_delay,
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    cleanup_checkpoints(output_dir)

    final_best = callback.best_step if callback is not None else None
    s99_steps = final_best or trainer.args.max_steps
    record = {
        "task": run_label,
        "seed": args.seed,
        "phase": "atomic",
        "s99_steps": s99_steps,
        "threshold_steps": threshold_steps,
        "checkpoint": output_dir,
        "eval_steps": args.eval_steps,
        "eval_delay": eval_delay,
        "eval_jitter_fraction": args.eval_jitter_fraction,
        "max_steps": args.max_steps,
        "train_full_steps": args.train_full_steps,
        "eval_data_seed": eval_data_seed,
    }
    metrics_path = os.path.join(args.results_dir, f"{run_label}_seed{args.seed}_atomic.json")
    write_json(metrics_path, record)
    append_jsonl(os.path.join(args.results_dir, "runs.jsonl"), record)
    print(f"S99 ({run_label}) reached at step {s99_steps}")


if __name__ == "__main__":
    main()
