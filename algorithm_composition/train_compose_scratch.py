"""Train composed task C=A∘B from scratch."""

from __future__ import annotations

import argparse
import os

from transformers import set_seed

from data.gen_reverse_regions import GeneratorConfig
from utils.cli import add_shared_training_args
from utils.datasets import (
    HeartbeatEvalDataset,
    ReverseDatasetConfig,
    ReverseTaskDataset,
    build_mixed_task_schedule,
)
from utils.collators import CausalLMDataCollator
from utils.training import (
    append_jsonl,
    build_model_and_tokenizer,
    build_model_from_tokenizer,
    cleanup_checkpoints,
    ensure_dir,
    GreedyEvalCallback,
    run_iterative_training_loop,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train composed task C from scratch.")
    parser.add_argument("--family", type=str, required=True, choices=["NC", "C"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/c_scratch")
    parser.add_argument("--results_dir", type=str, default="results")
    add_shared_training_args(parser)
    return parser.parse_args()


def build_eval_schedule(samples: int) -> list[str]:
    schedule = []
    while len(schedule) < samples:
        schedule.append("C")
    return schedule[:samples]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, f"{args.family}_seed{args.seed}")
    ensure_dir(output_dir)

    model, tokenizer = build_model_and_tokenizer(context_length=args.context_length)

    generator_cfg = GeneratorConfig()
    dataset_cfg = ReverseDatasetConfig(
        generator=generator_cfg,
        max_length=args.context_length,
        dataset_size=args.dataset_size,
    )

    mix_fraction = max(0.0, min(float(args.atomic_mix_fraction), 1.0))
    train_task_schedule = None
    train_tasks = ("C",)
    if mix_fraction > 0.0:
        train_task_schedule = build_mixed_task_schedule(
            dataset_size=args.dataset_size,
            primary_task="C",
            auxiliary_tasks=("A", "B"),
            auxiliary_fraction=mix_fraction,
            seed=args.seed,
        )
        train_tasks = ("C", "A", "B")

    train_dataset = ReverseTaskDataset(
        family=args.family,
        tasks=train_tasks,
        tokenizer=tokenizer,
        seed=args.seed,
        config=dataset_cfg,
        task_schedule=train_task_schedule,
    )

    greedy_eval_dataset = ReverseTaskDataset(
        family=args.family,
        tasks=("C",),
        tokenizer=tokenizer,
        seed=args.seed + 1,
        config=ReverseDatasetConfig(generator=generator_cfg, max_length=args.context_length),
        task_schedule=build_eval_schedule(args.eval_samples),
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

    model_builder = lambda: build_model_from_tokenizer(tokenizer, args.context_length)
    trainer, callback, round_history = run_iterative_training_loop(
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
        success_threshold=args.success_threshold,
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    cleanup_checkpoints(output_dir)

    final_best = callback.best_step if callback is not None else None
    s99_steps = final_best or trainer.args.max_steps
    record = {
        "family": args.family,
        "seed": args.seed,
        "phase": "scratch_C",
        "s99_steps": s99_steps,
        "round_history": round_history,
        "checkpoint": output_dir,
    }
    metrics_path = os.path.join(args.results_dir, f"{args.family}_seed{args.seed}_scratch.json")
    write_json(metrics_path, record)
    append_jsonl(os.path.join(args.results_dir, "runs.jsonl"), record)
    print(f"S99 (scratch C) reached at step {s99_steps}")


if __name__ == "__main__":
    main()
