"""Fine-tune task C starting from the multi-task A&B checkpoint."""

from __future__ import annotations

import argparse
import math
import os

from transformers import LlamaForCausalLM, set_seed

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
    cleanup_checkpoints,
    ensure_dir,
    GreedyEvalCallback,
    read_json,
    run_iterative_training_loop,
    write_json,
)
from utils.tokenizer import build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune composed task C from AB checkpoint.")
    parser.add_argument("--family", type=str, required=True, choices=["NC", "C"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the multi-task A&B checkpoint.",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/c_finetune")
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

    checkpoint = args.checkpoint or os.path.join("artifacts/ab_pretrain", f"shared_seed{args.seed}")
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    output_dir = os.path.join(args.output_dir, f"{args.family}_seed{args.seed}")
    ensure_dir(output_dir)

    model_builder = lambda: LlamaForCausalLM.from_pretrained(checkpoint)
    model = model_builder()
    tokenizer = build_tokenizer()

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

    scratch_path = os.path.join(args.results_dir, f"{args.family}_seed{args.seed}_scratch.json")
    if not os.path.exists(scratch_path):
        raise FileNotFoundError(
            f"Scratch metrics not found at {scratch_path}. "
            "Run train_compose_scratch.py for this seed first."
        )
    scratch_metrics = read_json(scratch_path)
    s99_scratch = scratch_metrics["s99_steps"]
    delta = math.log(s99_scratch) - math.log(s99_steps)

    record = {
        "family": args.family,
        "seed": args.seed,
        "phase": "finetune_C",
        "s99_steps": s99_steps,
        "s99_scratch": s99_scratch,
        "delta_ab": delta,
        "round_history": round_history,
        "checkpoint": output_dir,
        "pretrain_checkpoint": checkpoint,
    }
    metrics_path = os.path.join(args.results_dir, f"{args.family}_seed{args.seed}_finetune.json")
    write_json(metrics_path, record)
    append_jsonl(os.path.join(args.results_dir, "runs.jsonl"), record)
    append_jsonl(os.path.join(args.results_dir, "delta.jsonl"), record)
    print(f"S99 (fine-tune C) reached at step {s99_steps}, Δ_AB={delta:.4f}")


if __name__ == "__main__":
    main()
