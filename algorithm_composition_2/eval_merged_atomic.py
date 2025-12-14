#!/usr/bin/env python
"""Evaluate a checkpoint on atomic tasks A and B without further fine-tuning."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from data.simple_tasks import GeneratorConfig
from utils.datasets import SimpleDatasetConfig, SimpleTaskDataset
from utils.training import greedy_autoregressive_eval, write_json
from utils.tokenizer import SimpleCharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on tasks A and B.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to merged (or any) checkpoint.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for eval dataset generation (used if eval_data_seed is unset).")
    parser.add_argument("--eval_data_seed", type=int, default=None, help="Optional explicit seed for eval data.")
    parser.add_argument("--tasks", type=str, default="A,B", help="Comma-separated tasks to evaluate (subset of A,B).")
    parser.add_argument("--eval_samples", type=int, default=256, help="Number of eval examples per task.")
    parser.add_argument("--context_length", type=int, default=256, help="Max sequence length.")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for greedy eval.")
    parser.add_argument("--results_path", type=str, default=None, help="Where to write metrics JSON.")
    return parser.parse_args()


def build_eval_dataset(tokenizer: SimpleCharTokenizer, task: str, seed: int, samples: int, context_length: int):
    cfg = SimpleDatasetConfig(generator=GeneratorConfig(), max_length=context_length, dataset_size=samples)
    return SimpleTaskDataset(
        tasks=(task,),
        tokenizer=tokenizer,
        seed=seed,
        config=cfg,
        task_schedule=[task] * samples,
    )


def main() -> None:
    args = parse_args()
    eval_seed = args.eval_data_seed if args.eval_data_seed is not None else args.seed + 1
    tasks = [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    tasks = [t for t in tasks if t in {"A", "B"}]
    if not tasks:
        raise SystemExit("No valid tasks provided; expected some of A,B.")

    tokenizer = SimpleCharTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics_all: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        dataset = build_eval_dataset(
            tokenizer=tokenizer,
            task=task,
            seed=eval_seed,
            samples=args.eval_samples,
            context_length=args.context_length,
        )
        metrics = greedy_autoregressive_eval(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_new_tokens=args.context_length,
            batch_size=args.eval_batch_size,
            match_target_length=False,
        )
        metrics_all[task] = metrics
        print(f"Task {task}: exact={metrics['eval_exact']:.3f} token_acc={metrics['eval_token_accuracy']:.3f}")

    base = os.path.splitext(os.path.basename(args.checkpoint.rstrip('/')))[0]
    results_path = args.results_path or os.path.join("results", f"eval_atomic_{base}.json")
    payload = {
        "checkpoint": args.checkpoint,
        "tasks": tasks,
        "eval_data_seed": eval_seed,
        "eval_samples": args.eval_samples,
        "context_length": args.context_length,
        "eval_batch_size": args.eval_batch_size,
        "metrics": metrics_all,
    }
    write_json(results_path, payload)
    print(f"Wrote eval metrics to {results_path}")


if __name__ == "__main__":
    main()
