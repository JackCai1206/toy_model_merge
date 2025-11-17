"""Command-line helpers shared across training scripts."""

from __future__ import annotations

import argparse


def add_shared_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the training hyper-parameters that all scripts share."""
    parser.add_argument("--dataset_size", type=int, default=200_000)
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument(
        "--atomic_mix_fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of training samples swapped with atomic tasks A/B when training composed "
            "families (ignored when set to 0)."
        ),
    )
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200_000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--greedy_eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--greedy_eval_max_new_tokens",
        type=int,
        default=None,
        help="Override greedy eval generation length (defaults to context_length).",
    )
    parser.add_argument(
        "--greedy_eval_match_target_length",
        action="store_true",
        help="Set greedy eval max_new_tokens per batch to the ground-truth target length.",
    )
    parser.add_argument("--eval_refine_rounds", type=int, default=4)
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.99,
        help="Metric threshold considered a successful run (default: 0.99).",
    )
    parser.add_argument(
        "--rollback_branches",
        type=int,
        default=1,
        help="Number of branches to spawn on the first rollback (set to 1 to disable).",
    )
    return parser


__all__ = ["add_shared_training_args"]
