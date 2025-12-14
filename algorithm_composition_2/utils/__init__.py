"""Utility modules for the composition experiments."""

from .cli import add_shared_training_args
from .collators import CausalLMDataCollator
from .datasets import (
    HeartbeatEvalDataset,
    SimpleDatasetConfig,
    SimpleTaskDataset,
    TASK_TO_ID,
    build_mixed_task_schedule,
)
from .tokenizer import SimpleCharTokenizer, build_tokenizer, encode_prompt_with_sep
from .training import (
    GreedyEvalCallback,
    append_jsonl,
    build_model_and_tokenizer,
    build_model_from_tokenizer,
    cleanup_checkpoints,
    ensure_dir,
    read_json,
    run_iterative_training_loop,
    write_json,
)

__all__ = [
    "add_shared_training_args",
    "append_jsonl",
    "build_mixed_task_schedule",
    "build_model_and_tokenizer",
    "build_model_from_tokenizer",
    "cleanup_checkpoints",
    "CausalLMDataCollator",
    "ensure_dir",
    "GreedyEvalCallback",
    "HeartbeatEvalDataset",
    "SimpleCharTokenizer",
    "SimpleDatasetConfig",
    "SimpleTaskDataset",
    "TASK_TO_ID",
    "build_tokenizer",
    "encode_prompt_with_sep",
    "read_json",
    "run_iterative_training_loop",
    "write_json",
]
