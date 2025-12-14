"""Utilities for merging checkpoints trained on individual tasks."""

from __future__ import annotations

import json
import os
from typing import Dict, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, set_seed

from utils.tokenizer import SimpleCharTokenizer


def average_state_dicts(
    state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Average two compatible state dicts."""

    keys_a = set(state_a)
    keys_b = set(state_b)
    if keys_a != keys_b:
        missing_a = sorted(keys_b - keys_a)
        missing_b = sorted(keys_a - keys_b)
        raise ValueError(f"State dicts differ. Missing from A: {missing_a[:5]} Missing from B: {missing_b[:5]}")

    merged: Dict[str, torch.Tensor] = {}
    for key, tensor_a in state_a.items():
        tensor_b = state_b[key]
        if torch.is_floating_point(tensor_a):
            merged[key] = (tensor_a + tensor_b.to(tensor_a.device)) / 2.0
        else:
            # Keep non-floating values (e.g., buffers) from the first model.
            merged[key] = tensor_a
    return merged


def task_arithmetic_merge(
    base_state: Dict[str, torch.Tensor],
    deltas: Sequence[Dict[str, torch.Tensor]],
    delta_scale: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Combine multiple task deltas using task arithmetic.

    merged = base + scale * sum(delta_i) where delta_i = state_i - base.
    If delta_scale is None, scale defaults to 1 / len(deltas) (average delta).
    """

    if not deltas:
        raise ValueError("At least one delta state dict is required for task arithmetic merge.")
    scale = delta_scale if delta_scale is not None else 1.0 / len(deltas)
    merged: Dict[str, torch.Tensor] = {}
    keys_base = set(base_state)
    for delta in deltas:
        if set(delta) != keys_base:
            raise ValueError("State dict keys differ between base and delta states.")

    for key, base_tensor in base_state.items():
        if not torch.is_floating_point(base_tensor):
            merged[key] = base_tensor
            continue
        # Sum deltas for this tensor.
        delta_sum = None
        for delta in deltas:
            value = delta[key].to(base_tensor.device) - base_tensor
            delta_sum = value if delta_sum is None else delta_sum + value
        merged[key] = base_tensor + scale * delta_sum
    return merged


def _load_base_model(reference_checkpoint: str, base_checkpoint: str | None, base_seed: int | None):
    """Load or instantiate the base model used for task arithmetic."""

    if base_checkpoint:
        return AutoModelForCausalLM.from_pretrained(base_checkpoint)

    config = AutoConfig.from_pretrained(reference_checkpoint)
    if base_seed is not None:
        set_seed(base_seed)
    return AutoModelForCausalLM.from_config(config)


def merge_checkpoints(
    checkpoint_a: str,
    checkpoint_b: str,
    output_dir: str,
    *,
    base_checkpoint: str | None = None,
    base_seed: int | None = None,
    delta_scale: float | None = None,
    save_tokenizer: bool = True,
) -> str:
    """Merge two checkpoints using task arithmetic and persist the merged model."""

    os.makedirs(output_dir, exist_ok=True)
    base_model = _load_base_model(
        reference_checkpoint=checkpoint_a,
        base_checkpoint=base_checkpoint,
        base_seed=base_seed,
    )
    model_a = AutoModelForCausalLM.from_pretrained(checkpoint_a)
    model_b = AutoModelForCausalLM.from_pretrained(checkpoint_b)
    deltas = [model_a.state_dict(), model_b.state_dict()]
    merged_state = task_arithmetic_merge(
        base_state=base_model.state_dict(),
        deltas=deltas,
        delta_scale=delta_scale,
    )

    base_model.load_state_dict(merged_state)
    base_model.save_pretrained(output_dir)
    if save_tokenizer:
        try:
            tokenizer = SimpleCharTokenizer.from_pretrained(checkpoint_a)
            tokenizer.save_pretrained(output_dir)
        except Exception as exc:  # pragma: no cover - defensive
            # Fall back to copying files if a custom tokenizer class is not registered.
            tokenizer = SimpleCharTokenizer()
            tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "merge_tokenizer_warning.txt"), "w", encoding="utf-8") as handle:
                handle.write(f"Failed to load tokenizer from {checkpoint_a}: {exc}\n")

    metadata = {
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "tokenizer": checkpoint_a if save_tokenizer else None,
        "base_checkpoint": base_checkpoint,
        "base_seed": base_seed,
        "delta_scale": delta_scale if delta_scale is not None else f"1/{len(deltas)}",
        "method": "task_arithmetic",
    }
    with open(os.path.join(output_dir, "merge_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return output_dir


__all__ = ["average_state_dicts", "task_arithmetic_merge", "merge_checkpoints"]
