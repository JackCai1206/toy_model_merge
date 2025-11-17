"""Dataset utilities wrapping the synthetic generator for Trainer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from data.gen_reverse_regions import (
    GeneratorConfig,
    RegionSample,
    apply_task,
    generate_sample,
    tokens_to_text,
)
from utils.tokenizer import SimpleCharTokenizer, encode_prompt_with_sep


class HeartbeatEvalDataset(Dataset):
    """Single-sample dataset ensuring Trainer eval loops trigger callbacks."""

    def __init__(self, tokenizer: SimpleCharTokenizer) -> None:
        pad = tokenizer.pad_token_id
        self._sample = {
            "input_ids": torch.tensor([pad], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "labels": torch.tensor([-100], dtype=torch.long),
        }

    def __len__(self) -> int:  # pragma: no cover - trivial
        return 1

    def __getitem__(self, idx: int):  # pragma: no cover - deterministic
        return self._sample


TASK_TO_ID = {"A": 0, "B": 1, "C": 2}


def _encode_prompt_and_target(
    tokenizer: SimpleCharTokenizer, prompt: str, target: str, max_length: int
) -> Dict[str, torch.Tensor]:
    prompt_ids = encode_prompt_with_sep(tokenizer, prompt)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    target_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


@dataclass
class ReverseDatasetConfig:
    generator: GeneratorConfig
    max_length: int = 256
    dataset_size: int = 100_000


class ReverseTaskDataset(Dataset):
    """Map-style dataset built on top of the generator."""

    def __init__(
        self,
        family: str,
        tasks: Sequence[str],
        tokenizer: SimpleCharTokenizer,
        seed: int,
        config: ReverseDatasetConfig | None = None,
        task_schedule: Sequence[str] | None = None,
    ) -> None:
        self.family = family.upper()
        self.tasks = [task.upper() for task in tasks]
        self.tokenizer = tokenizer
        self.seed = seed
        self.config = config or ReverseDatasetConfig(generator=GeneratorConfig())
        self.task_schedule = (
            [task.upper() for task in task_schedule] if task_schedule is not None else None
        )
        if self.task_schedule:
            self._size = len(self.task_schedule)
        else:
            self._size = self.config.dataset_size

        self.task_ids = [TASK_TO_ID[task] for task in (self.task_schedule or self._default_schedule())]

    def _default_schedule(self) -> List[str]:
        schedule = []
        while len(schedule) < self._size:
            schedule.extend(self.tasks)
        return schedule[: self._size]

    def __len__(self) -> int:
        return self._size

    def _task_for_index(self, idx: int, rng: random.Random) -> str:
        if self.task_schedule:
            return self.task_schedule[idx]
        return self.tasks[idx % len(self.tasks)] if len(self.tasks) > 1 else self.tasks[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed * 9973 + idx * 17)
        max_len = self.config.max_length
        for _ in range(50):
            task = self._task_for_index(idx, rng)
            try:
                sample = generate_sample(rng, self.family, self.config.generator)
            except ValueError:
                continue
            target_tokens = apply_task(sample.tokens, sample.spans, task)
            encoded = self._format_example(sample, target_tokens, task)
            if max_len is None or max_len <= 0 or encoded["input_ids"].shape[0] <= max_len:
                return encoded
        raise RuntimeError(
            f"Failed to sample sequence under max_length={max_len} after multiple attempts."
        )

    def _format_example(self, sample: RegionSample, target_tokens: Sequence[str], task: str):
        input_text = tokens_to_text(sample.tokens)
        target_text = tokens_to_text(target_tokens)
        prompt = f"{sample.family}:{task}|{input_text}"
        encoded = _encode_prompt_and_target(
            tokenizer=self.tokenizer,
            prompt=prompt,
            target=target_text,
            max_length=self.config.max_length,
        )
        encoded["task_id"] = torch.tensor(TASK_TO_ID[task], dtype=torch.long)
        return encoded

    def get_prompt_and_target(self, idx: int) -> Dict[str, str]:
        """Deterministically regenerate prompt/target strings for a dataset example."""

        rng = random.Random(self.seed * 9973 + idx * 17)
        task = self._task_for_index(idx, rng)
        sample = generate_sample(rng, self.family, self.config.generator)
        target_tokens = apply_task(sample.tokens, sample.spans, task)
        input_text = tokens_to_text(sample.tokens)
        target_text = tokens_to_text(target_tokens)
        prompt = f"{sample.family}:{task}|{input_text}"
        return {
            "prompt": prompt,
            "target": target_text,
            "task": task,
            "task_id": TASK_TO_ID[task],
        }


def build_mixed_task_schedule(
    dataset_size: int,
    primary_task: str,
    auxiliary_tasks: Sequence[str],
    auxiliary_fraction: float,
    seed: int,
) -> List[str]:
    """Build a deterministic schedule mixing primary and auxiliary tasks.

    Args:
        dataset_size: Total number of samples required.
        primary_task: Task label that should occupy the remaining slots (e.g., "C").
        auxiliary_tasks: Iterable of task labels to interleave (e.g., ("A", "B")).
        auxiliary_fraction: Fraction of the dataset assigned to auxiliary tasks (0-1 range).
        seed: Random seed controlling how indices are selected.

    Returns:
        List of uppercase task labels with length ``dataset_size``.
    """

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    if not auxiliary_tasks:
        raise ValueError("auxiliary_tasks must be non-empty.")

    clamped_fraction = max(0.0, min(float(auxiliary_fraction), 1.0))
    if clamped_fraction <= 0.0:
        raise ValueError("auxiliary_fraction must be greater than zero to build a mix schedule.")

    primary = primary_task.upper()
    aux_tasks = [task.upper() for task in auxiliary_tasks]
    total_aux = int(round(dataset_size * clamped_fraction))
    total_aux = max(len(aux_tasks), total_aux)
    total_aux = min(dataset_size, total_aux)

    schedule: List[str] = [primary] * dataset_size
    rng = random.Random(seed * 7919 + dataset_size * 17)
    positions = rng.sample(range(dataset_size), total_aux)
    for idx, pos in enumerate(positions):
        schedule[pos] = aux_tasks[idx % len(aux_tasks)]
    return schedule
