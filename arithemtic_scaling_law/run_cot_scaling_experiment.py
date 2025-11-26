"""Scaling-law driver for collapsed CoT arithmetic expressions.

Implements the experiment loop described in the spec:
* Generate Train/Val/Test pools for each complexity level k under different CoT regimes.
* Pretrain models on levels ≤k for each regime.
* Measure sample complexity for k→k+1 by fine-tuning with varying N and logging accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from transformers import LlamaForCausalLM, set_seed

from arithemtic_scaling_law.generate_bracketed_cot import MODULUS, generate_dataset, generate_example
from arithemtic_scaling_law.recursive_sample_complexity import (
    measure_sample_complexity_with_recursive_rollback,
)
from algorithm_composition.utils.collators import CausalLMDataCollator
from algorithm_composition.utils.tokenizer import SimpleCharTokenizer, build_tokenizer, encode_prompt_with_sep
from algorithm_composition.utils.training import (
    build_model_from_tokenizer,
    cleanup_checkpoints,
    ensure_dir,
    write_json,
)


# -----------------------------
# Dataset utilities
# -----------------------------


def _encode_prompt_and_target(
    tokenizer: SimpleCharTokenizer, prompt: str, target: str, max_length: int | None
) -> Dict[str, torch.Tensor]:
    prompt_ids = encode_prompt_with_sep(tokenizer, prompt)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    target_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + target_ids
    if max_length and max_length > 0 and len(input_ids) > max_length:
        raise ValueError(
            f"Sequence length {len(input_ids)} exceeds max_length={max_length} for prompt '{prompt}'."
        )

    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _strip_after_eos(ids: List[int], eos_token_id: int) -> List[int]:
    trimmed: List[int] = []
    for token_id in ids:
        if token_id == eos_token_id:
            break
        trimmed.append(token_id)
    return trimmed


def _parse_answer(text: str) -> int | None:
    match = re.findall(r"-?\d+", text)
    if not match:
        return None
    return int(match[-1]) % MODULUS


class ArithmeticCoTDataset(Dataset):
    """Map-style dataset for bracketed arithmetic with collapsed CoT."""

    def __init__(
        self,
        path: str,
        tokenizer: SimpleCharTokenizer,
        max_length: int,
        levels: Iterable[int] | None = None,
        sample_size: int | None = None,
        seed: int = 0,
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")

        with open(path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]

        level_set = set(levels) if levels is not None else None
        filtered = [rec for rec in records if level_set is None or rec.get("complexity_k") in level_set]
        if sample_size is not None and sample_size > 0 and len(filtered) > sample_size:
            rng = random.Random(seed * 7919 + len(filtered))
            indices = rng.sample(range(len(filtered)), sample_size)
            filtered = [filtered[idx] for idx in indices]

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: List[Dict] = filtered
        self.formatted: List[Dict] = [self._format_record(rec) for rec in self.records]

    def _format_record(self, record: Dict) -> Dict:
        expr = record["expression"].replace(" ", "")
        steps = record.get("visible_cot", [])

        def _clean_step(step: str) -> str:
            # Strip "Step X:" prefix and all whitespace to minimize token count.
            step_no_prefix = re.sub(r"(?i)step\s*\d+:\s*", "", step)
            step_no_compute = re.sub(r"(?i)compute\s*", "", step_no_prefix)
            step_no_mod = re.sub(r"(?i)\(mod\s*\d+\)", "", step_no_compute)
            step_compact = re.sub(r"\s+", "", step_no_mod)
            return step_compact.strip(".")

        cot_text = "|".join(_clean_step(step).upper() for step in steps if step)
        cot_part = f"C{cot_text}" if cot_text else "C"
        prompt = f"E{expr}".upper()
        target = f"{cot_part}|A{record['answer']}".upper()
        return {
            "prompt": prompt,
            "target": target,
            "answer": int(record["answer"]) % MODULUS,
            "complexity_k": int(record["complexity_k"]),
        }

    def __len__(self) -> int:
        return len(self.formatted)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.formatted[idx]
        encoded = _encode_prompt_and_target(
            tokenizer=self.tokenizer,
            prompt=item["prompt"],
            target=item["target"],
            max_length=self.max_length,
        )
        encoded["answer"] = torch.tensor(item["answer"], dtype=torch.long)
        encoded["complexity_k"] = torch.tensor(item["complexity_k"], dtype=torch.long)
        return encoded

    def get_prompt_and_target(self, idx: int) -> Dict[str, str | int]:
        item = self.formatted[idx]
        return {
            "prompt": item["prompt"],
            "target": item["target"],
            "answer": item["answer"],
            "complexity_k": item["complexity_k"],
        }


class OnlineArithmeticDataset(IterableDataset):
    """Freshly generates samples every epoch for the requested levels."""

    def __init__(
        self,
        *,
        levels: Iterable[int],
        tokenizer: SimpleCharTokenizer,
        max_length: int,
        examples_per_epoch: int,
        regime: Regime,
        seed: int,
    ) -> None:
        self.levels = list(levels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples_per_epoch = examples_per_epoch
        self.regime = regime
        self.seed = seed

    def __iter__(self):
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        offset = worker.id if worker else 0
        rng = random.Random(self.seed + 7919 * offset)
        for _ in range(self.examples_per_epoch):
            encoded, _ = self._sample_example(rng)
            yield encoded

    def __len__(self) -> int:
        return self.examples_per_epoch

    def get_prompt_and_target(self, idx: int) -> Dict[str, str | int]:
        """Deterministically generate a single example for evaluation."""
        _, text_fields = self._sample_example(random.Random(self.seed + idx))
        return text_fields

    def _sample_example(
        self, rng: random.Random
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, str | int]]:
        while True:
            k = rng.choice(self.levels)
            example = generate_example(
                k=k,
                rng=rng,
                q_keep=self.regime.q_keep,
                max_block_size=self.regime.max_block_size,
            )
            expr = example["expression"].replace(" ", "")

            def _clean_step(step: str) -> str:
                step_no_prefix = re.sub(r"(?i)step\\s*\\d+:\\s*", "", step)
                step_no_compute = re.sub(r"(?i)compute\\s*", "", step_no_prefix)
                step_no_mod = re.sub(r"(?i)\\(mod\\s*\\d+\\)", "", step_no_compute)
                step_compact = re.sub(r"\\s+", "", step_no_mod)
                return step_compact.strip(".")

            cot_text = "|".join(_clean_step(step).upper() for step in example["visible_cot"] if step)
            cot_part = f"C{cot_text}" if cot_text else "C"
            prompt = f"E{expr}".upper()
            target = f"{cot_part}|A{example['answer']}".upper()
            try:
                encoded = _encode_prompt_and_target(
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    target=target,
                    max_length=self.max_length,
                )
            except ValueError:
                # Occasionally a sampled prompt may exceed the desired max_length; resample.
                continue

            encoded["answer"] = torch.tensor(int(example["answer"]) % MODULUS, dtype=torch.long)
            encoded["complexity_k"] = torch.tensor(k, dtype=torch.long)
            text_fields = {
                "prompt": prompt,
                "target": target,
                "answer": int(example["answer"]) % MODULUS,
                "complexity_k": k,
            }
            return encoded, text_fields


def greedy_eval_arithmetic(
    model,
    tokenizer: SimpleCharTokenizer,
    dataset: ArithmeticCoTDataset,
    max_new_tokens: int,
    batch_size: int,
    match_target_length: bool = False,
) -> Dict[str, float]:
    """Greedy decoding evaluation returning expression-level accuracy."""

    device = next(model.parameters()).device
    model.eval()
    total = len(dataset)
    correct = 0
    token_hits = 0
    token_total = 0
    exact_matches = 0

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            batch_items = [dataset.get_prompt_and_target(idx) for idx in range(start, end)]
            prompts = [encode_prompt_with_sep(tokenizer, item["prompt"]) for item in batch_items]
            prompt_lens = [len(p) for p in prompts]
            max_prompt = max(prompt_lens)
            batch = len(prompts)

            input_ids = torch.full(
                (batch, max_prompt), tokenizer.pad_token_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, ids in enumerate(prompts):
                length = len(ids)
                # Left-pad to avoid right-padding warnings for decoder-only generation.
                offset = max_prompt - length
                input_ids[row, offset:] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[row, offset:] = 1

            gen_max = max_new_tokens
            if match_target_length:
                target_lengths = [
                    len(tokenizer.encode(item["target"], add_special_tokens=False)) for item in batch_items
                ]
                gen_max = max(gen_max, max(target_lengths, default=0))

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            for row, item in enumerate(batch_items):
                prompt_len = prompt_lens[row]
                gen_ids = generated[row].tolist()[prompt_len:]
                gen_ids = _strip_after_eos(gen_ids, tokenizer.eos_token_id)
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred_answer = _parse_answer(gen_text)
                if pred_answer is not None and pred_answer == item["answer"]:
                    correct += 1
                target_ids = tokenizer.encode(item["target"], add_special_tokens=False)
                for pos, target_id in enumerate(target_ids):
                    token_total += 1
                    if pos < len(gen_ids) and gen_ids[pos] == target_id:
                        token_hits += 1
                if gen_ids == target_ids:
                    exact_matches += 1

    return {
        "eval_acc_expr": correct / max(total, 1),
        "eval_token_accuracy": token_hits / max(token_total, 1),
        "eval_exact_full": exact_matches / max(total, 1),
    }


def make_eval_fn(
    eval_dataset: ArithmeticCoTDataset,
    tokenizer: SimpleCharTokenizer,
    max_new_tokens: int,
    batch_size: int,
    match_target_length: bool,
):
    return lambda model: greedy_eval_arithmetic(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        match_target_length=match_target_length,
    )


# -----------------------------
# Experiment driver
# -----------------------------


@dataclass(frozen=True)
class Regime:
    q_keep: float
    max_block_size: int

    @property
    def slug(self) -> str:
        q_str = str(self.q_keep).replace(".", "")
        return f"q{q_str}_b{self.max_block_size}"

    def as_dict(self) -> Dict[str, float | int]:
        return {"q_keep": self.q_keep, "max_block_size": self.max_block_size}


def maybe_generate_split(
    *,
    regime: Regime,
    split: str,
    path: str,
    k_max: int,
    examples_per_k: int,
    seed: int,
    force: bool,
) -> None:
    if os.path.exists(path) and not force:
        return
    ensure_dir(os.path.dirname(path))
    generate_dataset(
        k_min=1,
        k_max=k_max,
        examples_per_k=examples_per_k,
        q_keep=regime.q_keep,
        max_block_size=regime.max_block_size,
        seed=seed,
        output_path=path,
    )


def build_datasets_for_level(
    *,
    data_dir: str,
    regime: Regime,
    k: int,
    tokenizer: SimpleCharTokenizer,
    max_length: int,
    train_examples: int,
    eval_examples: int,
    seeds: Dict[str, int],
) -> Dict[str, ArithmeticCoTDataset]:
    base = os.path.join(data_dir, regime.slug)
    train_path = os.path.join(base, "train.jsonl")
    val_path = os.path.join(base, "val.jsonl")
    test_path = os.path.join(base, "test.jsonl")

    train_ds = OnlineArithmeticDataset(
        levels=range(1, k + 1),
        tokenizer=tokenizer,
        max_length=max_length,
        examples_per_epoch=train_examples * k,
        regime=regime,
        seed=seeds["train"],
    )
    val_ds = ArithmeticCoTDataset(
        path=val_path,
        tokenizer=tokenizer,
        max_length=max_length,
        levels=range(1, k + 1),
        sample_size=eval_examples * k,
        seed=seeds["val"],
    )
    test_ds = ArithmeticCoTDataset(
        path=test_path,
        tokenizer=tokenizer,
        max_length=max_length,
        levels=range(1, k + 1),
        sample_size=eval_examples * k,
        seed=seeds["test"],
    )
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def train_level(
    *,
    k: int,
    regime: Regime,
    tokenizer: SimpleCharTokenizer,
    data_collator: CausalLMDataCollator,
    args: argparse.Namespace,
    base_checkpoint: str | None = None,
) -> Dict[str, object]:
    """Train (or continue training) a model on complexity level k only."""

    run_dir = os.path.join(args.artifacts_dir, regime.slug, f"level_k{k}")
    ensure_dir(run_dir)

    train_ds = OnlineArithmeticDataset(
        levels=[k],
        tokenizer=tokenizer,
        max_length=args.context_length,
        examples_per_epoch=args.train_examples_per_level,
        regime=regime,
        seed=args.seed + 11 * k,
    )
    val_ds = OnlineArithmeticDataset(
        levels=[k],
        tokenizer=tokenizer,
        max_length=args.context_length,
        examples_per_epoch=args.eval_examples_per_level,
        regime=regime,
        seed=args.seed + 13 * k,
    )
    test_ds = OnlineArithmeticDataset(
        levels=[k],
        tokenizer=tokenizer,
        max_length=args.context_length,
        examples_per_epoch=args.eval_examples_per_level,
        regime=regime,
        seed=args.seed + 17 * k,
    )

    eval_fn = make_eval_fn(
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        max_new_tokens=args.greedy_eval_max_new_tokens,
        batch_size=args.greedy_eval_batch_size,
        match_target_length=args.greedy_eval_match_target_length,
    )

    if base_checkpoint:
        model_builder = lambda: LlamaForCausalLM.from_pretrained(base_checkpoint)
        initial_model = model_builder()
    else:
        model_builder = lambda: build_model_from_tokenizer(tokenizer, args.context_length)
        initial_model = None

    trainer, callback, round_history = measure_sample_complexity_with_recursive_rollback(
        model_builder=model_builder,
        initial_model=initial_model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        greedy_eval_fn=eval_fn,
        output_dir=run_dir,
        per_device_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        initial_eval_steps=args.eval_steps,
        eval_refine_rounds=args.eval_refine_rounds,
        metric_name="eval_acc_expr",
        rollback_branches=args.rollback_branches,
        success_threshold=args.acc_target,
    )

    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    cleanup_checkpoints(run_dir)

    final_best = callback.best_step if callback is not None else None
    if final_best is None:
        raise RuntimeError(f"Training did not reach threshold for k={k} under regime {regime.slug} before max_steps.")
    s99_steps = final_best or trainer.args.max_steps
    test_metrics = greedy_eval_arithmetic(
        model=LlamaForCausalLM.from_pretrained(run_dir),
        tokenizer=tokenizer,
        dataset=test_ds,
        max_new_tokens=args.greedy_eval_max_new_tokens,
        batch_size=args.greedy_eval_batch_size,
        match_target_length=args.greedy_eval_match_target_length,
    )

    return {
        "checkpoint": run_dir,
        "round_history": round_history,
        "s99_steps": s99_steps,
        "metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaling-law experiments for collapsed CoT arithmetic.")
    parser.add_argument("--k_max", type=int, default=6)
    parser.add_argument("--q_keep", type=float, default=1.0, help="q_keep value for dataset corruption.")
    parser.add_argument("--max_block_size", type=int, default=1, help="max_block_size for dataset generation.")
    parser.add_argument("--train_examples_per_level", type=int, default=20000)
    parser.add_argument("--eval_examples_per_level", type=int, default=2000)
    parser.add_argument("--data_dir", type=str, default="arithemtic_scaling_law/data")
    parser.add_argument("--artifacts_dir", type=str, default="arithemtic_scaling_law/artifacts")
    parser.add_argument("--results_dir", type=str, default="arithemtic_scaling_law/results")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--eval_refine_rounds", type=int, default=4)
    parser.add_argument("--rollback_branches", type=int, default=1)
    parser.add_argument("--greedy_eval_batch_size", type=int, default=16)
    parser.add_argument("--greedy_eval_max_new_tokens", type=int, default=64)
    parser.add_argument("--greedy_eval_match_target_length", action="store_true")
    parser.add_argument("--max_steps", type=int, default=200000, help="Upper bound on steps per level.")
    parser.add_argument("--acc_target", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force_regen", action="store_true", help="Regenerate datasets even if they exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    regime = Regime(args.q_keep, args.max_block_size)
    tokenizer = build_tokenizer(extra_chars=["(", ")", "+", "*"])
    data_collator = CausalLMDataCollator(tokenizer=tokenizer)

    seeds = {"train": args.seed + 1, "val": args.seed + 2, "test": args.seed + 3}
    base_checkpoint: str | None = None

    for k in range(1, args.k_max):
        record = train_level(
            k=k,
            regime=regime,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=args,
            base_checkpoint=base_checkpoint,
        )
        base_checkpoint = record["checkpoint"]
        ensure_dir(args.results_dir)
        write_json(
            os.path.join(args.results_dir, f"level_k{k}_{regime.slug}.json"),
            record,
        )


if __name__ == "__main__":
    main()
