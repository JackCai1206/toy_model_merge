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
from typing import Any, Dict, Iterable, List
import shutil

import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from transformers import LlamaForCausalLM, set_seed

from arithemtic_scaling_law.generate_bracketed_cot import MODULUS, generate_dataset, generate_example
from arithemtic_scaling_law.recursive_sample_complexity import train_with_eval_threshold
from algorithm_composition.utils.collators import CausalLMDataCollator
from algorithm_composition.utils.tokenizer import SimpleCharTokenizer, build_tokenizer, encode_prompt_with_sep
from algorithm_composition.utils.training import (
    build_model_from_tokenizer,
    cleanup_checkpoints,
    ensure_dir,
    list_checkpoint_steps,
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


def _parse_answer(gen_ids: List[int], tokenizer: SimpleCharTokenizer) -> int | None:
    """Extract the final numeric answer from generated tokens.

    First look for digits immediately following the last 'A' token to avoid
    picking up intermediate step numbers; fall back to regex over decoded text.
    """
    tokens = tokenizer.convert_ids_to_tokens(gen_ids)
    digits: List[str] = []
    last_a_idx = None
    for idx in range(len(tokens) - 1, -1, -1):
        if tokens[idx] == "A":
            last_a_idx = idx
            break
    if last_a_idx is not None:
        for tok in tokens[last_a_idx + 1 :]:
            if tok.isdigit():
                digits.append(tok)
            elif digits:
                break
        if digits:
            return int("".join(digits)) % MODULUS

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    eq_matches = re.findall(r"=\s*(-?\d+)", gen_text)
    if eq_matches:
        return int(eq_matches[-1]) % MODULUS

    match = re.findall(r"-?\d+", gen_text)
    if not match:
        return None
    return int(match[-1]) % MODULUS


class SampleTooLongError(RuntimeError):
    """Raised when we cannot fit a sampled prompt/target within the context window."""


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
            step_no_prefix = re.sub(r"(?i)step\s*\d+:\s*", "", step)
            step_no_compute = re.sub(r"(?i)compute\s*", "", step_no_prefix)
            step_no_mod = re.sub(r"(?i)\(mod\s*\d+\)", "", step_no_compute)
            step_compact = re.sub(r"\s+", "", step_no_mod)
            return step_compact.strip(".")

        cleaned_steps = [_clean_step(step) for step in steps if step]
        if cleaned_steps:
            target = "=".join(cleaned_steps)
        else:
            target = str(int(record["answer"]) % MODULUS)

        prompt = f"{expr}="
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
    """Freshly generates samples every epoch for the requested levels, optionally mixing in prior levels."""

    def __init__(
        self,
        *,
        levels: Iterable[int],
        tokenizer: SimpleCharTokenizer,
        max_length: int,
        examples_per_epoch: int,
        regime: Regime,
        seed: int,
        focus_level: int | None = None,
        mix_prev_fraction: float = 0.0,
        mix_prev_decay: float = 0.8,
        max_resample_attempts: int = 1000,
    ) -> None:
        level_list = sorted(set(levels))
        if focus_level is not None and focus_level not in level_list:
            level_list.append(focus_level)
            level_list.sort()
        self.levels = level_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples_per_epoch = examples_per_epoch
        self.regime = regime
        self.seed = seed
        self.focus_level = focus_level if focus_level is None or focus_level in self.levels else None
        self.mix_prev_fraction = min(1.0, max(0.0, float(mix_prev_fraction)))
        self.mix_prev_decay = float(mix_prev_decay) if mix_prev_decay > 0 else 1.0
        self._prev_levels = (
            [lvl for lvl in self.levels if self.focus_level is not None and lvl < self.focus_level]
            if self.focus_level is not None
            else []
        )
        self._prev_level_weights = [
            self.mix_prev_decay ** max(1, self.focus_level - lvl) for lvl in self._prev_levels
        ] if self._prev_levels else []
        self.max_resample_attempts = max(1, int(max_resample_attempts))

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

    def _sample_level(self, rng: random.Random) -> int:
        if (
            self.focus_level is None
            or not self._prev_levels
            or self.mix_prev_fraction <= 0.0
            or rng.random() >= self.mix_prev_fraction
        ):
            # Default: stick to the focus level when provided, otherwise uniform across levels.
            return self.focus_level if self.focus_level is not None else rng.choice(self.levels)
        # Sample a previous level with exponential decay so recent levels are favored.
        return rng.choices(self._prev_levels, weights=self._prev_level_weights, k=1)[0]

    def _sample_example(
        self, rng: random.Random
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, str | int]]:
        attempts = 0
        last_error: Exception | None = None
        last_level: int | None = None
        while attempts < self.max_resample_attempts:
            attempts += 1
            k = self._sample_level(rng)
            example = generate_example(
                k=k,
                rng=rng,
                q_keep=self.regime.q_keep,
                max_steps_per_block=self.regime.max_steps_per_block,
            )
            expr = example["expression"].replace(" ", "")
            prompt = f"{expr}="
            steps = [step.replace(" ", "") for step in example["visible_cot"] if step]
            target = "=".join(steps) if steps else str(int(example["answer"]) % MODULUS)
            try:
                encoded = _encode_prompt_and_target(
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    target=target,
                    max_length=self.max_length,
                )
            except ValueError as exc:
                last_error = exc
                last_level = k
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

        level_info = last_level if last_level is not None else (self.focus_level or -1)
        raise SampleTooLongError(
            f"Failed to sample an example for k={level_info} within max_length={self.max_length} "
            f"after {self.max_resample_attempts} attempts. Last error: {last_error}"
        )


def greedy_eval_arithmetic(
    model,
    tokenizer: SimpleCharTokenizer,
    dataset: ArithmeticCoTDataset,
    max_new_tokens: int,
    batch_size: int,
    match_target_length: bool = False,
    per_level_breakdown: bool = False,
    include_counts: bool = False,
    sample_print_limit: int = 0,
    sample_print_context: str | None = None,
) -> Dict[str, float]:
    """Greedy decoding evaluation returning expression-level accuracy.

    When `sample_print_limit` > 0, prints up to that many prompt/target/prediction triples for
    quick manual inspection each time this eval function is called.
    """

    device = next(model.parameters()).device
    model.eval()
    total = len(dataset)
    correct = 0
    token_hits = 0
    token_total = 0
    exact_matches = 0
    per_level_total: Dict[int, int] = {}
    per_level_correct: Dict[int, int] = {}
    sample_limit = max(0, int(sample_print_limit or 0))
    sample_logs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            batch_items = [dataset.get_prompt_and_target(idx) for idx in range(start, end)]
            if not batch_items:
                continue
            prompts = [encode_prompt_with_sep(tokenizer, item["prompt"]) for item in batch_items]
            prompt_lens = [len(p) for p in prompts]
            max_prompt = max(prompt_lens)
            batch = len(prompts)
            input_ids = torch.full((batch, max_prompt), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            for row, ids in enumerate(prompts):
                length = len(ids)
                offset = max_prompt - length
                input_ids[row, offset:] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[row, offset:] = 1

            gen_max = max_new_tokens
            if match_target_length:
                target_lengths = [len(tokenizer.encode(item["target"], add_special_tokens=False)) for item in batch_items]
                gen_max = max(gen_max, max(target_lengths, default=0))

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            input_length = input_ids.shape[1]
            for row, item in enumerate(batch_items):
                gen_ids = generated[row].tolist()[input_length:]
                gen_ids = _strip_after_eos(gen_ids, tokenizer.eos_token_id)
                pred_answer = _parse_answer(gen_ids, tokenizer)
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if pred_answer is not None and pred_answer == item["answer"]:
                    correct += 1
                target_ids = tokenizer.encode(item["target"], add_special_tokens=False)
                for pos, target_id in enumerate(target_ids):
                    token_total += 1
                    if pos < len(gen_ids) and gen_ids[pos] == target_id:
                        token_hits += 1
                if gen_ids == target_ids:
                    exact_matches += 1
                if sample_limit and len(sample_logs) < sample_limit:
                    sample_logs.append(
                        {
                            "prompt": item.get("prompt"),
                            "target": item.get("target"),
                            "generated": generated_text,
                            "pred_answer": pred_answer,
                            "target_answer": item.get("answer"),
                            "exact_match": gen_ids == target_ids,
                            "expr_correct": pred_answer is not None and pred_answer == item["answer"],
                            "complexity_k": item.get("complexity_k"),
                        }
                    )
                if per_level_breakdown:
                    level = int(item.get("complexity_k", -1)) if isinstance(item, dict) else -1
                    per_level_total[level] = per_level_total.get(level, 0) + 1
                    if pred_answer is not None and pred_answer == item["answer"]:
                        per_level_correct[level] = per_level_correct.get(level, 0) + 1

    metrics: Dict[str, float] = {
        "eval_acc_expr": correct / max(total, 1),
        "eval_token_accuracy": token_hits / max(token_total, 1),
        "eval_exact_full": exact_matches / max(total, 1),
    }
    if per_level_breakdown:
        for level, count in sorted(per_level_total.items()):
            hits = per_level_correct.get(level, 0)
            metrics[f"eval_acc_expr_k{level}"] = hits / max(count, 1)
    if include_counts:
        metrics["eval_count_examples"] = total
        metrics["eval_count_token_hits"] = token_hits
        metrics["eval_count_token_total"] = token_total
        metrics["eval_count_exact_matches"] = exact_matches
    if sample_logs:
        context = sample_print_context or "eval"
        print(
            f"[{context}] showing {len(sample_logs)} sample(s) (limit={sample_limit})",
            flush=True,
        )
        for idx, sample in enumerate(sample_logs, start=1):
            print(
                f"[{context} sample {idx}] k={sample.get('complexity_k', '?')} expr_correct={sample['expr_correct']} exact={sample['exact_match']}",
                flush=True,
            )
            print(f"prompt: {sample['prompt']}", flush=True)
            print(f"target:\n{sample['target']}", flush=True)
            print(f"generated:\n{sample['generated']}", flush=True)
            print(
                f"parsed_answer={sample['pred_answer']} target_answer={sample['target_answer']}",
                flush=True,
            )
            print("-" * 60, flush=True)

    return metrics


def make_eval_fn(
    eval_dataset: ArithmeticCoTDataset,
    tokenizer: SimpleCharTokenizer,
    max_new_tokens: int,
    batch_size: int,
    match_target_length: bool,
    sample_print_limit: int = 0,
    sample_print_context: str | None = None,
):
    return lambda model: greedy_eval_arithmetic(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        match_target_length=match_target_length,
        sample_print_limit=sample_print_limit,
        sample_print_context=sample_print_context,
    )


def load_model_for_eval(path: str):
    """Load a checkpoint and place it on GPU if available so eval isn't CPU-bound."""

    model = LlamaForCausalLM.from_pretrained(path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def evaluate_levels_with_early_stop(
    *,
    model,
    tokenizer: SimpleCharTokenizer,
    regime: Regime,
    levels: Iterable[int],
    examples_per_level: int,
    context_length: int,
    max_new_tokens: int,
    batch_size: int,
    match_target_length: bool,
    seed_base: int = 0,
    show_progress: bool = False,
    stop_threshold: float | None = None,
    sample_print_limit: int = 0,
) -> Dict[str, float]:
    """Run greedy eval level-by-level and optionally stop early once accuracy dips to/below the threshold."""

    level_list = sorted(levels)
    aggregate_examples = 0
    aggregate_correct = 0.0
    aggregate_exact = 0.0
    aggregate_token_hits = 0
    aggregate_token_total = 0
    metrics: Dict[str, float] = {}
    levels_evaluated: List[int] = []

    for idx, level in enumerate(level_list, start=1):
        if show_progress:
            print(f"[final eval] level k={level} ({idx}/{len(level_list)})", flush=True)
        level_seed = seed_base + 23 * level
        level_ds = OnlineArithmeticDataset(
            levels=[level],
            tokenizer=tokenizer,
            max_length=context_length,
            examples_per_epoch=examples_per_level,
            regime=regime,
            seed=level_seed,
        )
        try:
            level_metrics = greedy_eval_arithmetic(
                model=model,
                tokenizer=tokenizer,
                dataset=level_ds,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                match_target_length=match_target_length,
                include_counts=True,
                sample_print_limit=sample_print_limit,
                sample_print_context=f"final_eval_k{level}",
            )
        except SampleTooLongError as exc:
            print(
                f"[final eval] stopping at k={level} because samples exceed context length: {exc}",
                flush=True,
            )
            break

        levels_evaluated.append(level)
        level_count = len(level_ds)
        metrics[f"eval_acc_expr_k{level}"] = level_metrics["eval_acc_expr"]
        metrics[f"eval_exact_full_k{level}"] = level_metrics["eval_exact_full"]
        metrics[f"eval_token_accuracy_k{level}"] = level_metrics["eval_token_accuracy"]

        aggregate_examples += level_count
        aggregate_correct += level_metrics["eval_acc_expr"] * level_count
        aggregate_exact += level_metrics["eval_exact_full"] * level_count
        aggregate_token_hits += int(level_metrics.get("eval_count_token_hits", 0))
        aggregate_token_total += int(level_metrics.get("eval_count_token_total", 0))

        acc = level_metrics["eval_acc_expr"]
        if show_progress:
            print(f"[final eval] level k={level} accuracy={acc:.4f}", flush=True)

        if stop_threshold is not None and acc <= stop_threshold:
            if show_progress:
                print(
                    f"[final eval] stopping after k={level} (accuracy={acc:.4f} <= threshold={stop_threshold:.2f})",
                    flush=True,
                )
            break

    metrics["eval_levels_evaluated"] = len(levels_evaluated)
    metrics["eval_acc_expr"] = aggregate_correct / max(aggregate_examples, 1)
    metrics["eval_token_accuracy"] = aggregate_token_hits / max(aggregate_token_total, 1)
    return metrics


@dataclass(frozen=True)
class Regime:
    q_keep: float
    max_steps_per_block: int

    @property
    def slug(self) -> str:
        q_str = str(self.q_keep).replace(".", "")
        return f"q{q_str}_b{self.max_steps_per_block}"

    def as_dict(self) -> Dict[str, float | int]:
        return {"q_keep": self.q_keep, "max_steps_per_block": self.max_steps_per_block}


def maybe_generate_split(
    *,
    regime: Regime,
    split: str,
    path: str,
    k_min: int,
    k_max: int,
    examples_per_k: int,
    seed: int,
    force: bool,
) -> None:
    if os.path.exists(path) and not force:
        return
    ensure_dir(os.path.dirname(path))
    generate_dataset(
        k_min=k_min,
        k_max=k_max,
        examples_per_k=examples_per_k,
        q_keep=regime.q_keep,
        max_steps_per_block=regime.max_steps_per_block,
        seed=seed,
        output_path=path,
    )


def build_datasets_for_level(
    *,
    data_dir: str,
    regime: Regime,
    k: int,
    k_min: int,
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
    level_range = range(k_min, k + 1)
    level_count = max(k - k_min + 1, 1)

    train_ds = OnlineArithmeticDataset(
        levels=level_range,
        tokenizer=tokenizer,
        max_length=max_length,
        examples_per_epoch=train_examples * level_count,
        regime=regime,
        seed=seeds["train"],
    )
    val_ds = ArithmeticCoTDataset(
        path=val_path,
        tokenizer=tokenizer,
        max_length=max_length,
        levels=level_range,
        sample_size=eval_examples * level_count,
        seed=seeds["val"],
    )
    test_ds = ArithmeticCoTDataset(
        path=test_path,
        tokenizer=tokenizer,
        max_length=max_length,
        levels=level_range,
        sample_size=eval_examples * level_count,
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
    eval_steps: int,
    eval_delay: int,
    base_checkpoint: str | None = None,
) -> Dict[str, object]:
    """Train (or continue training) a model on complexity level k only."""

    run_dir = os.path.join(args.artifacts_dir, regime.slug, f"level_k{k}")
    ensure_dir(run_dir)
    final_eval_examples = (
        args.final_eval_examples_per_level
        if args.final_eval_examples_per_level is not None
        else args.eval_examples_per_level
    )
    level_range = range(args.k_min, k + 1)
    def _warmup_for(eval_steps: int) -> int:
        if eval_steps < args.eval_steps:
            return max(1, int(0.1 * eval_steps))
        return args.warmup_steps

    train_ds = OnlineArithmeticDataset(
        levels=level_range,
        tokenizer=tokenizer,
        max_length=args.context_length,
        examples_per_epoch=args.train_examples_per_level,
        regime=regime,
        seed=args.seed + 11 * k,
        focus_level=k,
        mix_prev_fraction=args.prev_level_mix_fraction,
        mix_prev_decay=args.prev_level_mix_decay,
    )
    val_ds = OnlineArithmeticDataset(
        levels=level_range,
        tokenizer=tokenizer,
        max_length=args.context_length,
        examples_per_epoch=args.eval_examples_per_level,
        regime=regime,
        seed=args.seed + 13 * k,
        focus_level=k,
        mix_prev_fraction=args.prev_level_mix_fraction,
        mix_prev_decay=args.prev_level_mix_decay,
    )
    test_ds = OnlineArithmeticDataset(
        levels=level_range,
        tokenizer=tokenizer,
        max_length=args.context_length,
        examples_per_epoch=final_eval_examples,
        regime=regime,
        seed=args.seed + 17 * k,
        focus_level=k,
        mix_prev_fraction=args.prev_level_mix_fraction,
        mix_prev_decay=args.prev_level_mix_decay,
    )

    eval_fn = make_eval_fn(
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        max_new_tokens=args.greedy_eval_max_new_tokens,
        batch_size=args.greedy_eval_batch_size,
        match_target_length=args.greedy_eval_match_target_length,
        sample_print_limit=args.eval_print_examples,
        sample_print_context=f"train_eval_k{k}",
    )

    # Model construction. Request FlashAttention2 via Transformers when loading a checkpoint.
    if base_checkpoint:
        model_builder = lambda: LlamaForCausalLM.from_pretrained(
            base_checkpoint, attn_implementation="flash_attention_2"
        )
        initial_model = model_builder()
    else:
        model_builder = lambda: build_model_from_tokenizer(tokenizer, args.context_length)
        initial_model = None

    trainer = None
    threshold_steps: List[int] = []
    callback = None
    while True:
        level_warmup = _warmup_for(eval_steps)
        trainer, callback, threshold_steps = train_with_eval_threshold(
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
            eval_steps=eval_steps,
            eval_delay=eval_delay,
            metric_name="eval_acc_expr",
            warmup_steps=level_warmup,
            success_threshold=args.acc_target,
            torch_compile=True,
            use_liger_kernel=True,
        )
        first_hit = threshold_steps[0] if threshold_steps else None
        if first_hit is not None and first_hit <= eval_steps and eval_steps > args.eval_steps_min:
            new_steps = max(args.eval_steps_min, eval_steps // 2)
            if new_steps < eval_steps:
                print(
                    f"[level k={k}] Early threshold at step {first_hit}; shrinking eval_steps to {new_steps} and rerunning.",
                    flush=True,
                )
                eval_steps = new_steps
                level_warmup = _warmup_for(eval_steps)
                if os.path.isdir(run_dir):
                    shutil.rmtree(run_dir, ignore_errors=True)
                ensure_dir(run_dir)
                continue
        break

    # Persist only the latest checkpoint from the best branch.
    latest_steps = list_checkpoint_steps(trainer.args.output_dir)
    latest_checkpoint = (
        os.path.join(trainer.args.output_dir, f"checkpoint-{latest_steps[-1]}")
        if latest_steps
        else trainer.args.output_dir
    )
    if latest_steps:
        print(
            f"Finalizing level k={k}: using latest checkpoint {os.path.basename(latest_checkpoint)} from {trainer.args.output_dir}",
            flush=True,
        )
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    # Remove all checkpoint-* directories so only a single final checkpoint remains.
    cleanup_checkpoints(trainer.args.output_dir, keep=0)
    if trainer.args.output_dir != run_dir:
        cleanup_checkpoints(run_dir, keep=0)

    final_best = threshold_steps[0] if threshold_steps else None
    if final_best is None:
        raise RuntimeError(f"Training did not reach threshold for k={k} under regime {regime.slug} before max_steps.")
    s99_steps = final_best or trainer.args.max_steps
    eval_model = load_model_for_eval(run_dir)
    test_metrics = greedy_eval_arithmetic(
        model=eval_model,
        tokenizer=tokenizer,
        dataset=test_ds,
        max_new_tokens=args.greedy_eval_max_new_tokens,
        batch_size=args.greedy_eval_batch_size,
        match_target_length=args.greedy_eval_match_target_length,
        sample_print_limit=args.eval_print_examples,
        sample_print_context=f"test_k{k}",
    )
    all_levels_metrics = evaluate_levels_with_early_stop(
        model=eval_model,
        tokenizer=tokenizer,
        regime=regime,
        levels=range(args.k_min, args.k_max),
        examples_per_level=final_eval_examples,
        context_length=args.context_length,
        max_new_tokens=args.greedy_eval_max_new_tokens,
        batch_size=args.greedy_eval_batch_size,
        match_target_length=args.greedy_eval_match_target_length,
        seed_base=args.seed + 19 * k,
        show_progress=True,
        stop_threshold=args.final_eval_stop_threshold,
        sample_print_limit=args.eval_print_examples,
    )

    return {
        "checkpoint": run_dir,
        "k": k,
        "threshold_steps": threshold_steps,
        "s99_steps": s99_steps,
        "eval_steps": eval_steps,
        "eval_delay": eval_delay,
        "warmup_steps_used": _warmup_for(eval_steps),
        "max_steps": args.max_steps,
        "metrics": test_metrics,
        "metrics_all_levels": all_levels_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaling-law experiments for collapsed CoT arithmetic.")
    parser.add_argument("--k_min", type=int, default=1, help="Minimum complexity level to start training (inclusive).")
    parser.add_argument("--k_max", type=int, default=6)
    parser.add_argument("--run_name", type=str, default=None, help="Human-friendly run label (e.g., seed_123).")
    parser.add_argument("--run_group", type=str, default=None, help="Group folder to cluster seeds/runs.")
    parser.add_argument("--q_keep", type=float, default=1.0, help="q_keep value for dataset corruption.")
    parser.add_argument(
        "--max_steps_per_block",
        type=int,
        default=1,
        help="Upper bound on atomic steps collapsed into a single visible calculation.",
    )
    parser.add_argument("--train_examples_per_level", type=int, default=20000)
    parser.add_argument("--eval_examples_per_level", type=int, default=2000)
    parser.add_argument(
        "--final_eval_examples_per_level",
        type=int,
        default=500,
        help="Examples per level for the final greedy eval only; set to eval_examples_per_level to match training eval size.",
    )
    parser.add_argument(
        "--final_eval_stop_threshold",
        type=float,
        default=None,
        help="Stop evaluating higher levels once accuracy falls to/below this threshold. "
        "Use a negative value or omit to disable early stopping (default: disabled).",
    )
    parser.add_argument("--data_dir", type=str, default="arithemtic_scaling_law/data")
    parser.add_argument("--artifacts_dir", type=str, default="arithemtic_scaling_law/artifacts")
    parser.add_argument("--results_dir", type=str, default="arithemtic_scaling_law/results")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument(
        "--eval_steps_min",
        type=int,
        default=1,
        help="Minimum eval interval when adaptively shrinking after early hits.",
    )
    parser.add_argument(
        "--eval_steps_max",
        type=int,
        default=None,
        help="Upper bound on eval interval when adaptively growing after late hits (default: unlimited).",
    )
    parser.add_argument(
        "--eval_jitter_fraction",
        type=float,
        default=0.0,
        help="Fraction of eval_steps to use as max jitter for the first eval offset (per seed/level).",
    )
    parser.add_argument(
        "--prev_level_mix_fraction",
        type=float,
        default=0.2,
        help="Fraction of samples drawn from previous levels (EMA weighted) when training level k.",
    )
    parser.add_argument(
        "--prev_level_mix_decay",
        type=float,
        default=0.8,
        help="Exponential decay factor for weighting earlier levels when mixing in previous-level samples.",
    )
    parser.add_argument("--greedy_eval_batch_size", type=int, default=16)
    parser.add_argument("--greedy_eval_max_new_tokens", type=int, default=64)
    parser.add_argument("--greedy_eval_match_target_length", action="store_true")
    parser.add_argument(
        "--eval_print_examples",
        type=int,
        default=3,
        help="Number of prompt/target/prediction triples to print each time an eval runs (set 0 to disable).",
    )
    parser.add_argument("--max_steps", type=int, default=200000, help="Upper bound on steps per level.")
    parser.add_argument("--warmup_steps", type=int, default=600, help="Optimizer warmup steps.")
    parser.add_argument("--acc_target", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_min < 1:
        raise ValueError(f"k_min must be >= 1 (got {args.k_min}).")
    if args.k_min >= args.k_max:
        raise ValueError(f"k_min ({args.k_min}) must be less than k_max ({args.k_max}).")
    if args.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative (got {args.warmup_steps}).")
    set_seed(args.seed)

    regime = Regime(args.q_keep, args.max_steps_per_block)
    run_name = args.run_name or f"seed_{args.seed}"
    run_group = args.run_group or f"run_{regime.slug}"
    if args.run_group is None and args.k_min != 1:
        run_group = f"{run_group}_kmin{args.k_min}"
    tokenizer = build_tokenizer(extra_chars=["(", ")", "+", "*"])
    data_collator = CausalLMDataCollator(tokenizer=tokenizer)

    seeds = {"train": args.seed + 1, "val": args.seed + 2, "test": args.seed + 3}
    base_checkpoint: str | None = None
    total_levels = args.k_max - args.k_min

    current_eval_steps = max(args.eval_steps_min, args.eval_steps)
    eval_steps_max = args.eval_steps_max if args.eval_steps_max is None or args.eval_steps_max > 0 else None

    for idx, k in enumerate(range(args.k_min, args.k_max), start=1):
        print(
            f"=== Training level k={k} ({idx}/{total_levels}) | regime={regime.slug} | run_group={run_group} | run_name={run_name} ===",
            flush=True,
        )
        rng = random.Random(args.seed * 7919 + k)
        level_eval_steps = current_eval_steps
        attempt = 0
        first_hit: int | None = None
        while True:
            jitter_max = int(level_eval_steps * max(0.0, min(1.0, float(args.eval_jitter_fraction))))
            eval_delay = rng.randint(0, max(0, jitter_max)) if jitter_max > 0 else 0
            attempt += 1
            record = train_level(
                k=k,
                regime=regime,
                tokenizer=tokenizer,
                data_collator=data_collator,
                args=args,
                eval_steps=level_eval_steps,
                eval_delay=eval_delay,
                base_checkpoint=base_checkpoint,
            )
            threshold_steps = record.get("threshold_steps") or []
            first_hit = threshold_steps[0] if threshold_steps else None
            if (
                first_hit is not None
                and first_hit <= level_eval_steps
                and level_eval_steps > args.eval_steps_min
            ):
                new_steps = max(args.eval_steps_min, level_eval_steps // 2)
                if new_steps < level_eval_steps:
                    print(
                        f"[level k={k}] Early threshold at step {first_hit}; shrinking eval_steps to {new_steps} and rerunning (attempt {attempt+1}).",
                        flush=True,
                    )
                    level_eval_steps = new_steps
                    continue
            break

        base_checkpoint = record["checkpoint"]
        ensure_dir(args.results_dir)
        record.update(
            {
                "run_name": run_name,
                "run_group": run_group,
                "seed": args.seed,
                "regime": regime.as_dict(),
                "regime_slug": regime.slug,
                "results_dir": args.results_dir,
                "artifacts_dir": args.artifacts_dir,
            }
        )
        write_json(
            os.path.join(args.results_dir, f"level_k{k}_{regime.slug}.json"),
            record,
        )
        metrics = record.get("metrics", {})
        s99 = record.get("s99_steps")
        def _fmt(v: float | None) -> str:
            try:
                return f"{float(v):.4f}"
            except Exception:
                return "n/a"
        print(
            f"[level k={k}] s99_steps={s99} | "
            f"eval_acc_expr={_fmt(metrics.get('eval_acc_expr'))} | "
            f"eval_exact_full={_fmt(metrics.get('eval_exact_full'))}",
            flush=True,
        )

        if first_hit is not None and first_hit > 3 * level_eval_steps:
            grown = int(level_eval_steps * 2)
            if eval_steps_max is not None:
                grown = min(grown, eval_steps_max)
            if grown != level_eval_steps:
                print(
                    f"[level k={k}] Late threshold at step {first_hit}; increasing eval_steps to {grown} for subsequent levels.",
                    flush=True,
                )
            current_eval_steps = grown
        else:
            current_eval_steps = level_eval_steps


if __name__ == "__main__":
    main()
