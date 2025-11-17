"""Causal ablation sweep to localize where the reverse operation is encoded.

Given a checkpoint and a task family (C or NC), this script measures baseline
exact-match accuracy on synthetic eval data, then ablates each transformer layer
to see how much accuracy drops. It separately zeros the attention output and the
MLP output for every layer. Results are printed and optionally saved to JSON so
they can be compared across checkpoints (e.g., AB pretrain vs. C/NC fine-tunes).
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM

from data.gen_reverse_regions import GeneratorConfig
from utils.collators import CausalLMDataCollator
from utils.datasets import ReverseDatasetConfig, ReverseTaskDataset
from utils.tokenizer import build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-layer ablation sweep for reverse tasks.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to a trained checkpoint directory."
    )
    parser.add_argument(
        "--family", type=str, default="GENERAL", choices=["C", "NC", "GENERAL"], help="Eval family."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Tasks to evaluate (e.g., C, A, B, or A+B). Provide multiple entries to run each task "
            "individually; include '+' within an entry to evaluate tasks jointly (e.g., A+B). "
            "Defaults to ['A','B'] for GENERAL and ['C'] otherwise."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=200, help="Eval examples to sample.")
    parser.add_argument("--batch_size", type=int, default=16, help="Eval batch size.")
    parser.add_argument("--context_length", type=int, default=256, help="Max sequence length.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (auto, cpu, or cuda).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON results (layer drops, baseline).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for deterministic dataset sampling.",
    )
    parser.add_argument(
        "--mode",
        choices=["split", "coarse", "both"],
        default="coarse",
        help="Ablation granularity: split attn/MLP, coarse (both), or both.",
    )
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def build_dataloader(
    family: str,
    tasks: Tuple[str, ...],
    tokenizer,
    num_samples: int,
    batch_size: int,
    context_length: int,
    seed: int,
) -> DataLoader:
    cfg = ReverseDatasetConfig(generator=GeneratorConfig(), max_length=context_length, dataset_size=num_samples)
    dataset = ReverseTaskDataset(
        family=family,
        tasks=tasks,
        tokenizer=tokenizer,
        seed=seed,
        config=cfg,
    )
    collator = CausalLMDataCollator(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def exact_match(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    """Return (correct, total) counts for sequence-level exact match.

    Aligns logits to labels by shifting one to the right (causal LM). Only
    positions with labels != -100 are evaluated.
    """

    # Shift logits to align with labels (predict token t at position t-1).
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != -100
    preds = logits.argmax(dim=-1)

    per_token_ok = (~mask) | (preds == labels)
    per_seq_ok = per_token_ok.all(dim=1) & mask.any(dim=1)
    return int(per_seq_ok.sum().item()), int(mask.any(dim=1).sum().item())


@contextmanager
def apply_hooks(hooks: Iterable[Tuple[torch.nn.Module, callable]]):
    handles = [module.register_forward_hook(fn) for module, fn in hooks]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def evaluate(model: LlamaForCausalLM, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            c, t = exact_match(outputs.logits, batch["labels"])
            correct += c
            total += t
    return correct / total if total else 0.0


def ablation_sweep(
    model: LlamaForCausalLM, dataloader: DataLoader, device: torch.device, mode: str
) -> Dict:
    """Run per-layer ablations; return accuracy drops."""

    # Baseline accuracy with no ablation.
    baseline = evaluate(model, dataloader, device)
    results: Dict[str, List[Dict[str, float]]] = {"baseline": baseline}
    zero = lambda _module, _in, out: torch.zeros_like(out)

    if mode in {"split", "both"}:
        results["attention"] = []
        results["mlp"] = []

        # Zero the attention output (o_proj) for each layer independently.
        for idx, layer in enumerate(model.model.layers):
            attn_module = layer.self_attn.o_proj
            with apply_hooks([(attn_module, zero)]):
                acc = evaluate(model, dataloader, device)
            results["attention"].append({"layer": idx, "acc": acc, "drop": baseline - acc})

        # Zero the MLP output for each layer independently.
        for idx, layer in enumerate(model.model.layers):
            with apply_hooks([(layer.mlp, zero)]):
                acc = evaluate(model, dataloader, device)
            results["mlp"].append({"layer": idx, "acc": acc, "drop": baseline - acc})

    if mode in {"coarse", "both"}:
        results["joint"] = []
        # Zero both attention and MLP outputs for each layer.
        for idx, layer in enumerate(model.model.layers):
            hooks = [(layer.self_attn.o_proj, zero), (layer.mlp, zero)]
            with apply_hooks(hooks):
                acc = evaluate(model, dataloader, device)
            results["joint"].append({"layer": idx, "acc": acc, "drop": baseline - acc})

    return results


def parse_task_entry(entry: str) -> Tuple[str, Tuple[str, ...]]:
    parts = [token.strip() for token in entry.split("+") if token.strip()]
    if not parts:
        raise ValueError("Encountered empty task entry; use names like A, B, C, or A+B.")
    label = "+".join(parts)
    return label, tuple(parts)


def build_task_specs(args: argparse.Namespace) -> List[Tuple[str, Tuple[str, ...]]]:
    """Return (label, tasks) pairs to evaluate for the requested family."""

    default_entries: List[str]
    if args.family == "GENERAL":
        default_entries = ["A", "B"]
    else:
        default_entries = ["C"]

    entries = list(args.tasks) if args.tasks is not None else default_entries
    specs = [parse_task_entry(entry) for entry in entries]
    return specs


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer = build_tokenizer()

    task_specs = build_task_specs(args)
    if not task_specs:
        raise ValueError("No tasks specified for ablation.")

    model = LlamaForCausalLM.from_pretrained(args.checkpoint)
    model.to(device)

    per_task_results: Dict[str, Dict] = {}

    for label, task_tuple in task_specs:
        dataloader = build_dataloader(
            family=args.family,
            tasks=task_tuple,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            context_length=args.context_length,
            seed=args.seed,
        )
        results = ablation_sweep(model, dataloader, device, args.mode)
        per_task_results[label] = results

        print(f"Task {label}: Baseline exact match {results['baseline']:.3f}")
        if "attention" in results:
            print("  Top attention drops (sorted):")
            attn_sorted = sorted(results["attention"], key=lambda r: r["drop"], reverse=True)
            for entry in attn_sorted[:5]:
                print(
                    f"    Layer {entry['layer']}: drop={entry['drop']:.3f} acc={entry['acc']:.3f}"
                )
        if "mlp" in results:
            print("  Top MLP drops (sorted):")
            mlp_sorted = sorted(results["mlp"], key=lambda r: r["drop"], reverse=True)
            for entry in mlp_sorted[:5]:
                print(
                    f"    Layer {entry['layer']}: drop={entry['drop']:.3f} acc={entry['acc']:.3f}"
                )
        if "joint" in results:
            print("  Top joint (attn+MLP) drops (sorted):")
            joint_sorted = sorted(results["joint"], key=lambda r: r["drop"], reverse=True)
            for entry in joint_sorted[:5]:
                print(
                    f"    Layer {entry['layer']}: drop={entry['drop']:.3f} acc={entry['acc']:.3f}"
                )

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "family": args.family,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "device": str(device),
            "per_task": per_task_results,
        }
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
