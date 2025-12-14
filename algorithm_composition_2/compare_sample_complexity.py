"""Compare sample complexity when fine-tuning from joint vs. merged checkpoints."""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from typing import Dict, Iterable, List
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for batch jobs.
import matplotlib.pyplot as plt
from transformers import set_seed

from finetune_composed import run_finetune
from utils.cli import add_shared_training_args
from utils.merging import merge_checkpoints
from utils.training import ensure_dir, write_json


def _parse_scale_list(raw: str | None, fallback: float | None) -> List[float]:
    if raw:
        vals = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(float(part))
        if vals:
            return vals
    return [fallback] if fallback is not None else []


def _format_scale(scale: float) -> str:
    text = f"{scale:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _color_map(labels: List[str], cmap_name: str = "plasma") -> Dict[str, str]:
    if not labels:
        return {}
    cmap = plt.get_cmap(cmap_name)
    xs = np.linspace(0.1, 0.9, len(labels))
    return {label: cmap(x) for label, x in zip(labels, xs)}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the composed task from (1) a joint A/B checkpoint and "
            "(2) a merged checkpoint built from separately trained A and B models."
        )
    )
    parser.add_argument("--seed", type=int, help="Seed for fine-tuning runs (required unless --aggregate or --plot_only is used with an explicit results_path).")
    parser.add_argument(
        "--joint_checkpoint",
        type=str,
        help="Path to the joint A/B checkpoint (defaults to artifacts/joint/joint_seed{seed}).",
    )
    parser.add_argument(
        "--atomic_a_checkpoint",
        type=str,
        help="Path to the task A checkpoint (defaults to artifacts/atomic/A_seed{seed}).",
    )
    parser.add_argument(
        "--atomic_b_checkpoint",
        type=str,
        help="Path to the task B checkpoint (defaults to artifacts/atomic/B_seed{seed}).",
    )
    parser.add_argument(
        "--merged_checkpoint",
        type=str,
        help="Where to write the merged checkpoint (defaults to artifacts/merged/merged_seed{seed}).",
    )
    parser.add_argument(
        "--merge_scale",
        type=float,
        default=None,
        help="Delta scaling factor when merging checkpoints (defaults to average delta, i.e., 1/num_deltas).",
    )
    parser.add_argument(
        "--merge_scales",
        type=str,
        default=None,
        help="Comma-separated list of delta scales to sweep (e.g., 0.1,0.5,1.0). Overrides --merge_scale when set.",
    )
    parser.add_argument(
        "--finetune_output_dir",
        type=str,
        default="artifacts/finetune",
        help="Base directory for fine-tuned checkpoints.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory for per-run metric files produced by fine-tuning.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Aggregate comparison JSON path (defaults to results/compare_seed{seed}.json).",
    )
    parser.add_argument(
        "--force_merge",
        action="store_true",
        help="Recompute the merged checkpoint even if the target directory already exists.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a bar plot comparing S99 steps from joint vs. merged fine-tuning.",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        help="Where to write the plot (defaults to results_path with .png).",
    )
    parser.add_argument(
        "--curve_plot_path",
        type=str,
        help="Where to write eval-vs-steps curve plot (defaults to results_path with _curves.png).",
    )
    parser.add_argument(
        "--curve_metric",
        type=str,
        default="eval_exact",
        help="Eval metric key to plot against training steps.",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only render the plot from an existing results file without re-running fine-tuning.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate existing compare_seed*.json files across seeds and plot with error bars.",
    )
    parser.add_argument(
        "--aggregate_results_glob",
        type=str,
        help="Glob for comparison JSON files (defaults to results_dir/compare_seed*.json).",
    )
    parser.add_argument(
        "--aggregate_plot_path",
        type=str,
        help="Where to write the aggregate plot (defaults to results_dir/compare_aggregate.png).",
    )
    parser.add_argument(
        "--aggregate_stats_path",
        type=str,
        help="Where to write aggregate stats JSON (defaults to results_dir/compare_aggregate.json).",
    )
    parser.add_argument(
        "--aggregate_curve_path",
        type=str,
        help="Where to write aggregate eval-history plot (defaults to results_dir/compare_aggregate_curves.png).",
    )
    add_shared_training_args(parser)
    return parser.parse_args()


def fill_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if not args.aggregate_results_glob:
        args.aggregate_results_glob = os.path.join(args.results_dir, "compare_seed*.json")
    if not args.aggregate_plot_path:
        args.aggregate_plot_path = os.path.join(args.results_dir, "compare_aggregate.png")
    if not args.aggregate_stats_path:
        args.aggregate_stats_path = os.path.join(args.results_dir, "compare_aggregate.json")
    if not args.aggregate_curve_path:
        args.aggregate_curve_path = os.path.join(args.results_dir, "compare_aggregate_curves.png")

    if args.results_path and not args.plot_path:
        args.plot_path = f"{os.path.splitext(args.results_path)[0]}.png"
    if args.results_path and not args.curve_plot_path:
        args.curve_plot_path = f"{os.path.splitext(args.results_path)[0]}_curves.png"

    if args.seed is None:
        # When aggregating or plotting only with explicit paths, leave seed-specific defaults untouched.
        return args

    if not args.joint_checkpoint:
        args.joint_checkpoint = f"artifacts/joint/joint_seed{args.seed}"
    if not args.atomic_a_checkpoint:
        args.atomic_a_checkpoint = f"artifacts/atomic/A_seed{args.seed}"
    if not args.atomic_b_checkpoint:
        args.atomic_b_checkpoint = f"artifacts/atomic/B_seed{args.seed}"
    if not args.merged_checkpoint:
        args.merged_checkpoint = f"artifacts/merged/merged_seed{args.seed}"
    if not args.results_path:
        args.results_path = f"results/compare_seed{args.seed}.json"
    if not args.plot_path:
        args.plot_path = f"{os.path.splitext(args.results_path)[0]}.png"
    if not args.curve_plot_path:
        args.curve_plot_path = f"{os.path.splitext(args.results_path)[0]}_curves.png"
    if not args.aggregate_results_glob:
        args.aggregate_results_glob = os.path.join(args.results_dir, "compare_seed*.json")
    if not args.aggregate_plot_path:
        args.aggregate_plot_path = os.path.join(args.results_dir, "compare_aggregate.png")
    if not args.aggregate_stats_path:
        args.aggregate_stats_path = os.path.join(args.results_dir, "compare_aggregate.json")
    return args


def ensure_merged_checkpoint(args: argparse.Namespace, target_dir: str, delta_scale: float | None) -> str:
    if os.path.isdir(target_dir) and not args.force_merge:
        return target_dir
    ensure_dir(target_dir)
    return merge_checkpoints(
        checkpoint_a=args.atomic_a_checkpoint,
        checkpoint_b=args.atomic_b_checkpoint,
        output_dir=target_dir,
        base_seed=args.seed,
        delta_scale=delta_scale,
    )


def plot_summary(summary: Dict, path: str) -> None:
    """Render a simple bar chart comparing S99 steps."""

    joint = summary["finetune_from_joint"]["s99_steps"]
    merged_entry = summary["finetune_from_merged"]
    merge_label = "from merged"
    if isinstance(merged_entry, dict) and "s99_steps" not in merged_entry:
        best = None
        best_key = None
        for key, value in merged_entry.items():
            if not isinstance(value, dict) or "s99_steps" not in value:
                continue
            if best is None or value["s99_steps"] < best["s99_steps"]:
                best = value
                best_key = key
        merged_entry = best
        if best_key is not None:
            merge_label = f"from merged (λ={best_key})"
    merged = merged_entry["s99_steps"] if merged_entry else float("inf")
    ratio = merged / joint if joint else float("inf")
    labels = ["from joint", merge_label]
    values = [joint, merged]

    ensure_dir(os.path.dirname(path) or ".")
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("S99 steps")
    ax.set_title(f"Seed {summary['seed']} (merged/joint = {ratio:.2f}x)")
    ax.bar_label(bars, fmt="%.0f", padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote plot to {path}")


def _describe(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    mean_val = float(statistics.mean(values))
    std_val = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean_val, "std": std_val, "count": len(values)}


def plot_aggregate(records: List[Dict], plot_path: str, stats_path: str) -> None:
    """Aggregate S99 steps across seeds and plot with error bars."""

    def _select_merged(rec: Dict) -> Dict | None:
        merged = rec.get("finetune_from_merged")
        if merged is None:
            return None
        if isinstance(merged, dict) and "s99_steps" not in merged:
            best = None
            best_key = None
            for key, value in merged.items():
                if not isinstance(value, dict) or "s99_steps" not in value:
                    continue
                if best is None or value["s99_steps"] < best["s99_steps"]:
                    best = value
                    best_key = key
            if best is None:
                return None
            copy = dict(best)
            copy["_merge_scale"] = best_key
            return copy
        return merged

    joint_values = [rec["finetune_from_joint"]["s99_steps"] for rec in records]
    merged_records = [_select_merged(rec) for rec in records]
    merged_values = [rec["s99_steps"] for rec in merged_records if rec]
    ratio_values = []
    for joint, merged in zip(joint_values, merged_records):
        if merged is None:
            continue
        merged_val = merged["s99_steps"]
        ratio_values.append(merged_val / joint if joint else float("inf"))

    joint_stats = _describe(joint_values)
    merged_stats = _describe(merged_values)
    ratio_stats = _describe(ratio_values)

    ensure_dir(os.path.dirname(plot_path) or ".")
    labels = ["from joint", "from merged"]
    data = [joint_values, merged_values]
    colors = ["#4c72b0", "#dd8452"]
    positions = [1, 2]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    rng = np.random.default_rng(seed=0)
    for pos, vals, color in zip(positions, data, colors):
        if not vals:
            continue
        jitter = rng.normal(loc=0.0, scale=0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            color=color,
            edgecolor="black",
            alpha=0.8,
            zorder=3,
            s=30,
            label="seed" if pos == positions[0] else None,
        )

    ax.set_xticks(positions, labels)
    ax.set_ylabel("S99 steps")
    ax.set_title(
        f"S99 across seeds (n={joint_stats['count']}, merged/joint={ratio_stats['mean']:.2f}±{ratio_stats['std']:.2f}x)"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Wrote aggregate plot to {plot_path}")

    stats_payload = {
        "count": joint_stats["count"],
        "joint": joint_stats,
        "merged": merged_stats,
        "merged_over_joint": ratio_stats,
        "seeds": [rec.get("seed") for rec in records],
    }
    ensure_dir(os.path.dirname(stats_path) or ".")
    write_json(stats_path, stats_payload)
    print(f"Wrote aggregate stats to {stats_path}")


def plot_aggregate_history(records: List[Dict], metric: str, path: str) -> None:
    """Overlay eval histories across seeds for joint vs. merged fine-tunes."""

    def _xy(rec: Dict, key: str):
        xs: List[int] = []
        ys: List[float] = []
        hist = rec.get(key, {}).get("eval_history") or []
        for entry in hist:
            if metric not in entry:
                continue
            xs.append(int(entry.get("step", 0)))
            ys.append(float(entry[metric]))
        return xs, ys

    def _select_merged(rec: Dict) -> Dict | None:
        merged = rec.get("finetune_from_merged")
        if merged is None:
            return None
        if isinstance(merged, dict) and "eval_history" not in merged:
            best = None
            for value in merged.values():
                if not isinstance(value, dict):
                    continue
                if best is None or value.get("s99_steps", float("inf")) < best.get("s99_steps", float("inf")):
                    best = value
            return best
        return merged

    joint_curves: List[Tuple[List[int], List[float]]] = []
    merged_curves: Dict[str, List[Tuple[List[int], List[float]]]] = {}
    label_order: List[str] = []
    for rec in records:
        jx, jy = _xy(rec, "finetune_from_joint")
        merged_entry = rec.get("finetune_from_merged")
        merge_scales = rec.get("merge_scales") or []
        if isinstance(merged_entry, dict) and "eval_history" not in merged_entry:
            for key, value in merged_entry.items():
                if not isinstance(value, dict):
                    continue
                mx, my = _xy({"tmp": value}, "tmp")
                if not mx:
                    continue
                merged_curves.setdefault(key, []).append((mx, my))
                if key not in label_order:
                    label_order.append(key)
        else:
            merged_rec = _select_merged(rec)
            mx, my = _xy({"tmp": merged_rec} if merged_rec else {"tmp": {}}, "tmp")
            if mx:
                key = "avg"
                if merged_rec and merged_rec.get("merge_scale") is not None:
                    key = _format_scale(merged_rec["merge_scale"])
                merged_curves.setdefault(key, []).append((mx, my))
                if key not in label_order:
                    label_order.append(key)

        if jx:
            joint_curves.append((jx, jy))

        for sc in merge_scales:
            lbl = "avg" if sc is None else _format_scale(sc)
            if lbl not in label_order:
                label_order.append(lbl)

    if not (joint_curves or merged_curves):
        print(f"No eval history found for metric '{metric}'; skipping aggregate history plot.")
        return

    ensure_dir(os.path.dirname(path) or ".")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for xs, ys in joint_curves:
        ax.plot(
            xs,
            ys,
            color="#4c72b0",
            alpha=0.5,
            linewidth=1.8,
            label="from joint" if ax.get_legend_handles_labels()[1].count("from joint") == 0 else None,
        )
    merge_labels = list(merged_curves.keys())
    if not merge_labels:
        merge_labels = label_order
    colors = _color_map(merge_labels or ["merged"], cmap_name="magma")
    for label, curves in merged_curves.items():
        color = colors.get(label, "#dd8452")
        for xs, ys in curves:
            ax.plot(
                xs,
                ys,
                color=color,
                alpha=0.5,
                linewidth=1.8,
                label=f"merged λ={label}"
                if ax.get_legend_handles_labels()[1].count(f"merged λ={label}") == 0
                else None,
            )

    ax.set_xlabel("Training steps")
    ax.set_ylabel(metric)
    ax.set_title(f"Eval history across seeds ({metric})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote aggregate history plot to {path}")


def plot_eval_curves(summary: Dict, metric: str, path: str) -> None:
    """Plot eval metric vs. training steps for joint vs. merged fine-tuning."""

    def _extract_xy(record: Dict):
        xs: List[int] = []
        ys: List[float] = []
        for entry in record.get("eval_history") or []:
            if metric not in entry:
                continue
            xs.append(int(entry.get("step", 0)))
            ys.append(float(entry[metric]))
        return xs, ys

    joint_x, joint_y = _extract_xy(summary.get("finetune_from_joint", {}))

    merged_entry = summary.get("finetune_from_merged")
    merged_items: List[Tuple[str, Dict]] = []
    merge_scales = summary.get("merge_scales") or []
    if isinstance(merged_entry, dict) and "s99_steps" not in merged_entry:
        for key, value in merged_entry.items():
            if isinstance(value, dict) and value.get("eval_history"):
                merged_items.append((key, value))
    elif isinstance(merged_entry, dict):
        label = "avg" if merged_entry.get("merge_scale") is None else _format_scale(
            merged_entry.get("merge_scale")
        )
        merged_items.append((label, merged_entry))

    if not (joint_x or merged_items):
        print(f"No eval history found for metric '{metric}'; skipping curve plot at {path}")
        return

    labels = [lbl for lbl, _ in merged_items]
    # Preserve user-provided scale order when possible.
    ordered_labels = [("avg" if sc is None else _format_scale(sc)) for sc in merge_scales] or labels
    colors = _color_map(ordered_labels, cmap_name="magma")

    ensure_dir(os.path.dirname(path) or ".")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    if joint_x:
        ax.plot(joint_x, joint_y, label="from joint", marker="o", color="#4c72b0", linewidth=2.4)
    for label, rec in merged_items:
        xs, ys = _extract_xy(rec)
        if not xs:
            continue
        color = colors.get(label, "#dd8452")
        ax.plot(
            xs,
            ys,
            label=f"merged λ={label}",
            marker="o",
            linewidth=2.0,
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel("Training steps")
    ax.set_ylabel(metric)
    ax.set_title(f"Seed {summary.get('seed', '?')} eval vs. steps ({metric})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote eval curve plot to {path}")


def main() -> None:
    args = fill_default_paths(parse_args())

    if args.aggregate:
        paths = sorted(glob.glob(args.aggregate_results_glob))
        if not paths:
            raise FileNotFoundError(f"No comparison files found matching {args.aggregate_results_glob}")
        records: List[Dict] = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        plot_aggregate(records, args.aggregate_plot_path, args.aggregate_stats_path)
        plot_aggregate_history(records, args.curve_metric, args.aggregate_curve_path)
        return

    if args.plot_only:
        if not args.results_path:
            raise ValueError("results_path must be provided for --plot_only.")
        if not os.path.isfile(args.results_path):
            raise FileNotFoundError(f"results_path not found: {args.results_path}")
        with open(args.results_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if args.plot:
            plot_summary(summary, args.plot_path)
            plot_eval_curves(summary, args.curve_metric, args.curve_plot_path)
        return

    if args.seed is None:
        raise ValueError("Seed is required unless --aggregate or --plot_only is specified.")

    set_seed(args.seed)

    merge_scales = _parse_scale_list(args.merge_scales, args.merge_scale)
    if not merge_scales:
        merge_scales = [None]
    multi_merge = len(merge_scales) > 1

    joint_output = os.path.join(args.finetune_output_dir, f"joint_seed{args.seed}")

    joint_record = run_finetune(
        seed=args.seed,
        init_checkpoint=args.joint_checkpoint,
        output_dir=joint_output,
        results_dir=args.results_dir,
        context_length=args.context_length,
        dataset_size=args.dataset_size,
        eval_samples=args.eval_samples,
        per_device_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        eval_refine_rounds=args.eval_refine_rounds,
        rollback_branches=args.rollback_branches,
        success_threshold=args.success_threshold,
        greedy_eval_batch_size=args.greedy_eval_batch_size,
        greedy_eval_max_new_tokens=args.greedy_eval_max_new_tokens,
        greedy_eval_match_target_length=args.greedy_eval_match_target_length,
        atomic_mix_fraction=args.atomic_mix_fraction,
        run_label="finetune_from_joint",
        eval_jitter_fraction=args.eval_jitter_fraction,
        train_full_steps=args.train_full_steps,
        eval_data_seed=args.eval_data_seed,
    )

    merged_records: Dict[str, Dict] = {}
    merged_checkpoint_base = args.merged_checkpoint or f"artifacts/merged/merged_seed{args.seed}"
    merged_output_base = os.path.join(args.finetune_output_dir, f"merged_seed{args.seed}")
    for scale in merge_scales:
        label = "avg" if scale is None else _format_scale(scale)
        target_dir = merged_checkpoint_base if not multi_merge else f"{merged_checkpoint_base}_lambda{label}"
        merged_checkpoint = ensure_merged_checkpoint(args, target_dir, scale)
        merged_output = merged_output_base if not multi_merge else f"{merged_output_base}_lambda{label}"
        run_label = "finetune_from_merged" if not multi_merge else f"finetune_from_merged_lambda{label}"

        merged_record = run_finetune(
            seed=args.seed,
            init_checkpoint=merged_checkpoint,
            output_dir=merged_output,
            results_dir=args.results_dir,
            context_length=args.context_length,
            dataset_size=args.dataset_size,
            eval_samples=args.eval_samples,
            per_device_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            grad_accum=args.grad_accum,
            max_steps=args.max_steps,
            eval_steps=args.eval_steps,
            eval_refine_rounds=args.eval_refine_rounds,
            rollback_branches=args.rollback_branches,
            success_threshold=args.success_threshold,
            greedy_eval_batch_size=args.greedy_eval_batch_size,
            greedy_eval_max_new_tokens=args.greedy_eval_max_new_tokens,
            greedy_eval_match_target_length=args.greedy_eval_match_target_length,
            atomic_mix_fraction=args.atomic_mix_fraction,
            run_label=run_label,
            eval_jitter_fraction=args.eval_jitter_fraction,
            train_full_steps=args.train_full_steps,
            merge_scale=scale,
            eval_data_seed=args.eval_data_seed,
        )
        merged_records[label] = merged_record

    merged_payload: Dict | List[Dict] | None
    if not merged_records:
        merged_payload = None
    elif multi_merge:
        merged_payload = merged_records
    else:
        merged_payload = next(iter(merged_records.values()))

    summary: Dict = {
        "seed": args.seed,
        "joint_checkpoint": args.joint_checkpoint,
        "atomic_checkpoints": {
            "A": args.atomic_a_checkpoint,
            "B": args.atomic_b_checkpoint,
        },
        "merged_checkpoint": merged_checkpoint_base,
        "merge_scales": merge_scales,
        "finetune_from_joint": joint_record,
        "finetune_from_merged": merged_payload,
    }
    ensure_dir(os.path.dirname(args.results_path))
    write_json(args.results_path, summary)
    print(f"Wrote comparison to {args.results_path}")
    if args.plot:
        plot_summary(summary, args.plot_path)
        plot_eval_curves(summary, args.curve_metric, args.curve_plot_path)


if __name__ == "__main__":
    main()
