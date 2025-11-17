"""Aggregate and visualize ablation drops across families.

Reads JSON outputs from ``analyze_ablation.py`` (see run_ablation_sbatch.sh),
computes per-layer statistics, and produces both distributional (violin) plots
and layer-layer correlation heatmaps. Useful for contrasting where the C vs.
NC vs. GENERAL (AB pretrain) models localize the reverse operation.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


AVAILABLE_METRICS = ["joint", "attention", "mlp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation drops across families.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/ablation",
        help=(
            "Default ablation results directory (used when a scenario-specific directory is not "
            "provided)."
        ),
    )
    parser.add_argument(
        "--atomic_results_dir",
        type=str,
        default=None,
        help=(
            "Directory containing ablation JSON files evaluated on atomic tasks (e.g., A/B). "
            "Defaults to --results_dir when omitted."
        ),
    )
    parser.add_argument(
        "--composed_results_dir",
        type=str,
        default=None,
        help=(
            "Directory containing ablation JSON files evaluated on the composed task (e.g., C). "
            "Defaults to --results_dir when omitted."
        ),
    )
    parser.add_argument(
        "--analysis_mode",
        type=str,
        choices=["atomic", "composed", "both"],
        default="atomic",
        help=(
            "Which localization experiments to summarize. Set to 'both' to emit plots for "
            "atomic and composed tasks in a single run."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=AVAILABLE_METRICS + ["all"],
        help="Which ablation series to aggregate (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/ablation_summary.png",
        help=(
            "Path template for violin plots. If it contains '{metric}' it will be filled in; "
            "otherwise the metric name is appended before the extension. Set to '' to skip."
        ),
    )
    parser.add_argument(
        "--correlation_output",
        type=str,
        default="plots/ablation_correlation.png",
        help=(
            "Path template for correlation plots. Supports '{metric}' substitution similar to --output. "
            "Set to '' to skip."
        ),
    )
    return parser.parse_args()


def resolve_output_path(base: str, metric: str, scenario: str | None = None) -> str:
    if not base:
        return ""
    fmt_kwargs = {
        "metric": metric,
        "scenario": scenario or metric,
    }
    if "{metric}" in base or "{scenario}" in base:
        return base.format(**fmt_kwargs)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"
    parts = []
    if scenario:
        parts.append(scenario)
    parts.append(metric)
    suffix = "_".join(parts)
    return f"{root}_{suffix}{ext}"


def load_records(results_dir: str, metric: str) -> List[Dict]:
    def iter_task_metrics(data: Dict) -> List[Tuple[str, Dict]]:
        per_task = data.get("per_task")
        if isinstance(per_task, dict) and per_task:
            return list(per_task.items())
        fallback = {}
        for key in ["baseline", *AVAILABLE_METRICS]:
            value = data.get(key)
            if value is not None:
                fallback[key] = value
        return [("", fallback)]

    records: List[Dict] = []
    for path in glob.glob(os.path.join(results_dir, "*", "*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        for task_label, metrics in iter_task_metrics(data):
            if not isinstance(metrics, dict):
                continue
            series = metrics.get(metric)
            if not series:
                continue
            family = data.get("family") or os.path.basename(os.path.dirname(path))
            label = family if not task_label else f"{family}/{task_label}"
            records.append(
                {
                    "family": label,
                    "parent_family": family,
                    "task": task_label,
                    "checkpoint": os.path.splitext(os.path.basename(path))[0],
                    "baseline": metrics.get("baseline"),
                    "series": series,
                }
            )
    return records


def collect_family_runs(records: List[Dict]) -> Dict[str, List[Dict[str, object]]]:
    """Return family -> list of run dicts containing labels and layer drops."""

    runs: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        family = rec["family"]
        layer_map = {entry["layer"]: entry["drop"] for entry in rec["series"]}
        run_label = rec["checkpoint"]
        if rec.get("task"):
            run_label = f"{run_label}/{rec['task']}"
        runs[family].append({"label": run_label, "layers": layer_map})
    return runs


def aggregate_layer_drops(
    family_runs: Dict[str, List[Dict[str, object]]]
) -> Dict[str, Dict[int, List[float]]]:
    """Return family -> layer -> list of drops aggregated across runs."""

    agg: Dict[str, Dict[int, List[float]]] = {}
    for family, run_entries in family_runs.items():
        per_layer: Dict[int, List[float]] = defaultdict(list)
        for entry in run_entries:
            run_map = entry["layers"]
            for layer, drop in run_map.items():
                per_layer[layer].append(drop)
        agg[family] = per_layer
    return agg


def build_seed_matrix(
    run_entries: List[Dict[str, object]]
) -> Tuple[List[int], np.ndarray, List[str]]:
    layers = sorted({layer for entry in run_entries for layer in entry["layers"].keys()})
    if not layers:
        return [], np.empty((0, 0)), []
    rows = []
    labels = []
    for entry in run_entries:
        row = [entry["layers"].get(layer, np.nan) for layer in layers]
        if any(math.isnan(value) for value in row):
            continue
        rows.append(row)
        labels.append(str(entry["label"]))
    if not rows:
        return layers, np.empty((0, 0)), []
    matrix = np.array(rows)
    return layers, matrix, labels


def order_runs_by_similarity(corr: np.ndarray) -> List[int]:
    n = corr.shape[0]
    if n <= 2:
        return list(range(n))
    remaining = set(range(n))
    first = max(range(n), key=lambda i: np.nansum(corr[i]))
    order = [first]
    remaining.remove(first)
    while remaining:
        last = order[-1]
        next_idx = max(remaining, key=lambda i: corr[last, i])
        order.append(next_idx)
        remaining.remove(next_idx)
    return order


def compute_seed_correlations(
    family_runs: Dict[str, List[Dict[str, object]]]
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Return family -> (run_ids, seed-wise correlation matrix)."""

    corr_data: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for family, run_entries in family_runs.items():
        if len(run_entries) < 2:
            continue
        layers, matrix, run_labels = build_seed_matrix(run_entries)
        if matrix.size == 0 or matrix.shape[0] < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(matrix)
        corr = np.nan_to_num(corr, nan=0.0)
        order = order_runs_by_similarity(corr)
        corr = corr[np.ix_(order, order)]
        run_labels = [run_labels[idx] for idx in order]
        corr_data[family] = (run_labels, corr)
    return corr_data


def summarize(records: List[Dict], scenario: str) -> None:
    if not records:
        print(f"[{scenario}] No records found for the requested metric.")
        return
    print(f"[{scenario}] Loaded {len(records)} ablation runs.")
    by_family: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        by_family[rec["family"]].append(rec)
    for family, fam_recs in by_family.items():
        # For each run, find the layer with max drop for the chosen metric.
        top_layers = []
        for rec in fam_recs:
            series = rec["series"]
            top = max(series, key=lambda e: e["drop"])
            top_layers.append(top["layer"])
        fam_mean_top = mean(top_layers)
        print(
            f"[{scenario}] {family}: runs={len(fam_recs)}, "
            f"mean top-layer={fam_mean_top:.2f}, "
            f"top-layer hist (first 10)={top_layers[:10]}"
        )


def plot_layer_violins(layer_drops: Dict[str, Dict[int, List[float]]], output: str) -> None:
    if not output:
        return
    if not layer_drops:
        print("No layer drops to plot.")
        return

    families = sorted(layer_drops.keys())
    n_families = len(families)
    ncols = min(3, n_families) or 1
    nrows = math.ceil(n_families / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, family in enumerate(families):
        ax = axes_flat[idx]
        layer_dict = layer_drops[family]
        if not layer_dict:
            ax.set_visible(False)
            continue
        layers = sorted(layer_dict.keys())
        data = [layer_dict[layer] for layer in layers]
        parts = ax.violinplot(
            data,
            positions=layers,
            showmeans=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_alpha(0.6)
        ax.set_title(family)
        ax.set_xlabel("Layer")
        ax.set_xticks(layers)
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy drop")
        else:
            ax.set_ylabel("")

    for extra_ax in axes_flat[n_families:]:
        extra_ax.set_visible(False)

    fig.suptitle("Per-layer ablation drops (distribution across runs)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Wrote violin plot to {output}")


def plot_seed_correlations(
    corr_data: Dict[str, Tuple[List[str], np.ndarray]], output: str
) -> None:
    if not output:
        return
    if not corr_data:
        print("No correlation data available (need >=2 runs per family).")
        return

    families = sorted(corr_data.keys())
    n_families = len(families)
    ncols = min(3, n_families) or 1
    nrows = math.ceil(n_families / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()
    vmin, vmax = -1.0, 1.0

    for idx, family in enumerate(families):
        ax = axes_flat[idx]
        run_labels, corr = corr_data[family]
        im = ax.imshow(corr, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ticks = list(range(len(run_labels)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(run_labels)
        ax.set_yticklabels(run_labels)
        ax.set_title(family)
        ax.set_xlabel("Seed/Run")
        if idx % ncols == 0:
            ax.set_ylabel("Seed/Run")
        else:
            ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for extra_ax in axes_flat[n_families:]:
        extra_ax.set_visible(False)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        ax=axes_flat[:n_families],
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Pearson r")

    fig.suptitle("Seed-wise correlation of ablation drops")
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Wrote correlation heatmap to {output}")


def main() -> None:
    args = parse_args()
    metrics = AVAILABLE_METRICS if args.metric == "all" else [args.metric]

    scenarios: List[Tuple[str, str]] = []
    if args.analysis_mode in {"atomic", "both"}:
        atomic_dir = args.atomic_results_dir or args.results_dir
        scenarios.append(("atomic", atomic_dir))
    if args.analysis_mode in {"composed", "both"}:
        composed_dir = args.composed_results_dir or args.results_dir
        scenarios.append(("composed", composed_dir))

    if not scenarios:
        raise ValueError("No scenarios requested. Check --analysis_mode.")

    for scenario_label, results_dir in scenarios:
        if not results_dir:
            print(f"Skipping scenario '{scenario_label}' (no directory provided).")
            continue
        if not os.path.isdir(results_dir):
            print(
                f"Skipping scenario '{scenario_label}' because directory '{results_dir}' does not exist."
            )
            continue

        for metric in metrics:
            print(f"=== Scenario: {scenario_label} | Metric: {metric} ===")
            records = load_records(results_dir, metric)
            summarize(records, scenario_label)
            if not records:
                continue
            family_runs = collect_family_runs(records)
            layer_drops = aggregate_layer_drops(family_runs)
            metric_output = resolve_output_path(args.output, metric, scenario_label)
            plot_layer_violins(layer_drops, metric_output)
            corr_data = compute_seed_correlations(family_runs)
            metric_corr_output = resolve_output_path(
                args.correlation_output, metric, scenario_label
            )
            plot_seed_correlations(corr_data, metric_corr_output)


if __name__ == "__main__":
    main()
