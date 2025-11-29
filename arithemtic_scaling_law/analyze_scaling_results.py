"""Plot sample-complexity curves for arithmetic scaling-law runs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval metrics from scaling-law results.")
    parser.add_argument("--results_dir", type=str, default=None, help="Legacy: directory of flat sample_complexity JSONs.")
    parser.add_argument(
        "--results_parent",
        type=str,
        default=None,
        help="Parent directory containing run/seed subfolders with level_k*.json results.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <results_dir>/plots or <results_parent>/plots")
    parser.add_argument(
        "--metric",
        type=str,
        default="eval_acc_expr",
        help="Metric key under 'metrics' to plot from finetuning results.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Regime slug filter, e.g., q10_b1. Required when aggregating levels.",
    )
    parser.add_argument(
        "--aggregate_levels",
        action="store_true",
        help="Aggregate per-level results across seeds/run folders (expects level_k*.json files).",
    )
    return parser.parse_args()


def load_results(results_dir: str, metric: str) -> List[Dict]:
    """Legacy loader for flat sample_complexity outputs (phase=sample_complexity)."""
    records: List[Dict] = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("phase") != "sample_complexity":
            continue
        metrics = data.get("metrics") or {}
        if metric not in metrics:
            continue
        regime = data.get("regime") or {}
        records.append(
            {
                "k": int(data.get("k", -1)),
                "n": int(data.get("n_samples", -1)),
                "metric": float(metrics[metric]),
                "regime": regime,
                "path": path,
            }
        )
    return records


def regime_slug(regime: Dict) -> str:
    q = regime.get("q_keep", "q?")
    block = regime.get("max_block_size", "?")
    q_str = str(q).replace(".", "")
    return f"q{q_str}_b{block}"


def group_by_k(records: List[Dict]) -> Dict[int, Dict[str, List[Tuple[int, float]]]]:
    grouped: Dict[int, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        grouped[rec["k"]][regime_slug(rec["regime"])].append((rec["n"], rec["metric"]))
    for k, regime_data in grouped.items():
        for slug, points in regime_data.items():
            regime_data[slug] = sorted(points, key=lambda t: t[0])
    return grouped


def plot_k_curves(k: int, curves: Dict[str, List[Tuple[int, float]]], metric: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6.4, 4.6))
    for slug, points in sorted(curves.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=slug)
    plt.title(f"k={k} sample complexity ({metric})")
    plt.xlabel("N fine-tune examples (k→k+1)")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Regime")
    out_path = os.path.join(output_dir, f"scaling_k{k}_{metric}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _parse_level_file(path: str, regime_filter: str | None) -> Dict | None:
    """Parse a level_k*.json file and return record if it matches the regime filter."""

    fname = os.path.basename(path)
    match = re.match(r"level_k(?P<k>\d+)_?(?P<slug>[^.]+)?\.json", fname)
    if not match:
        return None
    slug = match.group("slug")
    if regime_filter and slug != regime_filter:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["k"] = data.get("k", int(match.group("k")))
    data["regime_slug"] = slug
    data["path"] = path
    data.setdefault("run_name", os.path.basename(os.path.dirname(path)))
    return data


def load_level_records(results_parent: str, regime_filter: str | None) -> List[Dict]:
    pattern = os.path.join(results_parent, "**", "level_k*.json")
    records: List[Dict] = []
    for path in glob.glob(pattern, recursive=True):
        rec = _parse_level_file(path, regime_filter)
        if rec is not None:
            records.append(rec)
    return records


def aggregate_accuracy(records: List[Dict], metric_prefix: str) -> Dict[int, Dict[int, Dict[str, float]]]:
    """Return nested dict: target_level -> train_level -> stats dict."""

    acc: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        train_level = int(rec.get("k", -1))
        metrics_all = rec.get("metrics_all_levels") or {}
        for key, value in metrics_all.items():
            m = re.match(rf"{re.escape(metric_prefix)}_k(?P<level>\d+)", key)
            if not m:
                continue
            target_level = int(m.group("level"))
            acc[target_level][train_level].append(float(value))

    stats: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(dict)
    for target_level, train_map in acc.items():
        for train_level, values in train_map.items():
            arr = np.array(values, dtype=float)
            stats[target_level][train_level] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "count": len(values),
            }
    return stats


def aggregate_sample_complexity(records: List[Dict]) -> Dict[int, Dict[str, float]]:
    acc: Dict[int, List[float]] = defaultdict(list)
    for rec in records:
        k = int(rec.get("k", -1))
        s99 = rec.get("s99_steps")
        if s99 is None:
            continue
        acc[k].append(float(s99))
    stats: Dict[int, Dict[str, float]] = {}
    for k, values in acc.items():
        arr = np.array(values, dtype=float)
        stats[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "count": len(values)}
    return stats


def _build_run_eval_curves(records: List[Dict], metric_prefix: str) -> Dict[str, Dict]:
    """Return mapping of run_name -> {"train_k": int, "points": List[(eval_k, metric)]}."""
    raise NotImplementedError("_build_run_eval_curves no longer used.")


def plot_accuracy_progression(
    accuracy_stats: Dict[int, Dict[int, Dict[str, float]]], metric: str, regime: str, output_dir: str
) -> str:
    """Plot eval vectors for each training level (mean ± std over seeds)."""

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7.0, 4.6))

    targets = sorted(accuracy_stats.keys())
    train_levels = sorted({tl for tgt in targets for tl in accuracy_stats[tgt].keys()})

    for train_k in train_levels:
        xs: List[int] = []
        ys: List[float] = []
        errs: List[float] = []
        for target in targets:
            stats = accuracy_stats[target].get(train_k)
            if stats is None:
                continue
            xs.append(target)
            ys.append(stats["mean"])
            errs.append(stats["std"])
        if not xs:
            continue
        plt.errorbar(xs, ys, yerr=errs, fmt="o-", capsize=3, label=f"train k={train_k}")

    plt.title(f"{regime}: {metric} vs eval level (mean±std over seeds)")
    plt.xlabel("Eval level k")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Train level")
    out_path = os.path.join(output_dir, f"accuracy_progression_{regime}_{metric}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_sample_complexity(sc_stats: Dict[int, Dict[str, float]], regime: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ks = sorted(sc_stats.keys())
    means = [sc_stats[k]["mean"] for k in ks]
    stds = [sc_stats[k]["std"] for k in ks]
    plt.figure(figsize=(6.4, 4.0))
    plt.errorbar(ks, means, yerr=stds, fmt="o-", capsize=4, label="s99_steps")
    plt.title(f"{regime}: sample complexity (mean±std over seeds)")
    plt.xlabel("Train level k")
    plt.ylabel("Steps to threshold (s99_steps)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    out_path = os.path.join(output_dir, f"sample_complexity_{regime}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main() -> None:
    args = parse_args()

    if args.aggregate_levels:
        if not args.results_parent:
            raise SystemExit("--aggregate_levels requires --results_parent pointing to run/seed subfolders.")
        if not args.regime:
            raise SystemExit("--aggregate_levels requires --regime (e.g., q10_b1).")
        output_dir = args.output_dir or os.path.join(args.results_parent, "plots")
        records = load_level_records(args.results_parent, args.regime)
        if not records:
            raise SystemExit(
                f"No level_k*.json results for regime '{args.regime}' under {args.results_parent}"
            )

        acc_stats = aggregate_accuracy(records, args.metric)
        sc_stats = aggregate_sample_complexity(records)
        acc_plot = plot_accuracy_progression(acc_stats, args.metric, args.regime, output_dir)
        sc_plot = plot_sample_complexity(sc_stats, args.regime, output_dir)

        summary = {"accuracy": acc_stats, "sample_complexity": sc_stats}
        summary_path = os.path.join(output_dir, f"aggregate_{args.regime}_{args.metric}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print(f"Wrote accuracy progression plot: {acc_plot}")
        print(f"Wrote sample complexity plot: {sc_plot}")
        print(f"Wrote aggregate summary: {summary_path}")
        return

    results_dir = args.results_dir or "arithemtic_scaling_law/results"
    output_dir = args.output_dir or os.path.join(results_dir, "plots")

    records = load_results(results_dir, args.metric)
    if not records:
        raise SystemExit(f"No sample_complexity results with metric '{args.metric}' found under {results_dir}")

    grouped = group_by_k(records)
    for k, curves in sorted(grouped.items()):
        if not curves:
            continue
        out = plot_k_curves(k, curves, args.metric, output_dir)
        print(f"k={k}: wrote {out}")


if __name__ == "__main__":
    main()
