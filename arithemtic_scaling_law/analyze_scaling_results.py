"""Plot sample-complexity curves for arithmetic scaling-law runs."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval metrics from scaling-law results.")
    parser.add_argument("--results_dir", type=str, default="arithemtic_scaling_law/results")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <results_dir>/plots")
    parser.add_argument(
        "--metric",
        type=str,
        default="eval_acc_expr",
        help="Metric key under 'metrics' to plot from finetuning results.",
    )
    return parser.parse_args()


def load_results(results_dir: str, metric: str) -> List[Dict]:
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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")

    records = load_results(args.results_dir, args.metric)
    if not records:
        raise SystemExit(f"No sample_complexity results with metric '{args.metric}' found under {args.results_dir}")

    grouped = group_by_k(records)
    for k, curves in sorted(grouped.items()):
        if not curves:
            continue
        out = plot_k_curves(k, curves, args.metric, output_dir)
        print(f"k={k}: wrote {out}")


if __name__ == "__main__":
    main()
