"""Aggregate Δ_AB metrics and produce simple visualizations."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu, wilcoxon
except ImportError:  # pragma: no cover - optional dependency
    mannwhitneyu = None
    wilcoxon = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Δ_AB distributions.")
    parser.add_argument("--metrics_path", type=str, default="results/delta.jsonl")
    parser.add_argument("--families", nargs="+", default=["NC", "C"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--step_budget", type=int, default=100_000)
    return parser.parse_args()


def load_records(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No delta log found at {path}")
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def summarize(records: List[Dict], families: List[str], step_budget: int) -> Dict[str, Dict]:
    grouped: Dict[str, Dict] = {}
    for family in families:
        fam_records = [r for r in records if r.get("family") == family]
        deltas = [r["delta_ab"] for r in fam_records]
        s99_scratch = [r.get("s99_scratch") for r in fam_records]
        s99_start = [r.get("s99_steps") for r in fam_records]
        if not fam_records:
            grouped[family] = {}
            continue
        grouped[family] = {
            "count": len(fam_records),
            "delta_mean": mean(deltas),
            "delta_median": median(deltas),
            "delta_std": pstdev(deltas) if len(deltas) > 1 else 0.0,
            "scratch_mean": mean(s99_scratch),
            "start_mean": mean(s99_start),
            "success_fraction": sum(step <= step_budget for step in s99_start) / len(s99_start),
        }
    return grouped


def run_tests(records: List[Dict], families: List[str]) -> Dict[str, float]:
    tests = {}
    if wilcoxon is None or mannwhitneyu is None:
        return tests
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["family"]].append(rec["delta_ab"])
    for family in families:
        values = grouped.get(family, [])
        if len(values) > 0:
            try:
                _, p_value = wilcoxon(values)
                tests[f"{family}_wilcoxon"] = p_value
            except ValueError:
                tests[f"{family}_wilcoxon"] = math.nan
    if all(len(grouped.get(fam, [])) > 0 for fam in families[:2]):
        _, p_value = mannwhitneyu(grouped[families[0]], grouped[families[1]], alternative="two-sided")
        tests["mann_whitney"] = p_value
    return tests


def save_violin_plot(records: List[Dict], families: List[str], output_dir: str) -> str:
    data = []
    labels = []
    for family in families:
        values = [r["delta_ab"] for r in records if r.get("family") == family]
        if values:
            data.append(values)
            labels.append(family)
    if not data:
        raise ValueError("No data available for plotting.")
    plt.figure(figsize=(8, 4))
    parts = plt.violinplot(data, showmeans=True, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Δ_AB (log S99 reduction)")
    plt.title("Compositionality Signal")
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "delta_violin.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()
    records = load_records(args.metrics_path)
    summary = summarize(records, args.families, args.step_budget)
    tests = run_tests(records, args.families)
    plot_path = save_violin_plot(records, args.families, args.output_dir)

    print("=== Δ_AB Summary ===")
    for family, stats in summary.items():
        if not stats:
            print(f"{family}: no data")
            continue
        print(
            f"{family}: mean={stats['delta_mean']:.4f}, median={stats['delta_median']:.4f}, "
            f"std={stats['delta_std']:.4f}, count={stats['count']}, "
            f"success≤{args.step_budget}={stats['success_fraction']:.2%}"
        )
    if tests:
        print("\n=== Statistical Tests ===")
        for name, value in tests.items():
            print(f"{name}: p-value={value:.4g}")
    print(f"\nSaved violin plot to {plot_path}")


if __name__ == "__main__":
    main()
