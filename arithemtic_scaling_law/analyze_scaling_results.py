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
    slugs = sorted(curves.keys())
    cmap = plt.get_cmap("viridis")
    color_vals = np.linspace(0, 1, len(slugs) or 1)
    for color, slug in zip(cmap(color_vals), slugs):
        points = curves[slug]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=slug, color=color)
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
    """Estimate crossing distribution using interval-censored bootstrap.

    Each record is treated as having an interval [L, U] where U is the first
    threshold-crossing eval step and L is the prior eval step. Missing
    crossings are right-censored at max_steps.
    """

    def _infer_interval(rec: Dict) -> Tuple[float, float | None, float]:
        eval_steps = rec.get("eval_steps") or None
        thresholds: List[int] = list(rec.get("threshold_steps") or [])
        thresholds = sorted(int(s) for s in thresholds)
        max_steps = float(rec.get("max_steps") or rec.get("s99_steps") or thresholds[-1] if thresholds else 0)
        if thresholds:
            first = float(thresholds[0])
            interval = float(eval_steps or (thresholds[1] - thresholds[0] if len(thresholds) >= 2 else first))
            lower = max(0.0, first - interval)
            upper = first
            return lower, upper, max_steps
        # Right-censored at max_steps when never crossed.
        return float(max_steps), None, float(max_steps)

    def _bootstrap_median(intervals: List[Tuple[float, float | None]], max_steps: float, samples: int = 500) -> Dict[str, float]:
        if not intervals:
            return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        medians: List[float] = []
        rng = np.random.default_rng(seed=0)
        for _ in range(samples):
            draw_vals: List[float] = []
            for low, high in rng.choice(intervals, size=len(intervals), replace=True):
                upper = max_steps if high is None else float(high)
                lower = min(float(low), upper)
                if upper > lower:
                    val = rng.uniform(lower, upper)
                else:
                    val = upper
                draw_vals.append(val)
            medians.append(float(np.median(draw_vals)))
        med_arr = np.array(medians, dtype=float)
        return {
            "median": float(np.median(med_arr)),
            "ci_low": float(np.percentile(med_arr, 2.5)),
            "ci_high": float(np.percentile(med_arr, 97.5)),
        }

    acc: Dict[int, List[Tuple[float, float | None]]] = defaultdict(list)
    max_seen: Dict[int, float] = defaultdict(float)
    fallback: Dict[int, List[float]] = defaultdict(list)

    for rec in records:
        k = int(rec.get("k", -1))
        lower, upper, max_steps = _infer_interval(rec)
        if lower == upper == 0:
            continue
        max_seen[k] = max(max_seen[k], max_steps)
        acc[k].append((lower, upper))
        s99 = rec.get("s99_steps")
        if s99 is not None:
            fallback[k].append(float(s99))

    stats: Dict[int, Dict[str, float]] = {}
    for k, intervals in acc.items():
        max_steps = max_seen.get(k, max(fallback.get(k, [0.0])) or 0.0)
        midpoints = []
        for low, high in intervals:
            upper = max_steps if high is None else high
            lower = min(low, upper)
            midpoints.append((lower + upper) / 2.0)
        mid_arr = np.array(midpoints, dtype=float)

        # Trim extreme midpoint outliers to keep downstream std/CI reasonable.
        keep_mask = np.isfinite(mid_arr)
        positive_mask = keep_mask & (mid_arr > 0)
        if np.count_nonzero(positive_mask) >= 2:
            vals = mid_arr[positive_mask]
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            keep_mask &= (mid_arr >= lower) & (mid_arr <= upper)
        if np.count_nonzero(positive_mask) >= 1:
            med = float(np.median(mid_arr[positive_mask]))
            if med > 0:
                keep_mask &= mid_arr <= med * 50.0
        if not np.any(keep_mask):
            keep_mask = np.isfinite(mid_arr)

        trimmed_mid = mid_arr[keep_mask]
        trimmed_intervals = [iv for iv, keep in zip(intervals, keep_mask) if keep]
        intervals_for_boot = trimmed_intervals if trimmed_intervals else intervals
        mid_for_stats = trimmed_mid if trimmed_mid.size else mid_arr

        # Cap the effective max_steps for bootstrap to prevent heavy right-censor tails.
        cap = None
        if mid_for_stats.size:
            cap = float(np.median(mid_for_stats) * 10.0)
            if cap <= 0:
                cap = None

        if cap is not None:
            capped_intervals = []
            for low, high in intervals_for_boot:
                upper = max_steps if high is None else float(high)
                upper = min(upper, cap)
                lower = min(float(low), upper)
                capped_intervals.append((lower, upper))
            intervals_for_boot = capped_intervals
            capped_mid = np.array([(lo + hi) / 2.0 for lo, hi in intervals_for_boot], dtype=float)
            if capped_mid.size:
                mid_for_stats = capped_mid

        boot = _bootstrap_median(intervals_for_boot, max_steps)
        stats[k] = {
            "mean": float(np.mean(mid_for_stats)),
            "std": float(np.std(mid_for_stats)),
            "median": boot["median"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "count": int(mid_for_stats.size),
        }
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

    cmap = plt.get_cmap("plasma")
    color_vals = np.linspace(0, 1, len(train_levels) or 1)

    for color, train_k in zip(cmap(color_vals), train_levels):
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
        plt.errorbar(xs, ys, yerr=errs, fmt="o-", capsize=3, label=f"train k={train_k}", color=color)

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
    medians = [sc_stats[k].get("median", sc_stats[k]["mean"]) for k in ks]
    ci_lows = [sc_stats[k].get("ci_low", sc_stats[k]["mean"]) for k in ks]
    ci_highs = [sc_stats[k].get("ci_high", sc_stats[k]["mean"]) for k in ks]
    lower_err = [m - lo for m, lo in zip(medians, ci_lows)]
    upper_err = [hi - m for m, hi in zip(medians, ci_highs)]

    vals = np.array(medians, dtype=float)
    positive_mask = np.isfinite(vals) & (vals > 0)
    keep_mask = positive_mask.copy()
    # Drop extreme outliers (Tukey fence) before plotting on log scale.
    if np.count_nonzero(positive_mask) >= 2:
        inlier_vals = vals[positive_mask]
        q1, q3 = np.percentile(inlier_vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        keep_mask &= (vals >= lower) & (vals <= upper)
    # Drop values that are extreme relative to the median (handles small-sample cases).
    if np.count_nonzero(positive_mask) >= 1:
        med = float(np.median(vals[positive_mask]))
        if med > 0:
            keep_mask &= vals <= med * 50.0
    err_arr = np.array(upper_err, dtype=float)
    finite_err = np.isfinite(err_arr) & (err_arr >= 0)
    if np.count_nonzero(finite_err) >= 1:
        err_med = float(np.median(err_arr[finite_err]))
        if err_med > 0:
            keep_mask &= err_arr <= err_med * 50.0
        if np.count_nonzero(positive_mask) >= 1:
            keep_mask &= err_arr <= np.maximum(vals, err_med) * 50.0
    # Drop points with CI balloons far above their median.
    if np.count_nonzero(positive_mask) >= 1:
        ci_ratio = np.divide(ci_highs, np.maximum(vals, 1e-12))
        keep_mask &= ci_ratio <= 30.0
    if not np.any(keep_mask):
        keep_mask = positive_mask if np.any(positive_mask) else np.ones_like(vals, dtype=bool)

    filtered = [
        (k, m, lo, hi, le, ue)
        for idx, (k, m, lo, hi, le, ue) in enumerate(zip(ks, medians, ci_lows, ci_highs, lower_err, upper_err))
        if keep_mask[idx]
    ]
    if filtered:
        ks, medians, ci_lows, ci_highs, lower_err, upper_err = map(list, zip(*filtered))

    plt.figure(figsize=(6.4, 4.0))
    plt.errorbar(ks, medians, yerr=[lower_err, upper_err], fmt="o-", capsize=4, label="median ± 95% CI")
    plt.yscale("log")
    plt.title(f"{regime}: sample complexity (interval-censored bootstrap)")
    plt.xlabel("Train level k")
    plt.ylabel("Steps to threshold")
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
