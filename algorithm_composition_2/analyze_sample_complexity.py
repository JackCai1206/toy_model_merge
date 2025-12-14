"""Aggregate sample-complexity estimates across runs using interval-censored bootstrap."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sample complexity from runs.jsonl.")
    parser.add_argument(
        "--runs_path",
        type=str,
        default="results/runs.jsonl",
        help="Path to runs.jsonl (one JSON object per line).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to write the summary JSON (default: runs_path parent + sample_complexity_summary.json).",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        dest="group_by",
        default=None,
        help="Fields to group by. Repeat to add multiple keys. Defaults to ['phase', 'task'].",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=500,
        help="Bootstrap samples for median/CI estimation.",
    )
    return parser.parse_args()


def _load_records(path: str) -> List[Dict]:
    records: List[Dict] = []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"runs file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _infer_interval(rec: Dict) -> Tuple[float, float | None, float]:
    """Infer [lower, upper] crossing interval from a run record."""

    thresholds: List[int] = sorted(int(s) for s in rec.get("threshold_steps") or [])
    eval_steps = rec.get("eval_steps")
    eval_delay = rec.get("eval_delay") or 0
    max_steps = float(
        rec.get("max_steps")
        or rec.get("s99_steps")
        or (thresholds[-1] if thresholds else 0)
    )

    if thresholds:
        first = float(thresholds[0])
        interval = None
        if eval_steps:
            interval = float(eval_steps)
        elif len(thresholds) >= 2:
            interval = float(thresholds[1] - thresholds[0])
        else:
            interval = float(max(first - eval_delay, 1.0))
        interval = max(interval, 1.0)
        lower = max(0.0, first - interval)
        return lower, first, max_steps

    # Never crossed: right-censored at max_steps.
    return float(max_steps), None, float(max_steps)


def _bootstrap_median(
    intervals: List[Tuple[float, float | None]],
    max_steps: float,
    samples: int = 500,
) -> Dict[str, float]:
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


def _format_key(key: Tuple[str, ...], fields: Iterable[str]) -> str:
    parts = []
    for name, value in zip(fields, key):
        parts.append(f"{name}={value}")
    return "|".join(parts)


def aggregate_sample_complexity(
    records: List[Dict], group_by: List[str], bootstrap_samples: int = 500
) -> Dict[str, Dict]:
    """Aggregate interval-censored sample complexity grouped by provided fields."""

    grouped: Dict[Tuple[str, ...], List[Dict]] = defaultdict(list)
    for rec in records:
        key = tuple(str(rec.get(field, "<missing>")) for field in group_by)
        grouped[key].append(rec)

    summary: Dict[str, Dict] = {}
    for key, recs in grouped.items():
        intervals: List[Tuple[float, float | None]] = []
        max_seen = 0.0
        fallback: List[float] = []
        for rec in recs:
            low, high, max_steps = _infer_interval(rec)
            if low == high == 0:
                continue
            max_seen = max(max_seen, max_steps)
            intervals.append((low, high))
            if rec.get("s99_steps") is not None:
                fallback.append(float(rec["s99_steps"]))

        if not intervals and not fallback:
            continue

        max_steps = max_seen or (max(fallback) if fallback else 0.0)
        midpoints = []
        for low, high in intervals:
            upper = max_steps if high is None else float(high)
            lower = min(float(low), upper)
            midpoints.append((lower + upper) / 2.0)
        mid_arr = np.array(midpoints, dtype=float)

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

        boot = _bootstrap_median(intervals_for_boot, max_steps, samples=bootstrap_samples)
        label = _format_key(key, group_by)
        summary[label] = {
            "group_values": {field: value for field, value in zip(group_by, key)},
            "mean": float(np.mean(mid_for_stats)),
            "std": float(np.std(mid_for_stats)),
            "median": boot["median"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "count": int(mid_for_stats.size),
        }
    return summary


def main() -> None:
    args = parse_args()
    group_by = args.group_by or ["phase", "task"]
    records = _load_records(args.runs_path)
    if not records:
        raise SystemExit(f"No records found in {args.runs_path}")

    summary = aggregate_sample_complexity(
        records=records, group_by=group_by, bootstrap_samples=args.bootstrap_samples
    )
    output_path = args.output_path
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(args.runs_path))
        output_path = os.path.join(base_dir, "sample_complexity_summary.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "runs_path": args.runs_path,
        "group_by": group_by,
        "total_records": len(records),
        "summary": summary,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote sample complexity summary ({len(summary)} groups) to {output_path}")
    for label, stats in sorted(summary.items()):
        print(
            f"{label}: median={stats['median']:.1f} "
            f"(95% CI [{stats['ci_low']:.1f}, {stats['ci_high']:.1f}]), "
            f"mean={stats['mean']:.1f}, n={stats['count']}"
        )


if __name__ == "__main__":
    main()
