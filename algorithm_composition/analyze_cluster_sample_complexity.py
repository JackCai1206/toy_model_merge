import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List

AVAILABLE_METRICS = ["joint", "attention", "mlp"]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Link pretrain cluster assignments (behavior seeds) to downstream sample complexity "
            "for specified finetuning families."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory containing results, ablation, and cluster center files.",
    )
    parser.add_argument(
        "--pretrain_family",
        type=str,
        default="GENERAL",
        help="Family name used for the pretrain checkpoints (default: GENERAL).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["C", "NC"],
        help="Finetuning families/tasks to evaluate (default: C NC).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=AVAILABLE_METRICS + ["all"],
        help="Which ablation metric(s) to analyze. Choose 'all' to sweep all available metrics.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory for saving plots (boxplots per scenario/metric).",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default="results/sample_complexity_summary_{metric}.json",
        help="Path template for saving summary statistics (set to '' to skip).",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=2,
        help="Minimum number of finetune runs per cluster required to include in summaries (default: 2).",
    )
    return parser.parse_args()


def load_cluster_centers(path: str, family: str) -> Dict[int, list[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    centers: Dict[int, list[float]] = {}
    for entry in data:
        if entry.get("family") != family:
            continue
        centers[int(entry["cluster_id"])] = entry["center"]
    if not centers:
        raise ValueError(f"No cluster centers found for family '{family}' in {path}")
    return centers


def seed_from_filename(path: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    # shared_seed123 -> 123
    if "_seed" not in base:
        raise ValueError(f"Unexpected filename format: {base}")
    return int(base.split("_seed")[-1])


def load_pretrain_vectors(ablation_dir: str, metric: str) -> Dict[int, list[float]]:
    vectors: Dict[int, list[float]] = {}
    for path in sorted(glob.glob(os.path.join(ablation_dir, "*.json"))):
        seed = seed_from_filename(path)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        per_task = data.get("per_task", {})
        vector: list[float] = []
        valid = True
        for task in ("A", "B"):
            series = per_task.get(task, {}).get(metric)
            if not isinstance(series, list) or not series:
                valid = False
                break
            series = sorted(series, key=lambda item: item["layer"])
            vector.extend([float(entry["drop"]) for entry in series])
        if valid:
            vectors[seed] = vector
    if not vectors:
        raise ValueError(
            f"No valid ablation vectors found in {ablation_dir} for metric '{metric}'."
        )
    return vectors


def assign_cluster(vector: list[float], centers: Dict[int, list[float]]) -> int:
    vec = np.array(vector, dtype=np.float64)
    best_id = None
    best_dist = None
    for cid, center in centers.items():
        dist = np.linalg.norm(vec - np.array(center, dtype=np.float64))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = cid
    assert best_id is not None
    return int(best_id)


def load_sample_complexity(results_dir: str, scenario: str, seed: int):
    path = os.path.join(results_dir, f"{scenario}_seed{seed}_finetune.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "s99_steps": data.get("s99_steps"),
        "delta_ab": data.get("delta_ab"),
        "s99_scratch": data.get("s99_scratch"),
        "checkpoint": data.get("checkpoint"),
    }


def summarize_records(records, min_points: int):
    summary = defaultdict(lambda: defaultdict(dict))
    for record in records:
        scenario = record["scenario"]
        cluster_id = record["cluster_id"]
        summary[scenario][cluster_id].setdefault("values", []).append(record["s99_steps"])
    result = {}
    for scenario, cluster_dict in summary.items():
        result[scenario] = {}
        for cluster_id, payload in cluster_dict.items():
            values = [v for v in payload["values"] if v is not None]
            if len(values) < min_points:
                continue
            arr = np.array(values, dtype=np.float64)
            result[scenario][cluster_id] = {
                "count": len(values),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return result


def plot_box(records, metric: str, scenario: str, output_dir: str, min_points: int) -> None:
    scenario_records = [rec for rec in records if rec["scenario"] == scenario]
    clusters = sorted({rec["cluster_id"] for rec in scenario_records})
    data = []
    labels = []
    for cid in clusters:
        vals = [rec["s99_steps"] for rec in scenario_records if rec["cluster_id"] == cid]
        vals = [v for v in vals if v is not None]
        if len(vals) < min_points:
            continue
        data.append(vals)
        labels.append(f"C{cid} (n={len(vals)})")
    if not data:
        print(f"  Skipping boxplot for {scenario} ({metric}): insufficient data.")
        return
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.boxplot(data, tick_labels=labels, patch_artist=True)
    ax.set_title(f"Sample complexity vs. pretrain cluster\nScenario {scenario} | {metric}")
    ax.set_ylabel("s99_steps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"sample_complexity_box_{metric}_{scenario}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    args = parse_args()
    metrics = AVAILABLE_METRICS if args.metric == "all" else [args.metric]
    pretrain_ablation_dir = os.path.join(args.results_dir, "ablation", args.pretrain_family)

    for metric in metrics:
        print(f"=== Metric: {metric} | Pretrain family: {args.pretrain_family} ===")
        cluster_path = os.path.join(args.results_dir, f"cluster_centers_{metric}.json")
        if not os.path.exists(cluster_path):
            print(f"Skipping {metric}: missing cluster center file {cluster_path}")
            continue
        try:
            centers = load_cluster_centers(cluster_path, args.pretrain_family)
            pretrain_vectors = load_pretrain_vectors(pretrain_ablation_dir, metric)
        except ValueError as err:
            print(f"  {err}")
            continue

        records = []
        for seed, vector in pretrain_vectors.items():
            cluster_id = assign_cluster(vector, centers)
            for scenario in args.scenarios:
                sample = load_sample_complexity(args.results_dir, scenario, seed)
                if not sample or sample.get("s99_steps") is None:
                    continue
                records.append(
                    {
                        "metric": metric,
                        "seed": seed,
                        "scenario": scenario,
                        "cluster_id": cluster_id,
                        "s99_steps": sample["s99_steps"],
                        "delta_ab": sample.get("delta_ab"),
                    }
                )

        if not records:
            print(f"  No overlapping finetune runs for metric {metric}.")
            continue

        summary = summarize_records(records, args.min_points)
        if summary and args.summary_output:
            summary_path = args.summary_output.format(metric=metric)
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print(f"  Wrote summary to {summary_path}")

        for scenario in args.scenarios:
            plot_box(records, metric, scenario, args.output_dir, args.min_points)


if __name__ == "__main__":
    main()
