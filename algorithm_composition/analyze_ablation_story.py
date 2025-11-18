from __future__ import annotations

"""Tell the core ablation story in a single pass.

The script focuses on two ingredients:
1. Cluster the concatenated A/B layer-drop vectors produced by ablations on the
   pretrained (GENERAL) checkpoints.
2. Relate those clusters to downstream sample complexity (SC) when the same
   seeds are fine-tuned on C and NC.

Everything else from the previous analysis utilities has been stripped away so
that we can spotlight the localization patterns (cluster identities) and their
impact on downstream learning dynamics.
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

AVAILABLE_METRICS = ["joint", "attention", "mlp"]
PREFERRED_ORDER = ["joint", "mlp", "attention"]

EntryKey = Tuple[int, Optional[int]]


@dataclass
class PretrainRun:
    seed: int
    label: str
    vector: np.ndarray
    per_task_vectors: Dict[str, np.ndarray]


@dataclass
class ClusterSummary:
    cluster_id: int
    count: int
    seeds: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster concatenated A/B ablation drops in pretrain checkpoints and "
            "relate the resulting clusters to downstream sample complexity."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory containing the 'ablation' subdir and downstream SC JSON files.",
    )
    parser.add_argument(
        "--pretrain_family",
        type=str,
        default="GENERAL",
        help="Family name used for the pretrain checkpoints (default: GENERAL).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="auto",
        choices=AVAILABLE_METRICS + ["auto"],
        help=(
            "Which ablation series to use when building the concatenated vectors. Set to 'auto' to "
            "pick the first metric present for both tasks (priority: joint > attention > mlp)."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs=2,
        default=["A", "B"],
        metavar=("TASK_A", "TASK_B"),
        help="Two task names whose ablation drops will be concatenated (default: A B).",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=None,
        help="Number of k-means clusters to fit over the concatenated vectors (default: 4).",
    )
    parser.add_argument(
        "--auto_cluster_range",
        type=int,
        nargs=2,
        metavar=("MIN_K", "MAX_K"),
        default=None,
        help=(
            "If provided, sweep k in the inclusive range [MIN_K, MAX_K] and pick the clustering "
            "with the best silhouette score (ties broken by lower inertia)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for k-means initialization.",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap on the number of pretrain runs to load (sorted order).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["C", "NC"],
        help="Fine-tuning families to pull SC stats from (default: C NC).",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=2,
        help="Minimum number of SC measurements per cluster required for summaries/plots.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots/ablation_core",
        help="Directory for PCA and SC box plots. Set to '' to skip plotting.",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="none",
        choices=["none", "max", "l2"],
        help=(
            "Optional normalization applied to each run's concatenated drop vector before clustering. "
            "Use 'max' to divide by the max per-task drop or 'l2' for L2 normalization.",
        ),
    )
    return parser.parse_args()


def seed_from_filename(path: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    if "_seed" not in base:
        raise ValueError(f"Cannot parse seed from filename '{base}'. Expected '*_seed123.json'.")
    return int(base.split("_seed")[-1])


def make_entry_key(entry: Dict[str, float]) -> EntryKey:
    layer_val = entry.get("layer")
    if layer_val is None:
        raise ValueError("Ablation series entry missing 'layer' field.")
    head_val = entry.get("head")
    head = None if head_val is None else int(head_val)
    return int(layer_val), head


def sort_entry_keys(keys: Sequence[EntryKey]) -> List[EntryKey]:
    return sorted(keys, key=lambda item: (item[0], -1 if item[1] is None else item[1]))


def ensure_entry_order(
    task: str,
    entry_orders: Dict[str, List[EntryKey] | None],
    series: Sequence[Dict[str, float]],
    checkpoint: str,
) -> bool:
    keys = sort_entry_keys([make_entry_key(entry) for entry in series])
    current = entry_orders.get(task)
    if current is None:
        entry_orders[task] = keys
        return True
    if keys != current:
        print(
            f"Skipping {checkpoint}: layer/head order for task {task} differs from the first observed layout."
        )
        return False
    return True


def series_to_vector(series: Sequence[Dict[str, float]], entry_order: Sequence[EntryKey]) -> np.ndarray:
    value_map = {make_entry_key(entry): entry["drop"] for entry in series}
    return np.array([value_map[key] for key in entry_order], dtype=np.float64)


def has_head_dimensions(entry_order: Sequence[EntryKey]) -> bool:
    return any(head is not None for _, head in entry_order)


def aggregate_vector_by_layer(
    vector: np.ndarray,
    entry_order: Sequence[EntryKey],
    layer_order: Sequence[int],
    reducer: str = "max",
) -> np.ndarray:
    if vector.size != len(entry_order):
        raise ValueError("Vector length does not match entry order length for aggregation.")
    layer_values: Dict[int, List[float]] = {layer: [] for layer in layer_order}
    for value, (layer, _head) in zip(vector, entry_order):
        if layer in layer_values:
            layer_values[layer].append(float(value))
    aggregated: List[float] = []
    for layer in layer_order:
        values = layer_values[layer]
        if not values:
            aggregated.append(0.0)
            continue
        if reducer == "mean":
            aggregated.append(float(np.mean(values)))
        elif reducer == "sum":
            aggregated.append(float(np.sum(values)))
        else:  # max
            aggregated.append(float(np.max(values)))
    return np.array(aggregated, dtype=np.float64)


def discover_common_metrics(per_task: Dict[str, Dict], tasks: Sequence[str]) -> List[str]:
    common = set(AVAILABLE_METRICS)
    for task in tasks:
        task_entry = per_task.get(task) or {}
        metrics_here = {key for key, value in task_entry.items() if isinstance(value, list)}
        common &= metrics_here
    return sorted(common)


def select_metric(metric_arg: str, available: Iterable[str]) -> str | None:
    if metric_arg != "auto":
        return metric_arg if metric_arg in available else None
    for candidate in PREFERRED_ORDER:
        if candidate in available:
            return candidate
    return None


def load_pretrain_vectors(
    results_dir: str,
    family: str,
    metric: str,
    tasks: Sequence[str],
    max_runs: int | None,
) -> Tuple[List[PretrainRun], str, Dict[str, List[int]], Dict[str, List[EntryKey]]]:
    ablation_dir = os.path.join(results_dir, "ablation", family)
    paths = sorted(glob.glob(os.path.join(ablation_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No ablation JSON files found under {ablation_dir}. Have you run analyze_ablation.py?"
        )
    if max_runs is not None:
        paths = paths[:max_runs]

    entry_orders: Dict[str, List[EntryKey] | None] = {task: None for task in tasks}
    runs: List[PretrainRun] = []
    chosen_metric: str | None = None

    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        per_task = payload.get("per_task") or {}
        available_metrics = discover_common_metrics(per_task, tasks)
        if not available_metrics:
            continue

        if chosen_metric is None:
            chosen_metric = select_metric(metric, available_metrics)
            if chosen_metric is None:
                continue
        elif chosen_metric not in available_metrics:
            print(f"Skipping {path}: metric '{chosen_metric}' missing for at least one task.")
            continue

        partials: List[np.ndarray] = []
        per_task_vectors: Dict[str, np.ndarray] = {}
        skip = False
        for task in tasks:
            task_entry = per_task.get(task)
            if not isinstance(task_entry, dict):
                skip = True
                break
            series = task_entry.get(chosen_metric)
            if not isinstance(series, list) or not series:
                skip = True
                break
            if not ensure_entry_order(task, entry_orders, series, path):
                skip = True
                break
            order = entry_orders.get(task)
            if order is None:
                skip = True
                break
            vector_part = series_to_vector(series, order)
            partials.append(vector_part)
            per_task_vectors[task] = vector_part
        if skip:
            continue
        vector = np.concatenate(partials)
        seed = seed_from_filename(path)
        runs.append(
            PretrainRun(
                seed=seed,
                label=os.path.basename(path),
                vector=vector,
                per_task_vectors=per_task_vectors,
            )
        )

    if not runs or chosen_metric is None:
        available_str = ", ".join(AVAILABLE_METRICS)
        raise RuntimeError(
            "Could not find overlapping A/B ablation series. Requested metric was "
            f"'{metric}'. Available metrics per file must include at least one of [{available_str}]."
        )
    concrete_orders: Dict[str, List[int]] = {}
    concrete_entry_orders: Dict[str, List[EntryKey]] = {}
    for task, order in entry_orders.items():
        if order is None:
            concrete_orders[task] = []
            concrete_entry_orders[task] = []
            continue
        concrete_entry_orders[task] = order
        unique_layers = sorted({layer for layer, _ in order})
        concrete_orders[task] = unique_layers
    return runs, chosen_metric, concrete_orders, concrete_entry_orders


def kmeans(
    data: np.ndarray, k: int, seed: int = 0, n_init: int = 8, max_iter: int = 200
) -> Tuple[np.ndarray, np.ndarray, float]:
    if data.shape[0] < k:
        raise ValueError(f"Need at least {k} samples to fit k-means, but only {data.shape[0]} were loaded.")
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels = None
    best_centers = None

    for _ in range(n_init):
        centers = data[rng.choice(data.shape[0], size=k, replace=False)].copy()
        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cid in range(k):
                mask = labels == cid
                if np.any(mask):
                    new_centers[cid] = data[mask].mean(axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        inertia = float(np.sum((data - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia


def compute_silhouette_score(points: np.ndarray, labels: Sequence[int]) -> float:
    label_array = np.array(labels, dtype=int)
    unique_labels = np.unique(label_array)
    n_samples = points.shape[0]
    if n_samples < 2 or unique_labels.size < 2:
        return float("nan")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    cluster_indices = {label: np.where(label_array == label)[0] for label in unique_labels}
    intra = np.zeros(n_samples, dtype=np.float64)

    for label, idxs in cluster_indices.items():
        if idxs.size <= 1:
            intra[idxs] = 0.0
            continue
        sub = distances[np.ix_(idxs, idxs)]
        sums = sub.sum(axis=1) - np.diag(sub)
        intra[idxs] = sums / (idxs.size - 1)

    silhouettes = np.zeros(n_samples, dtype=np.float64)
    for label, idxs in cluster_indices.items():
        other_labels = [other for other in unique_labels if other != label]
        if not other_labels:
            silhouettes[idxs] = 0.0
            continue
        means = []
        for other in other_labels:
            other_idxs = cluster_indices[other]
            means.append(distances[np.ix_(idxs, other_idxs)].mean(axis=1))
        other_means = np.stack(means, axis=1)
        b = other_means.min(axis=1)
        a = intra[idxs]
        denom = np.maximum(a, b)
        sil = np.where(denom == 0.0, 0.0, (b - a) / denom)
        silhouettes[idxs] = sil
    return float(np.nanmean(silhouettes))


def select_cluster_configuration(
    matrix: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[int, np.ndarray, np.ndarray, float, float | None]:
    auto_range = args.auto_cluster_range
    if not auto_range:
        labels, centers, inertia = kmeans(matrix, k=args.num_clusters, seed=args.seed)
        return args.num_clusters, labels, centers, inertia, None

    min_k, max_k = auto_range
    min_k = max(2, min_k)
    max_k = min(max_k, matrix.shape[0] - 1)
    if min_k > max_k:
        print(
            "Auto cluster range invalid after adjusting for sample count; falling back to k="
            f"{args.num_clusters}."
        )
        labels, centers, inertia = kmeans(matrix, k=args.num_clusters, seed=args.seed)
        return args.num_clusters, labels, centers, inertia, None

    evaluations = []
    for k in range(min_k, max_k + 1):
        try:
            labels, centers, inertia = kmeans(matrix, k=k, seed=args.seed)
        except ValueError as err:
            print(f"  Skipping k={k}: {err}")
            continue
        score = compute_silhouette_score(matrix, labels)
        evaluations.append(
            {
                "k": k,
                "labels": labels,
                "centers": centers,
                "inertia": inertia,
                "silhouette": score,
            }
        )

    if not evaluations:
        print(
            "No valid cluster configurations found in the requested range; "
            f"falling back to k={args.num_clusters}."
        )
        labels, centers, inertia = kmeans(matrix, k=args.num_clusters, seed=args.seed)
        return args.num_clusters, labels, centers, inertia, None

    def score_key(entry):
        sil = entry["silhouette"]
        sil_value = -1.0 if np.isnan(sil) else sil
        return (sil_value, -entry["inertia"])

    best = max(evaluations, key=score_key)
    print("Auto cluster search results:")
    for entry in evaluations:
        sil = entry["silhouette"]
        sil_str = f"{sil:.3f}" if not np.isnan(sil) else "nan"
        prefix = "*" if entry is best else "-"
        print(
            f"  {prefix} k={entry['k']}: silhouette={sil_str}, inertia={entry['inertia']:.4f}"
        )

    return (
        best["k"],
        best["labels"],
        best["centers"],
        best["inertia"],
        best["silhouette"],
    )


def compute_pca(points: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 2:
        raise ValueError("Need at least two samples to compute PCA.")
    centered = points - points.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    transformed = centered @ components.T
    explained = (s[:n_components] ** 2) / (s**2).sum()
    return transformed, explained


def plot_cluster_pca(
    runs: Sequence[PretrainRun],
    labels: Sequence[int],
    embeddings: np.ndarray,
    explained: np.ndarray,
    plot_dir: str,
    metric: str,
) -> None:
    if not plot_dir:
        return
    os.makedirs(plot_dir, exist_ok=True)
    unique_clusters = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for cid in unique_clusters:
        mask = np.array(labels) == cid
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=f"C{cid} (n={mask.sum()})",
            alpha=0.75,
        )
    for idx, run in enumerate(runs):
        ax.annotate(str(run.seed), (embeddings[idx, 0], embeddings[idx, 1]), fontsize=7)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.set_title(f"A/B ablation clusters ({metric})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, f"ablation_clusters_pca_{metric}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def summarize_clusters(runs: Sequence[PretrainRun], labels: Sequence[int]) -> List[ClusterSummary]:
    summary: Dict[int, List[int]] = {}
    for run, label in zip(runs, labels):
        summary.setdefault(int(label), []).append(run.seed)
    lines = ["Cluster assignments (seed -> cluster):"]
    for run, label in sorted(zip(runs, labels), key=lambda item: item[0].seed):
        lines.append(f"  seed {run.seed}: C{int(label)}")
    print("\n".join(lines))
    return [
        ClusterSummary(cluster_id=cid, count=len(seeds), seeds=sorted(seeds))
        for cid, seeds in sorted(summary.items())
    ]


def summarize_cluster_localization(
    runs: Sequence[PretrainRun],
    labels: Sequence[int],
    tasks: Sequence[str],
    layer_orders: Dict[str, Sequence[int]],
    entry_orders: Dict[str, Sequence[EntryKey]],
) -> None:
    """Print per-cluster, per-task localization stats (argmax and center-of-mass layer).

    For each cluster and task, we compute, across member runs:
      - argmax layer: layer index with the largest drop for that task
      - center-of-mass (COM) layer: sum_l l * drop_l / sum_l drop_l
    This gives a compact numeric summary of "where" each task tends to localize.
    """

    if not runs or not tasks:
        return

    label_array = np.array(labels, dtype=int)
    unique_clusters = sorted(set(label_array.tolist()))
    if not unique_clusters:
        return

    print("\nCluster-wise localization statistics:")
    for cid in unique_clusters:
        idxs = np.where(label_array == cid)[0]
        if idxs.size == 0:
            continue
        print(f"  Cluster C{cid} (n={idxs.size}):")
        for task in tasks:
            layer_ids = layer_orders.get(task)
            entry_order = entry_orders.get(task)
            if not layer_ids or not entry_order:
                print(f"    Task {task}: no layer/head order available; skipping.")
                continue
            argmax_layers: List[float] = []
            com_layers: List[float] = []
            for idx in idxs:
                vec = runs[idx].per_task_vectors.get(task)
                if vec is None or vec.size == 0:
                    continue
                aggregated = aggregate_vector_by_layer(vec, entry_order, layer_ids)
                if aggregated.size != len(layer_ids):
                    continue
                argmax_layer = float(layer_ids[int(np.argmax(aggregated))])
                total = float(np.sum(aggregated))
                if total <= 0.0:
                    # If all drops are ~0, COM is undefined; just record argmax.
                    com_layer = argmax_layer
                else:
                    weights = np.array(layer_ids, dtype=np.float64)
                    com_layer = float(np.sum(weights * aggregated) / total)
                argmax_layers.append(argmax_layer)
                com_layers.append(com_layer)

            if not argmax_layers:
                print(f"    Task {task}: no valid vectors; skipping.")
                continue

            argmax_arr = np.array(argmax_layers, dtype=np.float64)
            com_arr = np.array(com_layers, dtype=np.float64)
            print(
                f"    Task {task}: argmax layer mean={argmax_arr.mean():.2f} std={argmax_arr.std(ddof=0):.2f}; "
                f"COM layer mean={com_arr.mean():.2f} std={com_arr.std(ddof=0):.2f}"
            )


def load_sample_complexity(results_dir: str, scenario: str, seed: int) -> Dict[str, float] | None:
    path = os.path.join(results_dir, f"{scenario}_seed{seed}_finetune.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("s99_steps") is None:
        return None
    return {
        "s99_steps": float(data["s99_steps"]),
        "delta_ab": data.get("delta_ab"),
        "checkpoint": data.get("checkpoint"),
    }


def collect_sample_records(
    runs: Sequence[PretrainRun],
    labels: Sequence[int],
    results_dir: str,
    scenarios: Sequence[str],
) -> List[Dict[str, object]]:
    by_seed = {run.seed: int(label) for run, label in zip(runs, labels)}
    records: List[Dict[str, object]] = []
    for scenario in scenarios:
        for seed, cluster_id in by_seed.items():
            sample = load_sample_complexity(results_dir, scenario, seed)
            if not sample:
                continue
            records.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "cluster_id": cluster_id,
                    "s99_steps": sample["s99_steps"],
                    "delta_ab": sample.get("delta_ab"),
                }
            )
    return records


def summarize_sample_complexity(
    records: Sequence[Dict[str, object]],
    min_points: int,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    summary: Dict[str, Dict[int, List[float]]] = {}
    for record in records:
        scenario = record["scenario"]
        cluster_id = int(record["cluster_id"])
        summary.setdefault(scenario, {}).setdefault(cluster_id, []).append(record["s99_steps"])

    stats: Dict[str, Dict[int, Dict[str, float]]] = {}
    for scenario, cluster_map in summary.items():
        stats[scenario] = {}
        for cluster_id, values in cluster_map.items():
            if len(values) < min_points:
                continue
            arr = np.array(values, dtype=np.float64)
            stats[scenario][cluster_id] = {
                "count": len(values),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return stats


def plot_sample_complexity_boxes(
    records: Sequence[Dict[str, object]],
    scenario: str,
    metric: str,
    plot_dir: str,
    min_points: int,
) -> None:
    if not plot_dir:
        return
    scenario_records = [rec for rec in records if rec["scenario"] == scenario]
    clusters = sorted({rec["cluster_id"] for rec in scenario_records})
    data: List[List[float]] = []
    labels: List[str] = []
    for cid in clusters:
        vals = [rec["s99_steps"] for rec in scenario_records if rec["cluster_id"] == cid]
        if len(vals) < min_points:
            continue
        data.append(vals)
        labels.append(f"C{cid} (n={len(vals)})")
    if not data:
        print(f"  Skipping SC boxplot for scenario {scenario}: insufficient data.")
        return
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("s99_steps")
    ax.set_title(f"Sample complexity vs. pretrain cluster\nScenario {scenario} | {metric}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, f"sample_complexity_{metric}_{scenario}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_cluster_layer_profiles(
    runs: Sequence[PretrainRun],
    labels: Sequence[int],
    tasks: Sequence[str],
    layer_orders: Dict[str, Sequence[int]],
    entry_orders: Dict[str, Sequence[EntryKey]],
    metric: str,
    plot_dir: str,
) -> None:
    if not plot_dir or not runs or not tasks:
        return
    unique_clusters = sorted(set(int(label) for label in labels))
    if not unique_clusters:
        return
    base_dir = os.path.join(plot_dir, "cluster_layer_profiles")
    os.makedirs(base_dir, exist_ok=True)
    label_array = np.array(labels)
    cmap = plt.colormaps.get_cmap("tab10")
    if len(tasks) == 1:
        task_colors = {tasks[0]: cmap(0.5)}
    else:
        positions = np.linspace(0.1, 0.9, len(tasks))
        task_colors = {task: cmap(pos) for task, pos in zip(tasks, positions)}

    for cluster_id in unique_clusters:
        mask = label_array == cluster_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        task_count = max(1, len(tasks))
        fig, axes = plt.subplots(
            1,
            task_count,
            figsize=(4.5 * task_count, 3.6),
            sharey=True,
            squeeze=False,
        )
        axes_list = axes.flatten()
        plotted_any = False

        for ax, task in zip(axes_list, tasks):
            layer_ids = layer_orders.get(task)
            entry_order = entry_orders.get(task)
            if not layer_ids or not entry_order:
                ax.set_visible(False)
                continue
            vectors = [runs[idx].per_task_vectors.get(task) for idx in indices]
            vectors = [vec for vec in vectors if vec is not None]
            if not vectors:
                ax.set_visible(False)
                continue
            aggregated_vectors = [
                aggregate_vector_by_layer(vec, entry_order, layer_ids) for vec in vectors
            ]
            matrix = np.stack(aggregated_vectors)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=0)
            color = task_colors.get(task, cmap(0.5))
            ax.plot(layer_ids, mean, color=color, label=f"{task} mean")
            ax.fill_between(
                layer_ids,
                mean - std,
                mean + std,
                color=color,
                alpha=0.25,
            )
            ax.set_xlabel("Layer")
            ax.set_title(f"Task {task}")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, loc="best")
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        axes_list[0].set_ylabel("Drop")
        fig.suptitle(f"Cluster C{cluster_id} (n={len(indices)}) | {metric}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = os.path.join(base_dir, f"cluster_C{cluster_id}_{metric}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Wrote {out_path}")


def plot_cluster_head_heatmaps(
    runs: Sequence[PretrainRun],
    labels: Sequence[int],
    tasks: Sequence[str],
    layer_orders: Dict[str, Sequence[int]],
    entry_orders: Dict[str, Sequence[EntryKey]],
    metric: str,
    plot_dir: str,
) -> None:
    if not plot_dir or not runs or not tasks:
        return
    base_dir = os.path.join(plot_dir, "cluster_head_heatmaps")
    os.makedirs(base_dir, exist_ok=True)
    label_array = np.array(labels)

    for cluster_id in sorted(set(int(label) for label in labels)):
        indices = np.where(label_array == cluster_id)[0]
        if indices.size == 0:
            continue
        for task in tasks:
            entry_order = entry_orders.get(task)
            layer_ids = layer_orders.get(task)
            if not entry_order or not layer_ids:
                continue
            if not has_head_dimensions(entry_order):
                continue
            head_ids = sorted({head for _, head in entry_order if head is not None})
            if not head_ids:
                continue
            layer_index = {layer: idx for idx, layer in enumerate(layer_ids)}
            head_index = {head: idx for idx, head in enumerate(head_ids)}
            grid_sum = np.zeros((len(layer_ids), len(head_ids)), dtype=np.float64)
            grid_count = np.zeros_like(grid_sum)

            for idx in indices:
                vec = runs[idx].per_task_vectors.get(task)
                if vec is None or vec.size == 0:
                    continue
                for value, (layer, head) in zip(vec, entry_order):
                    if head is None:
                        continue
                    i = layer_index.get(layer)
                    j = head_index.get(head)
                    if i is None or j is None:
                        continue
                    grid_sum[i, j] += float(value)
                    grid_count[i, j] += 1.0

            if not np.any(grid_count):
                continue
            mean_grid = np.divide(
                grid_sum,
                grid_count,
                out=np.zeros_like(grid_sum),
                where=grid_count > 0,
            )

            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            im = ax.imshow(mean_grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(head_ids)))
            ax.set_xticklabels(head_ids)
            ax.set_yticks(range(len(layer_ids)))
            ax.set_yticklabels(layer_ids)
            ax.set_xlabel("Head index")
            ax.set_ylabel("Layer")
            ax.set_title(f"Cluster C{cluster_id} | Task {task} | {metric}")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean drop")
            fig.tight_layout()
            out_path = os.path.join(
                base_dir, f"cluster_C{cluster_id}_task_{task}_{metric}_heatmap.png"
            )
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"  Wrote {out_path}")


def main() -> None:
    args = parse_args()
    runs, metric_used, layer_orders, entry_orders = load_pretrain_vectors(
        results_dir=args.results_dir,
        family=args.pretrain_family,
        metric=args.metric,
        tasks=args.tasks,
        max_runs=args.max_runs,
    )
    def normalize(vec: np.ndarray) -> np.ndarray:
        if args.normalize_mode == "none":
            return vec
        if args.normalize_mode == "max":
            max_val = np.max(np.abs(vec))
            return vec if max_val == 0.0 else vec / max_val
        if args.normalize_mode == "l2":
            norm = float(np.linalg.norm(vec))
            return vec if norm == 0.0 else vec / norm
        return vec

    matrix = np.stack([normalize(run.vector) for run in runs])
    (
        selected_k,
        labels,
        centers,
        inertia,
        silhouette,
    ) = select_cluster_configuration(matrix, args)
    if silhouette is None or np.isnan(silhouette):
        print(
            f"Fitted k-means on {len(runs)} runs: k={selected_k}, inertia={inertia:.4f}"
        )
    else:
        print(
            f"Fitted k-means on {len(runs)} runs: k={selected_k}, inertia={inertia:.4f}, "
            f"silhouette={silhouette:.3f}"
        )
    cluster_summaries = summarize_clusters(runs, labels)
    summarize_cluster_localization(runs, labels, args.tasks, layer_orders, entry_orders)

    if len(runs) >= 2:
        try:
            embeddings, explained = compute_pca(matrix, n_components=2)
        except ValueError as err:
            print(f"Skipping PCA plot: {err}")
        else:
            plot_cluster_pca(runs, labels, embeddings, explained, args.plot_dir, metric_used)

    plot_cluster_layer_profiles(
        runs=runs,
        labels=labels,
        tasks=args.tasks,
        layer_orders=layer_orders,
        entry_orders=entry_orders,
        metric=metric_used,
        plot_dir=args.plot_dir,
    )

    plot_cluster_head_heatmaps(
        runs=runs,
        labels=labels,
        tasks=args.tasks,
        layer_orders=layer_orders,
        entry_orders=entry_orders,
        metric=metric_used,
        plot_dir=args.plot_dir,
    )

    records = collect_sample_records(runs, labels, args.results_dir, args.scenarios)
    if not records:
        print("No overlapping downstream finetune runs with SC data were found; stopping after clustering.")
        return

    stats = summarize_sample_complexity(records, args.min_points)
    if not stats:
        print("Sample complexity records exist but none meet the min_points threshold for summaries.")
    else:
        print("\nDownstream sample complexity by cluster:")
        for scenario, cluster_map in sorted(stats.items()):
            print(f"  Scenario {scenario}:")
            for cluster_id, values in sorted(cluster_map.items()):
                print(
                    f"    C{cluster_id}: n={values['count']}, mean={values['mean']:.1f}, "
                    f"median={values['median']:.1f}, std={values['std']:.1f}, "
                    f"min={values['min']:.1f}, max={values['max']:.1f}"
                )

    for scenario in args.scenarios:
        plot_sample_complexity_boxes(records, scenario, metric_used, args.plot_dir, args.min_points)


if __name__ == "__main__":
    main()
