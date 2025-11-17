from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

AVAILABLE_METRICS = ["attention", "mlp", "joint"]


@dataclass
class VectorRecord:
    vector: np.ndarray
    family: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster concatenated ablation drops from tasks A and B and inspect their "
            "distribution across training scenarios."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/ablation",
        help="Directory containing per-family ablation JSON files (e.g., results/ablation).",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help=(
            "List of training scenarios/families to include. Use 'all' or omit to include "
            "every subdirectory under --results_dir."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=AVAILABLE_METRICS + ["all"],
        help="Which ablation series to use (set to 'all' to iterate over every metric).",
    )
    parser.add_argument(
        "--tasks",
        nargs=2,
        default=["A", "B"],
        metavar=("TASK_A", "TASK_B"),
        help="Names of the tasks whose drops should be concatenated (default: A B).",
    )
    parser.add_argument(
        "--compare_mode",
        type=str,
        choices=["combined", "per-family"],
        default="per-family",
        help=(
            "Whether to cluster all families together ('combined') or run separate clusters "
            "within each family before comparing them (default: per-family)."
        ),
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=2,
        help="Number of k-means clusters to fit (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for centroid initialization.",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs per family to load (useful for quick tests).",
    )
    parser.add_argument(
        "--scatter_output",
        type=str,
        default="plots/ablation_cluster_pca.png",
        help=(
            "Path template for the PCA scatter plot (set to '' to skip). Supports '{metric}' "
            "and '{family}' placeholders."
        ),
    )
    parser.add_argument(
        "--center_scatter_output",
        type=str,
        default="plots/ablation_cluster_center_compare.png",
        help=(
            "Path template for a PCA scatter of cluster centers across families (per metric). "
            "Set to '' to skip. Supports '{metric}'."
        ),
    )
    parser.add_argument(
        "--center_summary_output",
        type=str,
        default="results/cluster_centers_{metric}.json",
        help=(
            "Path template for saving cluster center vectors/metadata (set to '' to skip). "
            "Supports '{metric}'."
        ),
    )
    return parser.parse_args()


def discover_families(results_dir: str) -> List[str]:
    if not os.path.isdir(results_dir):
        return []
    families = [
        entry
        for entry in sorted(os.listdir(results_dir))
        if os.path.isdir(os.path.join(results_dir, entry))
    ]
    return families


def resolve_output_path(base: str, metric: str, family: str | None = None) -> str:
    if not base:
        return ""
    fmt_kwargs = {
        "metric": metric,
        "family": family or metric,
    }
    if "{metric}" in base or "{family}" in base:
        return base.format(**fmt_kwargs)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"
    suffix_parts = [metric]
    if family:
        suffix_parts.append(family)
    suffix = "_".join(suffix_parts)
    return f"{root}_{suffix}{ext}"


def ensure_layer_order(
    task: str,
    layer_orders: Dict[str, List[int] | None],
    series: Sequence[Dict[str, float]],
    checkpoint: str,
) -> bool:
    layers = sorted(entry["layer"] for entry in series)
    current = layer_orders.get(task)
    if current is None:
        layer_orders[task] = layers
        return True
    if layers != current:
        print(
            f"Skipping {checkpoint}: layer set for task {task} differs from the reference order {current}."
        )
        return False
    return True


def series_to_vector(series: Sequence[Dict[str, float]], layer_order: Sequence[int]) -> np.ndarray:
    value_map = {entry["layer"]: entry["drop"] for entry in series}
    return np.array([value_map[layer] for layer in layer_order], dtype=np.float64)


def load_vectors(
    results_dir: str,
    families: Sequence[str],
    metric: str,
    tasks: Sequence[str],
    max_runs: int | None,
) -> List[VectorRecord]:
    layer_orders: Dict[str, List[int] | None] = {task: None for task in tasks}
    records: List[VectorRecord] = []

    for family in families:
        family_dir = os.path.join(results_dir, family)
        paths = sorted(glob.glob(os.path.join(family_dir, "*.json")))
        if not paths:
            print(f"Warning: no JSON files found under {family_dir}.")
            continue
        if max_runs is not None:
            paths = paths[:max_runs]
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            per_task = data.get("per_task") or {}
            partials = []
            skip = False
            for task in tasks:
                task_entry = per_task.get(task)
                if not isinstance(task_entry, dict):
                    skip = True
                    break
                series = task_entry.get(metric)
                if not isinstance(series, list) or not series:
                    skip = True
                    break
                if not ensure_layer_order(task, layer_orders, series, path):
                    skip = True
                    break
                order = layer_orders.get(task)
                if order is None:
                    skip = True
                    break
                vector_part = series_to_vector(series, order)
                partials.append(vector_part)
            if skip:
                continue
            vector = np.concatenate(partials)
            label = os.path.splitext(os.path.basename(path))[0]
            records.append(VectorRecord(vector=vector, family=family, label=label))

    return records


def compute_pca(
    points: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] < 2:
        raise ValueError("Need at least two samples to compute PCA.")
    mean = points.mean(axis=0, keepdims=True)
    centered = points - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    transformed = centered @ components.T
    explained = (s[:n_components] ** 2) / (s**2).sum()
    return transformed, explained, components, mean


def kmeans(
    data: np.ndarray, k: int, seed: int = 0, n_init: int = 8, max_iter: int = 200
) -> Tuple[np.ndarray, np.ndarray, float]:
    if data.shape[0] < k:
        raise ValueError(f"Cannot fit {k} clusters to only {data.shape[0]} samples.")
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels = None
    best_centers = None

    for init in range(n_init):
        indices = rng.choice(data.shape[0], size=k, replace=False)
        centers = data[indices].copy()
        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster_id in range(k):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centers[cluster_id] = data[mask].mean(axis=0)
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


def ensure_out_dir(path: str) -> None:
    if path:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)


def plot_pca_scatter(
    embeddings: np.ndarray,
    components: np.ndarray,
    centers: np.ndarray,
    data_mean: np.ndarray,
    families: Sequence[str],
    explained_ratio: np.ndarray,
    output: str,
    title: str | None = None,
) -> None:
    if not output:
        return
    ensure_out_dir(output)
    unique_families = sorted(set(families))
    cmap_name = "tab10" if len(unique_families) <= 10 else "tab20"
    cmap = plt.colormaps.get_cmap(cmap_name)
    if len(unique_families) == 1:
        color_positions = [0.5]
    else:
        color_positions = np.linspace(0.0, 1.0, len(unique_families))
    family_to_color = {
        family: cmap(pos) for family, pos in zip(unique_families, color_positions)
    }
    markers = ["o", "s", "^", "D", "P", "X"]
    family_to_marker = {
        family: markers[idx % len(markers)] for idx, family in enumerate(unique_families)
    }
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    families_np = np.array(families)
    for family in unique_families:
        mask = families_np == family
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=f"{family}",
            color=family_to_color[family],
            marker=family_to_marker[family],
            alpha=0.75,
            edgecolor="k",
            linewidth=0.4,
        )
    embedded_centers = (centers - data_mean) @ components.T
    ax.scatter(
        embedded_centers[:, 0],
        embedded_centers[:, 1],
        marker="X",
        s=120,
        c="black",
        label="Cluster centers",
    )
    for idx, (cx, cy) in enumerate(embedded_centers):
        ax.text(cx, cy, f"C{idx}", color="black", fontsize=9, ha="center", va="bottom")
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% var)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Concatenated drop vectors: PCA projection")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Wrote PCA scatter to {output}")


def plot_center_comparison(
    center_records: Sequence[Dict[str, object]],
    metric: str,
    output: str,
) -> None:
    if not output or len(center_records) < 2:
        return
    ensure_out_dir(output)
    centers = [np.array(record["center"], dtype=np.float64) for record in center_records]
    if not centers:
        print(f"Skipping center comparison plot for {metric}: no centers available.")
        return
    matrix = np.stack(centers)
    if matrix.shape[1] < 2:
        print(
            f"Skipping center comparison plot for {metric}: need >=2 dimensions, got {matrix.shape[1]}."
        )
        return
    try:
        embeddings, explained, _components, _mean = compute_pca(matrix, n_components=2)
    except ValueError as err:
        print(f"Skipping center comparison plot for {metric}: {err}")
        return

    families = [str(record["family"]) for record in center_records]
    unique_families = sorted(set(families))
    cmap_name = "tab10" if len(unique_families) <= 10 else "tab20"
    cmap = plt.colormaps.get_cmap(cmap_name)
    if len(unique_families) == 1:
        color_positions = [0.5]
    else:
        color_positions = np.linspace(0.0, 1.0, len(unique_families))
    family_to_color = {
        family: cmap(pos) for family, pos in zip(unique_families, color_positions)
    }
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    cluster_ids = sorted({int(record["cluster_id"]) for record in center_records})
    cluster_to_marker = {cid: markers[cid % len(markers)] for cid in cluster_ids}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for record, point in zip(center_records, embeddings):
        family = str(record["family"])
        cluster_id = int(record["cluster_id"])
        count = int(record.get("count", 0))
        label = f"{family}-C{cluster_id}"
        ax.scatter(
            point[0],
            point[1],
            color=family_to_color[family],
            marker=cluster_to_marker[cluster_id],
            s=90 + count,
            edgecolor="k",
            linewidth=0.4,
            alpha=0.9,
        )
        ax.text(
            point[0],
            point[1],
            f"{label} (n={count})",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Center PC1")
    ax.set_ylabel("Center PC2")
    ax.set_title(f"Cluster centers comparison ({metric})")
    ax.grid(alpha=0.2)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=family_to_color[family],
            markersize=8,
            label=family,
        )
        for family in unique_families
    ]
    legend1 = ax.legend(handles=family_handles, title="Family", loc="upper left")
    ax.add_artist(legend1)

    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker=cluster_to_marker[cid],
            color="k",
            linestyle="",
            label=f"C{cid}",
        )
        for cid in cluster_ids
    ]
    ax.legend(handles=cluster_handles, title="Cluster", loc="lower right")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Wrote center comparison plot to {output}")


def write_center_summary(path: str, center_records: Sequence[Dict[str, object]]) -> None:
    if not path or not center_records:
        return
    ensure_out_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(center_records, handle, indent=2)
    print(f"Wrote cluster center summary to {path}")


def summarize_clusters(
    records: Sequence[VectorRecord],
    labels: Sequence[int],
    inertia: float,
) -> None:
    print(f"Fitted k-means: inertia={inertia:.4f}, clusters={len(set(labels))}")
    summary: Dict[int, Dict[str, int]] = {}
    for rec, label in zip(records, labels):
        family_counts = summary.setdefault(label, {})
        family_counts[rec.family] = family_counts.get(rec.family, 0) + 1
    for cluster_id, family_counts in sorted(summary.items()):
        total = sum(family_counts.values())
        parts = ", ".join(f"{fam}: {count}" for fam, count in sorted(family_counts.items()))
        print(f"  Cluster {cluster_id} (n={total}): {parts}")


def compute_cluster_counts(labels: Sequence[int]) -> Dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def run_clustering_workflow(
    records: Sequence[VectorRecord],
    metric: str,
    args: argparse.Namespace,
    scatter_output: str,
    title: str,
) -> Tuple[np.ndarray, Dict[int, int]]:
    matrix = np.stack([rec.vector for rec in records])
    labels, centers, inertia = kmeans(
        data=matrix,
        k=args.num_clusters,
        seed=args.seed,
    )
    summarize_clusters(records, labels, inertia)

    try:
        embeddings, explained, components, data_mean = compute_pca(matrix, n_components=2)
    except ValueError as err:
        print(f"Skipping PCA scatter for {title}: {err}")
    else:
        plot_pca_scatter(
            embeddings=embeddings,
            components=components,
            centers=centers,
            data_mean=data_mean,
            families=[rec.family for rec in records],
            explained_ratio=explained,
            output=scatter_output,
            title=title,
        )

    counts = compute_cluster_counts(labels)
    return centers, counts


def main() -> None:
    args = parse_args()
    if not args.families or any(fam.lower() == "all" for fam in args.families):
        families = discover_families(args.results_dir)
    else:
        families = list(args.families)

    if not families:
        print(
            "No families discovered. Ensure --results_dir points to a directory with per-family JSON files."
        )
        return

    print(f"Using families: {', '.join(families)}")
    metrics = AVAILABLE_METRICS if args.metric == "all" else [args.metric]

    for metric in metrics:
        print(f"=== Metric: {metric} ===")
        records = load_vectors(
            results_dir=args.results_dir,
            families=families,
            metric=metric,
            tasks=args.tasks,
            max_runs=args.max_runs,
        )
        if not records:
            print(f"Skipping metric {metric}: no eligible ablation runs found.")
            continue

        if args.compare_mode == "combined":
            if len(records) < args.num_clusters:
                print(
                    f"Skipping metric {metric}: need at least {args.num_clusters} vectors, got {len(records)}."
                )
                continue
            run_clustering_workflow(
                records=records,
                metric=metric,
                args=args,
                scatter_output=resolve_output_path(args.scatter_output, metric),
                title=f"All families ({metric})",
            )
            continue

        # per-family mode
        grouped: Dict[str, List[VectorRecord]] = defaultdict(list)
        for rec in records:
            grouped[rec.family].append(rec)

        center_records: List[Dict[str, object]] = []
        for family in sorted(grouped.keys()):
            family_records = grouped[family]
            if len(family_records) < args.num_clusters:
                print(
                    f"  Skipping family {family} for metric {metric}: need >= {args.num_clusters} runs, got {len(family_records)}."
                )
                continue
            centers, counts = run_clustering_workflow(
                records=family_records,
                metric=metric,
                args=args,
                scatter_output=resolve_output_path(args.scatter_output, metric, family),
                title=f"{family} ({metric})",
            )
            for cluster_id, center in enumerate(centers):
                center_records.append(
                    {
                        "metric": metric,
                        "family": family,
                        "cluster_id": int(cluster_id),
                        "count": int(counts.get(cluster_id, 0)),
                        "center": center.tolist(),
                    }
                )

        if center_records:
            summary_path = resolve_output_path(args.center_summary_output, metric)
            write_center_summary(summary_path, center_records)
            plot_center_comparison(
                center_records=center_records,
                metric=metric,
                output=resolve_output_path(args.center_scatter_output, metric),
            )
        else:
            print(f"No family-specific clusters to compare for metric {metric}.")


if __name__ == "__main__":
    main()
