# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Visualize and quantify hot spots learned during post-training.

Loads a model saved after post-training unsupervised learning (e.g. by the
post_training_hotspots_objects_with_stickers_comp_models_mujoco experiment) and,
for each object in one LM's memory, plots the sensor-channel point cloud colored by
hot spot value, with the logo (LM 1 input channel) nodes overlaid.

Hot spot values are running averages of the (signed) evidence differential between
the recognized object and its strongest competitor, so locations shared with other
objects should average out near 0, while distinguishing locations stay positive.
To check that hot spots emerge where they should, it compares the mean hot spot
value of tagged nodes within --logo-radius of a logo node with that of the other
tagged nodes, and reports how many of the top-k hot spot nodes lie near the logo
compared with the fraction of all tagged nodes that do.

Usage:
    python analysis/scripts/hotspot_visualization.py [MODEL_PATH] [options]

MODEL_PATH points at model.pt itself or the directory containing it. Defaults to
the output of the post-training experiment.

Options:
    --lm ID             Which learning module's models to use (default: 2).
    --top-k K           Number of top hot spot nodes to report (default: 10).
    --min-count N       Only rank nodes tagged at least N times (default: 1).
    --logo-radius R     Distance (m) from a logo node counted as "on the logo"
                        (default: 0.01).
    --output PATH       Where to save the figure
                        (default: ~/Desktop/hotspots.png).
    --show              Also open an interactive window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import KDTree

DEFAULT_MODEL_PATH = (
    "~/tbp/results/comp_benefits_figures/"
    "post_training_hotspots_objects_with_stickers_comp_models_mujoco/0"
)
LOGO_CHANNEL = "learning_module_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_MODEL_PATH),
        help="model.pt or the directory containing it",
    )
    parser.add_argument("--lm", type=int, default=2, help="LM whose models to use")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--logo-radius", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("~/Desktop/hotspots.png"),
        help="where to save the figure",
    )
    parser.add_argument("--show", action="store_true", help="open a window")
    return parser.parse_args()


def resolve_checkpoint(model_path: Path) -> Path:
    path = model_path.expanduser()
    for candidate in (path, path / "model.pt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Model checkpoint not found under: {path}")


def load_lm_memory(checkpoint: Path, lm_id: int) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lm_dict = state["lm_dict"]
    lm_key = lm_id if lm_id in lm_dict else str(lm_id)
    return lm_dict[lm_key]["graph_memory"]


def summarize_object(
    object_id: str, channels: dict, top_k: int, min_count: int, logo_radius: float
):
    """Print the hot spot statistics of one object.

    Returns:
        Dict with the sensor channel model and summary statistics.
    """
    sensor_channel = next(c for c in channels if c.startswith("patch"))
    model = channels[sensor_channel]
    pos = np.asarray(model.pos)
    values = model.hotspot_values
    counts = model.hotspot_counts
    tagged = counts > 0

    near_logo = np.zeros(len(pos), dtype=bool)
    if LOGO_CHANNEL in channels:
        logo_tree = KDTree(np.asarray(channels[LOGO_CHANNEL].pos))
        dist_to_logo, _ = logo_tree.query(pos)
        near_logo = dist_to_logo < logo_radius

    print(f"\n{object_id} ({sensor_channel}, {len(pos)} nodes)")
    print(f"  tagged nodes: {tagged.sum()}, total tags: {counts.sum()}")
    summary = {"model": model, "near_logo": near_logo}
    if not np.any(tagged):
        return summary

    rankable = counts >= min_count
    ranked = np.argsort(np.where(rankable, values, -np.inf))[::-1][:top_k]
    ranked = ranked[rankable[ranked]]
    print(f"  top {len(ranked)} hot spots (node: value, count, near logo):")
    for node in ranked:
        print(
            f"    {node:5d}: {values[node]:6.3f}, {counts[node]:3d}, "
            f"{bool(near_logo[node])}"
        )

    # Count-weighted mean differential, i.e. the mean over all tags.
    def mean_tag_value(mask):
        return (values[mask] * counts[mask]).sum() / max(counts[mask].sum(), 1)

    tagged_near, tagged_far = tagged & near_logo, tagged & ~near_logo
    summary.update(
        tags_near_logo=int(counts[tagged_near].sum()),
        tags_elsewhere=int(counts[tagged_far].sum()),
        mean_near_logo=mean_tag_value(tagged_near),
        mean_elsewhere=mean_tag_value(tagged_far),
    )
    print(
        f"  nodes near logo: {near_logo.mean():.1%} of all, "
        f"{near_logo[tagged].mean():.1%} of tagged, "
        f"{near_logo[ranked].mean():.1%} of top-{len(ranked)}"
    )
    print(
        f"  mean tag value: near logo {summary['mean_near_logo']:.3f} "
        f"({summary['tags_near_logo']} tags), elsewhere "
        f"{summary['mean_elsewhere']:.3f} ({summary['tags_elsewhere']} tags)"
    )
    return summary


def plot_object(ax, object_id: str, channels: dict, model):
    pos = np.asarray(model.pos)
    values = model.hotspot_values
    tagged = model.hotspot_counts > 0
    ax.scatter(*pos[~tagged].T, s=2, c="lightgray", alpha=0.3)
    order = np.argsort(values[tagged])
    limit = max(np.abs(values[tagged]).max(), 1e-6) if np.any(tagged) else 1.0
    scatter = ax.scatter(
        *pos[tagged][order].T,
        s=12,
        c=values[tagged][order],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    if LOGO_CHANNEL in channels:
        logo = np.asarray(channels[LOGO_CHANNEL].pos)
        ax.scatter(*logo.T, s=12, c="#00a0df", alpha=0.3, label="logo nodes")
    best = model.get_max_hotspot_node()
    if best is not None:
        ax.scatter(
            *pos[best[0]],
            s=250,
            c="lime",
            marker="*",
            edgecolors="black",
            label="max hot spot",
            zorder=10,
        )
    ax.set_title(object_id)
    ax.set_box_aspect((1, 1, 1))
    limits = np.array([pos.min(axis=0), pos.max(axis=0)])
    center, half_span = limits.mean(axis=0), (limits[1] - limits[0]).max() / 2
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.legend(loc="upper left", fontsize=7)
    return scatter


def main():
    args = parse_args()
    checkpoint = resolve_checkpoint(args.model_path)
    memory = load_lm_memory(checkpoint, args.lm)
    print(f"Loaded LM_{args.lm} graph memory from {checkpoint}")

    summaries = {
        object_id: summarize_object(
            object_id, memory[object_id], args.top_k, args.min_count, args.logo_radius
        )
        for object_id in sorted(memory)
    }
    object_ids = [o for o, s in summaries.items() if s["model"].hotspot_counts.any()]
    if not object_ids:
        print("\nNo hot spots tagged on any object.")
        return

    figure, axes = plt.subplots(
        1,
        len(object_ids),
        figsize=(6 * len(object_ids), 6),
        subplot_kw={"projection": "3d"},
    )
    axes = np.atleast_1d(axes)
    for ax, object_id in zip(axes, object_ids):
        scatter = plot_object(
            ax, object_id, memory[object_id], summaries[object_id]["model"]
        )
        figure.colorbar(scatter, ax=ax, shrink=0.5, label="hot spot value")

    figure.suptitle("Hot spots learned during post-training unsupervised learning")
    figure.tight_layout()
    output = args.output.expanduser()
    figure.savefig(output, dpi=150)
    print(f"\nSaved figure to {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
