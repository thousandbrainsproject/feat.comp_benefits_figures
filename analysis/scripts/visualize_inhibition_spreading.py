# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# ruff: noqa: DOC201,DOC501
"""Visualize how sensory input inhibits regions of learned object models.

Runs the ChildObjectsGoalGenerator's spreading of inhibition through nodes with
a similar hue on a continuous surface (as predicted from the surface normals,
principal curvatures and curvature directions of neighboring nodes) on the sensor
channel graphs of one learning module (LM) of a pretrained model, sensing the
features stored at a chosen seed node with the most likely hypothesis (MLH) at
that node. Saves (or, with --interactive, shows), prefixed with lm<ID>_:

- <object>_seeds.png: for the mug, the region inhibited by a spread from seeds
  on its side, next to its handle, on its rim, bottom, handle and inside wall,
  colored by the order in which the spread reached each node.
- compartments.png: for each object, the "compartments" that spreads divide
  its model into. Starting from random uninhibited nodes, spreads are run until
  every node is assigned to the compartment of the first spread that reached it.
- <object>_tolerance.png: for the mug, spreads from the same seeds with
  different max_dissimilar_neighbors.

The sensor channel is the one channel of the LM's graphs that receives input from
a sensor module (e.g. patch_2 for LM 2, patch_0 for LM 0). Objects the LM has no
model of are skipped, e.g. LM 0 only models the objects without stickers.

Usage:
    python analysis/scripts/visualize_inhibition_spreading.py [MODEL_PATH] [options]

MODEL_PATH points at a trained model: either model.pt itself, its pretrained
directory, or the experiment directory containing it. Defaults to the
supervised_pre_training_objects_with_stickers_comp_models_mujoco experiment in
~/tbp/results/monty/pretrained_models/my_trained_models.

Options:
    --lm ID           Which learning module's models to use (default: 2; e.g. 0
                      to compare the models of LM 0).
    --output-dir DIR  Where to save the figures (default:
                      ~/tbp/results/comp_benefits_figures/inhibition_spreading).
    --interactive     Show the figures in interactive windows (e.g. to rotate
                      the 3D views) instead of saving them.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.goal_generation import ChildObjectsGoalGenerator

DEFAULT_MODEL_PATH = (
    "~/tbp/results/monty/pretrained_models/my_trained_models/"
    "supervised_pre_training_objects_with_stickers_comp_models_mujoco"
)
DEFAULT_OUTPUT_DIR = "~/tbp/results/comp_benefits_figures/inhibition_spreading"
MUG = "023_mug"
COMPARTMENT_OBJECTS = [
    "023_mug",
    "024_mug_tbp_horz",
    "025_mug_tbp_vert",
    "001_cube",
    "002_cube_tbp",
    "011_cylinder",
    "016_sphere",
]
# Viewpoints (elevation, azimuth) of the three views of each model.
VIEWS = [(20, 30), (20, 210), (-60, 30)]
# Compartments with fewer nodes are drawn in grey rather than their own color.
MIN_COMPARTMENT_NODES = 2
PART_NAMES = ["side", "rim", "bottom", "handle", "inside"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_MODEL_PATH),
        help="model.pt, its pretrained directory, or the experiment directory",
    )
    parser.add_argument("--lm", type=int, default=2, help="LM whose models to use")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="where to save the figures",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="show the figures in interactive windows instead of saving them",
    )
    return parser.parse_args()


def resolve_checkpoint(model_path: Path) -> Path:
    """Resolve model.pt from a checkpoint, pretrained, or experiment path."""
    path = model_path.expanduser()
    for candidate in (path, path / "model.pt", path / "pretrained" / "model.pt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Model checkpoint not found under: {path}")


def load_lm_memory(checkpoint: Path, lm_id: int) -> dict:
    """Load one LM's graph memory, with a location tree for every graph."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lm_dict = state["lm_dict"]
    memory = lm_dict[lm_id if lm_id in lm_dict else str(lm_id)]["graph_memory"]
    for channels in memory.values():
        for graph in channels.values():
            if getattr(graph, "_location_tree", None) is None:
                graph._location_tree = KDTree(np.asarray(graph.pos))
    return memory


def sensor_channel(memory: dict) -> str:
    """Find the channel of an LM's graphs that receives input from a sensor module.

    Channels from other LMs store "object_id" features; the sensor channel stores
    the surface features that sensory spreading compares.
    """
    channels = {
        channel
        for graphs in memory.values()
        for channel, graph in graphs.items()
        if "object_id" not in graph.feature_mapping
        and "pose_vectors" in graph.feature_mapping
    }
    if len(channels) != 1:
        raise ValueError(f"Expected one sensor channel, found {sorted(channels)}")
    return channels.pop()


class FakeParentLM:
    """Serve a checkpoint's graph memory through the parent-LM interface.

    Implements just what the ChildObjectsGoalGenerator uses to spread
    inhibition, with the MLH set by the caller.
    """

    def __init__(self, memory: dict):
        self.memory = memory
        self.learning_module_id = "demo_lm"
        self.object_id_feature_names = {}
        # As configured for LMs 0 and 2 in evidence_3lm_heterarchy
        self.max_match_distance = 0.01
        self.mlh = None
        self.buffer = SimpleNamespace(update_stats=lambda *_args, **_kwargs: None)

    def _get_current_mlh(self):
        return self.mlh

    def get_all_known_object_ids(self):
        return list(self.memory)

    def get_input_channels_in_graph(self, graph_id):
        return list(self.memory[graph_id])

    def get_graph(self, graph_id, input_channel=None):
        if input_channel is None:
            return self.memory[graph_id]
        return self.memory[graph_id][input_channel]


class Spreader:
    """Spread inhibition from sensing the features stored at a node."""

    def __init__(self, memory: dict, **gsg_kwargs):
        self.lm = FakeParentLM(memory)
        self.channel = sensor_channel(memory)
        self.gsg = ChildObjectsGoalGenerator(**gsg_kwargs)
        self.gsg.parent_lm = self.lm

    def spread(self, graph_id: str, node: int) -> tuple[np.ndarray, np.ndarray]:
        """Spread from a node, sensing the features stored there.

        Returns:
            Whether each node is inhibited, and the order of each node in the
            spread (-1 for nodes it did not reach).
        """
        graph = self.lm.get_graph(graph_id, self.channel)
        features = {
            name: np.asarray(graph.x[node, start:end])
            for name, (start, end) in graph.feature_mapping.items()
        }
        percept = SimpleNamespace(
            sender_type="SM",
            sender_id=self.channel,
            morphological_features={
                "pose_vectors": features["pose_vectors"].reshape(3, 3)
            },
            non_morphological_features={
                "hsv": features["hsv"],
                "principal_curvatures_log": features["principal_curvatures_log"],
            },
        )
        self.lm.mlh = {
            "graph_id": graph_id,
            "mlh_id": 0,
            "location": np.asarray(graph.pos[node]),
            "rotation": Rotation.identity(),
        }
        self.gsg.reset()
        self.gsg._spread_from_observations([percept])
        num_nodes = len(graph.pos)
        order = np.full(num_nodes, -1)
        for record in self.gsg.spread_records:
            order[record.node_order] = np.arange(len(record.node_order))
        inhibited = self.gsg.get_inhibition_weights(graph_id, self.channel) > 0
        return inhibited, order

    def reliable(self, graph_id: str) -> np.ndarray:
        graph = self.lm.get_graph(graph_id, self.channel)
        return self.gsg._get_spread_features(graph)["reliable"]


def _fit_circle(points_2d: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.column_stack([2 * points_2d, np.ones(len(points_2d))])
    b = (points_2d**2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    return np.array([cx, cy]), float(np.sqrt(c + cx**2 + cy**2))


def label_mug_parts(memory: dict, graph_id: str, channel: str) -> np.ndarray:
    """Label the nodes of a mug's sensor graph by geometry.

    Fits the mug's cylinder (its axis is the direction most surface normals are
    perpendicular to), then labels nodes on the cylinder's wall as its outer
    side or inside wall by the direction of their normal, nodes beyond the
    wall as the handle, and nodes at either end as the bottom (the end with a
    closed disc) or rim.

    Returns:
        One of PART_NAMES, or "other", per node.
    """
    graph = memory[graph_id][channel]
    pos = np.asarray(graph.pos, dtype=np.float64)
    start, _ = graph.feature_mapping["pose_vectors"]
    normals = np.asarray(graph.x[:, start : start + 3], dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    _, vectors = np.linalg.eigh(normals.T @ normals)
    axis, u, v = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    height = pos @ axis
    plane = np.column_stack([pos @ u, pos @ v])
    on_wall = np.abs(normals @ axis) < 0.2
    inliers = on_wall.copy()
    for _ in range(5):
        center, radius = _fit_circle(plane[inliers])
        r = np.linalg.norm(plane - center, axis=1)
        inliers = on_wall & (np.abs(r - radius) < 0.005)
    radial = (plane - center) / r[:, None]
    normal_radial = (normals @ u) * radial[:, 0] + (normals @ v) * radial[:, 1]
    low, high = np.percentile(height, [0.5, 99.5])
    low_end, high_end = height < low + 0.008, height > high - 0.008
    low_is_bottom = np.mean(r[low_end] < radius / 2) > np.mean(r[high_end] < radius / 2)
    bottom, rim = (low_end, high_end) if low_is_bottom else (high_end, low_end)

    parts = np.full(len(pos), "other", dtype=object)
    wall = np.abs(r - radius) < 0.004
    parts[~bottom & ~rim & wall & (normal_radial > 0.7)] = "side"
    parts[~bottom & ~rim & (r < radius - 0.001) & (normal_radial < -0.7)] = "inside"
    parts[r > radius + 0.006] = "handle"
    parts[bottom & (r <= radius + 0.006)] = "bottom"
    parts[rim & (r <= radius + 0.006)] = "rim"
    # Side nodes at either end belong to the side, not the rim or bottom.
    parts[(bottom | rim) & wall & (normal_radial > 0.7)] = "side"
    return parts


def draw_views(fig, grid_row, pos, layers, title, num_rows, num_cols, col_offset=0):
    """Draw one model in len(VIEWS) 3D views on a row of a figure's subplot grid.

    Args:
        fig: The figure.
        grid_row: The row of the subplot grid.
        pos: The node positions.
        layers: (mask, colors, size, label) tuples, drawn in order.
        title: The row's title, shown above its first view.
        num_rows: The number of rows of the subplot grid.
        num_cols: The number of columns of the subplot grid.
        col_offset: The column of the first view.
    """
    for i, (elevation, azimuth) in enumerate(VIEWS):
        ax = fig.add_subplot(
            num_rows,
            num_cols,
            grid_row * num_cols + col_offset + i + 1,
            projection="3d",
        )
        for mask, colors, size, label in layers:
            if not np.any(mask):
                continue
            selected = pos[mask]
            layer_colors = colors
            if not isinstance(colors, str) and np.ndim(colors) > 1:
                layer_colors = colors[mask]
            ax.scatter(
                selected[:, 0],
                selected[:, 1],
                selected[:, 2],
                c=layer_colors,
                s=size,
                depthshade=False,
                label=label if i == 0 else None,
            )
        ax.view_init(elevation, azimuth)
        ax.set_axis_off()
        ax.set_aspect("equal")
        if i == 0:
            ax.set_title(title, fontsize=9, loc="left")
            if any(label for *_, label in layers):
                ax.legend(fontsize=7, loc="lower left", markerscale=2)


def spread_layers(inhibited, order, seed, num_nodes):
    """Layers drawing uninhibited nodes, inhibited ones by spread order, and the seed.

    Returns:
        The layers for draw_views.
    """
    reached = order >= 0
    order_colors = plt.get_cmap("viridis")(
        np.where(reached, order, 0) / max(order.max(), 1)
    )
    seed_mask = np.arange(num_nodes) == seed
    return [
        (~inhibited, "0.8", 3, "not inhibited"),
        (inhibited & ~reached, "tab:red", 6, None),
        (reached, order_colors, 6, "inhibited (first to last)"),
        (seed_mask, "magenta", 90, "seed"),
    ]


def coverage_text(inhibited, parts) -> str:
    return ", ".join(
        f"{name} {inhibited[parts == name].mean():.0%}"
        for name in PART_NAMES
        if np.any(parts == name)
    )


def choose_mug_seeds(memory, parts, spreader, num_candidates=10):
    """Pick a representative seed on each part of the mug.

    Spreads from a node's own neighborhood vary with the noise in the stored
    features, so rather than picking a single node, spreads are run from
    num_candidates random (reliable) nodes of each part, and the seed whose
    spread inhibits the median number of nodes is chosen.

    Returns:
        (description, node) pairs.
    """
    pos = np.asarray(memory[MUG][spreader.channel].pos)
    reliable = spreader.reliable(MUG)
    handle_distance = KDTree(pos[parts == "handle"]).query(pos)[0]
    side = parts == "side"
    regions = [
        ("side", side & (handle_distance > 0.02)),
        ("side, next to the handle", side & (handle_distance < 0.012)),
        ("rim", parts == "rim"),
        ("bottom", parts == "bottom"),
        ("handle", parts == "handle"),
        ("inside wall", parts == "inside"),
    ]
    rng = np.random.default_rng(0)
    seeds = []
    for description, region in regions:
        candidates = np.nonzero(region & reliable)[0]
        candidates = rng.choice(
            candidates, min(num_candidates, len(candidates)), replace=False
        )
        sizes = [spreader.spread(MUG, int(node))[0].sum() for node in candidates]
        median = candidates[np.argsort(sizes)[len(sizes) // 2]]
        seeds.append((description, int(median)))
    return seeds


def plot_mug_seeds(memory):
    spreader = Spreader(memory)
    channel = spreader.channel
    parts = label_mug_parts(memory, MUG, channel)
    pos = np.asarray(memory[MUG][channel].pos)
    seeds = choose_mug_seeds(memory, parts, spreader)
    num_cols = len(VIEWS) + 1
    fig = plt.figure(figsize=(4 * num_cols, 3.6 * len(seeds)))
    for row, (description, seed) in enumerate(seeds):
        inhibited, order = spreader.spread(MUG, seed)
        draw_views(
            fig,
            row,
            pos,
            spread_layers(inhibited, order, seed, len(pos)),
            f"Seed on the {description}: {inhibited.sum()} of {len(pos)} nodes "
            f"inhibited\n{coverage_text(inhibited, parts)}",
            len(seeds),
            num_cols,
        )
    # The part labels used for the coverage percentages, in the last column.
    ax_parts = fig.add_subplot(len(seeds), num_cols, num_cols, projection="3d")
    part_colors = {
        "side": "tab:blue",
        "rim": "tab:orange",
        "bottom": "tab:green",
        "handle": "tab:red",
        "inside": "tab:cyan",
        "other": "0.6",
    }
    for name, color in part_colors.items():
        mask = parts == name
        ax_parts.scatter(*pos[mask].T, c=color, s=3, label=name, depthshade=False)
    ax_parts.view_init(*VIEWS[0])
    ax_parts.set_axis_off()
    ax_parts.set_aspect("equal")
    ax_parts.set_title("Mug parts (for coverage)", fontsize=9, loc="left")
    ax_parts.legend(fontsize=7, loc="lower left", markerscale=3)
    fig.suptitle(
        f"{MUG} ({channel} graph): inhibition spread from sensing one location "
        "on each part of the mug\n(the seed with the median spread out of 10 random "
        "seeds per part; percentages: share of each part inhibited)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, f"{MUG}_seeds.png"


def compartments(spreader, graph_id, rng):
    """Divide a graph into the regions that spreads from its nodes inhibit.

    Returns:
        The compartment of each node (-1 for nodes no spread reached), and the
        seed of each compartment.
    """
    num_nodes = len(spreader.lm.get_graph(graph_id, spreader.channel).pos)
    reliable = spreader.reliable(graph_id)
    labels = np.full(num_nodes, -1)
    seeds = []
    for node in rng.permutation(num_nodes):
        if labels[node] >= 0 or not reliable[node]:
            continue
        inhibited, _ = spreader.spread(graph_id, int(node))
        new = inhibited & (labels < 0)
        if not np.any(new):
            continue
        labels[new] = len(seeds)
        seeds.append(int(node))
    return labels, seeds


def plot_compartments(memory):
    spreader = Spreader(memory)
    channel = spreader.channel
    objects = [graph_id for graph_id in COMPARTMENT_OBJECTS if graph_id in memory]
    fig = plt.figure(figsize=(4 * len(VIEWS), 3.6 * len(objects)))
    rng = np.random.default_rng(0)
    for row, graph_id in enumerate(objects):
        pos = np.asarray(memory[graph_id][channel].pos)
        labels, _ = compartments(spreader, graph_id, rng)
        ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        large = ids[counts >= MIN_COMPARTMENT_NODES]
        large = large[np.argsort(-counts[np.isin(ids, large)])]
        cmap = plt.get_cmap("tab10")
        node_colors = np.zeros((len(pos), 4))
        for rank, compartment in enumerate(large):
            node_colors[labels == compartment] = cmap(rank % cmap.N)
        in_large = np.isin(labels, large)
        sizes = [int(size) for size in np.sort(counts)[::-1][:6]]
        draw_views(
            fig,
            row,
            pos,
            [
                (~in_large, "0.75", 3, None),
                (in_large, node_colors, 5, None),
            ],
            f"{graph_id}: {len(large)} compartments of at least "
            f"{MIN_COMPARTMENT_NODES} nodes, covering {in_large.mean():.0%} of "
            f"{len(pos)} nodes\nlargest: {', '.join(map(str, sizes))} nodes "
            "(grey: smaller compartments)",
            len(objects),
            len(VIEWS),
        )
    fig.suptitle(
        f"Compartments of the {channel} graphs: regions inhibited together "
        "by a spread from sensing any of their nodes\n(one color per compartment, "
        "largest first)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, "compartments.png"


def plot_tolerance(memory):
    channel = sensor_channel(memory)
    parts = label_mug_parts(memory, MUG, channel)
    pos = np.asarray(memory[MUG][channel].pos)
    seeds = choose_mug_seeds(memory, parts, Spreader(memory))[:2]
    tolerances = [0, 1, 2]
    num_cols = len(VIEWS) * len(seeds)
    fig = plt.figure(figsize=(3.4 * num_cols, 3.6 * len(tolerances)))
    for row, tolerance in enumerate(tolerances):
        spreader = Spreader(memory, max_dissimilar_neighbors=tolerance)
        for col, (description, seed) in enumerate(seeds):
            inhibited, order = spreader.spread(MUG, seed)
            draw_views(
                fig,
                row,
                pos,
                spread_layers(inhibited, order, seed, len(pos)),
                f"max_dissimilar_neighbors={tolerance}, seed on the {description}:"
                f" {inhibited.sum()} inhibited\n{coverage_text(inhibited, parts)}",
                len(tolerances),
                num_cols,
                col_offset=col * len(VIEWS),
            )
    fig.suptitle(
        f"{MUG}: spreads from the same seeds with different max_dissimilar_neighbors"
        " (the default is 1)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, f"{MUG}_tolerance.png"


def main():
    args = parse_args()
    if not args.interactive:
        plt.switch_backend("Agg")
    # Matrix products of float64 arrays spuriously warn on some macOS builds.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    output_dir = args.output_dir.expanduser()
    if not args.interactive:
        output_dir.mkdir(parents=True, exist_ok=True)
    memory = load_lm_memory(resolve_checkpoint(args.model_path), args.lm)
    plots = [plot_compartments]
    if MUG in memory:
        plots = [plot_mug_seeds, plot_compartments, plot_tolerance]
    for plot in plots:
        fig, filename = plot(memory)
        if args.interactive:
            continue
        path = output_dir / f"lm{args.lm}_{filename}"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Saved {path}")
    if args.interactive:
        plt.show()


if __name__ == "__main__":
    main()
