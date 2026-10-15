# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# ruff: noqa: DOC201,DOC501
"""Visualize how sensory input explains regions of learned object models.

Runs the EvidenceGoalGenerator's propagation (of a salience reduction) through
neighboring nodes with a similar hue and surface normal on the sensor channel
graphs of one learning module (LM) of a pretrained model, sensing the features
stored at a chosen seed node with the most likely hypothesis (MLH) at that node.
Saves (or, with --interactive, shows), prefixed with the model and LM (e.g.
comp_models_mujoco_lm2_):

- <object>_seeds.png: for the mug, the region reached by a propagation from
  seeds on its side, next to its handle, on its rim, bottom, handle and inside
  wall, colored by the order in which the propagation reached each node.
- compartments.png: for each object, the "compartments" that propagations
  divide its model into. Starting from random unreached nodes, propagations are
  run until every node is assigned to the compartment of the first propagation
  that reached it.
- <object>_tolerance.png: for the mug, propagations from the same seeds with
  different max_dissimilar_neighbors.
- visits.png: for each object, the online visits inference would make to it:
  starting from a seed on the side of the mug (or a random node of other
  objects), the next seed is sampled uniformly from the nodes not yet
  reached, until all are, colored by the visit that first reached each.

By default, propagations run on the sensor channel: the one channel of the LM's graphs
that receives input from a sensor module (e.g. patch_2 for LM 2, patch_0 for
LM 0). --channels instead propagates on any of the LM's channels, including
those receiving object IDs from lower-level LMs (e.g. learning_module_0 and
learning_module_1 for LM 2 of comp_models_mujoco), where the propagation goes
through the neighboring nodes storing the received object ID. With several
channels, each seed senses its location on all of them at once (sensing the
features, or object ID, stored at each channel's node nearest the location), and
their nodes are superimposed in the figures, with one marker per channel. Objects
the LM has no model of, or that have none of the channels, are skipped, e.g. LM 0
only models the objects without stickers, and only the objects with stickers
have a learning_module_1 channel. The tolerance figure, which only affects
sensory propagation, is skipped when propagating on object-ID channels only.

Usage:
    python analysis/scripts/visualize_salience_propagation.py [MODEL_PATH] [options]

Options:
    --model NAME      Which pretrained model to use (default: comp_models_mujoco):
                      - comp_models_mujoco: the compositional objects with stickers
                        (supervised_pre_training_objects_with_stickers_comp_models_
                        mujoco), LM 2, sensed by a distant agent.
                      - surf_agent_77obj: the 77 YCB objects learned by a surface
                        agent (pretrained_ycb_v13/surf_agent_1lm_77obj), which
                        senses the surface from close by, looking straight down at
                        it. Its 10 "distinct" objects are drawn, including the YCB
                        mug.
    MODEL_PATH        Overrides the model's checkpoint: either model.pt itself, its
                      pretrained directory, or the experiment directory.
    --lm ID           Overrides which learning module's models to use (e.g. 0 to
                      compare the models of LM 0 of comp_models_mujoco).
    --channels CH [CH ...]
                      The input channels to propagate on, superimposed (default: the
                      sensor channel), e.g. learning_module_0 learning_module_1.
    --objects ID [ID ...]
                      Overrides the objects drawn in the compartments and visits
                      figures (e.g. 024_mug_tbp_horz 025_mug_tbp_vert, which have
                      a learning_module_1 channel).
    --mug ID          Overrides the mug of the seeds, tolerance and visits figures
                      (e.g. 024_mug_tbp_horz).
    --min-propagation-fraction F
                      Overrides the goal generator's min_propagation_fraction in
                      all figures (0 to not push propagation through).
    --output-dir DIR  Where to save the figures (default:
                      ~/tbp/results/comp_benefits_figures/salience_propagation).
    --interactive     Show the figures in interactive windows (e.g. to rotate
                      the 3D views) instead of saving them. Figures taller than the
                      window scroll with the mouse wheel, the arrow keys, or page
                      up / page down.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.goal_generation import EvidenceGoalGenerator

DEFAULT_OUTPUT_DIR = "~/tbp/results/comp_benefits_figures/salience_propagation"
MODELS = {
    "comp_models_mujoco": SimpleNamespace(
        path="~/tbp/results/monty/pretrained_models/my_trained_models/"
        "supervised_pre_training_objects_with_stickers_comp_models_mujoco",
        lm=2,
        mug="023_mug",
        # The base objects, without stickers
        objects=["023_mug", "001_cube", "006_disk", "011_cylinder", "016_sphere"],
    ),
    "surf_agent_77obj": SimpleNamespace(
        path="~/tbp/results/monty/pretrained_models/pretrained_ycb_v13/"
        "surf_agent_1lm_77obj",
        lm=0,
        mug="mug",
        # The objects of eval_distinctobj_random
        objects=[
            "mug",
            "bowl",
            "potted_meat_can",
            "spoon",
            "strawberry",
            "mustard_bottle",
            "dice",
            "golf_ball",
            "c_lego_duplo",
            "banana",
        ],
    ),
}
# The height (in inches) of an interactive window; taller figures scroll.
WINDOW_HEIGHT = 9.0
# How far (in inches) one step of the mouse wheel or arrow keys scrolls.
SCROLL_STEP = 0.6
# Viewpoints (elevation, azimuth) of the three views of each model.
VIEWS = [(20, 30), (20, 210), (-60, 30)]
# Compartments with fewer nodes are drawn in grey rather than their own color.
MIN_COMPARTMENT_NODES = 2
# Visits newly reaching fewer than this fraction of nodes are drawn in grey.
MIN_VISIT_FRACTION = 0.01
PART_NAMES = ["side", "rim", "bottom", "handle", "inside"]
# Overrides of the goal generator's parameters, for every figure.
GSG_KWARGS: dict = {}
# The input channels to propagate on, for every figure (the sensor channel if empty).
CHANNELS: list[str] = []
# The marker of each channel's nodes, in the order of the channels.
CHANNEL_MARKERS = ["o", "^", "s", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path",
        type=Path,
        nargs="?",
        default=None,
        help="overrides the model's checkpoint: model.pt, its pretrained "
        "directory, or the experiment directory",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="comp_models_mujoco",
        help="which pretrained model to use",
    )
    parser.add_argument(
        "--lm", type=int, default=None, help="overrides the LM whose models to use"
    )
    parser.add_argument(
        "--channels",
        "--channel",
        nargs="+",
        default=None,
        help="the input channels to propagate on, superimposed (default: the sensor "
        "channel)",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="overrides the objects drawn in the compartments and visits figures",
    )
    parser.add_argument(
        "--mug",
        default=None,
        help="overrides the mug of the seeds, tolerance and visits figures",
    )
    parser.add_argument(
        "--min-propagation-fraction",
        type=float,
        default=None,
        help="overrides the goal generator's min_propagation_fraction (0 to not "
        "push propagation through)",
    )
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


class SearchableGraph:
    """A stored graph with the nearest-neighbor search the goal generator uses.

    GridObjectModels (e.g. of the compositional objects) provide this search, but
    the GraphObjectModels of older models (e.g. of the surface agent) do not.
    """

    def __init__(self, graph):
        self.pos = np.asarray(graph.pos)
        self.x = np.asarray(graph.x)
        self.feature_mapping = graph.feature_mapping
        self._tree = KDTree(self.pos)

    def find_nearest_neighbors(
        self, search_locations, num_neighbors, return_distance=False
    ):
        distances, node_ids = self._tree.query(search_locations, k=num_neighbors)
        return distances if return_distance else node_ids


def load_lm_memory(checkpoint: Path, lm_id: int) -> dict:
    """Load one LM's graph memory, with nearest-neighbor search for every graph."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lm_dict = state["lm_dict"]
    memory = lm_dict[lm_id if lm_id in lm_dict else str(lm_id)]["graph_memory"]
    return {
        graph_id: {
            channel: SearchableGraph(graph) for channel, graph in channels.items()
        }
        for graph_id, channels in memory.items()
    }


def sensor_channel(memory: dict) -> str:
    """Find the channel of an LM's graphs that receives input from a sensor module.

    Channels from other LMs store "object_id" features; the sensor channel stores
    the surface features that sensory propagation compares.
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

    Implements just what the EvidenceGoalGenerator uses to propagate, with the
    MLH set by the caller.
    """

    def __init__(self, memory: dict):
        self.memory = memory
        self.learning_module_id = "demo_lm"
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


def is_object_id_channel(memory: dict, channel: str) -> bool:
    """Whether a channel receives object IDs from a lower-level LM."""
    return any(
        "object_id" in graphs[channel].feature_mapping
        for graphs in memory.values()
        if channel in graphs
    )


class Propagator:
    """Propagate on one or more channels from sensing a location.

    The nodes of a graph's channels are concatenated in the order of
    self.channels, so that each node index refers to one node of one channel.
    """

    def __init__(self, memory: dict, **gsg_kwargs):
        self.lm = FakeParentLM(memory)
        self.channels = CHANNELS or [sensor_channel(memory)]
        self.gsg = EvidenceGoalGenerator(**{**GSG_KWARGS, **gsg_kwargs})
        self.gsg.parent_lm = self.lm
        self.gsg.reset()
        self._nodes = {}

    def channels_in(self, graph_id: str) -> list[str]:
        """Get the channels to propagate on that a graph has."""
        graphs = self.lm.memory.get(graph_id, {})
        return [channel for channel in self.channels if channel in graphs]

    def nodes(self, graph_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the nodes of all of a graph's channels to propagate on.

        Returns:
            The position of each node, and the index of its channel in
            self.channels.
        """
        if graph_id not in self._nodes:
            channels = self.channels_in(graph_id)
            pos = [np.asarray(self.lm.get_graph(graph_id, c).pos) for c in channels]
            node_channels = [
                np.full(len(channel_pos), self.channels.index(channel))
                for channel, channel_pos in zip(channels, pos)
            ]
            self._nodes[graph_id] = (np.concatenate(pos), np.concatenate(node_channels))
        return self._nodes[graph_id]

    def propagate(self, graph_id: str, node: int) -> tuple[np.ndarray, np.ndarray]:
        """Propagate from sensing a node's location on all of a graph's channels.

        Each channel senses the features stored at its node nearest the location:
        surface features on a channel from a sensor module, or the ID of the child
        object stored there on a channel from a lower-level LM.

        Returns:
            Whether each node was reached (its salience reduced), and how far
            through its channel's propagation each node was reached, from 0
            (first) to 1 (last), or -1 for nodes no propagation reached.
        """
        location = self.nodes(graph_id)[0][node]
        channels = self.channels_in(graph_id)
        percepts = [self._percept(graph_id, channel, location) for channel in channels]
        self.lm.mlh = {
            "graph_id": graph_id,
            "mlh_id": 0,
            "location": location,
            "rotation": Rotation.identity(),
        }
        # Only the salience is reset, keeping the cached neighbors.
        self.gsg._salience = {}
        self.gsg.propagation_records = []
        self.gsg._propagate_from_observations(percepts)
        reached_nodes, order = [], []
        for channel in channels:
            num_nodes = len(self.lm.get_graph(graph_id, channel).pos)
            channel_order = np.full(num_nodes, -1.0)
            for record in self.gsg.propagation_records:
                if record.input_channel == channel:
                    reached = record.node_order
                    channel_order[reached] = np.arange(len(reached)) / max(
                        len(reached) - 1, 1
                    )
            reached_nodes.append(self.gsg.get_salience(graph_id, channel) < 1)
            order.append(channel_order)
        return np.concatenate(reached_nodes), np.concatenate(order)

    def _percept(self, graph_id: str, channel: str, location: np.ndarray):
        """Sense the features stored at a channel's node nearest a location."""
        graph = self.lm.get_graph(graph_id, channel)
        node = int(graph.find_nearest_neighbors(location, num_neighbors=1))
        features = {
            name: np.asarray(graph.x[node, start:end])
            for name, (start, end) in graph.feature_mapping.items()
        }
        return SimpleNamespace(
            sender_type="LM" if "object_id" in features else "SM",
            sender_id=channel,
            morphological_features={
                "pose_vectors": features["pose_vectors"].reshape(3, 3)
            },
            non_morphological_features={
                name: value
                for name, value in features.items()
                if name != "pose_vectors"
            },
        )


def channels_text(propagator: Propagator) -> str:
    return " + ".join(propagator.channels)


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
    normals = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
    )
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


def mug_parts(memory: dict, mug: str, pos: np.ndarray) -> np.ndarray:
    """Label locations on a mug by the part of their nearest sensor graph node.

    Only the sensor graph stores the surface normals that label_mug_parts fits
    the mug to; the graphs of channels from other LMs store child object poses.

    Returns:
        One of PART_NAMES, or "other", per location.
    """
    channel = sensor_channel(memory)
    parts = label_mug_parts(memory, mug, channel)
    sensor_pos = np.asarray(memory[mug][channel].pos)
    return parts[KDTree(sensor_pos).query(pos)[1]]


def draw_views(
    fig,
    grid_row,
    pos,
    layers,
    title,
    num_rows,
    num_cols,
    col_offset=0,
    node_channels=None,
    channel_names=(),
    alpha=1.0,
):
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
        node_channels: The index of each node's channel in channel_names, drawn
            with that channel's marker (see CHANNEL_MARKERS); None for one channel.
        channel_names: The names of the channels, shown in the legend if more
            than one.
        alpha: The opacity of the markers.
    """
    if node_channels is None:
        node_channels = np.zeros(len(pos), dtype=int)
    for i, (elevation, azimuth) in enumerate(VIEWS):
        ax = fig.add_subplot(
            num_rows,
            num_cols,
            grid_row * num_cols + col_offset + i + 1,
            projection="3d",
        )
        for mask, colors, size, layer_label in layers:
            # Labeled once, on the first view, for the legend.
            label = layer_label if i == 0 else None
            for channel in np.unique(node_channels[mask]):
                selected_mask = mask & (node_channels == channel)
                selected = pos[selected_mask]
                layer_colors = colors
                if not isinstance(colors, str) and np.ndim(colors) > 1:
                    layer_colors = colors[selected_mask]
                ax.scatter(
                    selected[:, 0],
                    selected[:, 1],
                    selected[:, 2],
                    c=layer_colors,
                    s=size,
                    marker=CHANNEL_MARKERS[channel % len(CHANNEL_MARKERS)],
                    depthshade=False,
                    label=label,
                    alpha=alpha,
                )
                label = None
        ax.view_init(elevation, azimuth)
        ax.set_axis_off()
        ax.set_aspect("equal")
        if i == 0:
            ax.set_title(title, fontsize=9, loc="left")
            handles, _ = ax.get_legend_handles_labels()
            if len(channel_names) > 1:
                handles += [
                    Line2D(
                        [],
                        [],
                        marker=CHANNEL_MARKERS[channel % len(CHANNEL_MARKERS)],
                        markersize=3,
                        color="0.4",
                        linestyle="none",
                        label=channel_names[channel],
                    )
                    for channel in np.unique(node_channels)
                ]
            if handles:
                ax.legend(handles=handles, fontsize=7, loc="lower left", markerscale=2)


def propagation_layers(reached_nodes, order, seed, num_nodes):
    """Layers drawing unreached nodes, reached ones by propagation order, and the seed.

    Returns:
        The layers for draw_views.
    """
    reached = order >= 0
    order_colors = plt.get_cmap("viridis")(np.where(reached, order, 0))
    seed_mask = np.arange(num_nodes) == seed
    return [
        (~reached_nodes, "0.8", 3, "not reached"),
        (reached_nodes & ~reached, "tab:red", 6, None),
        (reached, order_colors, 6, "reached (first to last)"),
        (seed_mask, "magenta", 90, "seed"),
    ]


def coverage_text(reached, parts) -> str:
    return ", ".join(
        f"{name} {reached[parts == name].mean():.0%}"
        for name in PART_NAMES
        if np.any(parts == name)
    )


def choose_mug_seeds(mug, parts, propagator, num_candidates=10):
    """Pick a representative seed on each part of the mug.

    Propagations from a node's own neighborhood vary with the noise in the
    stored features, so rather than picking a single node, propagations are run
    from num_candidates random nodes of each part, and the seed whose
    propagation reaches the median number of nodes is chosen.

    Returns:
        (description, node) pairs.
    """
    pos, _ = propagator.nodes(mug)
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
        candidates = np.nonzero(region)[0]
        candidates = rng.choice(
            candidates, min(num_candidates, len(candidates)), replace=False
        )
        sizes = [propagator.propagate(mug, int(node))[0].sum() for node in candidates]
        median = candidates[np.argsort(sizes)[len(sizes) // 2]]
        seeds.append((description, int(median)))
    return seeds


def plot_mug_seeds(memory, mug, _objects):
    propagator = Propagator(memory)
    pos, node_channels = propagator.nodes(mug)
    parts = mug_parts(memory, mug, pos)
    seeds = choose_mug_seeds(mug, parts, propagator)
    num_cols = len(VIEWS) + 1
    fig = plt.figure(figsize=(4 * num_cols, 3.6 * len(seeds)))
    for row, (description, seed) in enumerate(seeds):
        reached, order = propagator.propagate(mug, seed)
        draw_views(
            fig,
            row,
            pos,
            propagation_layers(reached, order, seed, len(pos)),
            f"Seed on the {description}: {reached.sum()} of {len(pos)} nodes "
            f"reached\n{coverage_text(reached, parts)}",
            len(seeds),
            num_cols,
            node_channels=node_channels,
            channel_names=propagator.channels,
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
        f"{mug} ({channels_text(propagator)}): propagation from sensing one "
        "location "
        "on each part of the mug\n(the seed with the median propagation out of 10 "
        "random seeds per part; percentages: share of each part reached)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, f"{mug}_seeds.png"


def compartments(propagator, graph_id, rng):
    """Divide a graph into the regions that propagations from its nodes reach.

    Returns:
        The compartment of each node (-1 for nodes no propagation reached), and
        the seed of each compartment.
    """
    num_nodes = len(propagator.nodes(graph_id)[0])
    labels = np.full(num_nodes, -1)
    seeds = []
    for node in rng.permutation(num_nodes):
        if labels[node] >= 0:
            continue
        reached, _ = propagator.propagate(graph_id, int(node))
        new = reached & (labels < 0)
        if not np.any(new):
            continue
        labels[new] = len(seeds)
        seeds.append(int(node))
    return labels, seeds


def plot_compartments(memory, _mug, objects):
    propagator = Propagator(memory)
    objects = [graph_id for graph_id in objects if propagator.channels_in(graph_id)]
    fig = plt.figure(figsize=(4 * len(VIEWS), 3.6 * len(objects)))
    rng = np.random.default_rng(0)
    for row, graph_id in enumerate(objects):
        pos, node_channels = propagator.nodes(graph_id)
        labels, _ = compartments(propagator, graph_id, rng)
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
            node_channels=node_channels,
            channel_names=propagator.channels,
        )
    fig.suptitle(
        f"Compartments of the {channels_text(propagator)} graphs: regions reached "
        "together "
        "by a propagation from sensing any of their nodes\n(one color per "
        "compartment, largest first)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, "compartments.png"


def simulate_visits(propagator, graph_id, first_seed, rng):
    """Visit every node of a graph as inference would, without salience recovering.

    Beginning with the first seed, the sensed features of a seed node propagate,
    and the next seed is sampled uniformly from the nodes not yet reached, until
    every node has been reached (or been a seed).

    Returns:
        The visit (numbered from 0) that first reached each node, and for each
        visit, its seed and the number of nodes its propagation reached.
    """
    num_nodes = len(propagator.nodes(graph_id)[0])
    visit = np.full(num_nodes, -1)
    visits = []
    seed = first_seed
    while seed is not None:
        reached, _ = propagator.propagate(graph_id, seed)
        reached[seed] = True
        visit[reached & (visit < 0)] = len(visits)
        visits.append((seed, int(reached.sum())))
        remaining = np.flatnonzero(visit < 0)
        seed = int(rng.choice(remaining)) if len(remaining) else None
    return visit, visits


def first_mug_seed(parts, pos, rng):
    """Sample a node on the side of a mug, away from its handle."""
    handle_distance = KDTree(pos[parts == "handle"]).query(pos)[0]
    return int(rng.choice(np.flatnonzero((parts == "side") & (handle_distance > 0.02))))


def plot_visits(memory, mug, objects):
    propagator = Propagator(memory)
    fraction = propagator.gsg.min_propagation_fraction or 0
    objects = [graph_id for graph_id in objects if propagator.channels_in(graph_id)]
    num_cols = len(VIEWS) + 1
    fig = plt.figure(figsize=(3.6 * num_cols, 3.6 * len(objects)))
    # tab20 pairs each color with a lighter shade of it; all of the strong colors
    # come first so that consecutive visits differ in hue, and the greys are left
    # out, being the color of the other visits.
    tab20 = plt.get_cmap("tab20").colors
    visit_colors = [tab20[i] for i in [*range(0, 20, 2), *range(1, 20, 2)]]
    visit_colors = [c for c in visit_colors if c not in (tab20[14], tab20[15])]
    for row, graph_id in enumerate(objects):
        rng = np.random.default_rng(0)
        pos, node_channels = propagator.nodes(graph_id)
        num_nodes = len(pos)
        parts = None
        if graph_id == mug:
            parts = mug_parts(memory, mug, pos)
            first_seed = first_mug_seed(parts, pos, rng)
        else:
            first_seed = int(rng.integers(num_nodes))
        visit, visits = simulate_visits(propagator, graph_id, first_seed, rng)
        new_counts = np.bincount(visit, minlength=len(visits))
        shown = np.flatnonzero(new_counts >= MIN_VISIT_FRACTION * num_nodes)
        shown = shown[: len(visit_colors)]
        node_colors = np.tile(mpl_colors.to_rgba("0.75"), (num_nodes, 1))
        legend = []
        for rank, index in enumerate(shown):
            color = mpl_colors.to_rgba(visit_colors[rank])
            node_colors[visit == index] = color
            label = (
                f"{index + 1}: {new_counts[index] / num_nodes:.0%} new"
                f" (propagation {visits[index][1] / num_nodes:.0%})"
            )
            if parts is not None:
                new_parts = parts[visit == index]
                names, counts = np.unique(new_parts, return_counts=True)
                label += f", {names[np.argmax(counts)]}"
            legend.append(Patch(color=color, label=label))
        rest = np.isin(visit, shown, invert=True)
        legend.append(
            Patch(
                color="0.75",
                label=f"{len(visits) - len(shown)} other visits: "
                f"{rest.mean():.0%} of nodes",
            )
        )
        cumulative = np.cumsum(new_counts) / num_nodes
        draw_views(
            fig,
            row,
            pos,
            [(np.ones(num_nodes, dtype=bool), node_colors, 4, None)],
            f"{graph_id}: {len(visits)} visits to reach all {num_nodes} nodes; "
            f"the first 5 reach {cumulative[min(4, len(visits) - 1)]:.0%}",
            len(objects),
            num_cols,
            node_channels=node_channels,
            channel_names=propagator.channels,
            alpha=0.5,
        )
        ax = fig.add_subplot(len(objects), num_cols, (row + 1) * num_cols)
        ax.set_axis_off()
        ax.legend(handles=legend, loc="center left", fontsize=7, frameon=False)
    fig.suptitle(
        f"Online visits to the {channels_text(propagator)} graphs, each propagating "
        "from a seed sampled "
        "from the nodes not yet reached (the first on the side of the mug)\n"
        "colored by the visit that first reached each node; legend: the order of "
        "the visit, the nodes it newly reached, and the nodes its propagation "
        f"reached\nmin_propagation_fraction={fraction:.0%}; grey: visits newly "
        f"reaching fewer than {MIN_VISIT_FRACTION:.0%} of nodes",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, "visits.png"


def plot_tolerance(memory, mug, _objects):
    default_propagator = Propagator(memory)
    pos, node_channels = default_propagator.nodes(mug)
    parts = mug_parts(memory, mug, pos)
    seeds = choose_mug_seeds(mug, parts, default_propagator)[:2]
    tolerances = [1, 2, 3]
    num_cols = len(VIEWS) * len(seeds)
    fig = plt.figure(figsize=(3.4 * num_cols, 3.6 * len(tolerances)))
    for row, tolerance in enumerate(tolerances):
        propagator = Propagator(memory, max_dissimilar_neighbors=tolerance)
        for col, (description, seed) in enumerate(seeds):
            reached, order = propagator.propagate(mug, seed)
            draw_views(
                fig,
                row,
                pos,
                propagation_layers(reached, order, seed, len(pos)),
                f"max_dissimilar_neighbors={tolerance}, seed on the {description}:"
                f" {reached.sum()} reached\n{coverage_text(reached, parts)}",
                len(tolerances),
                num_cols,
                col_offset=col * len(VIEWS),
                node_channels=node_channels,
                channel_names=propagator.channels,
            )
    fig.suptitle(
        f"{mug} ({channels_text(default_propagator)}): propagations from the same "
        "seeds with different max_dissimilar_neighbors (the default is 2)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, f"{mug}_tolerance.png"


def make_scrollable(fig, window_height=WINDOW_HEIGHT):
    """Show a tall figure in a window of a fixed height that scrolls through it.

    The figure keeps the layout of its full height; the window shows a slice of it,
    which the mouse wheel, the up / down arrow keys, and page up / page down move.
    Works with any interactive backend, by moving the axes and texts within the
    window rather than relying on the backend's own scroll bars.
    """
    full_height = fig.get_figheight()
    if full_height <= window_height:
        return
    # The original (rather than the active) positions, since 3D axes shrink their
    # active position to fit their aspect within the original one on every draw.
    # Vertical positions in inches from the bottom of the full-height figure.
    axes = [(ax, ax.get_position(original=True)) for ax in fig.axes]
    axes = [
        (ax, box.x0, box.width, box.y0 * full_height, box.height * full_height)
        for ax, box in axes
    ]
    texts = [(text, text.get_position()[1] * full_height) for text in fig.texts]
    fig.set_size_inches(fig.get_figwidth(), window_height, forward=True)
    max_offset = full_height - window_height
    state = {"offset": 0.0}  # How far (in inches) the window is from the top.

    def layout():
        bottom = max_offset - state["offset"]
        for ax, x0, width, y0, height in axes:
            y = (y0 - bottom) / window_height
            ax.set_position([x0, y, width, height / window_height])
            ax.set_visible(y < 1 and y + height / window_height > 0)
        for text, y in texts:
            text.set_y((y - bottom) / window_height)
        fig.canvas.draw_idle()

    def scroll(inches):
        state["offset"] = float(np.clip(state["offset"] + inches, 0, max_offset))
        layout()

    def on_scroll(event):
        scroll(-event.step * SCROLL_STEP)

    def on_key(event):
        steps = {
            "down": SCROLL_STEP,
            "up": -SCROLL_STEP,
            "pagedown": window_height * 0.9,
            "pageup": -window_height * 0.9,
        }
        if event.key in steps:
            scroll(steps[event.key])

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    # Keep references to the callbacks for as long as the figure lives.
    fig._scroll_callbacks = (on_scroll, on_key)
    layout()


def main():
    args = parse_args()
    if not args.interactive:
        plt.switch_backend("Agg")
    # Matrix products of float64 arrays spuriously warn on some macOS builds.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    output_dir = args.output_dir.expanduser()
    if not args.interactive:
        output_dir.mkdir(parents=True, exist_ok=True)
    model = MODELS[args.model]
    model_path = Path(model.path) if args.model_path is None else args.model_path
    lm_id = model.lm if args.lm is None else args.lm
    memory = load_lm_memory(resolve_checkpoint(model_path), lm_id)
    if args.min_propagation_fraction is not None:
        GSG_KWARGS["min_propagation_fraction"] = args.min_propagation_fraction or None
    prefix = f"{args.model}_lm{lm_id}_"
    if args.channels:
        known = {channel for graphs in memory.values() for channel in graphs}
        unknown = sorted(set(args.channels) - known)
        if unknown:
            raise ValueError(
                f"Unknown channels {unknown}; LM {lm_id}'s graphs have {sorted(known)}"
            )
        CHANNELS.extend(args.channels)
        prefix += "+".join(args.channels) + "_"
    channels = Propagator(memory).channels
    mug = args.mug or model.mug
    objects = args.objects or model.objects
    if not any(c in memory.get(graph_id, {}) for graph_id in objects for c in channels):
        raise ValueError(
            f"None of the objects {objects} have any of the channels {channels}; "
            "choose others with --objects"
        )
    plots = [plot_compartments, plot_visits]
    mug_channels = [c for c in channels if c in memory.get(mug, {})]
    if mug_channels:
        plots = [plot_mug_seeds, plot_compartments, plot_visits]
        if not all(is_object_id_channel(memory, c) for c in mug_channels):
            plots.append(plot_tolerance)
    for plot in plots:
        fig, filename = plot(memory, mug, objects)
        if args.interactive:
            make_scrollable(fig)
            continue
        path = output_dir / f"{prefix}{filename}"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Saved {path}")
    if args.interactive:
        plt.show()


if __name__ == "__main__":
    main()
