# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# ruff: noqa: DOC201,DOC501
r"""Visualize the object models learned by a Monty learning module.

Usage:
    python analysis/scripts/visualize_comp_model.py MODEL_PATH [options]

MODEL_PATH points at a trained model: either model.pt itself, its pretrained
directory, or the experiment directory containing it. For example:

    python analysis/scripts/visualize_comp_model.py \
        ~/tbp/results/monty/pretrained_models/my_trained_models/<experiment> \
        --lm 2 --show

Options:
    --lm ID         Which learning module's models to plot (default: 0).
    --objects ...   One or more object names to plot (default: all stored).
    --morphology    Draw the stored surface normal (3D channels) or oriented
                    edge direction (2D channels) as an arrow at each point.
    --show          Also open an interactive window: click and drag to
                    rotate a model, scroll to zoom the subplot under the
                    cursor.
    --output PATH   Where to save the figure
                    (default: ~/Desktop/<model_name>_lm<id>.png).
    --columns N     Max subplot columns (default: 5).
    --dpi N         Resolution of the saved figure (default: 150).

Each point is drawn with the features learned at that location:

- Patch channels are colored with the learned (HSV) color at each point.
- LM input channels (compositional models) are colored by the learned child
  object ID, using bold colors that do not occur on the objects themselves.

Channels are distinguished by marker shape.
"""

from __future__ import annotations

import argparse
import math
from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D

# Bold colors for learned object IDs; deliberately absent from the objects
# themselves so they cannot be confused with learned patch colors.
OBJECT_ID_COLORS = ["#FF2DAF", "#E8000B", "#00B140", "#FFC20A"]
PATCH_MARKER = "o"
LM_CHANNEL_MARKERS = ["*", "^", "s", "D", "v", "P"]
PATCH_POINT_SIZE = 3
OBJECT_ID_POINT_SIZE = 28
FALLBACK_COLOR = "#808080"
MORPHOLOGY_COLOR = "#333333"


def positive_int(value: str) -> int:
    """Parse a positive integer argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def resolve_checkpoint(model_path: Path) -> Path:
    """Resolve model.pt from a checkpoint, pretrained, or experiment path."""
    path = model_path.expanduser()
    for candidate in (path, path / "model.pt", path / "pretrained" / "model.pt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Model checkpoint not found under: {path}")


def as_numpy(value: Any) -> np.ndarray:
    """Convert a tensor or array-like graph value to a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_lm_dict(checkpoint: Path) -> dict:
    """Load the dictionary of all LM states from a checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {checkpoint}")
    lm_dict = state.get("lm_dict")
    if not isinstance(lm_dict, dict):
        raise KeyError(f"Checkpoint has no lm_dict: {checkpoint}")
    return lm_dict


def get_lm_memory(lm_dict: dict, lm_id: int) -> dict:
    """Return one LM's graph memory."""
    lm_key = lm_id if lm_id in lm_dict else str(lm_id)
    if lm_key not in lm_dict:
        raise KeyError(f"LM_{lm_id} not found; available: {list(lm_dict)}")
    memory = lm_dict[lm_key].get("graph_memory", {})
    if not isinstance(memory, dict) or not memory:
        populated = [
            key
            for key, value in lm_dict.items()
            if isinstance(value.get("graph_memory"), dict) and value["graph_memory"]
        ]
        raise ValueError(
            f"LM_{lm_id} has no learned models; LMs with models: {populated}"
        )
    return memory


def encode_object_name(name: str) -> int:
    """Encode an object name the way object_id features are stored in graphs."""
    return sum(ord(character) for character in name)


def decode_object_ids(lm_dict: dict) -> dict[int, str]:
    """Map numeric object_id feature values back to object names.

    Object IDs are encoded as the sum of the character codes of the object
    name, so every object name stored anywhere in the checkpoint provides
    one decodable ID.
    """
    names: dict[int, str] = {}
    for lm_state in lm_dict.values():
        memory = lm_state.get("graph_memory", {})
        if isinstance(memory, dict):
            for object_name in memory:
                names[encode_object_name(object_name)] = object_name
    return names


def select_objects(memory: dict, requested: list[str] | None) -> list[str]:
    """Return requested object names, or every stored object."""
    if requested is None:
        return list(memory)
    missing = [name for name in requested if name not in memory]
    if missing:
        raise ValueError(f"Objects not found: {missing}; available: {list(memory)}")
    return requested


def collect_channels(memory: dict, objects: list[str]) -> list[str]:
    """Return every channel across the given objects, patch channels first."""
    channels: list[str] = []
    for object_name in objects:
        for channel in memory[object_name]:
            if channel not in channels:
                channels.append(channel)
    return sorted(channels, key=lambda name: (not name.startswith("patch"), name))


def channel_graph(memory: dict, object_name: str, channel: str):
    """Return the graph stored for one object/channel, or None if absent."""
    wrapper = memory[object_name].get(channel)
    if wrapper is None:
        return None
    return getattr(wrapper, "_graph", wrapper)


def graph_points(graph, object_name: str, channel: str) -> np.ndarray:
    """Return a graph's finite Nx3 point array."""
    positions = getattr(graph, "pos", None)
    if positions is None:
        raise ValueError(f"{object_name!r}/{channel!r} has no point positions")
    points = as_numpy(positions).astype(float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"{object_name!r}/{channel!r} has invalid point shape {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError(f"{object_name!r}/{channel!r} contains non-finite points")
    return points


def feature_values(graph, feature: str) -> np.ndarray | None:
    """Return one named feature's per-point values, or None if not stored."""
    mapping = getattr(graph, "feature_mapping", None) or {}
    if feature not in mapping:
        return None
    start, end = mapping[feature]
    return as_numpy(graph.x)[:, start:end].astype(float)


def morphology_vectors(graph) -> np.ndarray | None:
    """Return the per-point normal (3D) or oriented edge (2D) unit vectors.

    Pose vectors are stored as flattened 3x3 matrices: row 0 is the surface
    normal and rows 1-2 span the tangent plane. Channels from 2D sensor
    modules (identified by their edge_strength feature) store the oriented
    edge direction in row 1, while row 0 is just the out-of-plane normal.
    """
    pose = feature_values(graph, "pose_vectors")
    if pose is None or pose.shape[1] != 9:
        return None
    pose = pose.reshape(-1, 3, 3)
    mapping = getattr(graph, "feature_mapping", None) or {}
    row = 1 if "edge_strength" in mapping else 0
    return pose[:, row, :]


def is_2d_channel(graph) -> bool:
    """Whether the channel comes from a 2D (edge-based) sensor module."""
    mapping = getattr(graph, "feature_mapping", None) or {}
    return "edge_strength" in mapping


def channel_markers(channels: list[str]) -> dict[str, str]:
    """Assign a marker shape to each channel; patches share circles."""
    lm_markers = cycle(LM_CHANNEL_MARKERS)
    return {
        channel: PATCH_MARKER if channel.startswith("patch") else next(lm_markers)
        for channel in channels
    }


def collect_object_id_colors(
    memory: dict,
    id_names: dict[int, str],
) -> dict[float, str]:
    """Assign a bold color to every object ID stored in the LM's memory.

    Colors are assigned across all stored objects (not just the plotted
    selection) so each ID keeps the same color regardless of --objects.
    """
    ids: set[float] = set()
    for stored_channels in memory.values():
        for wrapper in stored_channels.values():
            graph = getattr(wrapper, "_graph", wrapper)
            values = feature_values(graph, "object_id")
            if values is not None:
                ids.update(np.unique(values).tolist())
    ordered = sorted(ids, key=lambda value: id_names.get(int(value), str(value)))
    colors = cycle(OBJECT_ID_COLORS)
    return {value: next(colors) for value in ordered}


def point_colors(
    graph,
    object_id_colors: dict[float, str],
) -> tuple[np.ndarray | str, bool]:
    """Return per-point colors and whether they encode learned object IDs."""
    ids = feature_values(graph, "object_id")
    if ids is not None:
        return np.array([object_id_colors[value] for value in ids[:, 0]]), True
    hsv = feature_values(graph, "hsv")
    if hsv is not None:
        return hsv_to_rgb(np.clip(hsv, 0.0, 1.0)), False
    return FALLBACK_COLOR, False


def set_equal_limits(axis, points: np.ndarray) -> float:
    """Use equal cubic limits centered on the combined point cloud.

    Returns:
        The half-width of the limits, for scaling other plot elements.
    """
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 1e-6) * 1.05
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    return radius


def short_channel(channel: str) -> str:
    """Abbreviate channel names so subplot titles stay compact."""
    return channel.replace("learning_module_", "LM_")


def checkpoint_label(checkpoint: Path) -> str:
    """Return a useful title for any accepted checkpoint path shape."""
    if checkpoint.parent.name == "pretrained":
        return checkpoint.parent.parent.name
    return checkpoint.stem


def plot_object(
    axis,
    memory: dict,
    object_name: str,
    *,
    channels: list[str],
    markers: dict[str, str],
    object_id_colors: dict[float, str],
    show_morphology: bool,
) -> None:
    """Draw one object's channels, features, and optional morphology arrows."""
    graphs = {
        channel: graph
        for channel in channels
        if (graph := channel_graph(memory, object_name, channel)) is not None
    }
    if not graphs:
        raise ValueError(f"{object_name!r} has none of {channels}")
    points_by_channel = {
        channel: graph_points(graph, object_name, channel)
        for channel, graph in graphs.items()
    }
    radius = set_equal_limits(axis, np.concatenate(list(points_by_channel.values())))

    for channel, graph in graphs.items():
        points = points_by_channel[channel]
        if not len(points):
            continue
        colors, is_object_id = point_colors(graph, object_id_colors)
        axis.scatter(
            *points.T,
            c=colors,
            marker=markers[channel],
            s=OBJECT_ID_POINT_SIZE if is_object_id else PATCH_POINT_SIZE,
            alpha=0.9 if is_object_id or len(graphs) == 1 else 0.45,
            depthshade=False,
            linewidths=0,
        )
        if show_morphology and not is_object_id:
            vectors = morphology_vectors(graph)
            if vectors is not None:
                axis.quiver(
                    *points.T,
                    *vectors.T,
                    length=0.07 * radius,
                    normalize=True,
                    pivot="middle" if is_2d_channel(graph) else "tail",
                    color=MORPHOLOGY_COLOR,
                    linewidth=0.5,
                    alpha=0.6,
                )

    axis.view_init(elev=20, azim=20, vertical_axis="y")
    axis.set_axis_off()
    counts_line = " · ".join(
        f"{short_channel(channel)} {len(points):,}"
        for channel, points in points_by_channel.items()
    )
    total = sum(len(points) for points in points_by_channel.values())
    axis.set_title(f"{object_name}\n{total:,} total\n{counts_line}", fontsize=9)


def make_figure(
    memory: dict,
    objects: list[str],
    channels: list[str],
    *,
    object_id_colors: dict[float, str],
    id_names: dict[int, str],
    columns: int,
    checkpoint: Path,
    lm_id: int,
    show_morphology: bool,
) -> plt.Figure:
    """Create one feature-colored plot per object, with channel/ID legends."""
    markers = channel_markers(channels)
    columns = min(columns, len(objects))
    rows = math.ceil(len(objects) / columns)
    header_inches = 2.3 if object_id_colors else 1.4
    height = 3.6 * rows + header_inches
    # Keep a minimum width so the title and legends fit even for one column.
    width = max(3.4 * columns, 7.2)
    figure = plt.figure(figsize=(width, height))

    for index, object_name in enumerate(objects):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        plot_object(
            axis,
            memory,
            object_name,
            channels=channels,
            markers=markers,
            object_id_colors=object_id_colors,
            show_morphology=show_morphology,
        )

    channel_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            label=channel if marker == PATCH_MARKER else f"{channel} (object ID)",
            markerfacecolor="#666666",
            markeredgewidth=0,
            markersize=6 if marker == PATCH_MARKER else 9,
        )
        for channel, marker in markers.items()
    ]
    # Place header elements at fixed distances (in inches) from the top so the
    # layout holds up for both small and large grids.
    figure.suptitle(
        f"{checkpoint_label(checkpoint)}\nLM_{lm_id} channels",
        fontsize=14,
        y=1 - 0.08 / height,
        va="top",
    )
    figure.legend(
        handles=channel_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.66 / height),
        ncols=len(channel_handles),
        fontsize=9,
    )
    if object_id_colors:
        id_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                label=id_names.get(int(value), f"ID {value:g}"),
                markerfacecolor=color,
                markeredgewidth=0,
                markersize=7,
            )
            for value, color in object_id_colors.items()
        ]
        figure.legend(
            handles=id_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - 1.0 / height),
            ncols=len(id_handles),
            fontsize=9,
        )
    figure.subplots_adjust(
        top=1 - (header_inches - 0.35) / height, bottom=0.01, wspace=0.05, hspace=0.12
    )
    return figure


def attach_scroll_zoom(figure: plt.Figure, zoom_per_step: float = 1.15) -> None:
    """Make the scroll wheel zoom the 3D axes under the cursor.

    Matplotlib's 3D axes only support rotating (left-drag) and zooming via
    right-click-drag out of the box, so scroll zoom is wired up manually by
    shrinking or growing the axis limits around their center.
    """

    def on_scroll(event) -> None:
        axis = event.inaxes
        if axis is None or not hasattr(axis, "get_zlim3d"):
            return
        # event.step > 0 when scrolling up, which should zoom in.
        factor = zoom_per_step ** (-event.step)
        for get_limits, set_limits in (
            (axis.get_xlim3d, axis.set_xlim3d),
            (axis.get_ylim3d, axis.set_ylim3d),
            (axis.get_zlim3d, axis.set_zlim3d),
        ):
            low, high = get_limits()
            center = (low + high) / 2.0
            half_width = (high - low) / 2.0 * factor
            set_limits(center - half_width, center + half_width)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("scroll_event", on_scroll)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to model.pt, its pretrained directory, or its experiment directory.",
    )
    parser.add_argument(
        "--lm",
        type=int,
        default=0,
        help="ID of the learning module to visualize, e.g. 0, 1, or 2 (default: 0).",
    )
    parser.add_argument(
        "--objects", nargs="+", help="Object names to plot (default: all)."
    )
    parser.add_argument(
        "--morphology",
        action="store_true",
        help="Draw stored surface normals (3D channels) or oriented edges "
        "(2D channels) as arrows at each point.",
    )
    parser.add_argument("--columns", type=positive_int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output figure path (default: ~/Desktop/<model_name>_lm<id>.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive window (drag to rotate, scroll to zoom).",
    )
    parser.add_argument("--dpi", type=positive_int, default=150)
    return parser.parse_args()


def main() -> None:
    """Load an LM's graph memory, save its feature plot, and optionally show it."""
    args = parse_args()
    checkpoint = resolve_checkpoint(args.model_path)
    lm_dict = load_lm_dict(checkpoint)
    memory = get_lm_memory(lm_dict, args.lm)
    objects = select_objects(memory, args.objects)
    channels = collect_channels(memory, objects)
    id_names = decode_object_ids(lm_dict)
    object_id_colors = collect_object_id_colors(memory, id_names)

    print(
        f"Loaded {len(objects)} objects from LM_{args.lm} in {checkpoint} "
        f"(channels: {channels})"
    )
    if object_id_colors:
        decoded = {
            id_names.get(int(value), value): color
            for value, color in object_id_colors.items()
        }
        print(f"Learned object IDs: {decoded}")
    figure = make_figure(
        memory,
        objects,
        channels,
        object_id_colors=object_id_colors,
        id_names=id_names,
        columns=args.columns,
        checkpoint=checkpoint,
        lm_id=args.lm,
        show_morphology=args.morphology,
    )

    output = args.output
    if output is None:
        output = Path(f"~/Desktop/{checkpoint_label(checkpoint)}_lm{args.lm}.png")
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    print(f"Saved {output}")

    if args.show:
        attach_scroll_zoom(figure)
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
