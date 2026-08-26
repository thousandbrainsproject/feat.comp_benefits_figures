# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# ruff: noqa: DOC201,DOC501
"""Plot the object models learned by any LM, colored by input channel."""

from __future__ import annotations

import argparse
import math
from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

# Faded style for raw sensor-patch channels when LM channels are also present.
PATCH_STYLE = {"color": "#B3B3B3", "alpha": 0.3}
ACCENT_COLORS = ["#00A0DF", "#F737BD", "#7A5CFA", "#FFB800", "#2ECC71", "#E74C3C"]
POINT_SIZE = 3


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


def load_lm_memory(checkpoint: Path, lm_id: int) -> dict:
    """Load one LM's graph memory from a checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {checkpoint}")
    lm_dict = state.get("lm_dict")
    if not isinstance(lm_dict, dict):
        raise KeyError(f"Checkpoint has no lm_dict: {checkpoint}")

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


def build_channel_styles(channels: list[str]) -> dict[str, dict]:
    """Assign a plot style to each channel.

    With multiple channels, patch channels are drawn faded gray so LM inputs
    stand out; with a single channel it gets a solid accent color.
    """
    accents = cycle(ACCENT_COLORS)
    if len(channels) == 1:
        return {channels[0]: {"color": next(accents), "alpha": 1.0}}
    return {
        channel: dict(PATCH_STYLE)
        if channel.startswith("patch")
        else {"color": next(accents), "alpha": 1.0}
        for channel in channels
    }


def channel_points(memory: dict, object_name: str, channel: str) -> np.ndarray:
    """Return one channel's finite Nx3 point array, or an empty array if absent."""
    wrapper = memory[object_name].get(channel)
    if wrapper is None:
        return np.empty((0, 3))
    graph = getattr(wrapper, "_graph", wrapper)
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


def set_equal_limits(axis, points: np.ndarray) -> None:
    """Use equal cubic limits centered on the combined point cloud."""
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 1e-6) * 1.05
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def short_channel(channel: str) -> str:
    """Abbreviate channel names so subplot titles stay compact."""
    return channel.replace("learning_module_", "LM_")


def checkpoint_label(checkpoint: Path) -> str:
    """Return a useful title for any accepted checkpoint path shape."""
    if checkpoint.parent.name == "pretrained":
        return checkpoint.parent.parent.name
    return checkpoint.stem


def make_figure(
    memory: dict,
    objects: list[str],
    channel_styles: dict[str, dict],
    *,
    columns: int,
    checkpoint: Path,
    lm_id: int,
) -> plt.Figure:
    """Create one overlaid channel plot per object."""
    columns = min(columns, len(objects))
    rows = math.ceil(len(objects) / columns)
    header_inches = 1.4
    height = 3.6 * rows + header_inches
    figure = plt.figure(figsize=(3.4 * columns, height))

    for index, object_name in enumerate(objects):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        points_by_channel = {
            channel: channel_points(memory, object_name, channel)
            for channel in channel_styles
        }
        present = [points for points in points_by_channel.values() if len(points)]
        if not present:
            raise ValueError(f"{object_name!r} has none of {list(channel_styles)}")

        for channel, style in channel_styles.items():
            points = points_by_channel[channel]
            if len(points):
                axis.scatter(
                    *points.T,
                    c=style["color"],
                    alpha=style["alpha"],
                    s=POINT_SIZE,
                    depthshade=False,
                    linewidths=0,
                )

        set_equal_limits(axis, np.concatenate(present))
        axis.view_init(elev=20, azim=20, vertical_axis="y")
        axis.set_axis_off()
        counts = {channel: len(points) for channel, points in points_by_channel.items()}
        total = sum(counts.values())
        counts_line = " · ".join(
            f"{short_channel(channel)} {count:,}" for channel, count in counts.items()
        )
        axis.set_title(
            f"{object_name}\n{total:,} total\n{counts_line}",
            fontsize=9,
        )

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            label=channel,
            markerfacecolor=style["color"],
            markeredgewidth=0,
            alpha=style["alpha"],
            markersize=6,
        )
        for channel, style in channel_styles.items()
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
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.68 / height),
        ncols=len(legend),
    )
    figure.subplots_adjust(
        top=1 - 1.05 / height, bottom=0.01, wspace=0.05, hspace=0.12
    )
    return figure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--columns", type=positive_int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output figure path (default: ~/Desktop/<model_name>_lm<id>.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively instead of saving it.",
    )
    parser.add_argument("--dpi", type=positive_int, default=150)
    return parser.parse_args()


def main() -> None:
    """Load an LM's graph memory and save or show its channel-colored grid."""
    args = parse_args()
    checkpoint = resolve_checkpoint(args.model_path)
    memory = load_lm_memory(checkpoint, args.lm)
    objects = select_objects(memory, args.objects)
    channels = collect_channels(memory, objects)
    channel_styles = build_channel_styles(channels)

    print(
        f"Loaded {len(objects)} objects from LM_{args.lm} in {checkpoint} "
        f"(channels: {channels})"
    )
    figure = make_figure(
        memory,
        objects,
        channel_styles,
        columns=args.columns,
        checkpoint=checkpoint,
        lm_id=args.lm,
    )

    if args.show:
        plt.show()
        return

    output = args.output
    if output is None:
        output = Path(f"~/Desktop/{checkpoint_label(checkpoint)}_lm{args.lm}.png")
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
