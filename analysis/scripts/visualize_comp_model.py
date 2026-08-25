# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# ruff: noqa: DOC201,DOC501
"""Plot LM_2 object models with points colored by input channel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

CHANNEL_STYLES = {
    "patch_2": {"color": "#B3B3B3", "alpha": 0.3},
    "learning_module_0": {"color": "#00A0DF", "alpha": 1.0},
    "learning_module_1": {"color": "#F737BD", "alpha": 1.0},
}
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


def load_lm2_memory(checkpoint: Path) -> dict:
    """Load LM_2 graph memory from a checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {checkpoint}")
    lm_dict = state.get("lm_dict")
    if not isinstance(lm_dict, dict):
        raise KeyError(f"Checkpoint has no lm_dict: {checkpoint}")

    lm_key = 2 if 2 in lm_dict else "2"
    if lm_key not in lm_dict:
        raise KeyError(f"LM_2 not found; available: {list(lm_dict)}")
    memory = lm_dict[lm_key].get("graph_memory", {})
    if not isinstance(memory, dict) or not memory:
        raise ValueError("LM_2 has no graph memory")
    return memory


def select_objects(memory: dict, requested: list[str] | None) -> list[str]:
    """Return requested object names, or every stored object."""
    if requested is None:
        return list(memory)
    missing = [name for name in requested if name not in memory]
    if missing:
        raise ValueError(f"Objects not found: {missing}; available: {list(memory)}")
    return requested


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


def checkpoint_label(checkpoint: Path) -> str:
    """Return a useful title for any accepted checkpoint path shape."""
    if checkpoint.parent.name == "pretrained":
        return checkpoint.parent.parent.name
    return checkpoint.stem


def make_figure(
    memory: dict,
    objects: list[str],
    columns: int,
    checkpoint: Path,
) -> plt.Figure:
    """Create one overlaid LM_2 channel plot per object."""
    columns = min(columns, len(objects))
    rows = math.ceil(len(objects) / columns)
    figure = plt.figure(
        figsize=(3.4 * columns, 3.6 * rows + 1.4),
    )

    for index, object_name in enumerate(objects):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        points_by_channel = {
            channel: channel_points(memory, object_name, channel)
            for channel in CHANNEL_STYLES
        }
        present = [points for points in points_by_channel.values() if len(points)]
        if not present:
            raise ValueError(f"{object_name!r} has none of {list(CHANNEL_STYLES)}")

        for channel, style in CHANNEL_STYLES.items():
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
        axis.set_title(
            f"{object_name}\n{total:,} total\n"
            f"patch_2 {counts['patch_2']:,} · "
            f"LM_0 {counts['learning_module_0']:,} · "
            f"LM_1 {counts['learning_module_1']:,}",
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
        for channel, style in CHANNEL_STYLES.items()
    ]
    figure.suptitle(
        f"{checkpoint_label(checkpoint)}\nLM_2 channels", fontsize=14, y=0.99
    )
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncols=len(legend),
    )
    figure.subplots_adjust(top=0.89, bottom=0.01, wspace=0.05, hspace=0.12)
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
        "--objects", nargs="+", help="Object names to plot (default: all)."
    )
    parser.add_argument("--columns", type=positive_int, default=5)
    parser.add_argument("--output", type=Path, help="Save instead of showing the plot.")
    parser.add_argument("--dpi", type=positive_int, default=150)
    return parser.parse_args()


def main() -> None:
    """Load LM_2 and show or save its channel-colored object grid."""
    args = parse_args()
    checkpoint = resolve_checkpoint(args.model_path)
    memory = load_lm2_memory(checkpoint)
    objects = select_objects(memory, args.objects)

    ignored = sorted(
        {
            channel
            for object_name in objects
            for channel in memory[object_name]
            if channel not in CHANNEL_STYLES
        }
    )
    if ignored:
        print(f"Ignoring unexpected LM_2 channels: {ignored}")
    print(f"Loaded {len(objects)} objects from LM_2 in {checkpoint}")
    figure = make_figure(memory, objects, args.columns, checkpoint)

    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=args.dpi)
        plt.close(figure)
        print(f"Saved {output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
