# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Draw every object graph a pretrained model holds, one figure per LM.

Each learning module gets a grid of 3D scatters, one per learned object.
Nodes learned from a sensor patch are drawn at their stored color (or
colored by a chosen feature); nodes learned from a lower-level LM are drawn
as larger markers labelled with the child object they stand for. Surface
normals can be drawn for every k-th node.

Run from the repo root, e.g. ``python -m analysis.scripts.visualize_models
--normals 20`` for the current chain's newest trained stage (see
``analysis.models.CURRENT_CHAIN``), or name a model: a ``model.pt`` path, a
run directory, or a run name under ``MODELS_DIR``. Figures go to
``<run_dir>/visualizations/``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from analysis.models import (
    LearnedObject,
    child_labels,
    default_model,
    load_learned_objects,
    resolve_model_path,
)
from analysis.panels import OKABE_ITO, add_colorbar, equal_aspect_bounds_3d, style_3d

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from mpl_toolkits.mplot3d import Axes3D

# Objects per figure row.
N_COLUMNS = 4
# Color nodes by their stored color rather than by a feature.
STORED_COLOR = "color"


def create_model_figures(
    model: str | os.PathLike | None = None,
    lms: Sequence[str] | None = None,
    objects: Sequence[str] | None = None,
    color_by: str = STORED_COLOR,
    normals_every: int = 0,
    elev: float = 30.0,
    azim: float = -60.0,
    out_dir: os.PathLike | None = None,
) -> list[Path]:
    """Draw a pretrained model's object graphs, one figure per learning module.

    Args:
        model: See :func:`analysis.models.resolve_model_path`; the newest
            trained stage of the current chain when None.
        lms: The learning modules to draw (``"LM_0"`` ...); all when None.
        objects: The object ids to draw; all when None.
        color_by: ``"color"`` for the nodes' stored color, else the name of a
            stored feature to color sensor-learned nodes by (a vector
            feature's first component).
        normals_every: Draw the surface normal of every k-th node; 0 for none.
        elev: Elevation of the 3D view, in degrees.
        azim: Azimuth of the 3D view, in degrees.
        out_dir: Where to save the figures; ``<run_dir>/visualizations`` by
            default.

    Returns:
        The saved figure paths, one per learning module drawn.
    """
    model_path = default_model() if model is None else resolve_model_path(model)
    run_dir = model_path.parent.parent
    out_dir = Path(out_dir) if out_dir else run_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    learned = load_learned_objects(model_path)

    saved = []
    for lm in lms or learned:
        by_object = learned[lm]
        if objects:
            by_object = {name: by_object[name] for name in objects if name in by_object}
        if not by_object:
            continue
        # Sensor-learned nodes of every object share one color scale, and
        # every child-object label gets one color across the figure.
        norm = _shared_norm(by_object, color_by)
        label_colors = _label_colors(by_object, learned)

        n = len(by_object)
        ncols = min(N_COLUMNS, n)
        nrows = math.ceil(n / ncols)
        fig = plt.figure(figsize=(4.8 * ncols, 4.6 * nrows))
        fig.suptitle(f"{run_dir.name} -- {lm}", fontsize=13, fontweight="bold")
        axes = []
        for index, channels in enumerate(by_object.values()):
            ax = fig.add_subplot(nrows, ncols, index + 1, projection="3d")
            axes.append(ax)
            draw_learned_object(
                ax, channels, learned, label_colors, color_by, norm, normals_every
            )
            ax.view_init(elev=elev, azim=azim)
        if norm is not None:
            mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
            add_colorbar(mappable, axes[-1], color_by)
        path = out_dir / f"learned_objects_{lm}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")
        saved.append(path)
    return saved


def draw_learned_object(
    ax: Axes3D,
    channels: dict[str, LearnedObject],
    learned: dict[str, dict[str, dict[str, LearnedObject]]],
    label_colors: dict[str, str],
    color_by: str,
    norm: Normalize | None,
    normals_every: int,
) -> None:
    """Draw one object's graph, every input channel on the same axes.

    Args:
        ax: The 3D axes to draw on.
        channels: The object's graphs, one per input channel.
        learned: Everything the model learned, to name child objects by.
        label_colors: The color of each child-object label
            (``"<channel>: <child>"``), shared across the figure.
        color_by: See :func:`create_model_figures`.
        norm: The shared color scale when coloring by a feature.
        normals_every: Draw the normal of every k-th node; 0 for none.
    """
    bounds = equal_aspect_bounds_3d([c.points.locations for c in channels.values()])
    span = max(hi - lo for lo, hi in bounds)
    labels = child_labels(channels, learned)
    for channel, obj in channels.items():
        xyz = obj.points.locations
        if channel in labels:
            # Nodes are child objects the lower LM recognized; name them.
            node_labels = np.array(labels[channel])
            for label in sorted(set(node_labels)):
                ax.scatter(
                    *xyz[node_labels == label].T,
                    s=40,
                    color=label_colors[label],
                    edgecolors="black",
                    linewidths=0.5,
                    label=label,
                    depthshade=False,
                )
        elif norm is None:
            ax.scatter(*xyz.T, s=4, c=obj.colors, depthshade=False)
        else:
            values = _feature_column(obj, color_by)
            ax.scatter(
                *xyz.T, s=4, c=values, cmap="viridis", norm=norm, depthshade=False
            )
        if normals_every and "pose_vectors" in obj.points.features:
            every = slice(None, None, normals_every)
            ax.quiver(
                *xyz[every].T,
                *obj.normals[every].T,
                length=0.04 * span,
                color="black",
                linewidth=0.5,
                arrow_length_ratio=0.0,
            )
    total = sum(len(c) for c in channels.values())
    first = next(iter(channels.values()))
    style_3d(ax, *bounds, title=f"{first.object_id} ({total} nodes)")
    if labels:
        ax.legend(fontsize=7, loc="upper left")


def _label_colors(
    by_object: dict[str, dict[str, LearnedObject]],
    learned: dict[str, dict[str, dict[str, LearnedObject]]],
) -> dict[str, str]:
    labels = sorted(
        {
            label
            for channels in by_object.values()
            for node_labels in child_labels(channels, learned).values()
            for label in node_labels
        }
    )
    palette = OKABE_ITO * (1 + len(labels) // len(OKABE_ITO))
    return dict(zip(labels, palette))


def _feature_column(obj: LearnedObject, feature: str) -> np.ndarray:
    values = obj.points[feature]
    return values[:, 0] if values.ndim == 2 else values


def _shared_norm(
    by_object: dict[str, dict[str, LearnedObject]], color_by: str
) -> Normalize | None:
    if color_by == STORED_COLOR:
        return None
    columns = [
        _feature_column(obj, color_by)
        for channels in by_object.values()
        for obj in channels.values()
        if color_by in obj.points.features
    ]
    if not columns:
        raise KeyError(f"no learned object stores a {color_by!r} feature")
    values = np.concatenate(columns)
    return Normalize(vmin=float(values.min()), vmax=float(values.max()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "model",
        nargs="?",
        help="model.pt, run directory, or run name under MODELS_DIR "
        "(default: the current chain's newest trained stage)",
    )
    parser.add_argument(
        "--lm", nargs="+", help="learning modules to draw (default: all)"
    )
    parser.add_argument(
        "--objects", nargs="+", help="object ids to draw (default: all)"
    )
    parser.add_argument(
        "--color-by",
        default=STORED_COLOR,
        help="'color' for the stored color, or a feature name (default: color)",
    )
    parser.add_argument(
        "--normals", type=int, default=0, metavar="K", help="draw every K-th normal"
    )
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument(
        "--out", help="output directory (default: <run_dir>/visualizations)"
    )
    args = parser.parse_args()
    create_model_figures(
        args.model,
        lms=args.lm,
        objects=args.objects,
        color_by=args.color_by,
        normals_every=args.normals,
        elev=args.elev,
        azim=args.azim,
        out_dir=args.out,
    )
