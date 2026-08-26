# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Inspect a pretrained model's object graphs interactively, with vedo.

One learning module at a time: a window with one renderer per learned
object, orbitable with the mouse. Nodes learned from a sensor patch show
their stored color (or a chosen feature); nodes learned from a lower-level
LM are spheres colored per child object with a label. Press ``n`` to toggle
surface normals, ``q`` to close.

Needs the ``viz`` extra (``uv sync --extra viz``). Run from the repo root,
e.g. ``python -m analysis.scripts.inspect_models --lm LM_0`` for the current
chain's newest trained stage (see ``analysis.models.CURRENT_CHAIN``), or
name a model: a ``model.pt`` path, a run directory, or a run name under
``MODELS_DIR``. ``--screenshot`` renders offscreen to a file instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import vedo

from analysis.models import (
    LearnedObject,
    child_labels,
    default_model,
    load_learned_objects,
    resolve_model_path,
)
from analysis.panels import OKABE_ITO

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

# Color nodes by their stored color rather than by a feature.
STORED_COLOR = "color"


def inspect_model(
    model: str | os.PathLike | None = None,
    lm: str | None = None,
    objects: Sequence[str] | None = None,
    color_by: str = STORED_COLOR,
    normals_every: int = 10,
    point_size: int = 6,
    screenshot: os.PathLike | None = None,
) -> None:
    """Open one learning module's object graphs in a vedo window.

    Args:
        model: See :func:`analysis.models.resolve_model_path`; the newest
            trained stage of the current chain when None.
        lm: The learning module to show (``"LM_0"`` ...); the first one that
            learned anything when None.
        objects: The object ids to show; all when None.
        color_by: ``"color"`` for the nodes' stored color, else the name of a
            stored feature to color sensor-learned nodes by (a vector
            feature's first component).
        normals_every: Which nodes get a normal arrow when normals are shown
            (every k-th); 0 for none.
        point_size: Screen-space radius of the sensor-learned nodes.
        screenshot: Render offscreen and save to this path instead of
            opening a window.

    Raises:
        ValueError: If the model holds no objects for the requested module.
    """
    model_path = default_model() if model is None else resolve_model_path(model)
    learned = load_learned_objects(model_path)
    if lm is None:
        lm = next((name for name, objs in learned.items() if objs), None)
    by_object = learned.get(lm or "", {})
    if objects:
        by_object = {name: by_object[name] for name in objects if name in by_object}
    if not by_object:
        raise ValueError(f"{model_path} holds no objects for {lm!r}")

    plotter = vedo.Plotter(
        N=len(by_object),
        title=f"{model_path.parent.parent.name} -- {lm}",
        axes=1,
        offscreen=screenshot is not None,
    )
    normal_arrows: list[vedo.Arrows] = []
    # Every child-object label gets one color across the window.
    labels = sorted(
        {
            label
            for channels in by_object.values()
            for node_labels in child_labels(channels, learned).values()
            for label in node_labels
        }
    )
    label_colors = dict(zip(labels, OKABE_ITO * (1 + len(labels) // len(OKABE_ITO))))
    for at, (object_id, channels) in enumerate(by_object.items()):
        actors, arrows = _object_actors(
            channels, learned, label_colors, color_by, normals_every, point_size
        )
        normal_arrows.extend(arrows)
        total = sum(len(c) for c in channels.values())
        plotter.at(at).show(*actors, f"{object_id} ({total} nodes)", resetcam=True)

    def toggle_normals(event: vedo.Event) -> None:
        if event.keypress == "n":
            for arrow in normal_arrows:
                arrow.toggle()
            plotter.render()

    plotter.add_callback("key press", toggle_normals)
    if screenshot is not None:
        plotter.screenshot(str(screenshot))
        print(f"Saved {screenshot}")
    else:
        plotter.interactive()
    plotter.close()


def _object_actors(
    channels: dict[str, LearnedObject],
    learned: dict[str, dict[str, dict[str, LearnedObject]]],
    label_colors: dict[str, str],
    color_by: str,
    normals_every: int,
    point_size: int,
) -> tuple[list, list[vedo.Arrows]]:
    """Build one object's vedo actors: nodes per channel, plus hidden normals.

    Returns:
        The actors to show, and the normal-arrow actors among them (hidden
        until toggled).
    """
    actors: list = []
    arrows: list[vedo.Arrows] = []
    child_spheres: list[vedo.Spheres] = []
    extent = np.ptp(
        np.vstack([c.points.locations for c in channels.values()]), axis=0
    ).max()
    labels = child_labels(channels, learned)
    for channel, obj in channels.items():
        xyz = obj.points.locations
        if channel in labels:
            node_labels = np.array(labels[channel])
            for label in sorted(set(node_labels)):
                spheres = vedo.Spheres(
                    xyz[node_labels == label], r=0.012 * extent, c=label_colors[label]
                )
                spheres.legend(label)
                child_spheres.append(spheres)
        else:
            points = vedo.Points(xyz, r=point_size)
            if color_by == STORED_COLOR:
                points.pointcolors = (obj.colors * 255).astype(np.uint8)
            else:
                values = obj.points[color_by]
                points.cmap("viridis", values[:, 0] if values.ndim == 2 else values)
                points.add_scalarbar(title=color_by)
            actors.append(points)
        if normals_every and "pose_vectors" in obj.points.features:
            every = slice(None, None, normals_every)
            start = xyz[every]
            end = start + 0.05 * extent * obj.normals[every]
            arrow = vedo.Arrows(start, end, c="black", s=0.3).off()
            arrows.append(arrow)
            actors.append(arrow)
    if child_spheres:
        actors.extend(child_spheres)
        actors.append(vedo.LegendBox(child_spheres, width=0.3, pos="bottom-left"))
    return actors, arrows


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
        "--lm", help="learning module to show (default: first with objects)"
    )
    parser.add_argument(
        "--objects", nargs="+", help="object ids to show (default: all)"
    )
    parser.add_argument(
        "--color-by",
        default=STORED_COLOR,
        help="'color' for the stored color, or a feature name (default: color)",
    )
    parser.add_argument(
        "--normals",
        type=int,
        default=10,
        metavar="K",
        help="every K-th normal, toggled with 'n'",
    )
    parser.add_argument("--point-size", type=int, default=6)
    parser.add_argument(
        "--screenshot", help="render offscreen to this file instead of a window"
    )
    args = parser.parse_args()
    inspect_model(
        args.model,
        lm=args.lm,
        objects=args.objects,
        color_by=args.color_by,
        normals_every=args.normals,
        point_size=args.point_size,
        screenshot=args.screenshot,
    )
