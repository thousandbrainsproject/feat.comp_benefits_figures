# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Tile a run's first frame per episode, as a sanity check on positioning.

One panel per episode: the chosen sensor module's first recorded frame
(the view after the positioning procedures have run, before any policy
step), titled with the episode's target object and rotation. The module
needs ``save_raw_obs`` on. Run from the repo root, e.g.::

    python -m analysis.scripts.initial_views <run>

where the run is e.g. ``reorient_logo_sweep/reorient_logo_sweep_no_gsg``.

The run is a directory or a name under ``RESULTS_DIR``; the figure goes to
``<run_dir>/visualizations/initial_views_<module>.png`` unless ``--output``
says otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.cli import run_directory
from analysis.telemetry import EpisodeTelemetry, available_episodes
from tbp.monty.frameworks.loggers.npz_handler import materialize


def first_frame(ep: EpisodeTelemetry, sensor_module: str) -> np.ndarray:
    """The module's first recorded RGBA frame.

    Returns:
        The frame as an array.
    """
    return np.asarray(ep(f"{sensor_module}/raw_observations/0/rgba"))


def episode_title(ep: EpisodeTelemetry, episode: int) -> str:
    """A short title: episode number, object, and rotation.

    Returns:
        The title.
    """
    target = materialize(ep.blocks["target"])
    rotation = "/".join(
        f"{a:g}" for a in np.asarray(target["primary_target_rotation_euler"])
    )
    name = str(target["primary_target_object"]).split("_", 1)[-1]
    return f"ep {episode}: {name} @ {rotation}"


def create_figure(
    run_dir: Path, sensor_module: str = "SM_3", columns: int = 6
) -> plt.Figure:
    """Tile the first frames of every episode in the run.

    Returns:
        The figure.
    """
    episodes = available_episodes(run_dir)
    rows = int(np.ceil(len(episodes) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.4 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, episode in zip(np.ravel(axes), episodes):
        ep = EpisodeTelemetry.load(run_dir, episode)
        ax.imshow(first_frame(ep, sensor_module))
        ax.set_title(episode_title(ep, episode), fontsize=8)
    fig.suptitle(f"{run_dir.name}: first {sensor_module} frame per episode")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=run_directory)
    parser.add_argument("--sensor-module", default="SM_3")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (
        args.run / "visualizations" / f"initial_views_{args.sensor_module}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    create_figure(args.run, args.sensor_module, args.columns).savefig(output, dpi=110)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
