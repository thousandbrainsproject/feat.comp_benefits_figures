# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Compare eval runs that differ in the patch module's face-on re-orientation.

The runs are expected to be paired: the same objects in the same predefined
rotations (one rotation per epoch), so each episode index means the same
trial in every run. Per episode, the script reads:

* from ``eval_stats.csv``: the parent LM's outcome and steps, the target
  object and its rotation;
* from the telemetry: the first episode step on which the sticker LM's most
  likely hypothesis was the correct logo, whether it was still correct at the
  end, and the step it reached its terminal state; the patch module's
  ``gsg`` block (face-on goals proposed and achieved, mean view angle and
  the share of steps above the generator's threshold); and the steps on
  which the agent was repositioned.

It prints one table per run and a paired table, writes them as CSV next to
the figure, and draws, against the rotation of each trial: the parent LM's
outcome, the steps until the sticker LM first held the correct logo, and the
face-on goals fired. Run from the repo root, e.g.::

    python -m analysis.scripts.compare_reorientation \
        reorient_cube_sweep_no_gsg reorient_cube_sweep_sm_gsg

The runs are directories or names under ``RESULTS_DIR``; the figure goes to
``~/tbp/projects/comp_benefits_figures/figures/`` unless ``--output`` says
otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from analysis.cli import run_directory
from analysis.scripts.visualize_view_angle import jump_steps
from analysis.telemetry import EpisodeTelemetry, available_episodes
from tbp.monty.frameworks.loggers.npz_handler import materialize

if TYPE_CHECKING:
    import os

    from matplotlib.axes import Axes

DEFAULT_FIGURE_DIR = Path("~/tbp/projects/comp_benefits_figures/figures").expanduser()
# The sticker each compositional object carries, by its name's logo tag.
LOGOS = {"tbp": "021_logo_tbp", "numenta": "022_logo_numenta"}
CORRECT = ("correct", "correct_mlh")


def expected_logo(target: str) -> str | None:
    """The sticker LM's target object for a compositional object.

    Returns:
        The logo graph id, or None when the name carries no logo tag.
    """
    for tag, logo in LOGOS.items():
        if f"_{tag}" in target:
            return logo
    return None


def rotation_label(euler: Any) -> str:
    """A compact label for an eval row's ``primary_target_rotation_euler``.

    Returns:
        The angles as ``x/y/z``.
    """
    angles = np.asarray(_parse(euler), dtype=float)
    return "/".join(f"{a:g}" for a in angles)


def rotation_magnitude(euler: Any) -> float:
    """How far the object was turned from its canonical pose, in degrees.

    Returns:
        The rotation's angle.
    """
    angles = np.asarray(_parse(euler), dtype=float)
    rotation = Rotation.from_euler("xyz", angles, degrees=True)
    return float(np.degrees(rotation.magnitude()))


def _parse(value: Any) -> np.ndarray:
    # eval_stats stores the angles as numpy prints them: "[0 60 0]".
    if isinstance(value, str):
        return np.array(value.strip("[]()").replace(",", " ").split(), dtype=float)
    return np.asarray(value, dtype=float)


def gsg_threshold(run_dir: Path, patch_module: str) -> float | None:
    """The face-on generator's ``max_view_angle`` from the run's saved config.

    Returns:
        The threshold in degrees, or None when the run had no generator.
    """
    config = run_dir / "config.yaml"
    if not config.is_file():
        return None
    cfg = yaml.safe_load(config.read_text())
    index = int(patch_module.removeprefix("SM_"))
    try:
        gsg = cfg["experiment"]["config"]["monty_config"]["sensor_modules"][
            f"sensor_module_{index}"
        ]["gsg"]
    except (KeyError, TypeError):
        return None
    if not gsg:
        return None
    return float(gsg.get("max_view_angle", 45.0))


def sticker_lm_record(ep: EpisodeTelemetry, lm: str, logo: str | None) -> dict:
    """When the sticker LM held the correct logo, from its MLH per processed step.

    Returns:
        ``first_correct_mlh_step`` (episode step, NaN if never), ``final_mlh``,
        ``final_mlh_correct``, and ``ts_step`` (the processed step the module's
        terminal state was reached on, NaN if none).
    """
    block = ep.blocks[lm]
    mlh = materialize(block["current_mlh"])
    ids = [m["graph_id"] for m in mlh]
    steps = ep.episode_steps(f"{lm}/current_mlh")
    hits = [i for i, g in enumerate(ids) if g == logo]
    ts = block.get("individual_ts_reached_at_step")
    return dict(
        first_correct_mlh_step=float(steps[hits[0]]) if hits else np.nan,
        final_mlh=ids[-1] if ids else None,
        final_mlh_correct=bool(ids) and ids[-1] == logo,
        ts_step=np.nan if ts is None else float(ts),
        ts_object=block.get("individual_ts_object"),
    )


def patch_module_record(ep: EpisodeTelemetry, sm: str, threshold: float | None) -> dict:
    """What the patch module's face-on generator saw and did this episode.

    Returns:
        ``goals``, ``goals_achieved``, ``mean_view_angle``,
        ``frac_steep`` (share of measured steps above the threshold; NaN
        without one), and ``jumps`` (agent repositionings of any origin).
    """
    jumps = len(jump_steps(ep))
    block = ep.blocks.get(sm, {})
    # A module without a face-on generator records an empty gsg block.
    gsg = materialize(block["gsg"]) if "gsg" in block else {}
    if "view_angle" not in gsg:
        return dict(
            goals=0,
            goals_achieved=0,
            mean_view_angle=np.nan,
            frac_steep=np.nan,
            jumps=jumps,
        )
    angles = np.asarray(gsg["view_angle"], dtype=float)
    measured = angles[np.isfinite(angles)]
    goals = gsg["goals"]
    return dict(
        goals=len(goals),
        goals_achieved=sum(bool(g["achieved"]) for g in goals),
        mean_view_angle=float(measured.mean()) if len(measured) else np.nan,
        frac_steep=(
            float(np.mean(measured > threshold))
            if threshold is not None and len(measured)
            else np.nan
        ),
        jumps=jumps,
    )


def episode_table(
    run_dir: os.PathLike,
    parent_lm: str = "LM_2",
    sticker_lm: str = "LM_1",
    patch_module: str = "SM_0",
) -> pd.DataFrame:
    """One row per episode of a run, with the outcome and policy measures.

    Args:
        run_dir: The run's output directory.
        parent_lm: The LM whose outcome counts (the compositional parent).
        sticker_lm: The LM that reads the sticker.
        patch_module: The sensor module carrying the face-on generator.

    Returns:
        The table, indexed by episode.
    """
    run_dir = Path(run_dir)
    stats = pd.read_csv(run_dir / "eval_stats.csv")
    parent = stats[stats.lm_id == parent_lm].reset_index(drop=True)
    threshold = gsg_threshold(run_dir, patch_module)
    rows = []
    episodes = available_episodes(run_dir)
    for episode, row in parent.iterrows():
        target = str(row.primary_target_object)
        logo = expected_logo(target)
        record = dict(
            episode=episode,
            object=target,
            rotation=rotation_label(row.primary_target_rotation_euler),
            rotation_deg=rotation_magnitude(row.primary_target_rotation_euler),
            outcome=row.primary_performance,
            correct=row.primary_performance in CORRECT,
            converged=row.primary_performance == "correct",
            steps=int(row.num_steps),
        )
        if episode in episodes:
            ep = EpisodeTelemetry.load(run_dir, episode)
            record.update(sticker_lm_record(ep, sticker_lm, logo))
            record.update(patch_module_record(ep, patch_module, threshold))
        rows.append(record)
    return pd.DataFrame(rows).set_index("episode")


def paired_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Side-by-side outcome, sticker step and goals of every run per trial.

    Returns:
        One row per episode with the trial columns from the first run and
        the per-run measures suffixed by the run label.
    """
    first = next(iter(tables.values()))
    out = first[["object", "rotation", "rotation_deg"]].copy()
    for label, table in tables.items():
        out[f"outcome [{label}]"] = table["outcome"]
        out[f"steps [{label}]"] = table["steps"]
        out[f"sticker step [{label}]"] = table.get("first_correct_mlh_step")
        out[f"goals [{label}]"] = table.get("goals")
    return out


def summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-run totals over the episodes.

    Returns:
        A table with one row per run.
    """
    rows = {}
    for label, t in tables.items():
        sticker_step = t.get("first_correct_mlh_step", pd.Series(dtype=float))
        rows[label] = dict(
            episodes=len(t),
            parent_correct=int(t.correct.sum()),
            parent_converged=int(t.converged.sum()),
            mean_steps=float(t.steps.mean()),
            sticker_final_correct=int(t.get("final_mlh_correct", pd.Series()).sum()),
            sticker_ever_correct=int(sticker_step.notna().sum()),
            median_sticker_step=float(sticker_step.median()),
            goals=int(t.get("goals", pd.Series()).sum()),
            goals_achieved=int(t.get("goals_achieved", pd.Series()).sum()),
            jumps=int(t.get("jumps", pd.Series()).sum()),
        )
    return pd.DataFrame(rows).T


def _draw_outcomes(ax: Axes, tables: dict[str, pd.DataFrame]) -> None:
    labels = list(tables)
    first = tables[labels[0]]
    trials = [f"{o.split('_')[-1]} {r}" for o, r in zip(first.object, first.rotation)]
    x = np.arange(len(trials))
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        t = tables[label]
        score = np.where(t.converged, 1.0, np.where(t.correct, 0.5, 0.0))
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, score, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(trials, rotation=90, fontsize=7)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["wrong", "correct MLH", "converged"])
    ax.set_title("Parent LM outcome per trial (object, rotation x/y/z)")
    ax.legend(fontsize=8)


# Horizontal offset between the runs' markers, in degrees, so they do not
# hide each other at the same rotation.
JITTER = 1.0


def _jittered(t: pd.DataFrame, i: int, n: int) -> np.ndarray:
    return t.rotation_deg.to_numpy(dtype=float) + (i - (n - 1) / 2) * JITTER


def _draw_sticker_steps(ax: Axes, tables: dict[str, pd.DataFrame]) -> None:
    for i, (label, t) in enumerate(tables.items()):
        if "first_correct_mlh_step" not in t:
            continue
        x = _jittered(t, i, len(tables))
        y = t.first_correct_mlh_step.to_numpy(dtype=float)
        never = np.isnan(y)
        ax.plot(x, y, "o", label=label, alpha=0.8)
        if never.any():
            top = np.nanmax(y) * 1.05 if np.isfinite(y).any() else 1.0
            ax.plot(
                x[never],
                np.full(never.sum(), top),
                "x",
                color=ax.lines[-1].get_color(),
            )
    ax.set_xlabel("rotation from canonical pose (deg)")
    ax.set_ylabel("episode step")
    ax.set_title("First step the sticker LM's MLH was the correct logo (x: never)")
    ax.legend(fontsize=8)


def _draw_goals(ax: Axes, tables: dict[str, pd.DataFrame]) -> None:
    for i, (label, t) in enumerate(tables.items()):
        if "goals" not in t:
            continue
        x = _jittered(t, i, len(tables))
        ax.plot(x, t.goals, "o", label=f"{label}: proposed", alpha=0.8)
        ax.plot(
            x,
            t.goals_achieved,
            "+",
            color=ax.lines[-1].get_color(),
            label=f"{label}: achieved",
        )
    ax.set_xlabel("rotation from canonical pose (deg)")
    ax.set_ylabel("face-on goals")
    ax.set_title("Face-on goals per episode")
    ax.legend(fontsize=8)


def create_figure(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    """The three comparison panels.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    _draw_outcomes(axes[0], tables)
    _draw_sticker_steps(axes[1], tables)
    _draw_goals(axes[2], tables)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=run_directory)
    parser.add_argument("--parent-lm", default="LM_2")
    parser.add_argument("--sticker-lm", default="LM_1")
    parser.add_argument("--patch-module", default="SM_0")
    parser.add_argument("--output", type=Path, default=None, help="Figure path.")
    args = parser.parse_args()

    tables = {
        run.name: episode_table(run, args.parent_lm, args.sticker_lm, args.patch_module)
        for run in args.runs
    }
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for label, table in tables.items():
        print(f"\n== {label}")
        print(table.to_string())
    print("\n== summary")
    print(summary(tables).to_string())

    output = args.output or DEFAULT_FIGURE_DIR / (
        "reorientation_" + "_vs_".join(t for t in tables) + ".png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    paired_table(tables).to_csv(output.with_suffix(".csv"))
    summary(tables).to_csv(output.with_name(output.stem + "_summary.csv"))
    fig = create_figure(tables)
    fig.savefig(output, dpi=120)
    print(f"\nFigure: {output}\nTables: {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
