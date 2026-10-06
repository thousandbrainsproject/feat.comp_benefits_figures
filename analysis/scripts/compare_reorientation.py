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

It prints one table per run, a paired table, a summary per run and a table
aggregated per object rotation, writes them as CSV next to the figures, and
draws two figures: per trial, the parent LM's outcome, the steps until the
sticker LM first held the correct logo and the face-on goals fired; and per
rotation, the parent LM's correct and converged counts, mean steps and
goals, one bar group per run; and mean steps against tilt about Y, about X,
and pooled over both axes, vertically aligned by angle; and correct versus
incorrect trial counts per angle, pooled over axes. Run from the repo root, e.g.::

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
    from collections.abc import Sequence

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


def summary(
    tables: dict[str, pd.DataFrame], baseline: Sequence[str] = ()
) -> pd.DataFrame:
    """Per-run totals over the episodes.

    Args:
        tables: Episode table per run label.
        baseline: Rotation labels (``x/y/z``) of baseline trials; they count
            toward the outcomes but are left out of ``mean_steps``.

    Returns:
        A table with one row per run.
    """
    rows = {}
    for label, t in tables.items():
        sticker_step = t.get("first_correct_mlh_step", pd.Series(dtype=float))
        tilted = t[~t.rotation.isin(baseline)]
        rows[label] = dict(
            episodes=len(t),
            parent_correct=int(t.correct.sum()),
            parent_converged=int(t.converged.sum()),
            mean_steps=float(tilted.steps.mean()),
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


def by_rotation(
    tables: dict[str, pd.DataFrame], baseline: Sequence[str] = ()
) -> pd.DataFrame:
    """The runs' measures aggregated over the trials at each object rotation.

    Args:
        tables: Episode table per run label.
        baseline: Rotation labels to list first, whatever their position.

    Returns:
        One row per rotation, baseline first then in order of appearance,
        with ``n`` trials and,
        per run, ``correct`` and ``converged`` counts, mean ``steps``,
        ``sticker_final_correct`` and ``goals``, the columns named
        ``measure [run]``.
    """
    first = next(iter(tables.values()))
    seen = list(dict.fromkeys(first.rotation))
    order = [r for r in baseline if r in seen] + [r for r in seen if r not in baseline]
    out = pd.DataFrame(
        {"n": first.groupby("rotation").size().loc[order]}, index=order
    ).rename_axis("rotation")
    for label, t in tables.items():
        g = t.groupby("rotation")
        out[f"correct [{label}]"] = g["correct"].sum().loc[order]
        out[f"converged [{label}]"] = g["converged"].sum().loc[order]
        out[f"steps [{label}]"] = g["steps"].mean().loc[order].round(1)
        if "final_mlh_correct" in t:
            out[f"sticker_final_correct [{label}]"] = (
                g["final_mlh_correct"].sum().loc[order]
            )
        if "goals" in t:
            out[f"goals [{label}]"] = g["goals"].sum().loc[order]
    return out


def create_rotation_figure(
    tables: dict[str, pd.DataFrame],
    parent_lm: str = "LM_2",
    baseline: Sequence[str] = (),
) -> plt.Figure:
    """Outcome, steps and goals per object rotation, one bar group per run.

    Args:
        tables: Episode table per run label.
        parent_lm: The module the outcomes describe, for the title.
        baseline: Rotation labels drawn first.

    Returns:
        The figure.
    """
    table = by_rotation(tables, baseline)
    labels = list(tables)
    x = np.arange(len(table))
    width = 0.8 / len(labels)
    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(table)), 9), sharex=True)
    for i, label in enumerate(labels):
        xs = x + (i - (len(labels) - 1) / 2) * width
        converged = table[f"converged [{label}]"].to_numpy(dtype=float)
        correct = table[f"correct [{label}]"].to_numpy(dtype=float)
        bars = axes[0].bar(xs, converged, width, label=label)
        axes[0].bar(
            xs,
            correct - converged,
            width,
            bottom=converged,
            color=bars[0].get_facecolor(),
            alpha=0.4,
        )
        axes[1].bar(xs, table[f"steps [{label}]"], width, label=label)
        if f"goals [{label}]" in table:
            axes[2].bar(xs, table[f"goals [{label}]"], width, label=label)
    n = int(table.n.max())
    axes[0].set_ylabel(f"trials correct (of {n})\nsolid: converged, light: MLH only")
    axes[0].set_yticks(range(n + 1))
    axes[0].set_title(f"{parent_lm} performance per object rotation")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("mean matching steps")
    axes[2].set_ylabel("face-on goals")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(table.index)
    axes[2].set_xlabel("object rotation x/y/z (deg)")
    fig.tight_layout()
    return fig


def tilt_axis(rotation: str) -> tuple[str, float] | None:
    """Split a single-axis rotation label into its axis and angle.

    Returns:
        ``("x", 45.0)`` for ``"45/0/0"``, ``("y", 45.0)`` for ``"0/45/0"``,
        and so on; None for the identity or a rotation about several axes.
    """
    angles = [float(a) for a in rotation.split("/")]
    nonzero = [(axis, a) for axis, a in zip("xyz", angles) if a != 0]
    return nonzero[0] if len(nonzero) == 1 else None


def steps_by_axis(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean matching steps per single-axis tilt, per run, plus the axes pooled.

    Returns:
        Columns ``axis`` (``x``, ``y`` or ``pooled``), ``tilt`` in degrees,
        and one mean-steps column per run label. Multi-axis rotations and
        the identity are left out.
    """
    frames = []
    for label, t in tables.items():
        parsed = t.rotation.map(tilt_axis)
        keep = parsed.notna()
        rows = pd.DataFrame(
            {
                "axis": [p[0] for p in parsed[keep]],
                "tilt": [p[1] for p in parsed[keep]],
                label: t.steps[keep].to_numpy(dtype=float),
            }
        )
        pooled = rows.assign(axis="pooled")
        frames.append(pd.concat([rows, pooled]).groupby(["axis", "tilt"])[label].mean())
    return pd.concat(frames, axis=1).reset_index()


def create_steps_by_axis_figure(
    tables: dict[str, pd.DataFrame], parent_lm: str = "LM_2"
) -> plt.Figure:
    """Mean steps against tilt: about Y, about X, and both axes pooled.

    The rows share the tilt axis so the same angle lines up vertically.

    Returns:
        The figure.
    """
    table = steps_by_axis(tables)
    labels = list(tables)
    tilts = sorted(table.tilt.unique())
    x = np.arange(len(tilts))
    width = 0.8 / len(labels)
    rows = [
        ("y", "tilt about Y"),
        ("x", "tilt about X"),
        ("pooled", "both axes pooled"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(max(6, 1.1 * len(tilts)), 9), sharex=True)
    for ax, (axis, title) in zip(axes, rows):
        part = table[table.axis == axis].set_index("tilt").reindex(tilts)
        for i, label in enumerate(labels):
            ax.bar(
                x + (i - (len(labels) - 1) / 2) * width, part[label], width, label=label
            )
        ax.set_title(title)
        ax.set_ylabel("mean matching steps")
    axes[0].legend(fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{t:g}" for t in tilts])
    axes[-1].set_xlabel("degrees off the canonical pose")
    fig.suptitle(f"{parent_lm} matching steps by tilt")
    fig.tight_layout()
    return fig


def accuracy_by_tilt(
    tables: dict[str, pd.DataFrame], baseline: Sequence[str] = ()
) -> pd.DataFrame:
    """Correct and incorrect trial counts per angle from the canonical pose.

    Trials at the same angle are pooled whatever the axis of rotation.

    Args:
        tables: Episode table per run label.
        baseline: Rotation labels (``x/y/z``) to leave out.

    Returns:
        Columns ``tilt``, and per run ``correct [run]`` (converged or correct
        MLH) and ``incorrect [run]``.
    """
    first = next(iter(tables.values()))
    first = first[~first.rotation.isin(baseline)]
    tilts = sorted(first.rotation_deg.round(1).unique())
    out = pd.DataFrame({"tilt": tilts})
    for label, table in tables.items():
        t = table[~table.rotation.isin(baseline)]
        g = t.groupby(t.rotation_deg.round(1))
        out[f"correct [{label}]"] = g["correct"].sum().reindex(tilts).to_numpy()
        out[f"incorrect [{label}]"] = (
            (g["correct"].size() - g["correct"].sum()).reindex(tilts).to_numpy()
        )
    return out


def create_accuracy_by_tilt_figure(
    tables: dict[str, pd.DataFrame],
    parent_lm: str = "LM_2",
    baseline: Sequence[str] = (),
) -> plt.Figure:
    """Stacked bars per angle: correct trials in the run's color, wrong in black.

    Args:
        tables: Episode table per run label.
        parent_lm: The module the outcomes describe, for the title.
        baseline: Rotation labels to leave out.

    Returns:
        The figure.
    """
    table = accuracy_by_tilt(tables, baseline)
    labels = list(tables)
    x = np.arange(len(table))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(table)), 4.5))
    for i, label in enumerate(labels):
        xs = x + (i - (len(labels) - 1) / 2) * width
        correct = table[f"correct [{label}]"]
        ax.bar(xs, correct, width, label=label)
        ax.bar(
            xs,
            table[f"incorrect [{label}]"],
            width,
            bottom=correct,
            color="black",
            label="Incorrect" if i == 0 else None,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}" for t in table.tilt])
    ax.set_xlabel("degrees off the canonical pose")
    ax.set_ylabel("trials")
    ax.set_title(f"{parent_lm}: correct (converged or MLH) vs incorrect, per tilt")
    # Every bar can reach the top, so the legend goes beside the axes.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig


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
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="display name per run, in order (default: the run directory names)",
    )
    parser.add_argument(
        "--baseline",
        nargs="*",
        default=(),
        metavar="X/Y/Z",
        help="rotations that are baselines: listed first, left out of mean steps",
    )
    args = parser.parse_args()

    labels = args.labels or [run.name for run in args.runs]
    if len(labels) != len(args.runs):
        parser.error("--labels needs one name per run")
    tables = {
        label: episode_table(run, args.parent_lm, args.sticker_lm, args.patch_module)
        for label, run in zip(labels, args.runs)
    }
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for label, table in tables.items():
        print(f"\n== {label}")
        print(table.to_string())
    print("\n== summary")
    print(summary(tables, args.baseline).to_string())
    print("\n== by rotation")
    print(by_rotation(tables, args.baseline).to_string())

    output = args.output or DEFAULT_FIGURE_DIR / (
        "reorientation_" + "_vs_".join(run.name for run in args.runs) + ".png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    paired_table(tables).to_csv(output.with_suffix(".csv"))
    summary(tables, args.baseline).to_csv(
        output.with_name(output.stem + "_summary.csv")
    )
    create_figure(tables).savefig(output, dpi=120)
    rotation_output = output.with_name(output.stem + "_by_rotation.png")
    by_rotation(tables, args.baseline).to_csv(rotation_output.with_suffix(".csv"))
    create_rotation_figure(tables, args.parent_lm, args.baseline).savefig(
        rotation_output, dpi=120
    )
    axis_output = output.with_name(output.stem + "_steps_by_axis.png")
    steps_by_axis(tables).to_csv(axis_output.with_suffix(".csv"), index=False)
    create_steps_by_axis_figure(tables, args.parent_lm).savefig(axis_output, dpi=120)
    accuracy_output = output.with_name(output.stem + "_accuracy_by_tilt.png")
    accuracy_by_tilt(tables, args.baseline).to_csv(
        accuracy_output.with_suffix(".csv"), index=False
    )
    create_accuracy_by_tilt_figure(tables, args.parent_lm, args.baseline).savefig(
        accuracy_output, dpi=120
    )
    print(
        f"\nFigures: {output}, {rotation_output}, {axis_output}, {accuracy_output}"
        f"\nTables: {output.with_suffix('.csv')}, "
        f"{output.with_name(output.stem + '_summary.csv')}, "
        f"{rotation_output.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
