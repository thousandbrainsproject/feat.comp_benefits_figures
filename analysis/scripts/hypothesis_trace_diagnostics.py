# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Diagnose where evidence distinguishing an object arrives during post-training.

Reads the per-step hypothesis trace histories saved by HypothesisTracer (with
history_dir set), e.g. by the
post_training_hotspots_objects_with_stickers_comp_models_mujoco experiment, and for
each episode relates the evidence differential to the inputs of each step.

The differential used here is anchored on the true target object rather than the
most likely hypothesis (MLH): the target's evidence gain minus the largest gain of
any other object. Steps are split by whether LM_1 passed a logo object ID to the LM,
and by whether the MLH location lay within --logo-radius of a logo node on the MLH
object's model (only meaningful once the MLH pose is right).

Usage:
    python analysis/scripts/hypothesis_trace_diagnostics.py [RUN_DIR] [options]

RUN_DIR is the experiment's output directory, containing hypothesis_traces/ and
0/model.pt. Defaults to the output of the post-training experiment.

Options:
    --lm ID             Learning module whose traces to analyze (default: 2).
    --logo-radius R     Distance (m) from a logo node counted as "on the logo"
                        (default: 0.01).
    --steps-per-view N  Steps per view of the exploration policy, used to mark
                        view boundaries (default: 50).
    --output PATH       Where to save the figure
                        (default: ~/Desktop/hypothesis_trace_diagnostics.png).
    --show              Also open an interactive window.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import KDTree

DEFAULT_RUN_DIR = (
    "~/tbp/results/comp_benefits_figures/"
    "post_training_hotspots_objects_with_stickers_comp_models_mujoco"
)
LOGO_CHANNEL = "learning_module_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path(DEFAULT_RUN_DIR))
    parser.add_argument("--lm", type=int, default=2)
    parser.add_argument("--logo-radius", type=float, default=0.01)
    parser.add_argument("--steps-per-view", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("~/Desktop/hypothesis_trace_diagnostics.png"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_logo_trees(checkpoint: Path, lm_id: int) -> dict[str, KDTree]:
    """KD-trees over the logo-channel nodes of each object in one LM's memory.

    Returns:
        Mapping from object ID to a KD-tree, for objects with a logo channel.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lm_dict = state["lm_dict"]
    memory = lm_dict[lm_id if lm_id in lm_dict else str(lm_id)]["graph_memory"]
    return {
        object_id: KDTree(np.asarray(channels[LOGO_CHANNEL].pos))
        for object_id, channels in memory.items()
        if LOGO_CHANNEL in channels
    }


def analyze_episode(episode: dict, logo_trees: dict, logo_radius: float) -> dict:
    """Compute per-step diagnostics of one episode.

    Returns:
        Dict of per-step arrays and episode metadata.
    """
    target = episode["primary_target"]
    steps = episode["steps"]
    target_diff, cumulative_lead, logo_input, mlh_near_logo = [], [], [], []
    mlh_correct, frozen = [], []
    for step in steps:
        gains, evidence = step["gains"], step["evidence"]
        others = [g for g in gains if g != target]
        if target in gains and others:
            target_diff.append(gains[target] - max(gains[g] for g in others))
            cumulative_lead.append(
                evidence[target] - max(evidence[g] for g in others if g in evidence)
            )
        else:
            target_diff.append(np.nan)
            cumulative_lead.append(np.nan)
        inputs = step["inputs"] or {}
        logo_input.append(inputs.get(LOGO_CHANNEL))
        tree = logo_trees.get(step["graph_id"])
        mlh_near_logo.append(
            tree is not None and tree.query(step["location"])[0] < logo_radius
        )
        mlh_correct.append(step["graph_id"] == target)
        frozen.append(step["frozen"])
    return {
        "episode": episode["episode"],
        "target": target,
        "converged": episode["converged_graph_id"],
        "terminal_state": episode["terminal_state"],
        "num_tagged_steps": episode["num_tagged_steps"],
        "target_diff": np.array(target_diff, dtype=float),
        "cumulative_lead": np.array(cumulative_lead, dtype=float),
        "logo_input": logo_input,
        "has_logo_input": np.array([x is not None for x in logo_input]),
        "mlh_near_logo": np.array(mlh_near_logo),
        "mlh_correct": np.array(mlh_correct),
        "frozen": np.array(frozen),
        "differential": np.array([s["differential"] for s in steps], dtype=float),
        "episode_step": np.array(
            [
                np.nan if s.get("episode_step") is None else s["episode_step"]
                for s in steps
            ],
            dtype=float,
        ),
        "body_location": np.array([s["body_location"] for s in steps], dtype=float),
    }


def mean_or_nan(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else float("nan")


def print_episode(result: dict, steps_per_view: int) -> None:
    diff = result["target_diff"]
    logo = result["has_logo_input"]
    near = result["mlh_near_logo"] & result["mlh_correct"]
    frozen_steps = np.flatnonzero(result["frozen"])
    converged_at = int(frozen_steps[0]) if len(frozen_steps) else None
    print(
        f"\nEpisode {result['episode']}: target {result['target']}, "
        f"{len(diff)} steps, terminal state {result['terminal_state']}, "
        f"converged on {result['converged']} at step {converged_at}, "
        f"{result['num_tagged_steps']} steps tagged"
    )
    print(f"  MLH is the target on {result['mlh_correct'].mean():.0%} of steps")
    print(f"  final evidence lead of target: {result['cumulative_lead'][-1]:.2f}")
    print(
        f"  logo input on {logo.mean():.0%} of steps: "
        f"{dict(Counter(x for x in result['logo_input'] if x is not None))}"
    )
    print(
        "  mean target differential: "
        f"with logo input {mean_or_nan(diff[logo]):.3f} ({logo.sum()} steps), "
        f"without {mean_or_nan(diff[~logo]):.3f} ({(~logo).sum()} steps)"
    )
    print(
        "  mean target differential: "
        f"MLH (correct) near logo {mean_or_nan(diff[near]):.3f} ({near.sum()} steps), "
        f"elsewhere {mean_or_nan(diff[~near]):.3f} ({(~near).sum()} steps)"
    )
    print("  per view: mean target differential, fraction with logo input")
    num_views = int(np.ceil(len(diff) / steps_per_view))
    cells = []
    for view in range(num_views):
        view_slice = slice(view * steps_per_view, (view + 1) * steps_per_view)
        cells.append(
            f"{view:2d}: {mean_or_nan(diff[view_slice]):+.2f} "
            f"({logo[view_slice].mean():.0%})"
        )
    for row in range(0, len(cells), 5):
        print("    " + "   ".join(cells[row : row + 5]))
    print_dwelling(result, steps_per_view)
    top = np.argsort(np.nan_to_num(diff, nan=-np.inf))[::-1][:5]
    print("  top 5 steps by target differential (step: value, logo input, near):")
    for i in top:
        print(
            f"    {i:4d}: {diff[i]:+.3f}, {result['logo_input'][i]}, "
            f"{bool(result['mlh_near_logo'][i])}"
        )


def view_dwelling(result: dict, steps_per_view: int) -> list[tuple[float, int]]:
    """Measure how long, and over how much surface, the sensor dwells per view.

    Returns:
        For each view, the number of Monty steps per traced (LM) step, and the
        number of distinct 1 cm voxels in which the LM sensed a location.
    """
    episode_step = result["episode_step"]
    locations = result["body_location"]
    dwelling = []
    for start in range(0, len(episode_step), steps_per_view):
        view = slice(start, start + steps_per_view)
        view_steps = episode_step[view]
        monty_steps = view_steps[-1] - (episode_step[start - 1] if start else 0)
        voxels = {tuple(v) for v in np.floor(locations[view] / 0.01).astype(int)}
        dwelling.append((monty_steps / len(view_steps), len(voxels)))
    return dwelling


def print_dwelling(result: dict, steps_per_view: int) -> None:
    if np.all(np.isnan(result["episode_step"])):
        return
    dwelling = view_dwelling(result, steps_per_view)
    ratio = np.nanmax(result["episode_step"]) / len(result["episode_step"])
    print(
        f"  Monty steps per traced step: {ratio:.2f} overall; per view "
        "(Monty steps per traced step, 1 cm voxels visited):"
    )
    cells = [
        f"{view:2d}: {steps:4.1f}, {voxels:3d}"
        for view, (steps, voxels) in enumerate(dwelling)
    ]
    for row in range(0, len(cells), 5):
        print("    " + "   ".join(cells[row : row + 5]))


def plot_episodes(results: list[dict], steps_per_view: int):
    figure, axes = plt.subplots(
        len(results), 1, figsize=(12, 3 * len(results)), sharex=True, squeeze=False
    )
    for ax, result in zip(axes[:, 0], results):
        steps = np.arange(len(result["target_diff"]))
        ax.bar(steps, result["target_diff"], width=1.0, color="gray", label="step diff")
        logo = result["has_logo_input"]
        ax.scatter(
            steps[logo],
            result["target_diff"][logo],
            s=6,
            c="#00a0df",
            zorder=3,
            label="logo input",
        )
        lead_ax = ax.twinx()
        lead_ax.plot(steps, result["cumulative_lead"], c="crimson", label="lead")
        lead_ax.set_ylabel("target lead", color="crimson")
        for boundary in range(steps_per_view, len(steps), steps_per_view):
            ax.axvline(boundary, c="black", lw=0.5, alpha=0.3)
        frozen = np.flatnonzero(result["frozen"])
        if len(frozen):
            ax.axvline(frozen[0], c="lime", lw=2, label="converged")
        ax.set_ylabel("target diff")
        ax.set_title(
            f"Episode {result['episode']}: {result['target']} "
            f"(converged on {result['converged']})"
        )
        ax.legend(loc="upper left", fontsize=7)
    axes[-1, 0].set_xlabel("traced step")
    figure.tight_layout()
    return figure


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    trace_files = sorted(
        (run_dir / "hypothesis_traces").glob(f"learning_module_{args.lm}_episode_*")
    )
    if not trace_files:
        raise FileNotFoundError(f"No trace histories in {run_dir}/hypothesis_traces")
    logo_trees = load_logo_trees(run_dir / "0" / "model.pt", args.lm)

    results = []
    for path in trace_files:
        result = analyze_episode(
            json.loads(path.read_text()), logo_trees, args.logo_radius
        )
        print_episode(result, args.steps_per_view)
        results.append(result)

    figure = plot_episodes(results, args.steps_per_view)
    output = args.output.expanduser()
    figure.savefig(output, dpi=120)
    print(f"\nSaved figure to {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
