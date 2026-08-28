# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Plot how face-on a learning module's patch viewed the surface, and its goals.

One figure per run. The top panel is the angle between the camera's viewing
direction and the patch's surface normal on every step the module processed
(from the motor telemetry's agent state and the module's stored normals),
the module's smoothed view angle where a ``FaceOnGoalGenerator`` recorded
one, the steps on which the agent was repositioned (jumps), the steps on
which the module proposed a face-on goal, and the step it recognized its
object.

Below it, one row per face-on goal the module got executed: the view
finder's frame just before and just after the jump, with the goal drawn
into the image -- the sensed surface point, the surface normal it wants to
look along (an arrow from that point to where the camera should go), and
the current view ray (the image center) -- and a 3D panel of the same
geometry: the face's points, the camera before and after the jump, its
view ray before, and the normal axis it aligned to.

Run from the repo root, e.g. ``python -m analysis.scripts.visualize_view_angle
debug_cube_face_on --learning-module LM_2``; the run is a directory or a
name under ``RESULTS_DIR``. The figure goes to ``<run_dir>/visualizations/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import quaternion as qt

from analysis.telemetry import EpisodeTelemetry
from tbp.monty.frameworks.loggers.npz_handler import materialize

if TYPE_CHECKING:
    import os

    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D

# Face-on jumps drawn in detail, at most.
MAX_JUMPS_SHOWN = 3
# View finder pinhole model: hfov 90 degrees at zoom 1 (see DepthTo3DLocations).
FOCAL = (1.0, 1.0)
# Colors: the surface point, the normal axis / goal, the current view ray.
SURFACE_COLOR = "#00B140"
GOAL_COLOR = "#FF2DAF"
VIEW_COLOR = "#00A6D6"
CLOUD_COLOR = "#B0A090"


def create_view_angle_figure(
    run_dir: os.PathLike,
    learning_module: str = "LM_2",
    sensor_module: str = "SM_3",
    episode: int = 0,
) -> Path:
    """Plot the module's view angle over the episode and its face-on goals.

    Args:
        run_dir: Experiment directory.
        learning_module: The module whose patch normals and goals to read.
        sensor_module: The view finder module whose frames to draw into.
        episode: Episode number to visualize.

    Returns:
        Path to the saved figure,
        ``<run_dir>/visualizations/view_angle_<module>_<episode>.png``.
    """
    run_dir = Path(run_dir)
    ep = EpisodeTelemetry.load(run_dir, episode)
    steps, angles = view_angles(ep, learning_module)
    jumps = jump_steps(ep)
    face_on = face_on_jumps(ep, learning_module, jumps)[:MAX_JUMPS_SHOWN]

    nrows = 1 + len(face_on)
    fig = plt.figure(figsize=(13, 4.5 + 4.2 * len(face_on)))
    grid = fig.add_gridspec(nrows, 3, height_ratios=[1.1] + [1] * len(face_on))
    ax = fig.add_subplot(grid[0, :])
    draw_angle_trace(ax, ep, learning_module, steps, angles, jumps, face_on)
    ax.set_title(f"{run_dir.name} -- {learning_module} view angle")

    raw = ep.blocks[sensor_module]["raw_observations"]
    for row, (jump, goal) in enumerate(face_on, start=1):
        # The goal is computed from the observation at the jump step and the
        # jump is executed in that same step, so that frame is "before" and
        # the next one "after".
        for column, (label, step) in enumerate((("before", jump), ("after", jump + 1))):
            fax = fig.add_subplot(grid[row, column])
            obs = materialize(raw[min(max(step, 0), len(raw) - 1)])
            draw_goal_in_frame(fax, obs, goal)
            fax.set_title(f"jump @ {jump}: {label} (step {step})", fontsize=9)
        gax = fig.add_subplot(grid[row, 2], projection="3d")
        draw_goal_geometry(gax, ep, jump, goal, materialize(raw[jump]))
    fig.tight_layout()
    out = run_dir / "visualizations" / f"view_angle_{learning_module}_{episode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def draw_angle_trace(
    ax: Axes,
    ep: EpisodeTelemetry,
    learning_module: str,
    steps: np.ndarray,
    angles: np.ndarray,
    jumps: list[int],
    face_on: list[tuple[int, dict]],
) -> None:
    """Draw the view-angle trace with jumps, goals, and recognition marked."""
    lm = ep.blocks[learning_module]
    ax.plot(
        steps, angles, ".-", color="#4C72B0", label="view angle (per processed step)"
    )
    if "smoothed_view_angle" in lm:
        smoothed = np.asarray(materialize(lm["smoothed_view_angle"]), dtype=float)
        if len(smoothed) == len(steps):
            ax.plot(
                steps,
                smoothed,
                "-",
                color="#DD8452",
                linewidth=2,
                label="smoothed (GSG)",
            )
    face_on_steps = {jump for jump, _ in face_on}
    for j in jumps:
        color = GOAL_COLOR if j in face_on_steps else "black"
        label = (
            ("face-on jump" if j in face_on_steps else "other jump")
            if j
            == min(k for k in jumps if (k in face_on_steps) == (j in face_on_steps))
            else None
        )
        ax.axvline(j, color=color, linestyle="--", linewidth=1.2, label=label)
    recognized = lm["individual_ts_reached_at_step"]
    if recognized is not None:
        ax.axvline(
            int(recognized),
            color="#55A868",
            linewidth=2,
            label=f"{learning_module} recognized",
        )
    ax.axhline(45, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("episode step")
    ax.set_ylabel("angle to surface normal (deg)")
    ax.set_ylim(0, 95)
    ax.legend(loc="upper right", fontsize=8)


def draw_goal_in_frame(ax: Axes, obs: dict, goal: dict) -> None:
    """Draw the face-on goal into a view finder frame.

    The sensed surface point, the normal axis to align to (arrow from the
    surface point to the goal camera position), and the current view ray
    (the image center), all projected through the frame's camera.
    """
    image = np.asarray(obs["rgba"])
    ax.imshow(image)
    shape = image.shape[:2]
    cam_to_world = np.asarray(obs["cam_to_world"], dtype=float)
    surface = np.asarray(goal["info"]["proposed_surface_loc"], dtype=float)
    camera_goal = np.asarray(goal["location"], dtype=float)
    (su, sv), s_ok = project(surface, cam_to_world, shape)
    (gu, gv), g_ok = project(camera_goal, cam_to_world, shape)
    center = ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    ax.plot(
        *center,
        "x",
        color=VIEW_COLOR,
        markersize=10,
        markeredgewidth=2,
        label="current view ray",
    )
    if s_ok:
        ax.plot(
            su,
            sv,
            "o",
            color=SURFACE_COLOR,
            markersize=8,
            markeredgecolor="black",
            label="surface point",
        )
        if g_ok:
            # The goal camera position usually projects outside the frame:
            # draw the normal axis toward it, clipped to the frame border.
            (eu, ev), inside = clip_to_frame((su, sv), (gu, gv), shape)
            ax.annotate(
                "",
                xy=(eu, ev),
                xytext=(su, sv),
                arrowprops=dict(arrowstyle="-|>", color=GOAL_COLOR, linewidth=2),
            )
            ax.plot(
                eu,
                ev,
                "*",
                color=GOAL_COLOR,
                markersize=13,
                markeredgecolor="black",
                markerfacecolor=GOAL_COLOR if inside else "none",
                label="goal camera position" + ("" if inside else " (off frame)"),
            )
        else:
            # The goal position sits on the camera's own axis: the normal
            # points straight at the viewer, so the axis collapses onto the point.
            ax.plot(
                su,
                sv,
                "*",
                color=GOAL_COLOR,
                markersize=13,
                markeredgecolor="black",
                label="goal camera position (toward viewer)",
            )
    ax.text(
        4,
        shape[0] - 6,
        f"view angle at goal: {goal['info'].get('view_angle', float('nan')):.0f}°",
        color="white",
        fontsize=8,
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )
    ax.set_xlim(0, shape[1] - 1)
    ax.set_ylim(shape[0] - 1, 0)
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=6)


def clip_to_frame(
    start: tuple[float, float], end: tuple[float, float], shape: tuple[int, int]
) -> tuple[tuple[float, float], bool]:
    """Shorten a segment from ``start`` so it ends inside the frame.

    Returns:
        The (possibly shortened) end point, and whether the original end
        already lay inside the frame.
    """
    sx, sy = start
    dx, dy = end[0] - sx, end[1] - sy
    limits = ((0, shape[1] - 1), (0, shape[0] - 1))
    t = 1.0
    for delta, origin, (low, high) in ((dx, sx, limits[0]), (dy, sy, limits[1])):
        if delta > 0:
            t = min(t, (high - origin) / delta)
        elif delta < 0:
            t = min(t, (low - origin) / delta)
    t = max(t, 0.0)
    return (sx + t * dx, sy + t * dy), t >= 1.0


def draw_goal_geometry(
    ax: Axes3D, ep: EpisodeTelemetry, jump: int, goal: dict, obs: dict
) -> None:
    """Draw the goal geometry in 3D: the face, the camera before/after, the normal."""
    actions = materialize(ep.blocks["motor_system"]["action_sequence"])
    before = np.asarray(actions[jump][1]["agent_id_0"]["position"], dtype=float)
    after_step = min(jump + 1, len(actions) - 1)
    after = np.asarray(actions[after_step][1]["agent_id_0"]["position"], dtype=float)
    surface = np.asarray(goal["info"]["proposed_surface_loc"], dtype=float)
    camera_goal = np.asarray(goal["location"], dtype=float)
    cloud = np.asarray(obs["semantic_3d"], dtype=float)
    cloud = cloud[cloud[:, 3] > 0][::40, :3]
    if len(cloud):
        ax.scatter(*cloud.T, s=2, color=CLOUD_COLOR, alpha=0.5, depthshade=False)
    ax.plot(
        *np.vstack([before, surface]).T,
        color=VIEW_COLOR,
        linewidth=2,
        label="view ray before",
    )
    ax.plot(
        *np.vstack([surface, camera_goal]).T,
        color=GOAL_COLOR,
        linewidth=2.5,
        label="normal axis (goal)",
    )
    ax.scatter(
        *surface,
        color=SURFACE_COLOR,
        s=50,
        edgecolors="black",
        label="surface point",
        depthshade=False,
    )
    ax.scatter(
        *before,
        color=VIEW_COLOR,
        s=60,
        marker="^",
        edgecolors="black",
        label="camera before",
        depthshade=False,
    )
    ax.scatter(
        *after,
        color=GOAL_COLOR,
        s=90,
        marker="*",
        edgecolors="black",
        label="camera after",
        depthshade=False,
    )
    points = np.vstack(
        [cloud, before[None], after[None], surface[None], camera_goal[None]]
    )
    center = (points.min(0) + points.max(0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) / 2, 0.02) * 1.1
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-35, vertical_axis="y")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title(f"geometry of jump @ {jump}", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")


def project(
    point: np.ndarray, cam_to_world: np.ndarray, shape: tuple[int, int]
) -> tuple[tuple[float, float], bool]:
    """Project a body-frame point into a frame taken with ``cam_to_world``.

    Inverts DepthTo3DLocations' unprojection: camera-frame ``x = fx * u * d``,
    ``y = fy * v * d``, ``z = -d`` with ``u, v`` in [-1, 1].

    Returns:
        The (column, row) pixel coordinates, and whether the point lies in
        front of the camera (a point at or behind it has no projection).
    """
    camera = np.linalg.inv(cam_to_world) @ np.append(point, 1.0)
    depth = -camera[2]
    if depth <= 1e-6:
        return (np.nan, np.nan), False
    u = camera[0] / (FOCAL[0] * depth)
    v = camera[1] / (FOCAL[1] * depth)
    return ((u + 1) / 2 * (shape[1] - 1), (1 - v) / 2 * (shape[0] - 1)), True


def face_on_jumps(
    ep: EpisodeTelemetry, learning_module: str, jumps: list[int]
) -> list[tuple[int, dict]]:
    """Pair each jump the module's face-on goal caused with that goal.

    A jump is the module's when its goal reached the motor system on that
    step; the goal is the module's most recently logged one.

    Returns:
        ``(jump step, goal record)`` pairs, ascending.
    """
    lm = ep.blocks[learning_module]
    if "goal_states" not in lm or not len(lm["goal_states"]):
        return []
    goals_in = materialize(ep.blocks["motor_system"]["goals_in"])
    lm_id = f"learning_module_{learning_module.rsplit('_', maxsplit=1)[-1]}"
    goals = materialize(lm["goal_states"])
    lm_steps = np.flatnonzero(np.asarray(lm["lm_processed_steps"], dtype=bool))
    logged = {}
    for goal in goals:
        k = goal["info"].get("matching_step_when_output_goal_set")
        if k is not None and 0 <= int(k) < len(lm_steps):
            logged[int(lm_steps[int(k)])] = goal
    pairs = []
    for jump in jumps:
        senders = goals_in[jump].get("sender_ids", []) if jump < len(goals_in) else []
        if lm_id not in [str(s) for s in senders]:
            continue
        earlier = [s for s in logged if s <= jump]
        if earlier:
            pairs.append((jump, logged[max(earlier)]))
    return pairs


def view_angles(
    ep: EpisodeTelemetry, learning_module: str
) -> tuple[np.ndarray, np.ndarray]:
    """Angle between the camera's viewing direction and the patch normal.

    Read off the motor telemetry's agent state (agent and sensor rotations;
    the camera looks along its -Z) and the module's stored pose vectors.

    Returns:
        The processed episode steps and the angle, in degrees, at each.
    """
    actions = materialize(ep.blocks["motor_system"]["action_sequence"])
    lm = ep.blocks[learning_module]
    steps = np.flatnonzero(np.asarray(lm["lm_processed_steps"], dtype=bool))
    channel = next(c for c in lm if c.startswith("patch"))
    normals = np.asarray(materialize(lm[channel])["pose_vectors"], dtype=float)[:, :3]
    angles = []
    for row, step in enumerate(steps):
        agent = actions[step][1]["agent_id_0"]
        rotation = qt.quaternion(
            *np.asarray(agent["rotation"], dtype=float)
        ) * qt.quaternion(
            *np.asarray(agent["sensors"][channel]["rotation"], dtype=float)
        )
        view = qt.rotate_vectors(rotation, np.array([0.0, 0.0, -1.0]))
        cosine = -np.dot(view, normals[row]) / (np.linalg.norm(normals[row]) + 1e-12)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return steps, np.asarray(angles)


def jump_steps(ep: EpisodeTelemetry) -> list[int]:
    """Episode steps on which the agent was repositioned (a pose-setting action).

    Returns:
        The steps, ascending.
    """
    actions = materialize(ep.blocks["motor_system"]["action_sequence"])
    return [
        s
        for s in range(len(actions))
        if any(
            isinstance(a, dict) and str(a.get("action", "")).startswith("set_")
            for a in actions[s][0]
        )
    ]


if __name__ == "__main__":
    import argparse

    from analysis.cli import run_directory

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run", type=run_directory, help="run directory or name under RESULTS_DIR"
    )
    parser.add_argument("--learning-module", default="LM_2")
    parser.add_argument("--sensor-module", default="SM_3")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    create_view_angle_figure(
        args.run, args.learning_module, args.sensor_module, args.episode
    )
