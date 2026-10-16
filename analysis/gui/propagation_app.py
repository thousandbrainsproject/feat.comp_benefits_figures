# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Interactive explorer of the EvidenceGoalGenerator's salience propagation.

A local web app (Dash) that runs the goal generator's propagation on synthetic
3D primitives (see `primitives`) or on learned object models (see
`learned_models`) and lets you watch it unfold node by node. It complements
analysis/scripts/visualize_salience_propagation.py, which plots the same
propagation on learned models for the paper figures.

Usage:
    python analysis/gui/propagation_app.py [--condition KEY] [--port PORT] [--debug]

then open http://127.0.0.1:8050 in a browser. Requires `dash` (pip install dash).

In the app:

- Pick a condition: one of seven primitives (from a uniform curved surface to
  object-ID channels with noisy or striped IDs), or a learned model (the 10
  distinct YCB objects learned by the surface agent, or the compositional
  objects learned by LM 2 of the compositional Monty, with their object-ID
  channels), and the channel whose graph to display, or all of them
  superimposed (told apart by their node symbols).
- Click a node (or pick a landmark) to sense its location: that is one step of
  the LM, on which every channel receives the features or object ID stored at
  its nearest node, which then propagate, reducing the salience of the nodes
  they reach (by salience_reduction, less for nodes reached later if
  salience_reduction_falloff > 0), while the salience of every other node
  relaxes by one step towards rest (0.5). Override the sensed hue, normal
  (tilted away from the stored one) or object ID to see the goal generator
  refuse to propagate from input the MLH does not predict.
- "Sample goal" samples a goal's target as the goal generator would (in
  proportion to salience, across all channels), and shows its hypothetical
  propagation, through the target's channel, which raises the salience of the
  part the target is on to 1 (not an LM step). Or, with "Clicking a node" set
  to select a goal, click a node to take it as the target, as if it had been
  sampled (not a node with no salience, which is never sampled); with every
  channel superimposed, a dropdown asks which channel's model to take it from
  when several channels have a node there. "Go to goal" then senses the
  target's location (one LM step), whose input reduces that salience: going to
  the same part again and again drives it below the rest of the object, until
  goals are sampled elsewhere.
- Set the age of the MLH hypothesis (as burst sampling tracks it), which every
  LM step increases by one: nothing propagates, and no goal is sampled, until
  the MLH has survived its first chance of being deleted, i.e. its age exceeds
  min_age.
- Scrub the slider, or press play, to step through the propagation in the order
  the nodes were reached. The seeds, the node being reached, the sinks (reached
  nodes that do not propagate on), the boundary the propagation stopped at, and
  the frontier a push-through started from are each marked, and the status bar
  says when a push-through is under way.
- Color the nodes by the propagation, by their hue, their surface normals, their
  current salience (bright: salient, likely to be sampled as a goal; grey: not;
  "Emphasize low salience" inverts this), their object ID, or the region of the
  primitive they belong to; toggle normal glyphs, the propagation tree, and the
  other channels' nodes.
- "Step" relaxes the salience by one step (or ten) without sensing; "Reset"
  restores it to rest everywhere, and resets the view (orbiting, panning or zooming
  the scene otherwise persists across sensing, stepping and recoloring).
  The goal generator's parameters can be changed on the fly (the salience is
  kept).
- Hover a node for its features, its neighbors (how many are similar, dissimilar,
  or on the far side of the surface), whether it propagates, and its salience.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from matplotlib import colors as mcolors
from plotly.colors import qualitative, sample_colorscale

from analysis.gui import learned_models
from analysis.gui.primitives import CONDITION_KEYS, build_condition
from analysis.gui.propagation_session import MIN_AGE, PARAMETERS, Session

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.goal_generation import PropagationRecord

UNREACHED_COLOR = "#d4d4d4"
SEED_COLOR = "#ff00ff"
CURRENT_COLOR = "#ffd600"
SINK_COLOR = "#111111"
BOUNDARY_COLOR = "#e41a1c"
FRONTIER_COLOR = "#ff6f00"
GOAL_COLOR = "#00b140"
TREE_COLOR = "rgba(60, 60, 60, 0.45)"
GLYPH_COLOR = "rgba(0, 90, 160, 0.5)"
# Salience from 0 (grey, hard to see), through rest (0.5), to 1 (bright and
# obvious, highlighted by a goal).
SALIENCE_COLORSCALE = [[0.0, "#e6e6e6"], [0.5, "#fdae6b"], [1.0, "#d94801"]]
ID_COLORS = {
    "square": "#1f77b4",
    "circle": "#ff7f0e",
    "logo_A": "#2ca02c",
    "logo_B": "#d62728",
}
REGION_COLORS = {
    "surface": "#9ecae1",
    "hue island": "#fd8d3c",
    "tilted island": "#756bb1",
    "smooth bump": "#74c476",
    "small hue island": "#e6550d",
    "small tilted island": "#54278f",
    "far side": "#bcbddc",
    # Parts of a learned mug.
    "side": "#9ecae1",
    "handle": "#fd8d3c",
    "rim": "#74c476",
    "bottom": "#756bb1",
    "inside": "#e6550d",
    "other": "#bbbbbb",
}
COLOR_MODES = ["Propagation", "Hue", "Normals", "Salience", "Object ID", "Region"]
# The "Displayed channel" value showing every channel's graph, superimposed.
ALL_CHANNELS = "__all__"
# The node symbols telling the channels apart when they are superimposed.
CHANNEL_SYMBOLS = ["circle", "square", "diamond", "cross", "x"]
CLICK_MODES = [
    {"label": "Sense its location", "value": "sense"},
    {"label": "Select it as a goal", "value": "goal"},
]
OVERLAYS = {
    "seeds": "Seeds",
    "current": "Node being reached",
    "sinks": "Sinks (reached, don't propagate on)",
    "boundary": "Boundary (unreached neighbors)",
    "frontier": "Push-through frontier",
    "tree": "Propagation tree (edges)",
    "glyphs": "Normal glyphs",
    "ghost": "Other channels",
}
DEFAULT_OVERLAYS = ["seeds", "current", "sinks", "boundary", "frontier"]
DEFAULT_CAMERA = {
    "eye": {"x": 0.0, "y": -0.6, "z": 2.2},
    "up": {"x": 0, "y": 1, "z": 0},
}
SPEEDS = [1, 2, 5, 10, 25, 50]
PARAMETER_STEPS = {
    "num_propagation_neighbors": 1,
    "salience_relaxation_steps": 1,
    "salience_reduction": 0.05,
    "salience_reduction_falloff": 0.1,
    "lm_hue_tolerance": 0.01,
    "lm_normal_tolerance": 5,
    "max_dissimilar_neighbors": 1,
    "min_propagation_fraction": 0.01,
}

SESSION = Session(CONDITION_KEYS[0])
CONDITION_OPTIONS = [
    {"label": build_condition(key).title, "value": key} for key in CONDITION_KEYS
] + learned_models.condition_options()


# ------------------------------------------------------------------- figure


def displayed_channels(session: Session, channel: str) -> list[str]:
    """The channels whose graphs are shown: one, or all of them superimposed.

    Returns:
        The channels, in the order their node traces are drawn.
    """
    return list(session.channels) if channel == ALL_CHANNELS else [channel]


def channel_options(session: Session) -> list[dict]:
    """Dropdown options for the displayed channel.

    Returns:
        Dash dropdown options: each channel, then all of them if there are several.
    """
    options = [{"label": channel, "value": channel} for channel in session.channels]
    if len(session.channels) > 1:
        options.append({"label": "all (superimposed)", "value": ALL_CHANNELS})
    return options


def last_record(session: Session, channel: str) -> PropagationRecord | None:
    """The last propagation (of a sensing or a goal) on a channel, if any.

    Returns:
        The propagation record, or None.
    """
    if session.sensing is None:
        return None
    return session.sensing.records.get(channel)


def color_table(names, preset: dict) -> dict[str, str]:
    """Assign a color to each name: its preset one, or the next of a palette.

    Returns:
        A color per distinct name.
    """
    table = {}
    extra = 0
    for name in sorted(set(names)):
        if name in preset:
            table[name] = preset[name]
        else:
            table[name] = qualitative.Plotly[extra % len(qualitative.Plotly)]
            extra += 1
    return table


def marker_size(num_nodes: int) -> float:
    """Marker size for a graph: smaller the more nodes it has.

    Returns:
        The size, in pixels.
    """
    return float(np.clip(135 / np.sqrt(max(num_nodes, 1)), 3.6, 9.0))


def stage_colors(record: PropagationRecord, progress: int) -> list[str]:
    """Color the reached nodes by stage and hops.

    Returns:
        One color per node of `record.node_order[:progress]`: a viridis ramp
        (dark to light) over the hops from the seeds, and a yellow-to-red ramp
        over the hops of each push-through.
    """
    colors = np.empty(len(record.node_order), dtype=object)
    for stage in np.unique(record.stage):
        in_stage = np.flatnonzero(record.stage == stage)
        hops = record.hops[in_stage]
        fraction = hops / max(hops.max(), 1)
        if stage == 0:
            colors[in_stage] = sample_colorscale("Viridis", fraction.tolist())
        else:
            colors[in_stage] = sample_colorscale(
                "YlOrRd", (0.35 + 0.65 * fraction).tolist()
            )
    return colors[:progress].tolist()


def salience_marker(salience: np.ndarray, emphasize_low: bool) -> dict:
    """Marker settings coloring nodes by salience.

    Args:
        salience: The salience of each node.
        emphasize_low: Whether to make the nodes with low salience the bright
            ones (by default, the salient nodes are bright and the rest grey).

    Returns:
        Keys to update the nodes' marker with.
    """
    colorscale = SALIENCE_COLORSCALE
    if emphasize_low:
        colorscale = [[1 - stop, color] for stop, color in reversed(colorscale)]
    return {
        "color": salience,
        "colorscale": colorscale,
        "cmin": 0,
        "cmax": 1,
        "colorbar": {"title": "salience", "thickness": 12, "len": 0.5, "x": 1.0},
    }


def hover_texts(
    session: Session, channel: str, progress: int, show_channel: bool = False
) -> list[str]:
    graph = session.graphs[channel]
    labels = session.condition.labels[channel]
    salience = session.salience(channel)
    is_id = session.is_id_channel(channel)
    propagation_text = {}
    if session.sensing and channel in session.sensing.records:
        record = session.sensing.records[channel]
        for position, node in enumerate(record.node_order[:progress]):
            stage = record.stage[position]
            where = "from seeds" if stage == 0 else f"push-through {stage}"
            change = (
                "salience raised to 1 (goal)"
                if record.hypothetical
                else f"salience -{record.reduction[position]:.2f}"
            )
            propagation_text[int(node)] = (
                f"reached #{position + 1}, {record.hops[position]} hop(s) {where}; "
                + change
            )
    has_hue = "hsv" in graph.feature_mapping
    texts = []
    for node in range(len(graph.pos)):
        lines = [
            f"<b>node {node}</b> · "
            + (f"{channel} · " if show_channel else "")
            + str(labels[node])
        ]
        if is_id:
            object_id = int(graph.feature("object_id")[node, 0])
            lines.append(f"object ID: {session.id_name(object_id)}")
        else:
            normal = graph.normals[node]
            text = f"normal ({normal[0]:+.2f}, {normal[1]:+.2f}, {normal[2]:+.2f})"
            if has_hue:
                text = f"hue {graph.feature('hsv')[node, 0]:.3f} · " + text
            lines.append(text)
        lines.append(f"salience: {salience[node]:.2f}")
        if node in propagation_text:
            lines.append(propagation_text[node])
        texts.append("<br>".join(lines))
    return texts


def line_trace(starts, ends, color, width, name, showlegend=False, **kwargs):
    """A single Scatter3d of disconnected segments from starts to ends.

    Returns:
        The trace.
    """
    if len(starts) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", name=name, showlegend=False)
    points = np.empty((len(starts) * 3, 3))
    points[0::3] = starts
    points[1::3] = ends
    points[2::3] = np.nan
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        line={"color": color, "width": width},
        hoverinfo="skip",
        name=name,
        showlegend=showlegend,
        **kwargs,
    )


def marker_trace(pos, nodes, name, symbol, color, size, line=None):
    return go.Scatter3d(
        x=pos[nodes, 0],
        y=pos[nodes, 1],
        z=pos[nodes, 2],
        mode="markers",
        marker={
            "symbol": symbol,
            "color": color,
            "size": size,
            "line": line or {"width": 0},
        },
        hoverinfo="skip",
        name=name,
        showlegend=True,
    )


def legend_marker(name, color, symbol="circle", size=6):
    return go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode="markers",
        marker={"symbol": symbol, "color": color, "size": size},
        name=name,
        showlegend=True,
        hoverinfo="skip",
    )


def node_marker(
    session: Session,
    channel: str,
    progress: int,
    color_mode: str,
    emphasize_low: bool,
    tables: dict[str, dict[str, str]],
) -> tuple[dict, list]:
    """Color a channel's nodes.

    Args:
        session: The session.
        channel: The channel.
        progress: How many nodes of the channel's last propagation to color as
            reached.
        color_mode: One of COLOR_MODES.
        emphasize_low: See `build_figure`.
        tables: The color of each object ID name ("Object ID") and region label
            ("Region"), shared by the displayed channels.

    Returns:
        The nodes' marker settings, and the legend entries for their colors.
    """
    graph = session.graphs[channel]
    num_nodes = len(graph.pos)
    record = last_record(session, channel)
    marker: dict = {"size": marker_size(num_nodes), "line": {"width": 0}}
    legend = []
    if color_mode == "Propagation":
        colors = np.full(num_nodes, UNREACHED_COLOR, dtype=object)
        if record is not None and progress:
            colors[record.node_order[:progress]] = stage_colors(record, progress)
        marker["color"] = colors.tolist()
        legend.append(legend_marker("reached: hops from seeds (dark→light)", "#3b528b"))
        legend.append(legend_marker("pushed through: hops from frontier", "#fd8d3c"))
    elif color_mode == "Hue" and "hsv" in graph.feature_mapping:
        hsv = np.clip(graph.feature("hsv"), 0, 1)
        marker["color"] = [mcolors.to_hex(c) for c in mcolors.hsv_to_rgb(hsv)]
    elif color_mode == "Normals":
        rgb = (graph.normals + 1) / 2
        marker["color"] = [mcolors.to_hex(c) for c in rgb]
    elif color_mode == "Salience":
        salience = session.salience(channel, progress if record else None)
        marker.update(salience_marker(salience, emphasize_low))
    elif color_mode == "Object ID" and session.is_id_channel(channel):
        names = [session.id_name(i) for i in graph.feature("object_id")[:, 0]]
        colors = tables["Object ID"]
        marker["color"] = [colors[name] for name in names]
        legend.extend(legend_marker(name, colors[name]) for name in sorted(set(names)))
    else:  # Region, or a mode the channel has no features for.
        labels = session.condition.labels[channel]
        colors = tables["Region"]
        marker["color"] = [colors[label] for label in labels]
        legend.extend(
            legend_marker(label, colors[label]) for label in sorted(set(labels))
        )
    return marker, legend


def channel_overlays(
    session: Session,
    channel: str,
    record: PropagationRecord | None,
    progress: int,
    overlays,
) -> list:
    """The traces drawn over a channel's nodes: glyphs, propagation, goal target.

    Returns:
        The traces.
    """
    graph = session.graphs[channel]
    traces = []
    if "glyphs" in overlays:
        length = 0.004
        traces.append(
            line_trace(
                graph.pos,
                graph.pos + length * graph.normals,
                GLYPH_COLOR,
                1.5,
                "normals",
            )
        )
    if record is not None and progress:
        traces.extend(
            propagation_overlays(session, channel, record, progress, overlays)
        )
    if session.goal is not None and session.goal[0] == channel:
        traces.append(
            marker_trace(
                graph.pos,
                [session.goal[1]],
                "goal target",
                "diamond-open",
                GOAL_COLOR,
                14,
                {"width": 4, "color": GOAL_COLOR},
            )
        )
    return traces


def ghost_traces(session: Session, channel: str) -> list:
    """Faint traces of the nodes of every channel but the displayed one.

    Returns:
        The traces.
    """
    return [
        go.Scatter3d(
            x=other_graph.pos[:, 0],
            y=other_graph.pos[:, 1],
            z=other_graph.pos[:, 2],
            mode="markers",
            marker={"size": 2, "color": "#999999", "opacity": 0.3},
            hoverinfo="skip",
            name=f"{other} (other channel)",
            showlegend=True,
        )
        for other, other_graph in session.graphs.items()
        if other != channel
    ]


def without_duplicate_legends(traces: list) -> list:
    """Show each legend entry once, when several channels draw the same overlay.

    Returns:
        The traces, with all but the first of each legend name hidden from the
        legend.
    """
    seen = set()
    for trace in traces:
        if trace.showlegend:
            if trace.name in seen:
                trace.showlegend = False
            seen.add(trace.name)
    return traces


def build_figure(
    session: Session,
    channel: str,
    progress: int,
    color_mode: str,
    overlays: list,
    camera: dict | None = None,
    emphasize_low: bool = False,
) -> go.Figure:
    """Build the 3D figure of a channel's graph, or of all of them superimposed.

    The first traces are the nodes of the displayed channels, one trace per
    channel in the order of `displayed_channels` (see `event_node`).

    Args:
        session: The session.
        channel: The channel whose graph to show, or ALL_CHANNELS.
        progress: How many nodes of the last propagation to show as reached (on
            each channel, when they are superimposed).
        color_mode: One of COLOR_MODES.
        overlays: The keys of OVERLAYS to draw.
        camera: The view the user orbited to (as plotly reports it in
            relayoutData["scene.camera"]), or None for the default view.
        emphasize_low: In the "Salience" color mode, whether to make the nodes
            with low salience the bright ones rather than the salient nodes.

    Returns:
        The figure.
    """
    channels = displayed_channels(session, channel)
    superimposed = channel == ALL_CHANNELS
    id_names = [
        session.id_name(object_id)
        for shown in channels
        if session.is_id_channel(shown)
        for object_id in session.graphs[shown].feature("object_id")[:, 0]
    ]
    tables = {
        "Object ID": color_table(id_names, ID_COLORS),
        "Region": color_table(
            [label for shown in channels for label in session.condition.labels[shown]],
            REGION_COLORS,
        ),
    }
    node_traces, traces = [], []
    for index, shown in enumerate(channels):
        pos = session.graphs[shown].pos
        record = last_record(session, shown)
        shown_progress = 0 if record is None else min(progress, len(record.node_order))
        marker, legend = node_marker(
            session, shown, shown_progress, color_mode, emphasize_low, tables
        )
        if superimposed:
            marker["symbol"] = CHANNEL_SYMBOLS[index % len(CHANNEL_SYMBOLS)]
            if index:
                marker.pop("colorbar", None)
        node_traces.append(
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="markers",
                marker=marker,
                hovertext=hover_texts(session, shown, shown_progress, superimposed),
                hoverinfo="text",
                name=f"{shown} nodes" if superimposed else "nodes",
                showlegend=superimposed,
            )
        )
        traces.extend(legend)
        traces.extend(
            channel_overlays(session, shown, record, shown_progress, overlays)
        )
        if "ghost" in overlays and not superimposed:
            traces.extend(ghost_traces(session, shown))

    figure = go.Figure(node_traces + without_duplicate_legends(traces))
    axis = {"visible": False, "showbackground": False}
    figure.update_layout(
        scene={
            "xaxis": axis,
            "yaxis": axis,
            "zaxis": axis,
            "aspectmode": "data",
            "camera": camera or DEFAULT_CAMERA,
        },
        # Changing the revision makes plotly adopt the camera given above rather
        # than keep the one the user orbited to; that is wanted exactly when the
        # view is reset (camera None) after the user orbited (camera set).
        uirevision=f"{session.condition.key}:{'user' if camera else 'default'}",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={
            "x": 0.0,
            "y": 1.0,
            "font": {"size": 11},
            "itemsizing": "constant",
            "bgcolor": "rgba(255,255,255,0.7)",
        },
        height=760,
        clickmode="event",
    )
    return figure


def propagation_overlays(
    session: Session, channel: str, record: PropagationRecord, progress: int, overlays
) -> list:
    """The traces marking the special nodes of a propagation, up to a point in it.

    Returns:
        The traces.
    """
    pos = session.graphs[channel].pos
    reached_nodes = record.node_order[:progress]
    traces = []
    if "tree" in overlays:
        has_parent = record.predecessor[:progress] >= 0
        children = reached_nodes[has_parent]
        parents = record.predecessor[:progress][has_parent]
        traces.append(
            line_trace(pos[parents], pos[children], TREE_COLOR, 2, "propagation tree")
        )
    if "boundary" in overlays:
        reached = np.zeros(len(pos), dtype=bool)
        reached[reached_nodes] = True
        _, adjacency = session.edge_matrices(channel)
        touched = np.asarray(adjacency.T @ reached.astype(float)).ravel() > 0
        boundary = np.flatnonzero(touched & ~reached)
        traces.append(
            marker_trace(
                pos,
                boundary,
                "boundary: unreached neighbors of reached nodes",
                "circle-open",
                BOUNDARY_COLOR,
                6,
                {"width": 2, "color": BOUNDARY_COLOR},
            )
        )
    if "sinks" in overlays:
        sinks = np.intersect1d(reached_nodes, record.sinks)
        traces.append(
            marker_trace(
                pos, sinks, "sinks: reached, don't propagate on", "x", SINK_COLOR, 4
            )
        )
    if "frontier" in overlays:
        frontier = reached_nodes[
            (record.stage[:progress] > 0) & (record.hops[:progress] == 0)
        ]
        traces.append(
            marker_trace(
                pos, frontier, "push-through frontier", "square", FRONTIER_COLOR, 5
            )
        )
    if "seeds" in overlays:
        seeds = np.intersect1d(reached_nodes, record.seeds)
        traces.append(marker_trace(pos, seeds, "seeds", "diamond", SEED_COLOR, 7))
    if "current" in overlays:
        current = reached_nodes[-1:]
        traces.append(
            marker_trace(
                pos,
                current,
                "node being reached",
                "circle-open",
                CURRENT_COLOR,
                16,
                {"width": 4, "color": CURRENT_COLOR},
            )
        )
    return traces


# ------------------------------------------------------------------- text


def status_text(session: Session, channel: str, progress: int):
    """The status bar: where in the propagation the slider is, and what is going on.

    Returns:
        Dash children and a background color.
    """
    if session.sensing is None:
        return (
            "Click a node (or pick a landmark) to sense its location, or sample a "
            "goal.",
            "#f0f0f0",
        )
    if channel == ALL_CHANNELS:
        return superimposed_status(session, progress)
    record = session.sensing.records.get(channel)
    if record is None:
        reason = session.sensing.no_propagation.get(channel, "unknown")
        return f"No propagation on {channel}: {reason}.", "#fde0dd"
    total = len(record.node_order)
    progress = min(progress, total)
    if progress == 0:
        return f"Step 0 of {total}: nothing reached yet.", "#f0f0f0"
    position = progress - 1
    node = int(record.node_order[position])
    stage = int(record.stage[position])
    hops = int(record.hops[position])
    parts = [
        html.B(("goal " if record.hypothetical else "") + f"{progress} / {total}"),
        f" · node {node}: ",
    ]
    reduction = (
        " · salience raised to 1"
        if record.hypothetical
        else f" · salience -{record.reduction[position]:.2f}"
    )
    if stage == 0:
        parts.append("seed" if hops == 0 else f"{hops} hop(s) from the seeds")
        parts.append(reduction)
        if node in record.sinks:
            parts.append(" · sink (does not propagate on)")
        return parts, "#e3f6e8" if record.hypothetical else "#e8f1fb"
    stalled_at = int(np.count_nonzero(record.stage < stage))
    frontier = int(np.count_nonzero((record.stage == stage) & (record.hops == 0)))
    parts.append(
        "frontier node (pushed through to)"
        if hops == 0
        else f"{hops} hop(s) from the frontier"
    )
    parts.append(reduction)
    parts.append(
        html.Span(
            f"  PUSH-THROUGH {stage}: the propagation stalled at {stalled_at} nodes, "
            f"short of min_propagation_fraction ({record.min_size:.1f} nodes), so it "
            f"continued from the {frontier} unreached neighbors of the reached "
            "nodes, whatever their features, until that many nodes were reached.",
            style={"marginLeft": "12px", "fontWeight": "bold", "color": "#b35900"},
        )
    )
    return parts, "#ffe9cc"


def superimposed_status(session: Session, progress: int):
    """The status bar with every channel superimposed: each one's progress.

    Returns:
        Dash children and a background color.
    """
    sensing = session.sensing
    entries = []
    for channel in session.channels:
        record = sensing.records.get(channel)
        if record is None:
            entries.append(f"{channel}: no propagation")
        else:
            total = len(record.node_order)
            entries.append(f"{channel}: {min(progress, total)} / {total}")
    kind = "Goal" if sensing.goal_target is not None else "Sensing"
    return (
        [html.B(f"{kind}, all channels"), " · " + " · ".join(entries)],
        "#e3f6e8" if sensing.goal_target is not None else "#e8f1fb",
    )


def summary_children(session: Session):
    """The sensing summary: what each channel received and how it propagated.

    Returns:
        Dash children.
    """
    children = [
        html.Div(
            f"LM step {session.step} · {session.condition.title}",
            style={"fontWeight": "bold"},
        ),
        html.Div(
            f"MLH age {session.mlh_age}: "
            + (
                f"survived deletion (> min_age {MIN_AGE}), so input propagates "
                "and goals are sampled"
                if session.mlh_established
                else f"too young to propagate or sample goals (needs > min_age "
                f"{MIN_AGE})"
            ),
            style={"color": "#333" if session.mlh_established else "#b30000"},
        ),
    ]
    active = {
        name: value for name, value in session.overrides.items() if value is not None
    }
    if active:
        children.append(
            html.Div(
                "Overriding sensed "
                + ", ".join(
                    f"{name.replace('_', ' ')} = "
                    + (session.id_name(value) if name == "object_id" else f"{value:g}")
                    for name, value in active.items()
                ),
                style={"color": "#b35900"},
            )
        )
    if session.sensing is None:
        children.append(html.Div("No location sensed yet."))
        return children
    sensing = session.sensing
    if sensing.goal_target is not None:
        children.extend(goal_summary(session, sensing))
        return children
    children.append(
        html.Div(
            "Sensed location: ("
            + ", ".join(f"{c:.3f}" for c in sensing.location)
            + f") at step {sensing.step}"
        )
    )
    for channel in session.channels:
        lines = [html.B(channel), html.Br(), f"received {sensing.percepts[channel]}"]
        record = sensing.records.get(channel)
        if record is None:
            lines += [
                html.Br(),
                html.I(f"no propagation: {sensing.no_propagation[channel]}"),
            ]
        else:
            pushes = int(record.stage.max())
            salience = session.salience(channel)
            lines += [
                html.Br(),
                f"{len(record.node_order)} nodes reached from {len(record.seeds)} "
                f"seeds; {len(record.sinks)} sinks",
                html.Br(),
                f"salience reduced by {record.reduction.max():.2f} (first) to "
                f"{record.reduction.min():.2f} (last); "
                f"{int(np.count_nonzero(salience <= 0))} node(s) at 0",
            ]
            if pushes:
                lines += [
                    html.Br(),
                    html.Span(
                        f"{pushes} push-through(s) to reach {record.min_size:.1f} "
                        "nodes",
                        style={"color": "#b35900"},
                    ),
                ]
        children.append(html.Div(lines, style={"marginTop": "6px"}))
    return children


def goal_summary(session: Session, sensing) -> list:
    """The goal summary: its target, and the part its propagation highlighted.

    Returns:
        Dash children.
    """
    channel, _ = sensing.goal_target
    record = sensing.records.get(channel)
    lines = [
        html.B(f"Goal sampled at step {sensing.step}"),
        html.Br(),
        f"on {channel}: {sensing.percepts[channel].removeprefix('goal target: ')}",
        html.Br(),
        "at ("
        + ", ".join(f"{c:.3f}" for c in sensing.location)
        + ")"
        + ("" if session.goal else " (gone to, or superseded by a sensing)"),
        html.Br(),
    ]
    if record is None:
        lines.append(html.I(sensing.no_propagation[channel]))
    else:
        lines.append(
            f"hypothetical propagation from {len(record.seeds)} seeds raised "
            f"{len(record.node_order)} nodes to salience 1"
        )
        if int(record.stage.max()):
            lines += [
                html.Br(),
                html.Span(
                    f"{int(record.stage.max())} push-through(s) to reach "
                    f"{record.min_size:.1f} nodes",
                    style={"color": "#b35900"},
                ),
            ]
    lines += [
        html.Br(),
        html.I("Other channels are not highlighted; Go to goal senses the target."),
    ]
    return [html.Div(lines, style={"marginTop": "6px", "color": "#0b6b2e"})]


def details_children(details: dict[str, str]):
    return html.Table(
        [
            html.Tr(
                [
                    html.Td(name, style={"color": "#666", "paddingRight": "8px"}),
                    html.Td(value),
                ]
            )
            for name, value in details.items()
        ],
        style={"fontSize": "12px"},
    )


def slider_marks(record: PropagationRecord | None) -> dict:
    if record is None:
        return {0: "0"}
    total = len(record.node_order)
    marks = {0: "0", total: str(total)}
    for start in np.flatnonzero(np.diff(record.stage) > 0) + 1:
        marks[int(start)] = {
            "label": f"push {record.stage[start]}",
            "style": {"color": "#b35900", "fontWeight": "bold"},
        }
    return marks


# ------------------------------------------------------------------- layout


def section(title, *children):
    return html.Div(
        [
            html.Div(title, style={"fontWeight": "bold", "margin": "10px 0 4px"}),
            *children,
        ]
    )


def parameter_inputs():
    rows = []
    for name, default in PARAMETERS.items():
        rows.append(
            html.Div(
                [
                    html.Label(name, style={"fontSize": "11px", "flex": "1"}),
                    dcc.Input(
                        id={"type": "param", "name": name},
                        type="number",
                        value=default,
                        step=PARAMETER_STEPS[name],
                        min=0,
                        debounce=True,
                        style={"width": "70px", "fontSize": "11px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "margin": "2px 0"},
            )
        )
    return rows


def id_options(session: Session) -> list[dict]:
    """Dropdown options for the object IDs the condition's ID channels store.

    Returns:
        Dash dropdown options.
    """
    return [
        {"label": f"{session.id_name(object_id)} ({object_id})", "value": object_id}
        for object_id in session.stored_ids()
    ]


def override_inputs(session: Session):
    def row(label, control):
        return html.Div(
            [html.Label(label, style={"fontSize": "11px", "flex": "1"}), control],
            style={"display": "flex", "alignItems": "center", "margin": "2px 0"},
        )

    number_style = {"width": "70px", "fontSize": "11px"}
    return [
        row(
            "hue (0 to 1)",
            dcc.Input(
                id="override-hue",
                type="number",
                min=0,
                max=1,
                step=0.01,
                placeholder="stored",
                debounce=True,
                style=number_style,
            ),
        ),
        row(
            "normal tilt (°)",
            dcc.Input(
                id="override-tilt",
                type="number",
                min=0,
                max=180,
                step=5,
                placeholder="stored",
                debounce=True,
                style=number_style,
            ),
        ),
        row(
            "object ID",
            dcc.Dropdown(
                id="override-id",
                options=id_options(session),
                placeholder="stored",
                style={"width": "170px", "fontSize": "11px"},
            ),
        ),
    ]


def build_layout():
    condition = SESSION.condition
    return html.Div(
        style={"fontFamily": "Helvetica, Arial, sans-serif", "fontSize": "13px"},
        children=[
            dcc.Store(id="version", data=0),
            # The view the user orbited to, kept across re-renders until reset.
            dcc.Store(id="camera", data=None),
            dcc.Interval(id="player", interval=120, disabled=True),
            html.Div(
                [
                    html.H3(
                        "Salience propagation explorer (EvidenceGoalGenerator)",
                        style={"margin": "8px 0 2px"},
                    ),
                    html.Div(id="description", children=condition.description),
                ],
                style={"padding": "0 14px"},
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "padding": "8px 14px"},
                children=[
                    # ---------------------------------------------- controls
                    html.Div(
                        style={"width": "290px", "flex": "none"},
                        children=[
                            section(
                                "Condition",
                                dcc.Dropdown(
                                    id="condition",
                                    options=CONDITION_OPTIONS,
                                    value=condition.key,
                                    clearable=False,
                                ),
                                html.Div(
                                    "Synthetic primitives, or learned models (loaded "
                                    "from their checkpoint on first use).",
                                    style={"fontSize": "11px", "color": "#666"},
                                ),
                            ),
                            section(
                                "Displayed channel",
                                dcc.Dropdown(
                                    id="channel",
                                    options=channel_options(SESSION),
                                    value=condition.channels[0],
                                    clearable=False,
                                ),
                            ),
                            section(
                                "Clicking a node",
                                dcc.RadioItems(
                                    id="click-mode",
                                    options=CLICK_MODES,
                                    value="sense",
                                    inline=True,
                                    labelStyle={"marginRight": "8px"},
                                ),
                            ),
                            section(
                                "Sense a landmark (or click any node)",
                                dcc.Dropdown(
                                    id="landmark",
                                    options=list(condition.landmarks),
                                    placeholder="choose a location…",
                                ),
                            ),
                            section(
                                "Sensed features (override)",
                                *override_inputs(SESSION),
                                html.Div(
                                    "By default each channel senses what its nearest "
                                    "node stores. Override to see the goal generator "
                                    "reject input the MLH does not predict.",
                                    style={"fontSize": "11px", "color": "#666"},
                                ),
                            ),
                            section(
                                "MLH age",
                                html.Div(
                                    [
                                        html.Label(
                                            "age of the MLH hypothesis",
                                            style={"fontSize": "11px", "flex": "1"},
                                        ),
                                        dcc.Input(
                                            id="mlh-age",
                                            type="number",
                                            value=SESSION.mlh_age,
                                            min=0,
                                            step=1,
                                            debounce=True,
                                            style={"width": "70px", "fontSize": "11px"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "margin": "2px 0",
                                    },
                                ),
                                html.Div(
                                    "Burst sampling can delete a hypothesis once its "
                                    f"age reaches min_age ({MIN_AGE}); the goal "
                                    "generator only propagates (and samples goals) "
                                    "once the MLH has survived that, i.e. its age "
                                    f"exceeds {MIN_AGE}. "
                                    "Every LM step (sensing, or Step) ages the MLH "
                                    f"by one: set it to {MIN_AGE - 1} and sense twice "
                                    "to see it start propagating.",
                                    style={"fontSize": "11px", "color": "#666"},
                                ),
                            ),
                            section(
                                "Color nodes by",
                                dcc.RadioItems(
                                    id="color-mode",
                                    options=COLOR_MODES,
                                    value="Propagation",
                                    inline=True,
                                    labelStyle={"marginRight": "8px"},
                                ),
                                dcc.Checklist(
                                    id="salience-emphasis",
                                    options=[
                                        {
                                            "label": "Emphasize low salience (invert "
                                            "the salience colors)",
                                            "value": "low",
                                        }
                                    ],
                                    value=[],
                                    style={"fontSize": "11px", "marginTop": "4px"},
                                ),
                                html.Div(
                                    "Salience: bright nodes are salient (likely to be "
                                    "sampled as a goal), grey ones are not; orange "
                                    "is at rest (0.5).",
                                    style={"fontSize": "11px", "color": "#666"},
                                ),
                            ),
                            section(
                                "Overlays",
                                dcc.Checklist(
                                    id="overlays",
                                    options=[
                                        {"label": label, "value": key}
                                        for key, label in OVERLAYS.items()
                                    ],
                                    value=DEFAULT_OVERLAYS,
                                    labelStyle={"display": "block"},
                                ),
                            ),
                            section(
                                "Salience",
                                html.Div(
                                    [
                                        html.Button("Step (relax)", id="step-btn"),
                                        html.Button(
                                            "Step x10",
                                            id="step10-btn",
                                            style={"marginLeft": "6px"},
                                        ),
                                        html.Button(
                                            "Reset",
                                            id="reset-btn",
                                            style={"marginLeft": "6px"},
                                        ),
                                        html.Span(
                                            id="step-counter",
                                            style={
                                                "marginLeft": "10px",
                                                "color": "#555",
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    "Sensing a location is also a step: the salience "
                                    "of the nodes the propagation does not reach "
                                    "relaxes by one step towards rest (0.5), from "
                                    "above or below. Reset restores it to rest "
                                    "everywhere.",
                                    style={"fontSize": "11px", "color": "#666"},
                                ),
                            ),
                            section(
                                "Goals",
                                html.Div(
                                    [
                                        html.Button(
                                            "Sample goal", id="sample-goal-btn"
                                        ),
                                        html.Button(
                                            "Go to goal",
                                            id="go-to-goal-btn",
                                            style={"marginLeft": "6px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="goal-notice",
                                    style={
                                        "fontSize": "11px",
                                        "color": "#b30000",
                                        "marginTop": "4px",
                                    },
                                ),
                                html.Div(
                                    id="goal-choice",
                                    style={"display": "none"},
                                    children=dcc.Dropdown(
                                        id="goal-channel",
                                        options=[],
                                        placeholder="take the goal from…",
                                        style={"fontSize": "11px"},
                                    ),
                                ),
                                html.Div(
                                    "Sample goal: a target sampled in proportion to "
                                    "salience (across all channels, the displayed "
                                    "one switching to the target's), whose "
                                    "hypothetical propagation raises the salience "
                                    "of its part of the target's channel to 1; not "
                                    'an LM step. Or click a node with "Select it '
                                    'as a goal" to take it as the target (not a '
                                    "node with no salience). With all channels "
                                    "displayed, choose which channel's model to "
                                    "take it from when several have a node there. "
                                    "Go to goal: sense the target (one LM step), "
                                    "reducing that salience.",
                                    style={"fontSize": "11px", "color": "#666"},
                                ),
                            ),
                            section("Goal generator parameters", *parameter_inputs()),
                        ],
                    ),
                    # ---------------------------------------------- graph
                    html.Div(
                        style={"flex": "1", "minWidth": "0"},
                        children=[
                            dcc.Graph(
                                id="graph",
                                figure=build_figure(
                                    SESSION,
                                    condition.channels[0],
                                    0,
                                    "Propagation",
                                    DEFAULT_OVERLAYS,
                                ),
                                config={"displayModeBar": True, "scrollZoom": True},
                                style={"height": "760px"},
                            ),
                            html.Div(
                                id="status",
                                style={
                                    "padding": "6px 10px",
                                    "borderRadius": "4px",
                                    "background": "#f0f0f0",
                                    "minHeight": "20px",
                                },
                            ),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "8px",
                                },
                                children=[
                                    html.Button(
                                        "▶ Play", id="play-btn", style={"width": "80px"}
                                    ),
                                    dcc.Dropdown(
                                        id="speed",
                                        options=[
                                            {"label": f"{s} node(s)/tick", "value": s}
                                            for s in SPEEDS
                                        ],
                                        value=5,
                                        clearable=False,
                                        style={"width": "150px"},
                                    ),
                                    html.Div(
                                        dcc.Slider(
                                            id="progress",
                                            min=0,
                                            max=1,
                                            step=1,
                                            value=0,
                                            marks={0: "0"},
                                            tooltip={"placement": "bottom"},
                                            updatemode="drag",
                                        ),
                                        style={"flex": "1", "paddingTop": "10px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # ---------------------------------------------- info
                    html.Div(
                        style={"width": "330px", "flex": "none"},
                        children=[
                            section("Last sensing or goal", html.Div(id="summary")),
                            section("Hovered node", html.Div(id="details")),
                            section(
                                "Log",
                                html.Pre(
                                    id="log",
                                    style={
                                        "fontSize": "11px",
                                        "background": "#fafafa",
                                        "padding": "6px",
                                        "maxHeight": "220px",
                                        "overflowY": "auto",
                                        "whiteSpace": "pre-wrap",
                                    },
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ------------------------------------------------------------------- callbacks


def event_node(event_data, channels: list[str]) -> tuple[str, int] | None:
    """Return the channel and node of a click/hover event on a nodes trace, if any.

    The nodes traces are the first traces, one per displayed channel (see
    `build_figure`), each listing every node of its channel in order, so the
    curve number is the channel's index and the point number the node index.
    (The trace's ``customdata`` cannot be used: Dash looks it up client-side in
    the figure's data, where plotly.py >= 6 has serialized the numpy array as a
    base64 blob rather than a list.)

    Args:
        event_data: The graph's ``clickData`` or ``hoverData``.
        channels: The displayed channels (see `displayed_channels`).

    Returns:
        The (channel, node), or None if the event was not on a node.
    """
    if not event_data or not event_data.get("points"):
        return None
    point = event_data["points"][0]
    curve = point.get("curveNumber")
    if curve is None or curve >= len(channels) or point.get("pointNumber") is None:
        return None
    return channels[curve], int(point["pointNumber"])


def goal_choice_value(channel: str, node: int) -> str:
    return json.dumps([channel, int(node)])


def select_goal_at(
    session: Session, displayed: str, channel: str, node: int
) -> list[dict] | None:
    """Select a clicked node as a goal's target, or offer the channels to take it from.

    With every channel superimposed, a location several channels' graphs have a
    node at could be the target of a goal from any of them, so the user chooses
    which; otherwise the clicked node of the displayed channel is the target.

    Args:
        session: The session.
        displayed: The displayed channel (or ALL_CHANNELS).
        channel: The channel of the clicked node.
        node: The clicked node.

    Returns:
        The options of the channels to choose from, or None if the goal was
        selected (or refused) straight away.
    """
    candidates = session.goal_candidates(channel, node)
    if displayed != ALL_CHANNELS or len(candidates) == 1:
        session.select_goal(channel, node)
        return None
    if not session.goals_allowed():
        return None
    options = []
    for candidate, candidate_node in candidates:
        salience = float(session.salience(candidate)[candidate_node])
        label = session.condition.labels[candidate][candidate_node]
        options.append(
            {
                "label": f"{candidate}: node {candidate_node} ({label}), salience "
                f"{salience:.2f}"
                + (" - no salience, cannot be a goal" if salience <= 0 else ""),
                "value": goal_choice_value(candidate, candidate_node),
                "disabled": salience <= 0,
            }
        )
    if all(option["disabled"] for option in options):
        session.select_goal(channel, node)
        return None
    session.notice = (
        "This location is in several channels' models: choose which to take the "
        "goal from."
    )
    return options


def progress_state(session: Session, channel: str) -> tuple[int, dict]:
    """The slider's range over the displayed channels' last propagations.

    Returns:
        The number of nodes the slider steps through (the most of any displayed
        channel), and its marks.
    """
    records = [
        record
        for shown in displayed_channels(session, channel)
        if (record := last_record(session, shown)) is not None
    ]
    total = max((len(record.node_order) for record in records), default=0)
    if channel != ALL_CHANNELS and records:
        return total, slider_marks(records[0])
    return total, {0: "0", total: str(total)} if total else {0: "0"}


# The buttons whose action needs nothing but the session.
BUTTON_ACTIONS = {
    "step-btn": lambda session: session.step_salience(),
    "step10-btn": lambda session: session.step_salience(10),
    "reset-btn": lambda session: session.reset_salience(),
    "go-to-goal-btn": lambda session: session.go_to_goal(),
}


def handle_action(
    session: Session,
    trigger,
    condition_key,
    channel,
    click_data,
    landmark,
    parameters: dict,
    overrides: dict,
    progress,
    mlh_age=None,
    click_mode="sense",
    goal_choice=None,
) -> dict:
    """Apply a user action to the session, and work out what the UI must update.

    Args:
        session: The session.
        trigger: The ID of the component that triggered the action (None on load).
        condition_key: The selected condition.
        channel: The displayed channel (or ALL_CHANNELS).
        click_data: The graph's click data, if a node was clicked.
        landmark: The selected landmark, if one was.
        parameters: The goal generator's parameters, as entered.
        overrides: The sensed-feature overrides, as entered.
        progress: The slider's current value.
        mlh_age: The MLH age, as entered.
        click_mode: What clicking a node does: "sense" its location, or select
            it as a "goal".
        goal_choice: The channel's node chosen to take a goal from (see
            `goal_choice_value`), if one was.

    Returns:
        The new values of the UI's state: "channel", "condition" (the condition
        actually loaded, which differs from the selected one if loading it
        failed), "channel_options", "landmark_options", "id_options",
        "description" (the latter four None if unchanged), "marks" (the slider's),
        "total" (the number of nodes the slider steps through), "progress" (the
        slider's new value, or None to keep it), "mlh_age" (the MLH's age after
        the action), "goal_options" (the channels to choose a goal's target from,
        or None) and "notice" (why no goal was generated, if one was not).
    """
    session.notice = None
    goal_options = None
    channel_options_ = landmark_options = id_options_ = description = None
    if trigger == "condition" and condition_key != session.condition.key:
        try:
            session.set_condition(condition_key)
        except (FileNotFoundError, ValueError) as error:
            session.log.append(f"Could not load {condition_key}: {error}")
        else:
            channel = session.channels[0]
            channel_options_ = channel_options(session)
            landmark_options = list(session.condition.landmarks)
            id_options_ = id_options(session)
            description = session.condition.description
    elif trigger == "landmark" and landmark is not None:
        session.sense(session.condition.landmarks[landmark])
    elif trigger == "graph" and (
        hit := event_node(click_data, displayed_channels(session, channel))
    ):
        if click_mode == "goal":
            goal_options = select_goal_at(session, channel, *hit)
        else:
            session.sense(session.graphs[hit[0]].pos[hit[1]])
    elif trigger == "goal-channel" and goal_choice:
        session.select_goal(*json.loads(goal_choice))
    elif trigger in BUTTON_ACTIONS:
        BUTTON_ACTIONS[trigger](session)
    elif trigger == "sample-goal-btn":
        if session.sample_goal() is not None and channel != ALL_CHANNELS:
            channel = session.goal[0]
    elif isinstance(trigger, dict) and trigger.get("type") == "param":
        session.set_parameters(**parameters)
    elif isinstance(trigger, str) and trigger.startswith("override-"):
        session.set_overrides(**overrides)
    elif trigger == "mlh-age" and mlh_age is not None:
        session.set_mlh_age(mlh_age)
    if channel not in session.channels and channel != ALL_CHANNELS:
        channel = session.channels[0]
    total, marks = progress_state(session, channel)
    new_progress = None
    if trigger in (
        "graph",
        "landmark",
        "condition",
        "reset-btn",
        "sample-goal-btn",
        "go-to-goal-btn",
        "goal-channel",
    ):
        new_progress = total
    elif trigger == "channel":
        new_progress = min(progress, total) if progress else total
    return {
        "channel": channel,
        "condition": session.condition.key,
        "channel_options": channel_options_,
        "landmark_options": landmark_options,
        "id_options": id_options_,
        "description": description,
        "marks": marks,
        "total": total,
        "progress": new_progress,
        "mlh_age": session.mlh_age,
        "goal_options": goal_options,
        "notice": session.notice,
    }


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("version", "data"),
        Output("progress", "max"),
        Output("progress", "value"),
        Output("progress", "marks"),
        Output("channel", "options"),
        Output("channel", "value"),
        Output("landmark", "options"),
        Output("landmark", "value"),
        Output("description", "children"),
        Output("condition", "value"),
        Output("override-id", "options"),
        Output("override-id", "value"),
        Output("override-hue", "value"),
        Output("override-tilt", "value"),
        Output("graph", "clickData"),
        Output("player", "disabled"),
        Output("play-btn", "children"),
        Output("camera", "data"),
        Output("mlh-age", "value"),
        Output("goal-channel", "options"),
        Output("goal-channel", "value"),
        Output("goal-choice", "style"),
        Output("goal-notice", "children"),
        Input("condition", "value"),
        Input("channel", "value"),
        Input("graph", "clickData"),
        Input("landmark", "value"),
        Input("step-btn", "n_clicks"),
        Input("step10-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("sample-goal-btn", "n_clicks"),
        Input("go-to-goal-btn", "n_clicks"),
        Input({"type": "param", "name": ALL}, "value"),
        Input("override-hue", "value"),
        Input("override-tilt", "value"),
        Input("override-id", "value"),
        Input("mlh-age", "value"),
        Input("goal-channel", "value"),
        State("version", "data"),
        State("progress", "value"),
        State("click-mode", "value"),
    )
    def on_action(
        condition_key,
        channel,
        click_data,
        landmark,
        _step_clicks,
        _step10_clicks,
        _reset_clicks,
        _sample_goal_clicks,
        _go_to_goal_clicks,
        _parameter_values,
        override_hue,
        override_tilt,
        override_id,
        mlh_age,
        goal_choice,
        version,
        progress,
        click_mode,
    ):
        # The parameter inputs are the one pattern-matching input, i.e. a list.
        parameter_inputs = next(
            (entry for entry in ctx.inputs_list if isinstance(entry, list)), []
        )
        parameters = {
            entry["id"]["name"]: entry["value"]
            for entry in parameter_inputs
            if entry.get("value") is not None
        }
        overrides = {
            "hue": override_hue,
            "normal_tilt": override_tilt,
            "object_id": override_id,
        }
        result = handle_action(
            SESSION,
            ctx.triggered_id,
            condition_key,
            channel,
            click_data,
            landmark,
            parameters,
            overrides,
            progress,
            mlh_age,
            click_mode,
            goal_choice,
        )

        def updated(value):
            return no_update if value is None else value

        loaded = result["id_options"] is not None  # A new condition was loaded.
        # The view resets with the salience, and when another object is loaded.
        reset_view = loaded or ctx.triggered_id == "reset-btn"
        return (
            version + 1,
            max(result["total"], 1),
            updated(result["progress"]),
            result["marks"],
            updated(result["channel_options"]),
            result["channel"],
            updated(result["landmark_options"]),
            None,
            updated(result["description"]),
            result["condition"],
            updated(result["id_options"]),
            None if loaded else no_update,
            None if loaded else no_update,
            None if loaded else no_update,
            None,
            True,
            "▶ Play",
            None if reset_view else no_update,
            result["mlh_age"] if result["mlh_age"] != mlh_age else no_update,
            result["goal_options"] or [],
            None,
            {"display": "block", "marginTop": "4px"}
            if result["goal_options"]
            else {"display": "none"},
            result["notice"] or "",
        )

    @app.callback(
        Output("camera", "data", allow_duplicate=True),
        Input("graph", "relayoutData"),
        prevent_initial_call=True,
    )
    def on_relayout(relayout_data):
        # Remember the view whenever the user orbits, pans or zooms the scene.
        if relayout_data and "scene.camera" in relayout_data:
            return relayout_data["scene.camera"]
        return no_update

    @app.callback(
        Output("player", "disabled", allow_duplicate=True),
        Output("play-btn", "children", allow_duplicate=True),
        Output("progress", "value", allow_duplicate=True),
        Input("play-btn", "n_clicks"),
        State("player", "disabled"),
        State("progress", "value"),
        State("progress", "max"),
        prevent_initial_call=True,
    )
    def on_play(_clicks, disabled, progress, maximum):
        if not disabled:
            return True, "▶ Play", no_update
        restart = progress is None or progress >= maximum
        return False, "❚❚ Pause", 0 if restart else no_update

    @app.callback(
        Output("progress", "value", allow_duplicate=True),
        Output("player", "disabled", allow_duplicate=True),
        Output("play-btn", "children", allow_duplicate=True),
        Input("player", "n_intervals"),
        State("progress", "value"),
        State("progress", "max"),
        State("speed", "value"),
        prevent_initial_call=True,
    )
    def on_tick(_n, progress, maximum, speed):
        progress = min((progress or 0) + (speed or 1), maximum)
        if progress >= maximum:
            return progress, True, "▶ Play"
        return progress, no_update, no_update

    @app.callback(
        Output("graph", "figure"),
        Output("status", "children"),
        Output("status", "style"),
        Output("summary", "children"),
        Output("log", "children"),
        Output("step-counter", "children"),
        Input("version", "data"),
        Input("progress", "value"),
        Input("channel", "value"),
        Input("color-mode", "value"),
        Input("salience-emphasis", "value"),
        Input("overlays", "value"),
        State("camera", "data"),
    )
    def render(_version, progress, channel, color_mode, emphasis, overlays, camera):
        session = SESSION
        if channel not in session.channels and channel != ALL_CHANNELS:
            channel = session.channels[0]
        progress = int(progress or 0)
        figure = build_figure(
            session,
            channel,
            progress,
            color_mode,
            overlays or [],
            camera,
            emphasize_low="low" in (emphasis or []),
        )
        text, background = status_text(session, channel, progress)
        style = {
            "padding": "6px 10px",
            "borderRadius": "4px",
            "background": background,
            "minHeight": "20px",
        }
        return (
            figure,
            text,
            style,
            summary_children(session),
            "\n".join(session.log[-14:]),
            f"LM step {session.step}",
        )

    @app.callback(
        Output("details", "children"),
        Input("graph", "hoverData"),
        Input("version", "data"),
        State("channel", "value"),
    )
    def on_hover(hover_data, _version, channel):
        session = SESSION
        if not hover_data or (
            channel not in session.channels and channel != ALL_CHANNELS
        ):
            return "Hover over a node."
        hit = event_node(hover_data, displayed_channels(session, channel))
        if hit is None:
            return no_update
        return details_children(session.node_details(*hit))


def create_app() -> Dash:
    app = Dash(__name__, title="Salience propagation explorer")
    app.layout = build_layout()
    register_callbacks(app)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--condition", choices=CONDITION_KEYS, default=CONDITION_KEYS[0]
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.condition != SESSION.condition.key:
        SESSION.set_condition(args.condition)
    app = create_app()
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
