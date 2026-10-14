# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Learned object models as conditions for the spreading explorer.

Two sets of pretrained models are offered, both read from the checkpoints that
analysis/scripts/visualize_inhibition_spreading.py uses:

- "ycb": the 10 distinct YCB objects (mug, bowl, ...) learned by the surface
  agent, with one sensor channel.
- "comp": the compositional objects (cube, disk, cylinder, sphere, mug, plain or
  with a TBP or Numenta logo sticker) learned by the top-level LM (LM 2) of the
  compositional Monty, with a sensor channel and two object-ID channels (one per
  child LM, storing the IDs of the base objects and of the logos).

A checkpoint is loaded (and kept) the first time one of its objects is selected.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import KDTree

from analysis.gui.primitives import Condition, Graph
from analysis.scripts.visualize_inhibition_spreading import (
    MODELS,
    label_mug_parts,
    resolve_checkpoint,
)

PREFIX = "learned"
MODEL_SETS = {
    "ycb": {
        "title": "YCB, surface agent",
        "path": MODELS["surf_agent_77obj"].path,
        "lm": MODELS["surf_agent_77obj"].lm,
        "objects": MODELS["surf_agent_77obj"].objects,
    },
    "comp": {
        "title": "Compositional, LM 2",
        "path": MODELS["comp_models_mujoco"].path,
        "lm": MODELS["comp_models_mujoco"].lm,
        "objects": [
            "023_mug",
            "024_mug_tbp_horz",
            "025_mug_tbp_vert",
            "026_mug_numenta_horz",
            "027_mug_numenta_vert",
            "001_cube",
            "002_cube_tbp",
            "004_cube_numenta",
            "006_disk",
            "007_disk_tbp",
            "009_disk_numenta",
            "011_cylinder",
            "012_cylinder_tbp_horz",
            "013_cylinder_tbp_vert",
            "014_cylinder_numenta_horz",
            "015_cylinder_numenta_vert",
            "016_sphere",
            "017_sphere_tbp",
            "019_sphere_numenta",
        ],
    },
}
EXTREMES = {
    "top (+y)": (1, 1),
    "bottom (-y)": (1, -1),
    "front (+z)": (2, 1),
    "back (-z)": (2, -1),
    "left (-x)": (0, -1),
    "right (+x)": (0, 1),
}


def condition_key(model_set: str, object_name: str) -> str:
    return f"{PREFIX}:{model_set}:{object_name}"


def is_learned(key: str) -> bool:
    return key.startswith(PREFIX + ":")


def checkpoint_path(model_set: str) -> Path | None:
    """Find the checkpoint of a model set.

    Returns:
        The checkpoint, or None if it is not on this machine.
    """
    try:
        return resolve_checkpoint(Path(MODEL_SETS[model_set]["path"]))
    except FileNotFoundError:
        return None


def condition_options() -> list[dict]:
    """Dropdown options for every learned object, disabled if not available.

    Returns:
        Dash dropdown options.
    """
    options = []
    for model_set, spec in MODEL_SETS.items():
        available = checkpoint_path(model_set) is not None
        for object_name in spec["objects"]:
            label = f"{spec['title']} · {object_name}"
            if not available:
                label += " (checkpoint not found)"
            options.append(
                {
                    "label": label,
                    "value": condition_key(model_set, object_name),
                    "disabled": not available,
                }
            )
    return options


@functools.lru_cache(maxsize=None)
def load_model_set(model_set: str) -> tuple[dict[str, dict[str, Graph]], dict]:
    """Load the graphs of a model set's LM, and the names of the IDs it stores.

    Returns:
        The graph of each input channel of each object, and the name of each
        object ID (the IDs the LM's object-ID channels store are those of the
        objects its child LMs know, i.e. the sum of their names' characters).

    Raises:
        FileNotFoundError: If the checkpoint is not on this machine.
    """
    spec = MODEL_SETS[model_set]
    checkpoint = checkpoint_path(model_set)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found under {spec['path']}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lm_dict = state["lm_dict"]
    lm_id = spec["lm"]
    memory = lm_dict[lm_id if lm_id in lm_dict else str(lm_id)]["graph_memory"]
    graphs = {
        graph_id: {
            channel: Graph(
                np.asarray(graph.pos),
                {
                    name: np.asarray(graph.x)[:, start:end]
                    for name, (start, end) in graph.feature_mapping.items()
                },
            )
            for channel, graph in channels.items()
        }
        for graph_id, channels in memory.items()
    }
    id_names = {}
    for other_id, other_lm in lm_dict.items():
        if str(other_id) != str(lm_id):
            for name in other_lm["graph_memory"]:
                id_names[sum(ord(c) for c in name)] = name
    return graphs, id_names


def sensor_channels(graphs: dict[str, Graph]) -> list[str]:
    return [
        channel
        for channel, graph in graphs.items()
        if "object_id" not in graph.feature_mapping
        and "pose_vectors" in graph.feature_mapping
    ]


def label_nodes(object_name: str, graphs: dict[str, Graph]) -> dict[str, np.ndarray]:
    """Label the nodes of an object's graphs by part.

    Mugs are labeled by geometry (side, rim, bottom, handle, inside); the nodes
    of other objects all get the object's name. Object-ID channels take the label
    of the nearest sensor node.

    Returns:
        One label per node, per channel.
    """
    sensors = sensor_channels(graphs)
    labels = {}
    for channel in sensors:
        if "mug" in object_name:
            labels[channel] = label_mug_parts(
                {object_name: graphs}, object_name, channel
            )
        else:
            labels[channel] = np.full(
                len(graphs[channel].pos), object_name, dtype=object
            )
    if sensors:
        tree = KDTree(graphs[sensors[0]].pos)
        reference = labels[sensors[0]]
    for channel, graph in graphs.items():
        if channel not in labels:
            labels[channel] = (
                reference[tree.query(graph.pos)[1]]
                if sensors
                else np.full(len(graph.pos), object_name, dtype=object)
            )
    return labels


def landmarks(graph: Graph, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Locations worth seeding from: the center of each part, and the extremes.

    Returns:
        Named node positions.
    """
    pos = graph.pos
    points = {}
    for label in sorted(set(labels)):
        if label == "other":
            continue
        members = np.flatnonzero(labels == label)
        centroid = pos[members].mean(axis=0)
        # The member node nearest the centroid, which for a hollow part (e.g. a
        # mug's side) is not the centroid itself.
        points[f"{label} (center)"] = pos[
            members[np.argmin(np.linalg.norm(pos[members] - centroid, axis=1))]
        ]
    for name, (axis, sign) in EXTREMES.items():
        points[name] = pos[np.argmax(sign * pos[:, axis])]
    return points


def build_condition(key: str) -> Condition:
    """Build the condition of a learned object.

    Returns:
        The condition.

    Raises:
        ValueError: If the key does not name a learned object of the checkpoint.
            (`load_model_set` raises FileNotFoundError if the checkpoint is not
            on this machine.)
    """
    try:
        _, model_set, object_name = key.split(":", 2)
        spec = MODEL_SETS[model_set]
    except (ValueError, KeyError) as error:
        raise ValueError(f"Not a learned object: {key}") from error
    graphs_by_object, id_names = load_model_set(model_set)
    if object_name not in graphs_by_object:
        raise ValueError(f"{object_name} is not in the {spec['title']} checkpoint")
    graphs = graphs_by_object[object_name]
    labels = label_nodes(object_name, graphs)
    sensors = sensor_channels(graphs)
    reference = sensors[0] if sensors else next(iter(graphs))
    channels = ", ".join(
        f"{channel} ({len(graph.pos)} nodes)" for channel, graph in graphs.items()
    )
    return Condition(
        key,
        f"{object_name} ({spec['title']})",
        f"A learned model, from LM {spec['lm']} of {spec['path']}. "
        f"Channels: {channels}. The MLH is placed exactly at the sensed node, so "
        "each channel receives the features (or object ID) its nearest node "
        "stores, unless overridden.",
        graphs,
        labels,
        landmarks(graphs[reference], labels[reference]),
        id_names,
    )
