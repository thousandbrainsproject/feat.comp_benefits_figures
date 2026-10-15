# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Drive an `EvidenceGoalGenerator`'s salience propagation on one condition.

A `Session` holds one condition (a synthetic primitive, see `primitives`, or a
learned object, see `learned_models`) and one goal generator whose parent LM is
faked to serve the condition's graphs, with the most likely hypothesis (MLH)
placed wherever the user last sensed. Sensing a location is one step of the LM:
every channel receives the features (or object ID) stored at its node nearest
the location (unless overridden, to see the goal generator reject input the MLH
does not predict), the goal generator propagates from there, reducing the
salience of the nodes it reaches, and the salience of every other node recovers
by one step. The `PropagationRecord`s the goal generator logs are kept for
stepping through the propagations node by node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from analysis.gui import learned_models, primitives
from analysis.gui.primitives import Condition, Graph
from tbp.monty.frameworks.models.goal_generation import (
    EvidenceGoalGenerator,
    PropagationRecord,
)

MAX_MATCH_DISTANCE = 0.01
# What the sensed features can be overridden with (None: as stored at the node).
OVERRIDES = {"hue": None, "normal_tilt": None, "object_id": None}
# The goal generator's parameters the GUI exposes, with their defaults.
PARAMETERS = {
    "num_propagation_neighbors": 6,
    "num_feature_propagation_neighbors": 10,
    "salience_recovery_steps": 50,
    "salience_reduction": 0.2,
    "salience_reduction_falloff": 0.5,
    "max_hue_difference": 0.05,
    "max_normal_angle": 40.0,
    "min_far_side_angle": 135.0,
    "max_dissimilar_neighbors": 2,
    "min_propagation_fraction": 0.05,
}


class FakeParentLM:
    """Serve a condition's graphs through the parent-LM interface the GSG uses."""

    def __init__(self, graphs: dict[str, Graph]):
        self.graphs = graphs
        self.learning_module_id = "gui_lm"
        self.max_match_distance = MAX_MATCH_DISTANCE
        self.mlh = {
            "graph_id": "primitive",
            "mlh_id": 0,
            "location": np.zeros(3),
            "rotation": Rotation.identity(),
        }
        self.buffer = SimpleNamespace(update_stats=lambda *_args, **_kwargs: None)

    def _get_current_mlh(self):
        return self.mlh

    def get_all_known_object_ids(self):
        return ["primitive"]

    def get_input_channels_in_graph(self, _graph_id):
        return list(self.graphs)

    def get_graph(self, _graph_id, input_channel=None):
        return self.graphs if input_channel is None else self.graphs[input_channel]


@dataclass
class Sensing:
    """One sensing of a location: what each channel received, and what propagated.

    Attributes:
        step: The LM step the sensing took place on.
        location: The sensed location (the MLH location).
        percepts: For each channel, a summary of what it received.
        records: The propagation on each channel that propagated.
        no_propagation: For each channel that did not propagate, why.
        salience_before: For each channel, the salience of its nodes before the
            propagation (for showing the salience as a propagation progresses).
    """

    step: int
    location: np.ndarray
    percepts: dict[str, str]
    records: dict[str, PropagationRecord]
    no_propagation: dict[str, str]
    salience_before: dict[str, np.ndarray] = field(default_factory=dict)


class Session:
    """A condition, a goal generator propagating on it, and its history."""

    def __init__(self, condition_key: str, parameters: dict | None = None):
        self.parameters = {**PARAMETERS, **(parameters or {})}
        self.overrides = dict(OVERRIDES)
        self.step = 0
        self.sensing: Sensing | None = None
        self.log: list[str] = []
        self.set_condition(condition_key)

    # ------------------------------------------------------------- state

    def set_condition(self, condition_key: str) -> None:
        """Load a condition (a primitive or a learned object), fresh.

        Propagates `learned_models.build_condition`'s FileNotFoundError (the
        checkpoint is not available) and ValueError (no such condition).
        """
        if learned_models.is_learned(condition_key):
            condition = learned_models.build_condition(condition_key)
        else:
            condition = primitives.build_condition(condition_key)
        self.condition: Condition = condition
        self.lm = FakeParentLM(self.condition.graphs)
        self.gsg = self._make_gsg()
        self.step = 0
        self.sensing = None
        self.overrides = dict(OVERRIDES)
        self.log = [f"Loaded {self.condition.title}"]

    def set_overrides(self, **overrides) -> None:
        """Override what the channels sense (None: the features stored at the node).

        Args:
            **overrides: "hue" (0 to 1), "normal_tilt" (degrees the sensed normal
                is tilted away from the stored one) and/or "object_id" (the ID
                every object-ID channel receives).
        """
        changed = {
            name: value
            for name, value in overrides.items()
            if name in OVERRIDES and value != self.overrides[name]
        }
        if not changed:
            return
        self.overrides.update(changed)
        self.log.append(
            "Sensed-feature overrides: "
            + ", ".join(
                f"{name}={'stored' if value is None else value}"
                for name, value in changed.items()
            )
        )

    def set_parameters(self, **parameters) -> None:
        """Change the goal generator's parameters, keeping the salience."""
        changed = {
            name: value
            for name, value in parameters.items()
            if name in PARAMETERS and value != self.parameters[name]
        }
        if not changed:
            return
        self.parameters.update(changed)
        salience = self.gsg._salience
        self.gsg = self._make_gsg()
        self.gsg._salience = salience
        self.log.append(
            "Parameters: "
            + ", ".join(f"{name}={value}" for name, value in changed.items())
        )

    def _make_gsg(self) -> EvidenceGoalGenerator:
        parameters = dict(self.parameters)
        if not parameters["min_propagation_fraction"]:
            parameters["min_propagation_fraction"] = None
        gsg = EvidenceGoalGenerator(**parameters)
        gsg.parent_lm = self.lm
        gsg.reset()
        return gsg

    @property
    def graphs(self) -> dict[str, Graph]:
        return self.condition.graphs

    @property
    def channels(self) -> list[str]:
        return self.condition.channels

    def is_id_channel(self, channel) -> bool:
        return "object_id" in self.graphs[channel].feature_mapping

    @property
    def id_channels(self) -> list[str]:
        return [channel for channel in self.channels if self.is_id_channel(channel)]

    def id_name(self, object_id) -> str:
        """Name an object ID.

        Returns:
            The name, or the ID itself if its name is unknown.
        """
        return str(self.condition.id_names.get(int(object_id), int(object_id)))

    def stored_ids(self) -> list[int]:
        """Get the object IDs the condition's object-ID channels store.

        Returns:
            The IDs, sorted.
        """
        ids: set[int] = set()
        for channel in self.id_channels:
            ids.update(
                np.unique(self.graphs[channel].feature("object_id")[:, 0])
                .astype(int)
                .tolist()
            )
        return sorted(ids)

    # ----------------------------------------------------------- actions

    def reset_salience(self) -> None:
        """Restore every node's salience to 1 and forget the last sensing."""
        self.gsg._salience = {}
        self.sensing = None
        self.log.append(f"Step {self.step}: salience reset")

    def step_salience(self, num_steps: int = 1) -> None:
        """LM steps without sensing anything: the salience recovers."""
        for _ in range(num_steps):
            self.step += 1
            self.gsg._recover_salience()
        self.log.append(
            f"Step {self.step}: salience recovered"
            + (f" ({num_steps} steps)" if num_steps > 1 else "")
        )

    def sense(self, location) -> Sensing:
        """Sense a location: one LM step, with input to every channel.

        Each channel receives the features (or object ID) stored at its node nearest
        the location, as if the MLH were exactly right; the goal generator
        propagates from the nodes nearest the location, reducing the salience of
        the nodes it reaches, and the salience of the other nodes recovers by one
        step.

        Returns:
            What was sensed and what propagated.
        """
        location = np.asarray(location, dtype=float)
        self.step += 1
        self.lm.mlh = {**self.lm.mlh, "location": location}
        salience_before = {
            channel: self.salience(channel).copy() for channel in self.channels
        }
        percepts = [self._percept(channel, location) for channel in self.channels]
        self.gsg._update_salience(percepts)
        records = {
            record.input_channel: record for record in self.gsg.propagation_records
        }
        self.sensing = Sensing(
            step=self.step,
            location=location,
            percepts={
                channel: self._describe_percept(percept)
                for channel, percept in zip(self.channels, percepts)
            },
            records=records,
            no_propagation={
                channel: self._why_no_propagation(channel, location, percept)
                for channel, percept in zip(self.channels, percepts)
                if channel not in records
            },
            salience_before=salience_before,
        )
        propagated = ", ".join(
            f"{channel}: {len(record.node_order)} nodes"
            + (f" ({record.stage.max()} push-through(s))" if record.min_size else "")
            for channel, record in records.items()
        )
        self.log.append(
            f"Step {self.step}: sensed; propagated on {propagated or 'no channel'}"
        )
        return self.sensing

    def nearest_node(self, channel, location) -> int:
        return int(
            self.graphs[channel].find_nearest_neighbors(
                np.asarray(location, dtype=float), num_neighbors=1
            )
        )

    # ------------------------------------------------------------ queries

    def salience(self, channel, progress=None) -> np.ndarray:
        """Get the salience (0 to 1) of each node of a channel.

        Args:
            channel: The channel.
            progress: Number of nodes of the last propagation on the channel to
                count as reached, to show the salience as the propagation
                progresses; None for the current salience.

        Returns:
            The salience of each node (a copy).
        """
        num_nodes = len(self.graphs[channel].pos)
        current = self.gsg._get_salience("primitive", channel, num_nodes).copy()
        record = None if self.sensing is None else self.sensing.records.get(channel)
        if progress is None or record is None or progress >= len(record.node_order):
            return current
        # Mid-propagation: the nodes reached so far show their actual salience;
        # the ones not reached yet what they had before the propagation, plus the
        # recovery of the steps since (none on the step they were reached).
        before = self.sensing.salience_before[channel]
        if len(before) != num_nodes:
            return current
        pending = record.node_order[progress:]
        recovered = (self.step - self.sensing.step) / self.gsg.salience_recovery_steps
        current[pending] = np.minimum(before[pending] + recovered, 1.0)
        return current

    def edge_matrices(self, channel):
        """Get the propagation and surface (or neighbor) matrices of a channel.

        Returns:
            Sparse matrices with the edges a node propagates along, and the edges
            to all of its neighbors on the same side of the surface (for an
            object-ID channel: all of its neighbors, and None, as those
            propagations are not pushed through), see
            `EvidenceGoalGenerator._get_propagation_order`.
        """
        graph = self.graphs[channel]
        if self.is_id_channel(channel):
            return None, self.gsg._get_neighbor_matrix("primitive", channel, graph)
        features = self.gsg._get_propagation_features(graph)
        return self.gsg._get_similar_neighbors("primitive", channel, graph, features)

    def node_details(self, channel, node) -> dict[str, str]:
        """Describe a node: its features, neighbors, salience and propagation status.

        Returns:
            Labeled values, in display order.
        """
        graph = self.graphs[channel]
        gsg = self.gsg
        details = {
            "node": f"{node} on {channel}",
            "region": str(self.condition.labels[channel][node]),
        }
        if self.is_id_channel(channel):
            object_id = int(graph.feature("object_id")[node, 0])
            details["object ID"] = f"{self.id_name(object_id)} ({object_id})"
            neighbors = gsg._get_neighbor_matrix("primitive", channel, graph)
            targets = neighbors.indices[
                neighbors.indptr[node] : neighbors.indptr[node + 1]
            ]
            other = graph.feature("object_id")[targets, 0] != object_id
            details["neighbors"] = (
                f"{len(targets)}, of which {other.sum()} store another ID"
            )
            details["propagates"] = (
                "yes"
                if len(targets) and not other.any()
                else "no (not all neighbors store its ID)"
                if len(targets)
                else "no (no neighbors)"
            )
        else:
            if "hsv" in graph.feature_mapping:
                hsv = graph.feature("hsv")[node]
                details["hue"] = f"{hsv[0]:.3f} (s {hsv[1]:.2f}, v {hsv[2]:.2f})"
            normal = graph.normals[node]
            details["normal"] = "(" + ", ".join(f"{c:+.2f}" for c in normal) + ")"
            features = gsg._get_propagation_features(graph)
            all_neighbors = gsg._get_neighbor_matrix(
                "primitive",
                channel,
                graph,
                num_neighbors=gsg.num_feature_propagation_neighbors,
            )
            targets = all_neighbors.indices[
                all_neighbors.indptr[node] : all_neighbors.indptr[node + 1]
            ]
            reference = {name: values[node] for name, values in features.items()}
            far = gsg._far_side(features, targets, reference)
            similar = gsg._similar_features(features, targets[~far], reference)
            num_dissimilar = int((~similar).sum())
            details["neighbors"] = (
                f"{len(targets)}: {far.sum()} on the far side (ignored), "
                f"{similar.sum()} similar, {num_dissimilar} dissimilar"
            )
            if len(targets) == far.sum():
                details["propagates"] = "no (no neighbors on this side of the surface)"
            elif num_dissimilar > gsg.max_dissimilar_neighbors:
                details["propagates"] = (
                    f"no ({num_dissimilar} > max_dissimilar_neighbors="
                    f"{gsg.max_dissimilar_neighbors})"
                )
            else:
                details["propagates"] = f"yes, to its {similar.sum()} similar neighbors"
        salience = float(self.salience(channel)[node])
        if salience >= 1:
            details["salience"] = "1.00 (full)"
        else:
            remaining = int(np.ceil((1 - salience) * gsg.salience_recovery_steps))
            details["salience"] = f"{salience:.2f}" + (
                " (never sampled)" if salience <= 0 else ""
            )
            details["recovery"] = f"back to 1 in {remaining} step(s)"
        if self.sensing is not None and channel in self.sensing.records:
            record = self.sensing.records[channel]
            position = np.flatnonzero(record.node_order == node)
            if len(position):
                position = int(position[0])
                stage = int(record.stage[position])
                details["last propagation"] = (
                    f"reached #{position + 1} of {len(record.node_order)}, "
                    f"{record.hops[position]} hop(s) "
                    + ("from the seeds" if stage == 0 else f"into push-through {stage}")
                    + (" (seed)" if node in record.seeds else "")
                    + (" (sink)" if node in record.sinks else "")
                    + f"; salience reduced by {record.reduction[position]:.2f}"
                )
            else:
                details["last propagation"] = "not reached"
        return details

    # ------------------------------------------------------------ private

    def _percept(self, channel, location):
        """What a channel senses at a location: its nearest node's features.

        The features are copied from the node, then overridden as requested: the
        hue, the normal (tilted away from the node's normal, along with the rest
        of the pose) and/or the object ID.

        Returns:
            A percept, as the goal generator reads it.
        """
        graph = self.graphs[channel]
        node = self.nearest_node(channel, location)
        features = {
            name: graph.feature(name)[node].copy() for name in graph.feature_mapping
        }
        overridden = []
        pose = features["pose_vectors"].reshape(3, 3)
        tilt = self.overrides["normal_tilt"]
        if tilt:
            normal = pose[0] / max(np.linalg.norm(pose[0]), 1e-12)
            # Tilt about an axis perpendicular to the normal (the first curvature
            # direction if the pose defines one, else any perpendicular).
            axis = pose[1] - (pose[1] @ normal) * normal
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(normal, [1.0, 0.0, 0.0])
                if np.linalg.norm(axis) < 1e-6:
                    axis = np.cross(normal, [0.0, 1.0, 0.0])
            axis = axis / np.linalg.norm(axis)
            pose = Rotation.from_rotvec(np.radians(tilt) * axis).apply(pose)
            overridden.append(f"normal tilted {tilt:g}°")
        hue = self.overrides["hue"]
        if hue is not None and "hsv" in features:
            features["hsv"][0] = hue
            overridden.append(f"hue {hue:g}")
        object_id = self.overrides["object_id"]
        if object_id is not None and "object_id" in features:
            features["object_id"][0] = object_id
            overridden.append(f"object ID {self.id_name(object_id)}")
        return SimpleNamespace(
            sender_type="LM" if "object_id" in features else "SM",
            sender_id=channel,
            morphological_features={"pose_vectors": pose},
            non_morphological_features={
                name: value
                for name, value in features.items()
                if name != "pose_vectors"
            },
            overridden=overridden,
        )

    def _describe_percept(self, percept) -> str:
        features = percept.non_morphological_features
        if "object_id" in features:
            text = f"object ID {self.id_name(features['object_id'][0])}"
        else:
            normal = percept.morphological_features["pose_vectors"][0]
            text = "normal (" + ", ".join(f"{c:+.2f}" for c in normal) + ")"
            if "hsv" in features:
                text = f"hue {features['hsv'][0]:.2f}, " + text
        if percept.overridden:
            text += " [overridden: " + ", ".join(percept.overridden) + "]"
        return text

    def _why_no_propagation(self, channel, location, percept) -> str:
        graph = self.graphs[channel]
        gsg = self.gsg
        seeds, distances = gsg._get_nodes_nearest_mlh(graph, location)
        nearby = distances <= self.lm.max_match_distance
        if not nearby.any():
            return (
                f"no node within max_match_distance ({self.lm.max_match_distance} m) "
                f"of the sensed location (nearest: {distances.min():.3f} m)"
            )
        if self.is_id_channel(channel):
            received = int(percept.non_morphological_features["object_id"][0])
            stored = graph.feature("object_id")[seeds, 0].astype(int)
            if not np.any(stored[nearby] == received):
                return "the received ID is not stored by any nearby node"
            others = sorted({self.id_name(i) for i in stored if i != received})
            return (
                f"not all of the {len(seeds)} nearest nodes store the received ID "
                f"(some store {', '.join(map(str, others))})"
            )
        features = gsg._get_propagation_features(graph)
        sensed = gsg._get_sensed_propagation_features(percept, self.lm.mlh["rotation"])
        if not features:
            return "the channel stores no features to compare"
        if not gsg._similar_features(features, seeds[nearby], sensed).any():
            return "the sensed features are not similar to any nearby node's"
        return "no seeds (all nearest nodes are on the far side of the surface)"
