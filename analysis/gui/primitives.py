# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Synthetic object models ("primitives") that exercise salience propagation.

Each condition is a set of graphs, one per input channel, in the format the
`EvidenceGoalGenerator` reads from a learning module's graph memory: node
positions, a feature matrix and a `feature_mapping` from feature name to its
columns. All conditions share the same slightly curved base surface, so that
their propagations are comparable:

1. `uniform`: the base surface, with a uniform hue.
2. `islands`: with islands of a different hue, of abruptly tilted surface
   normals, and a smooth bump (whose normals change gradually).
3. `small_islands`: as above, with islands too small to reach
   `min_propagation_fraction` of the nodes, so that a propagation from inside
   one is pushed through.
4. `two_sided`: as above, with a far side (inverted normals 1 mm beneath the
   surface) alongside part of the surface, like the inside of a thin wall.
5. `object_ids`: the uniform surface, with two object-ID channels from lower-level
   LMs: "square" everywhere on one, and an island of "logo_A" on the other.
6. `noisy_ids`: as above, with a few "circle" and "logo_B" nodes randomly
   interspersed.
7. `striped_ids`: as above, with "circle" and "logo_B" forming bands that
   separate bands of "square" and "logo_A".

Nodes carry small feature noise (as learned models do), and are jittered off the
regular grid, so that nearest-neighbor ties do not shape the propagations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

SENSOR_CHANNEL = "patch"
ID_CHANNELS = ("learning_module_0", "learning_module_1")
OBJECT_IDS = {"square": 1, "circle": 2, "logo_A": 10, "logo_B": 11}
ID_NAMES = {value: name for name, value in OBJECT_IDS.items()}

SPACING = 0.002
# The base surface: a patch of a cylinder of this radius (m), curving along its
# width, with its normals pointing roughly along +z.
RADIUS = 0.08
WIDTH = 0.09
HEIGHT = 0.06
JITTER = 0.15  # Fraction of the spacing.
NORMAL_NOISE = 0.03  # Radians, per component.
HUE_NOISE = 0.006
BASE_HUE = 0.6
ISLAND_HUE = 0.08
TILT_DEGREES = 60.0
WALL_THICKNESS = 0.001


class Graph:
    """A graph with the interface the goal generator reads from graph memory."""

    def __init__(self, pos, features: dict[str, np.ndarray]):
        self.pos = np.ascontiguousarray(pos, dtype=float)
        self.feature_mapping = {}
        columns = []
        start = 0
        for name, raw_values in features.items():
            values = np.asarray(raw_values, dtype=float).reshape(len(self.pos), -1)
            self.feature_mapping[name] = [start, start + values.shape[1]]
            start += values.shape[1]
            columns.append(values)
        self.x = np.column_stack(columns)
        self._tree = KDTree(self.pos)

    def find_nearest_neighbors(
        self, search_locations, num_neighbors, return_distance=False
    ):
        distances, node_ids = self._tree.query(search_locations, k=num_neighbors)
        return distances if return_distance else node_ids

    def feature(self, name) -> np.ndarray:
        start, end = self.feature_mapping[name]
        return self.x[:, start:end]

    @property
    def normals(self) -> np.ndarray:
        return self.feature("pose_vectors")[:, :3]


@dataclass
class Condition:
    """One synthetic condition.

    Attributes:
        key: Identifier.
        title: Short name.
        description: What the condition exercises, and what to look for.
        graphs: The graph of each input channel.
        labels: For each channel, the region each node belongs to (e.g. "surface",
            "hue island", "far side"), for hover text and the legend.
        landmarks: Named 3D locations worth seeding a propagation from.
        id_names: The name of each object ID the object-ID channels store.
    """

    key: str
    title: str
    description: str
    graphs: dict[str, Graph]
    labels: dict[str, np.ndarray]
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)
    id_names: dict[int, str] = field(default_factory=lambda: dict(ID_NAMES))

    @property
    def channels(self) -> list[str]:
        return list(self.graphs)


# ------------------------------------------------------------------ geometry


class Surface:
    """The base surface, with the per-node quantities islands are built from."""

    def __init__(self, rng, spacing=SPACING):
        num_u = round(WIDTH / spacing) + 1
        num_v = round(HEIGHT / spacing) + 1
        iu, iv = np.meshgrid(np.arange(num_u), np.arange(num_v), indexing="ij")
        self.u = ((iu - (num_u - 1) / 2) * spacing).ravel()
        self.v = ((iv - (num_v - 1) / 2) * spacing).ravel()
        self.u = self.u + rng.normal(0, JITTER * spacing, self.u.shape)
        self.v = self.v + rng.normal(0, JITTER * spacing, self.v.shape)
        angle = self.u / RADIUS
        self.pos = np.column_stack(
            [RADIUS * np.sin(angle), self.v, RADIUS * (np.cos(angle) - 1)]
        )
        self.normals = np.column_stack(
            [np.sin(angle), np.zeros_like(angle), np.cos(angle)]
        )
        self.tangent_u = np.column_stack(
            [np.cos(angle), np.zeros_like(angle), -np.sin(angle)]
        )
        self.tangent_v = np.tile([0.0, 1.0, 0.0], (len(angle), 1))
        self.hue = np.full(len(angle), BASE_HUE)
        self.labels = np.full(len(angle), "surface", dtype=object)

    def __len__(self):
        return len(self.u)

    def disc(self, center_u, center_v, radius) -> np.ndarray:
        return np.hypot(self.u - center_u, self.v - center_v) <= radius

    def band(self, u_min, u_max) -> np.ndarray:
        return (self.u >= u_min) & (self.u < u_max)

    def hue_island(self, center_u, center_v, radius, label="hue island") -> None:
        mask = self.disc(center_u, center_v, radius)
        self.hue[mask] = ISLAND_HUE
        self.labels[mask] = label

    def tilt_island(self, center_u, center_v, radius, label="tilted island") -> None:
        """Abruptly tilt the normals of a disc of nodes about the surface's v axis."""
        mask = self.disc(center_u, center_v, radius)
        rotation = Rotation.from_rotvec(np.radians(TILT_DEGREES) * np.array([0, 1, 0]))
        self.normals[mask] = rotation.apply(self.normals[mask])
        self.labels[mask] = label

    def bump(self, center_u, center_v, sigma, amplitude, label="smooth bump") -> None:
        """Raise a Gaussian bump, whose normals change gradually from node to node."""
        du, dv = self.u - center_u, self.v - center_v
        r2 = du**2 + dv**2
        height = amplitude * np.exp(-r2 / (2 * sigma**2))
        gradient_u = -du / sigma**2 * height
        gradient_v = -dv / sigma**2 * height
        self.pos = self.pos + height[:, None] * self.normals
        normals = (
            self.normals
            - gradient_u[:, None] * self.tangent_u
            - gradient_v[:, None] * self.tangent_v
        )
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        mask = r2 <= (2 * sigma) ** 2
        self.labels[mask] = label

    def sensor_graph(self, rng, far_side=None) -> tuple[Graph, np.ndarray]:
        """Build the sensor-channel graph, with feature noise.

        Args:
            rng: Random generator for the noise.
            far_side: Mask of the nodes to duplicate on the far side of a thin
                wall (inverted normals, beneath the surface), or None.

        Returns:
            The graph and the label of each of its nodes.
        """
        pos, normals, hue, labels = self.pos, self.normals, self.hue, self.labels
        if far_side is not None:
            pos = np.vstack([pos, pos[far_side] - WALL_THICKNESS * normals[far_side]])
            normals = np.vstack([normals, -normals[far_side]])
            hue = np.concatenate([hue, hue[far_side]])
            labels = np.concatenate(
                [labels, np.full(np.count_nonzero(far_side), "far side", dtype=object)]
            )
        normals = normals + rng.normal(0, NORMAL_NOISE, normals.shape)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        hue = np.mod(hue + rng.normal(0, HUE_NOISE, hue.shape), 1.0)
        return Graph(pos, pose_features(normals, hue)), labels


def pose_features(normals, hue=None) -> dict[str, np.ndarray]:
    """Features storing the normals as the first pose vector (and hue as HSV).

    Returns:
        Feature arrays keyed by name.
    """
    num_nodes = len(normals)
    # The other two pose vectors complete an orthonormal frame.
    reference = np.where(
        np.abs(normals[:, [1]]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]
    )
    tangent = np.cross(normals, reference)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(normals, tangent)
    pose_vectors = np.column_stack([normals, tangent, bitangent])
    features = {"pose_vectors": pose_vectors}
    if hue is not None:
        features["hsv"] = np.column_stack([hue, np.ones(num_nodes), np.ones(num_nodes)])
    return features


def id_graph(surface: Surface, mask, object_ids, rng, spacing_factor=1.0) -> Graph:
    """Build an object-ID channel graph on a subset of the surface.

    Args:
        surface: The base surface.
        mask: The surface nodes the channel has nodes at.
        object_ids: The object ID stored at each surface node (only those in the
            mask are used).
        rng: Random generator for subsampling and jitter.
        spacing_factor: Keep about 1 / spacing_factor^2 of the masked nodes, for
            a coarser graph.

    Returns:
        The graph, storing "object_id" and the child object's pose vectors.
    """
    nodes = np.flatnonzero(mask)
    if spacing_factor > 1:
        keep = rng.random(len(nodes)) < 1 / spacing_factor**2
        nodes = nodes[keep]
    pos = surface.pos[nodes] + rng.normal(0, 0.2 * SPACING, (len(nodes), 3))
    normals = surface.normals[nodes]
    return Graph(
        pos,
        {"object_id": np.asarray(object_ids)[nodes], **pose_features(normals)},
    )


# ------------------------------------------------------------------ conditions


def _add_islands(surface: Surface) -> None:
    surface.hue_island(-0.025, 0.012, 0.011)
    surface.tilt_island(0.012, -0.015, 0.009)
    # Gentle enough for neighboring normals (up to two spacings apart) to differ by
    # well under the LM's surface normal tolerance (30 degrees), while the bump's
    # normals span about 25 degrees.
    surface.bump(0.03, 0.015, sigma=0.009, amplitude=0.006)


def _add_small_islands(surface: Surface) -> None:
    surface.hue_island(-0.005, 0.02, 0.0028, label="small hue island")
    surface.hue_island(0.008, 0.012, 0.0028, label="small hue island")
    surface.tilt_island(-0.02, -0.02, 0.0028, label="small tilted island")
    surface.tilt_island(0.035, -0.02, 0.0028, label="small tilted island")


def surface_point(surface: Surface, u, v, depth=0.0) -> np.ndarray:
    """Get the 3D location of the surface's node nearest a point of its grid.

    Args:
        surface: The surface (after any bumps have been raised).
        u: Position along the surface's (curved) width.
        v: Position along its height.
        depth: Distance beneath the surface, along the inverted normal.

    Returns:
        The location.
    """
    node = np.argmin(np.hypot(surface.u - u, surface.v - v))
    return surface.pos[node] - depth * surface.normals[node]


def _landmarks(
    surface: Surface, points: dict[str, tuple[float, float]]
) -> dict[str, np.ndarray]:
    return {
        "surface (center)": surface_point(surface, 0.0, -0.005),
        **{name: surface_point(surface, *uv) for name, uv in points.items()},
    }


def build_condition(key: str, seed: int = 0) -> Condition:
    """Build one of the conditions (see the module docstring).

    Returns:
        The condition.

    Raises:
        ValueError: If the key is not one of CONDITION_KEYS.
    """
    rng = np.random.default_rng(seed)
    surface = Surface(rng)
    if key == "uniform":
        graph, labels = surface.sensor_graph(rng)
        return Condition(
            key,
            "1. Curved surface, uniform hue",
            "A slightly curved surface whose hue is uniform (up to noise) and whose "
            "normals change gradually. A propagation from anywhere should cover the "
            "whole surface: neighboring normals differ by far less than the LM's "
            "surface normal tolerance.",
            {SENSOR_CHANNEL: graph},
            {SENSOR_CHANNEL: labels},
            _landmarks(surface, {}),
        )
    if key in ("islands", "small_islands", "two_sided"):
        _add_islands(surface)
        island_points = {
            "hue island": (-0.025, 0.012),
            "tilted island": (0.012, -0.015),
            "smooth bump": (0.03, 0.015),
        }
        if key == "islands":
            graph, labels = surface.sensor_graph(rng)
            return Condition(
                key,
                "2. Islands of different hue / normals",
                "Three islands: one of a different hue, one of abruptly tilted "
                "normals, and a smooth bump whose normals change gradually. A "
                "propagation from the surface stops at the first two (leaving them "
                "as holes) but flows over the bump; a propagation from inside an "
                "island covers only the island, as it is larger than "
                "min_propagation_fraction of the nodes.",
                {SENSOR_CHANNEL: graph},
                {SENSOR_CHANNEL: labels},
                _landmarks(surface, island_points),
            )
        _add_small_islands(surface)
        island_points |= {
            "small hue island": (-0.005, 0.02),
            "small tilted island": (-0.02, -0.02),
        }
        if key == "small_islands":
            graph, labels = surface.sensor_graph(rng)
            return Condition(
                key,
                "3. With small islands (push-through)",
                "Adds four islands of ~8 nodes each, fewer than "
                "min_propagation_fraction of the graph. A propagation seeded inside "
                "one stops at its edge, then is pushed through onto the surrounding "
                "surface until it has reached min_propagation_fraction of the nodes "
                "(orange/red). A propagation from the surface leaves the small "
                "islands as unreached holes, which keep their salience.",
                {SENSOR_CHANNEL: graph},
                {SENSOR_CHANNEL: labels},
                _landmarks(surface, island_points),
            )
        far_side = surface.band(-0.005, WIDTH)
        graph, labels = surface.sensor_graph(rng, far_side=far_side)
        landmarks = _landmarks(surface, island_points)
        landmarks["far side"] = surface_point(
            surface, 0.03, -0.01, depth=WALL_THICKNESS
        )
        return Condition(
            key,
            "4. With a far side (thin wall)",
            "The right part of the surface has a second side 1 mm beneath it, with "
            "inverted normals, like the inside of a thin wall. Far-side nodes are "
            "among the nearest neighbors of the surface's nodes, but neither stop a "
            "propagation (they are not counted as dissimilar neighbors) nor "
            "receive it, and a propagation is not pushed through to them either.",
            {SENSOR_CHANNEL: graph},
            {SENSOR_CHANNEL: labels},
            landmarks,
        )
    if key in ("object_ids", "noisy_ids", "striped_ids"):
        graph, labels = surface.sensor_graph(rng)
        island = surface.disc(0.012, 0.0, 0.018)
        ids_0 = np.full(len(surface), OBJECT_IDS["square"])
        ids_1 = np.full(len(surface), OBJECT_IDS["logo_A"])
        if key == "noisy_ids":
            noise_0 = rng.random(len(surface)) < 0.03
            noise_1 = rng.random(len(surface)) < 0.05
            ids_0[noise_0] = OBJECT_IDS["circle"]
            ids_1[noise_1] = OBJECT_IDS["logo_B"]
        if key == "striped_ids":
            ids_0[surface.band(-0.03, -0.015)] = OBJECT_IDS["circle"]
            ids_1[surface.band(0.006, 0.018)] = OBJECT_IDS["logo_B"]
        graph_0 = id_graph(surface, np.ones(len(surface), bool), ids_0, rng, 1.5)
        graph_1 = id_graph(surface, island, ids_1, rng)
        graphs = {
            SENSOR_CHANNEL: graph,
            ID_CHANNELS[0]: graph_0,
            ID_CHANNELS[1]: graph_1,
        }
        id_labels = {
            channel: np.array(
                [ID_NAMES[int(i)] for i in g.feature("object_id")[:, 0]], dtype=object
            )
            for channel, g in ((ID_CHANNELS[0], graph_0), (ID_CHANNELS[1], graph_1))
        }
        landmarks = _landmarks(
            surface, {"logo island": (0.012, 0.0), "surface (left)": (-0.03, 0.0)}
        )
        descriptions = {
            "object_ids": (
                "5. Object IDs on two channels",
                "Besides the sensor channel, two channels receive object IDs from "
                'lower-level LMs: learning_module_0 stores "square" at every node, '
                "and learning_module_1 only has nodes on an island, storing "
                '"logo_A". Sensing a location sends each channel the ID stored at '
                "its nearest node, which propagates through the nodes storing that ID "
                "(all of a node's neighbors must store it for it to propagate). "
                "Switch the displayed channel to see each propagation.",
            ),
            "noisy_ids": (
                "6. With noisy object IDs",
                '3% of learning_module_0\'s nodes store "circle" and 5% of '
                'learning_module_1\'s store "logo_B", at random. As a node only '
                "propagates if all of its neighbors store the received ID, each noisy "
                "node stops its whole neighborhood from propagating, fragmenting "
                "the propagation (and a noisy node among the seeds prevents it "
                "entirely).",
            ),
            "striped_ids": (
                "7. With bands of object IDs",
                'A band of "circle" splits learning_module_0 into two bands of '
                '"square", and a band of "logo_B" splits the island on '
                'learning_module_1 into two bands of "logo_A". A propagation of one ID '
                "stops at the band of the other and never jumps across it to the "
                "disjoint band of the same ID.",
            ),
        }
        title, description = descriptions[key]
        return Condition(
            key,
            title,
            description,
            graphs,
            {SENSOR_CHANNEL: labels, **id_labels},
            landmarks,
        )
    raise ValueError(f"Unknown condition {key!r}")


CONDITION_KEYS = [
    "uniform",
    "islands",
    "small_islands",
    "two_sided",
    "object_ids",
    "noisy_ids",
    "striped_ids",
]
