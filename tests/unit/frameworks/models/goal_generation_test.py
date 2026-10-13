# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Tests for the EvidenceGoalGenerator's hypothesis-testing policy.

The tests use mock graphs and a mocked parent LM to exercise the selection of
unexplained locations to test, the spreading of inhibition from received object
IDs and from sensory input, and the transform of a target location on the MLH
model into a Goal for the motor system.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as nptest
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.goal_generation import EvidenceGoalGenerator

SENSOR_CHANNEL = "patch"
TOP_ID = "object_a"
SECOND_ID = "object_b"


class FakeGraph:
    """Minimal stand-in for a learned object graph (one input channel).

    Mirrors the parts of the object-model interface used by the GSG: `pos`, `x`,
    `feature_mapping` and `find_nearest_neighbors`.
    """

    def __init__(self, pos, features=None):
        self.pos = np.asarray(pos, dtype=float)
        self.feature_mapping = {}
        columns = []
        num_columns = 0
        for name, raw_values in (features or {}).items():
            values = np.asarray(raw_values, dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            self.feature_mapping[name] = [num_columns, num_columns + values.shape[1]]
            num_columns += values.shape[1]
            columns.append(values)
        self.x = np.column_stack(columns) if columns else np.zeros((len(self.pos), 0))
        self._tree = KDTree(self.pos)

    def find_nearest_neighbors(
        self,
        search_locations,
        num_neighbors,
        return_distance=False,
    ):
        distances, nearest_node_ids = self._tree.query(
            search_locations, k=num_neighbors
        )
        if return_distance:
            return distances
        return nearest_node_ids


def identity_mlh(graph_id):
    """An MLH with identity rotation at the origin (transform is a no-op).

    Returns:
        The MLH dict.
    """
    return {
        "graph_id": graph_id,
        "mlh_id": 0,
        "location": np.zeros(3),
        "rotation": Rotation.identity(),
    }


def gsg_with_graphs(graphs, **gsg_kwargs) -> EvidenceGoalGenerator:
    """Build an EvidenceGoalGenerator around a mocked parent LM.

    Args:
        graphs: Nested dict of graph_id -> input_channel -> FakeGraph.
        **gsg_kwargs: Forwarded to the EvidenceGoalGenerator constructor.

    Returns:
        The GSG, with its parent LM mocked to serve the given graphs and to
        report object_a as the MLH.
    """
    lm = MagicMock()
    lm.learning_module_id = "learning_module_2"
    lm.max_match_distance = 0.01
    lm.buffer.get_first_sensory_input_channel.return_value = SENSOR_CHANNEL
    lm._get_current_mlh.return_value = identity_mlh(TOP_ID)
    lm.get_graph.side_effect = lambda graph_id, input_channel=None: (
        graphs[graph_id] if input_channel is None else graphs[graph_id][input_channel]
    )
    lm.get_input_channels_in_graph.side_effect = lambda graph_id: list(
        graphs[graph_id].keys()
    )
    gsg = EvidenceGoalGenerator(**gsg_kwargs)
    gsg.parent_lm = lm
    return gsg


def make_ctx() -> RuntimeContext:
    return RuntimeContext(rng=np.random.RandomState(42))


# A small flat point cloud used for spatially-identical sensor channels.
BASE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.05, 0.05, 0.0],
        [0.0, 0.05, 0.0],
    ]
)


class TargetLocInfoTest(unittest.TestCase):
    def test_lm_channel_target_uses_surface_normal_of_nearest_sensor_node(
        self,
    ) -> None:
        # Distinct normals per sensor node, so we can verify which one is used
        normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        pose_vectors = np.column_stack([normals, np.zeros((4, 3)), np.zeros((4, 3))])
        sensor = FakeGraph(BASE_POINTS, {"pose_vectors": pose_vectors})
        # An LM-channel node sitting just next to sensor node 2
        lm_graph = FakeGraph(
            np.array([[0.051, 0.049, 0.0]]), {"object_id": np.array([1])}
        )
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor, "learning_module_0": lm_graph},
            SECOND_ID: {SENSOR_CHANNEL: sensor, "learning_module_0": lm_graph},
        }
        gsg = gsg_with_graphs(graphs)

        target_info = gsg._get_target_loc_info(
            target_loc_id=0, input_channel="learning_module_0"
        )

        nptest.assert_allclose(target_info["target_loc"], [0.051, 0.049, 0.0])
        nptest.assert_allclose(
            target_info["target_surface_normal"],
            normals[2],
            err_msg="Surface normal should come from the nearest sensor-channel "
            "node, not the LM-channel node.",
        )

    def test_sensor_channel_target_uses_its_own_pose_vectors(self) -> None:
        normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        pose_vectors = np.column_stack([normals, np.zeros((4, 3)), np.zeros((4, 3))])
        sensor = FakeGraph(BASE_POINTS, {"pose_vectors": pose_vectors})
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor},
            SECOND_ID: {SENSOR_CHANNEL: sensor},
        }
        gsg = gsg_with_graphs(graphs)

        target_info = gsg._get_target_loc_info(
            target_loc_id=3, input_channel=SENSOR_CHANNEL
        )

        nptest.assert_allclose(target_info["target_loc"], BASE_POINTS[3])
        nptest.assert_allclose(target_info["target_surface_normal"], normals[3])


WHEEL_ID = 573
BODY_ID = 1099
LOGO_ID = 1096
NODE_SPACING = 0.002


def car_line_graph():
    """A line of nodes: a wheel, then the car body, then a second (disjoint) wheel.

    Returns:
        The LM-channel graph, with 10 wheel nodes, 15 body nodes and 10 wheel nodes.
    """
    object_ids = np.array([WHEEL_ID] * 10 + [BODY_ID] * 15 + [WHEEL_ID] * 10)
    positions = np.zeros((len(object_ids), 3))
    positions[:, 0] = np.arange(len(object_ids)) * NODE_SPACING
    return FakeGraph(positions, {"object_id": object_ids})


def logo_line_graph():
    positions = np.zeros((5, 3))
    positions[:, 0] = np.arange(5) * NODE_SPACING
    positions[:, 1] = 0.05
    return FakeGraph(positions, {"object_id": np.full(5, LOGO_ID)})


# Surface normal (first row) perpendicular to the line the car's nodes lie along.
LINE_POSE_VECTORS = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def oriented_sensor_graph(positions):
    pv = LINE_POSE_VECTORS.flatten()
    return FakeGraph(positions, {"pose_vectors": np.tile(pv, (len(positions), 1))})


def lm_percept(sender_id, object_id):
    return MagicMock(
        sender_type="LM",
        sender_id=sender_id,
        non_morphological_features={"object_id": object_id},
    )


class EvidenceGoalGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.car = car_line_graph()
        self.logo = logo_line_graph()
        # The sensor channel has fewer nodes than any child object, so that child
        # objects are ranked first when selecting a target.
        self.sensor = oriented_sensor_graph(self.car.pos[::9])
        self.graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: self.sensor,
                "learning_module_0": self.car,
                "learning_module_1": self.logo,
            }
        }
        self.mlh = self.mlh_at_node(4)
        self.sensor_percept = MagicMock(
            sender_type="SM",
            sender_id=SENSOR_CHANNEL,
            location=np.zeros(3),
            morphological_features={"pose_vectors": LINE_POSE_VECTORS},
            non_morphological_features={},
        )

        lm = MagicMock()
        lm.learning_module_id = "learning_module_2"
        lm.max_match_distance = 0.01
        lm.buffer.get_first_sensory_input_channel.return_value = SENSOR_CHANNEL
        lm.buffer.get_previous_input_percepts.return_value = None
        lm.buffer.get_num_matching_steps.return_value = 30
        lm.buffer.get_num_steps_post_output_goal_generated.return_value = 30
        lm.get_all_known_object_ids.side_effect = lambda: list(self.graphs.keys())
        lm._get_current_mlh.side_effect = lambda: self.mlh
        lm.get_output.return_value = MagicMock(confidence=1.0)
        lm.get_graph.side_effect = lambda graph_id, input_channel=None: (
            self.graphs[graph_id]
            if input_channel is None
            else self.graphs[graph_id][input_channel]
        )
        lm.get_input_channels_in_graph.side_effect = lambda graph_id: list(
            self.graphs[graph_id].keys()
        )
        self.gsg = EvidenceGoalGenerator(
            min_post_goal_success_steps=20,
            elapsed_steps_factor=10,
            num_spread_neighbors=6,
            inhibition_decay_steps=50,
        )
        self.gsg.parent_lm = lm

    def mlh_at_node(self, node_id, rotation=None):
        return {
            "graph_id": TOP_ID,
            "mlh_id": 0,
            "location": np.array(self.car.pos[node_id]),
            "rotation": Rotation.identity() if rotation is None else rotation,
        }

    def car_inhibition(self) -> np.ndarray:
        return self.gsg.get_inhibition_weights(TOP_ID, "learning_module_0")

    def set_steps_since_goal(self, steps) -> None:
        buffer = self.gsg.parent_lm.buffer
        buffer.get_num_steps_post_output_goal_generated.return_value = steps

    # ------------------------- Spreading -------------------------

    def test_recognized_child_inhibits_only_its_contiguous_region(self) -> None:
        self.gsg._spread_from_observations(
            [self.sensor_percept, lm_percept("learning_module_0", WHEEL_ID)]
        )

        inhibited = np.nonzero(self.car_inhibition())[0]
        nptest.assert_array_equal(
            inhibited,
            np.arange(10),
            "Inhibition should cover the recognized wheel, but stop at the car "
            "body and not reach the second, disjoint wheel.",
        )

    def test_spread_reaches_outlying_nodes(self) -> None:
        # The outlier is 10 node spacings beyond the end of the line, so no node
        # has it among its 6 nearest neighbors, but it has the end of the line
        # among its own.
        positions = np.zeros((21, 3))
        positions[:20, 0] = np.arange(20) * NODE_SPACING
        positions[20, 0] = -10 * NODE_SPACING
        line = FakeGraph(positions, {"object_id": np.full(21, WHEEL_ID)})
        self.graphs[TOP_ID]["learning_module_0"] = line
        self.mlh = {**self.mlh_at_node(0), "location": positions[10]}

        self.gsg._spread_from_observations([lm_percept("learning_module_0", WHEEL_ID)])

        weights = self.gsg.get_inhibition_weights(TOP_ID, "learning_module_0")
        self.assertTrue(np.all(weights > 0), "Every node should be inhibited.")

    def test_spread_is_recorded_in_the_order_nodes_were_reached(self) -> None:
        self.gsg._spread_from_observations([lm_percept("learning_module_0", WHEEL_ID)])

        (record,) = self.gsg.spread_records
        self.assertEqual(record.graph_id, TOP_ID)
        self.assertEqual(record.input_channel, "learning_module_0")
        self.assertEqual(record.object_id, WHEEL_ID)
        nptest.assert_array_equal(np.sort(record.node_order), np.arange(10))
        # The spread starts from the 6 nodes nearest the MLH (at node 4), so the
        # ends of the wheel are reached last.
        self.assertEqual(set(record.node_order[:6].tolist()) - set(range(1, 8)), set())
        self.assertIn(0, record.node_order[6:])
        self.assertIn(9, record.node_order[6:])

    def test_spread_records_only_hold_the_current_steps_spreads(self) -> None:
        def recorded_channels():
            return [record.input_channel for record in self.gsg.spread_records]

        self.gsg.step(
            make_ctx(), [self.sensor_percept, lm_percept("learning_module_0", WHEEL_ID)]
        )
        self.assertEqual(recorded_channels(), [SENSOR_CHANNEL, "learning_module_0"])

        self.gsg.step(make_ctx(), [self.sensor_percept])
        self.assertEqual(recorded_channels(), [SENSOR_CHANNEL])

    def test_no_spreading_when_received_id_is_not_predicted_by_mlh(self) -> None:
        self.mlh = self.mlh_at_node(17)  # On the car body

        self.gsg._spread_from_observations([lm_percept("learning_module_0", WHEEL_ID)])

        self.assertFalse(np.any(self.car_inhibition()))

    def test_no_spreading_when_seed_neighbors_store_different_ids(self) -> None:
        # At the border between the wheel and the body, the nearest nodes store
        # both IDs, even though the wheel is predicted at the MLH location.
        self.mlh = self.mlh_at_node(9)

        self.gsg._spread_from_observations([lm_percept("learning_module_0", WHEEL_ID)])

        self.assertFalse(np.any(self.car_inhibition()))

    def test_ids_from_channels_not_in_the_mlh_graph_are_ignored(self) -> None:
        self.gsg._spread_from_observations([lm_percept("learning_module_5", WHEEL_ID)])

        self.assertFalse(np.any(self.car_inhibition()))

    def test_inhibition_decays_linearly_to_zero(self) -> None:
        self.gsg._spread_from_observations([lm_percept("learning_module_0", WHEEL_ID)])
        self.assertEqual(self.car_inhibition()[0], 1.0)

        for _ in range(10):
            self.gsg._decay_inhibition()
        self.assertAlmostEqual(self.car_inhibition()[0], 0.8)

        for _ in range(40):
            self.gsg._decay_inhibition()
        self.assertEqual(self.car_inhibition()[0], 0.0)

    def test_reset_clears_inhibition(self) -> None:
        self.gsg._spread_from_observations([lm_percept("learning_module_0", WHEEL_ID)])

        self.gsg.reset()

        self.assertFalse(np.any(self.car_inhibition()))

    # ------------------------- Selection -------------------------

    def test_selects_child_with_most_nodes_first(self) -> None:
        channel, node_id, object_id = self.gsg._select_target(make_ctx(), TOP_ID)

        self.assertEqual(channel, "learning_module_0")
        self.assertEqual(object_id, WHEEL_ID)
        self.assertEqual(self.car.x[node_id, 0], WHEEL_ID)

    def test_selects_next_child_once_largest_is_inhibited(self) -> None:
        self.gsg._get_inhibition_steps(TOP_ID, "learning_module_0", 35)[:10] = 1
        self.gsg._get_inhibition_steps(TOP_ID, "learning_module_0", 35)[25:] = 1

        channel, _, object_id = self.gsg._select_target(make_ctx(), TOP_ID)
        self.assertEqual((channel, object_id), ("learning_module_0", BODY_ID))

        self.gsg._get_inhibition_steps(TOP_ID, "learning_module_0", 35)[:] = 1
        channel, _, object_id = self.gsg._select_target(make_ctx(), TOP_ID)
        self.assertEqual((channel, object_id), ("learning_module_1", LOGO_ID))

        self.gsg._get_inhibition_steps(TOP_ID, "learning_module_1", 5)[:] = 1
        channel, _, object_id = self.gsg._select_target(make_ctx(), TOP_ID)
        self.assertEqual((channel, object_id), (SENSOR_CHANNEL, None))

    def test_sensor_channel_is_ranked_by_its_number_of_nodes(self) -> None:
        self.graphs[TOP_ID][SENSOR_CHANNEL] = oriented_sensor_graph(self.car.pos)

        channel, _, object_id = self.gsg._select_target(make_ctx(), TOP_ID)

        self.assertEqual((channel, object_id), (SENSOR_CHANNEL, None))

    def test_only_uninhibited_nodes_are_selected(self) -> None:
        self.gsg._get_inhibition_steps(TOP_ID, "learning_module_0", 35)[:10] = 1

        for seed in range(10):
            ctx = RuntimeContext(rng=np.random.RandomState(seed))
            _, node_id, _ = self.gsg._select_target(ctx, TOP_ID)
            self.assertIn(node_id, range(25, 35))

    def test_no_goal_when_every_node_is_inhibited(self) -> None:
        self.gsg._get_inhibition_steps(TOP_ID, "learning_module_0", 35)[:] = 1
        self.gsg._get_inhibition_steps(TOP_ID, "learning_module_1", 5)[:] = 1
        self.gsg._get_inhibition_steps(TOP_ID, SENSOR_CHANNEL, 4)[:] = 1

        goal = self.gsg._generate_goal(make_ctx(), [self.sensor_percept])

        self.assertIsNone(goal)

    def test_goal_targets_the_selected_child_node(self) -> None:
        goal = self.gsg._generate_goal(make_ctx(), [self.sensor_percept])

        node_id = goal.info["target_node_id"]
        self.assertEqual(goal.info["model_frame_input_channel"], "learning_module_0")
        self.assertEqual(goal.info["target_child_object_id"], WHEEL_ID)
        nptest.assert_allclose(
            goal.info["model_frame_target_loc"], self.car.pos[node_id]
        )

    def test_goal_info_carries_hypothesis_identity_and_predicted_displacement(
        self,
    ) -> None:
        self.sensor_percept.location = np.array([0.01, 0.0, 0.0])

        goal = self.gsg._generate_goal(make_ctx(), [self.sensor_percept])

        # The identity of the hypothesis behind the goal is snapshotted so the
        # LM can attribute a failed jump to it later.
        self.assertEqual(goal.info["hypothesis_to_test_graph_id"], TOP_ID)
        self.assertEqual(goal.info["hypothesis_to_test_mlh_id"], 0)
        # The MLH has identity rotation, so a successful jump displaces the
        # sensor by the model-frame vector from the MLH location to the target.
        nptest.assert_allclose(
            goal.info["predicted_displacement"],
            goal.info["model_frame_target_loc"] - self.mlh["location"],
        )
        nptest.assert_allclose(
            goal.info["proposed_surface_loc"],
            self.sensor_percept.location + goal.info["predicted_displacement"],
        )

    # ------------------------- Triggering -------------------------

    def test_step_spreads_then_targets_an_unexplained_child(self) -> None:
        self.gsg.step(
            make_ctx(),
            [self.sensor_percept, lm_percept("learning_module_0", WHEEL_ID)],
        )

        (goal,) = self.gsg.output_goals()
        self.assertIn(goal.info["target_node_id"], range(25, 35))

    def test_no_goal_before_min_post_goal_success_steps(self) -> None:
        self.set_steps_since_goal(20)

        self.assertFalse(
            self.gsg._check_need_new_output_goal(make_ctx(), output_goal_achieved=False)
        )

    def test_no_goal_when_previous_goal_was_achieved(self) -> None:
        self.assertFalse(
            self.gsg._check_need_new_output_goal(make_ctx(), output_goal_achieved=True)
        )

    def test_goal_when_mlh_changes_or_enough_steps_elapse(self) -> None:
        self.gsg._generate_goal(make_ctx(), [self.sensor_percept])

        self.set_steps_since_goal(21)
        self.assertFalse(
            self.gsg._check_need_new_output_goal(
                make_ctx(), output_goal_achieved=False
            ),
            "No goal when the MLH is unchanged and the step interval isn't met.",
        )

        self.set_steps_since_goal(30)
        self.assertTrue(
            self.gsg._check_need_new_output_goal(make_ctx(), output_goal_achieved=False)
        )

        self.set_steps_since_goal(21)
        self.mlh = self.mlh_at_node(
            4, rotation=Rotation.from_euler("xyz", [0, 0, 90], degrees=True)
        )
        self.assertTrue(
            self.gsg._check_need_new_output_goal(make_ctx(), output_goal_achieved=False)
        )

    def test_targets_sensor_nodes_without_compositional_models(self) -> None:
        self.graphs = {TOP_ID: {SENSOR_CHANNEL: oriented_sensor_graph(self.car.pos)}}
        # A sensed normal the MLH does not predict, so nothing is inhibited.
        self.sensor_percept.morphological_features = {"pose_vectors": np.eye(3)}

        self.gsg.step(make_ctx(), [self.sensor_percept])

        (goal,) = self.gsg.output_goals()
        self.assertEqual(goal.info["model_frame_input_channel"], SENSOR_CHANNEL)
        self.assertIsNone(goal.info["target_child_object_id"])


SURFACE_SPACING = 0.002


def plane_points(origin, u, v, nu, nv):
    """A grid of nu x nv points spanning directions u and v from an origin.

    Returns:
        The points, shape (nu * nv, 3).
    """
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    return (
        np.asarray(origin, dtype=float)
        + iu.reshape(-1, 1) * SURFACE_SPACING * np.asarray(u, dtype=float)
        + iv.reshape(-1, 1) * SURFACE_SPACING * np.asarray(v, dtype=float)
    )


def surface_graph(positions, normals, hues=None, saturations=None):
    """A sensor-channel graph storing hue and surface normals.

    Args:
        positions: The node locations.
        normals: The surface normal of every node (or one for all).
        hues: The hue of every node; 0 if None.
        saturations: The saturation of every node; 1 if None.

    Returns:
        The graph.
    """
    num_nodes = len(positions)
    pose_vectors = np.zeros((num_nodes, 3, 3))
    pose_vectors[:, 0] = np.broadcast_to(
        np.asarray(normals, dtype=float), (num_nodes, 3)
    )
    hsv = np.column_stack(
        [
            np.zeros(num_nodes) if hues is None else hues,
            np.ones(num_nodes) if saturations is None else saturations,
            np.ones(num_nodes),
        ]
    )
    return FakeGraph(
        positions, {"pose_vectors": pose_vectors.reshape(num_nodes, 9), "hsv": hsv}
    )


def cylinder_surface(radius, num_around, num_along):
    """Points around a cylinder along z, with outward normals.

    Returns:
        The points and their normals.
    """
    angles = np.arange(num_around) * 2 * np.pi / num_around
    heights = np.arange(num_along) * SURFACE_SPACING
    angle, height = np.meshgrid(angles, heights, indexing="ij")
    angle, height = angle.ravel(), height.ravel()
    normals = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(angle.size)])
    points = np.column_stack([radius * normals[:, :2], height])
    return points, normals


class SensoryFeatureSpreadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = None
        self.mlh = None
        lm = MagicMock()
        lm.learning_module_id = "learning_module_0"
        lm.max_match_distance = 0.01
        lm.get_all_known_object_ids.side_effect = lambda: [TOP_ID]
        lm._get_current_mlh.side_effect = lambda: self.mlh
        lm.get_graph.side_effect = lambda _graph_id, input_channel=None: (
            {SENSOR_CHANNEL: self.graph} if input_channel is None else self.graph
        )
        lm.get_input_channels_in_graph.side_effect = lambda _graph_id: [SENSOR_CHANNEL]
        self.gsg = EvidenceGoalGenerator()
        self.gsg.parent_lm = lm

    def spread_from(self, node, rotation=None, **sensed_overrides) -> np.ndarray:
        """Spread from sensing the features stored at a node.

        Args:
            node: The node at the MLH location.
            rotation: The MLH rotation; the sensed normal is the stored one rotated
                into the body frame by its inverse. Identity if None.
            **sensed_overrides: Sensed features replacing the stored ones.

        Returns:
            Whether each node is inhibited.
        """
        rotation = Rotation.identity() if rotation is None else rotation
        mapping = self.graph.feature_mapping
        features = {
            name: self.graph.x[node, start:end]
            for name, (start, end) in mapping.items()
        }
        pose_vectors = features["pose_vectors"].reshape(3, 3)
        pose_vectors = rotation.inv().apply(pose_vectors)
        sensed = {
            "pose_vectors": pose_vectors,
            "hsv": features["hsv"],
            **sensed_overrides,
        }
        percept = MagicMock(
            sender_type="SM",
            sender_id=SENSOR_CHANNEL,
            morphological_features={"pose_vectors": sensed["pose_vectors"]},
            non_morphological_features={"hsv": sensed["hsv"]},
        )
        self.mlh = {
            "graph_id": TOP_ID,
            "mlh_id": 0,
            "location": np.array(self.graph.pos[node]),
            "rotation": rotation,
        }
        self.gsg.reset()
        self.gsg._spread_from_observations([percept])
        return self.gsg.get_inhibition_weights(TOP_ID, SENSOR_CHANNEL) > 0

    def test_spreads_across_a_face_but_not_around_an_edge(self) -> None:
        # Two faces of a cube meeting at an edge.
        top = plane_points([0, 0, 0.02], [1, 0, 0], [0, 1, 0], 10, 10)
        side = plane_points([0.02, 0, 0], [0, 0, 1], [0, 1, 0], 10, 10)
        self.graph = surface_graph(
            np.vstack([top, side]),
            np.vstack([np.tile([0, 0, 1], (100, 1)), np.tile([1, 0, 0], (100, 1))]),
        )

        inhibited = self.spread_from(55)

        self.assertTrue(np.all(inhibited[:100]), "The sensed face is inhibited.")
        self.assertFalse(np.any(inhibited[100:]), "The adjacent face is not.")

    def test_spreads_around_a_cylinder_but_not_onto_its_cap(self) -> None:
        radius = 0.02
        side, normals = cylinder_surface(radius, 60, 8)
        top = side[:, 2].max()
        cap = plane_points([-0.019, -0.019, top], [1, 0, 0], [0, 1, 0], 20, 20)
        cap = cap[np.linalg.norm(cap[:, :2], axis=1) < radius - SURFACE_SPACING]
        self.graph = surface_graph(
            np.vstack([side, cap]),
            np.vstack([normals, np.tile([0, 0, 1], (len(cap), 1))]),
        )

        inhibited = self.spread_from(4)

        self.assertTrue(
            np.all(inhibited[: len(side)]),
            "The whole side is inhibited, as neighboring normals are similar even "
            "though the normals around the cylinder are not.",
        )
        self.assertFalse(np.any(inhibited[len(side) :]), "The cap is not inhibited.")

    def test_stops_where_hue_differs(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 20, 10)
        right = points[:, 0] >= 0.02
        self.graph = surface_graph(points, [0, 0, 1], hues=np.where(right, 0.5, 0.0))

        inhibited = self.spread_from(25)

        nptest.assert_array_equal(inhibited, ~right)

    def test_achromatic_nodes_do_not_stop_spreading(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 20, 10)
        right = points[:, 0] >= 0.02
        self.graph = surface_graph(
            points,
            [0, 0, 1],
            hues=np.where(right, 0.5, 0.0),
            saturations=np.where(right, 0.0, 1.0),
        )

        inhibited = self.spread_from(25)

        self.assertTrue(
            np.all(inhibited),
            "The hue of the achromatic nodes is undefined, so is not compared.",
        )

    def test_stops_at_a_tightly_rounded_edge(self) -> None:
        # A face, rounded over an edge of radius 2mm into a perpendicular face.
        radius = 0.002
        top = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        edge_x = top[:, 0].max()
        angles = np.radians([30, 60])
        arc_normals = np.column_stack([np.sin(angles), np.zeros(2), np.cos(angles)])
        arc = np.vstack(
            [
                [edge_x, y, -radius] + radius * normal
                for y in np.unique(top[:, 1])
                for normal in arc_normals
            ]
        )
        side = plane_points(
            [edge_x + radius, 0, -radius - SURFACE_SPACING],
            [0, 0, -1],
            [0, 1, 0],
            5,
            10,
        )
        self.graph = surface_graph(
            np.vstack([top, arc, side]),
            np.vstack(
                [
                    np.tile([0, 0, 1], (len(top), 1)),
                    np.tile(arc_normals, (10, 1)),
                    np.tile([1, 0, 0], (len(side), 1)),
                ]
            ),
        )

        inhibited = self.spread_from(44)

        self.assertTrue(np.all(inhibited[: len(top)]))
        self.assertFalse(np.any(inhibited[len(top) + len(arc) :]))

    def test_far_side_of_a_thin_wall_neither_stops_nor_receives_spreading(
        self,
    ) -> None:
        outside = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        inside = outside - [0, 0, 0.001]
        self.graph = surface_graph(
            np.vstack([outside, inside]),
            np.vstack([np.tile([0, 0, 1], (100, 1)), np.tile([0, 0, -1], (100, 1))]),
        )

        inhibited = self.spread_from(55)

        self.assertTrue(np.all(inhibited[:100]))
        self.assertFalse(np.any(inhibited[100:]))

    def test_noisy_neighbors_do_not_stop_spreading(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        normals = np.tile([0.0, 0.0, 1.0], (100, 1))
        noisy = [44, 45]
        normals[noisy] = [np.sin(np.radians(60)), 0.0, np.cos(np.radians(60))]
        self.graph = surface_graph(points, normals)
        self.gsg.min_spread_fraction = None

        inhibited = self.spread_from(55)

        self.assertFalse(np.any(inhibited[noisy]), "The noisy nodes are not spread to.")
        self.assertTrue(np.all(np.delete(inhibited, noisy)))

    def test_spreading_that_stops_too_soon_is_pushed_through(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 30, 22)
        x, y = np.round(points[:, :2] / SURFACE_SPACING).astype(int).T
        ring = (x >= 7) & (x < 22) & (y >= 3) & (y < 18)
        patch = (x >= 13) & (x < 16) & (y >= 9) & (y < 12)
        ring &= ~patch
        hues = np.select([patch, ring], [0.5, 0.25], default=0.0)
        self.graph = surface_graph(points, [0, 0, 1], hues=hues)

        self.gsg.min_spread_fraction = None
        self.assertFalse(np.any(self.spread_from(np.flatnonzero(patch)[4])[~patch]))

        self.gsg.min_spread_fraction = 0.2
        inhibited = self.spread_from(np.flatnonzero(patch)[4])
        self.assertTrue(np.all(inhibited[patch]))
        self.assertFalse(np.any(inhibited[~(patch | ring)]))
        self.assertGreaterEqual(
            inhibited.sum(),
            132,
            "Spreading stops at the edge of the patch (9 nodes), short of 132 "
            "nodes, so is pushed through into the ring around it.",
        )
        self.assertLess(
            inhibited.sum(),
            150,
            "Once pushed through, spreading stops on reaching 132 nodes, rather than "
            "covering the ring (225 nodes).",
        )
        nptest.assert_array_equal(
            self.spread_from(0),
            ~(patch | ring),
            "Spreading from the plane around the ring stops at the ring, having "
            "reached enough nodes.",
        )

    def test_spreading_is_not_pushed_through_a_thin_wall(self) -> None:
        outside = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        inside = plane_points([0.008, 0.008, -0.001], [1, 0, 0], [0, 1, 0], 3, 3)
        self.graph = surface_graph(
            np.vstack([outside, inside]),
            np.vstack([np.tile([0, 0, 1], (100, 1)), np.tile([0, 0, -1], (9, 1))]),
        )
        self.gsg.min_spread_fraction = 0.2

        nptest.assert_array_equal(
            self.spread_from(102),
            np.arange(109) >= 100,
            "The inside patch's only neighbors are on the far side of the wall.",
        )

    def test_no_spreading_when_sensed_features_are_not_predicted(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        self.graph = surface_graph(points, [0, 0, 1])

        inhibited = self.spread_from(55, hsv=np.array([0.5, 1.0, 1.0]))

        self.assertFalse(np.any(inhibited))

    def test_sensed_normal_is_rotated_into_the_model_frame_by_the_mlh(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        self.graph = surface_graph(points, [0, 0, 1])
        rotation = Rotation.from_euler("x", 90, degrees=True)

        self.assertTrue(np.all(self.spread_from(55, rotation=rotation)))
        self.assertFalse(
            np.any(self.spread_from(55, rotation=rotation, pose_vectors=np.eye(3))),
            "The sensed normal, unrotated, is not the one the MLH predicts.",
        )

    def test_sensory_spread_is_recorded_without_an_object_id(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        self.graph = surface_graph(points, [0, 0, 1])

        self.spread_from(55)

        (record,) = self.gsg.spread_records
        self.assertEqual(record.input_channel, SENSOR_CHANNEL)
        self.assertIsNone(record.object_id)
        nptest.assert_array_equal(np.sort(record.node_order), np.arange(100))


if __name__ == "__main__":
    unittest.main()
