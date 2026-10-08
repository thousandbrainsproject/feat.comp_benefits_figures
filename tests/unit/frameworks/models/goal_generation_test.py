# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Tests for the feature-based graph-mismatch logic of the EvidenceGoalGenerator.

The spatial (Euclidean) mismatch path is additionally covered end-to-end in
tests/unit/frameworks/models/evidence_matching/evidence_lm_test.py; the tests here
use mock graphs to exercise the discrete (object ID), novel channel, and continuous
(hue) feature paths that are used when two graphs are spatially near-identical.
"""

from __future__ import annotations

import itertools
import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as nptest
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.goal_generation import (
    ChildObjectsGoalGenerator,
    CubeViewGoalGenerator,
    EvidenceGoalGenerator,
    TraceGoalGenerator,
)
from tbp.monty.frameworks.models.object_model import GridObjectModel
from tbp.monty.frameworks.utils.sensor_processing import log_sign

SKIP_ID_CHANNEL_HYPOTHESIS_TESTING = (
    "Object-ID channel hypothesis testing is under active development"
)
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
        report object_a and object_b as the top-two MLH objects.
    """
    lm = MagicMock()
    lm.learning_module_id = "learning_module_2"
    lm.object_id_feature_names = {}
    lm.max_match_distance = 0.01
    lm.buffer.get_first_sensory_input_channel.return_value = SENSOR_CHANNEL
    lm.get_top_two_mlh_ids.return_value = (TOP_ID, SECOND_ID)
    lm.get_mlh_for_object.side_effect = identity_mlh
    lm._get_current_mlh.return_value = identity_mlh(TOP_ID)
    lm.get_graph.side_effect = lambda graph_id, input_channel=None: (
        graphs[graph_id] if input_channel is None else graphs[graph_id][input_channel]
    )
    lm.get_input_channels_in_graph.side_effect = lambda graph_id: list(
        graphs[graph_id].keys()
    )
    gsg = EvidenceGoalGenerator(**gsg_kwargs)
    gsg.parent_lm = lm
    gsg.focus_on_pose = False
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


def sensor_graph(hues=None, saturations=None, values=None):
    """A sensor-channel graph over BASE_POINTS, optionally with HSV features.

    Saturations and values default to 1 (fully chromatic) when hues are given.

    Returns:
        The graph.
    """
    features = None
    if hues is not None:
        ones = np.ones_like(hues)
        features = {
            "hsv": np.column_stack(
                [
                    hues,
                    ones if saturations is None else saturations,
                    ones if values is None else values,
                ]
            )
        }
    return FakeGraph(BASE_POINTS, features)


class SpatialPathTest(unittest.TestCase):
    @unittest.skip(SKIP_ID_CHANNEL_HYPOTHESIS_TESTING)
    def test_spatial_target_returned_when_graphs_differ_spatially(self) -> None:
        # The top graph has an extra point 5cm away from anything in the second
        # graph (like a mug handle), so the spatial path should propose it.
        top_points = np.vstack([BASE_POINTS, [0.025, 0.1, 0.0]])
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: FakeGraph(top_points)},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, SENSOR_CHANNEL)
        self.assertEqual(target_loc_id, len(top_points) - 1)


class DiscreteFeaturePathTest(unittest.TestCase):
    def lm_channel_graphs(self, positions, top_object_ids, second_object_ids):
        """Build matching-position LM channel graphs with the given object IDs.

        Returns:
            Tuple of (top graph, second graph).
        """
        return (
            FakeGraph(positions, {"object_id": top_object_ids}),
            FakeGraph(positions, {"object_id": second_object_ids}),
        )

    def test_target_is_model_point_at_center_of_mismatch_cluster(self) -> None:
        # Three contiguous mismatching nodes (a "logo") plus one far-away
        # mismatching outlier; the target should be the middle of the cluster,
        # and the outlier should not drag the center away.
        lm_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.004, 0.0, 0.0],
                [0.008, 0.0, 0.0],
                [0.03, 0.03, 0.0],  # matching node
                [0.1, 0.0, 0.0],  # mismatching outlier, >1cm from the cluster
            ]
        )
        top_ids = np.array([1, 1, 1, 2, 3])
        second_ids = np.array([7, 7, 7, 2, 8])
        top_lm, second_lm = self.lm_channel_graphs(lm_positions, top_ids, second_ids)
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(), "learning_module_0": top_lm},
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": second_lm,
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_0")
        self.assertEqual(
            target_loc_id,
            1,
            "Target should be the model point closest to the cluster's center, "
            "with the far-away mismatching node excluded as an outlier.",
        )

    def test_channel_with_larger_mismatch_cluster_wins(self) -> None:
        cluster_of_two = np.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
        cluster_of_three = np.array(
            [[0.0, 0.02, 0.0], [0.005, 0.02, 0.0], [0.01, 0.02, 0.0]]
        )
        top_lm0, second_lm0 = self.lm_channel_graphs(
            cluster_of_two, np.array([1, 1]), np.array([2, 2])
        )
        top_lm1, second_lm1 = self.lm_channel_graphs(
            cluster_of_three, np.array([1, 1, 1]), np.array([2, 2, 2])
        )
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": top_lm0,
                "learning_module_1": top_lm1,
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": second_lm0,
                "learning_module_1": second_lm1,
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_1")
        self.assertEqual(target_loc_id, 1, "Middle of the three-node cluster.")

    def test_object_id_mismatch_prioritized_over_hue_mismatch(self) -> None:
        # The sensor channel has a large hue difference, but a single
        # mismatching object ID should still take precedence.
        lm_positions = np.array([[0.0, 0.0, 0.0]])
        top_lm, second_lm = self.lm_channel_graphs(
            lm_positions, np.array([1]), np.array([2])
        )
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(hues=np.array([0.0, 0.5, 0.0, 0.0])),
                "learning_module_0": top_lm,
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(hues=np.zeros(4)),
                "learning_module_0": second_lm,
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, _ = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_0")

    def test_no_goal_when_object_ids_match_and_no_other_features(self) -> None:
        lm_positions = np.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
        top_lm, second_lm = self.lm_channel_graphs(
            lm_positions, np.array([1, 1]), np.array([1, 1])
        )
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(), "learning_module_0": top_lm},
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": second_lm,
            },
        }
        gsg = gsg_with_graphs(graphs)

        self.assertIsNone(gsg._compute_graph_mismatch(make_ctx()))

    def test_matching_object_ids_beyond_max_match_distance_mismatch(self) -> None:
        # The second graph stores the same object ID, but only 2cm away from the
        # top graph's three nodes, i.e. beyond max_match_distance (1cm).
        top_lm = FakeGraph(
            np.array([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.008, 0.0, 0.0]]),
            {"object_id": np.array([1, 1, 1])},
        )
        second_lm = FakeGraph(
            np.array([[0.004, 0.02, 0.0]]), {"object_id": np.array([1])}
        )
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(), "learning_module_0": top_lm},
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": second_lm,
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())
        self.assertEqual(channel, "learning_module_0")
        self.assertEqual(target_loc_id, 1, "Middle of the three-node cluster.")

        gsg.parent_lm.max_match_distance = 0.03
        self.assertIsNone(
            gsg._compute_graph_mismatch(make_ctx()),
            "Within max_match_distance, matching object IDs are not a mismatch.",
        )


# A contiguous "logo" of LM-channel nodes around BASE_POINTS[2], plus a far-away
# outlier node that should be excluded from the cluster.
LOGO_POSITIONS = np.array(
    [
        [0.046, 0.05, 0.0],
        [0.05, 0.05, 0.0],
        [0.054, 0.05, 0.0],
        [0.0, 0.0, 0.0],  # outlier, >1cm from the cluster
    ]
)


def logo_graph(positions=LOGO_POSITIONS):
    """An LM-channel graph storing object IDs at the given positions.

    Returns:
        The graph.
    """
    return FakeGraph(positions, {"object_id": np.ones(len(positions))})


class NovelChannelPathTest(unittest.TestCase):
    def test_channel_only_in_top_graph_targets_center_of_its_cluster(self) -> None:
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(), "learning_module_1": logo_graph()},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_1")
        self.assertEqual(target_loc_id, 1, "Middle of the logo cluster.")

    def test_channel_only_in_second_graph_targets_top_graph_sensor_node(
        self,
    ) -> None:
        # The second MLH has a different pose to the top MLH, so its graph is
        # stored in a different frame; the logo should be mapped back into the
        # top MLH graph's frame, landing on the top graph's sensor node 2.
        second_rotation = Rotation.from_euler("z", 90, degrees=True)
        second_location = np.array([0.1, 0.0, 0.0])

        def mlh_for_object(graph_id):
            mlh = identity_mlh(graph_id)
            if graph_id == SECOND_ID:
                mlh["rotation"] = second_rotation
                mlh["location"] = second_location
            return mlh

        def to_second_frame(points):
            return second_rotation.apply(points) + second_location

        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph()},
            SECOND_ID: {
                SENSOR_CHANNEL: FakeGraph(to_second_frame(BASE_POINTS)),
                "learning_module_1": logo_graph(to_second_frame(LOGO_POSITIONS)),
            },
        }
        gsg = gsg_with_graphs(graphs)
        gsg.parent_lm.get_mlh_for_object.side_effect = mlh_for_object

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(
            channel,
            SENSOR_CHANNEL,
            "Targets must index the top MLH graph, which has no logo channel.",
        )
        self.assertEqual(target_loc_id, 2)

    def test_channel_without_object_ids_in_other_graph_is_novel(self) -> None:
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(), "learning_module_1": logo_graph()},
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_1": FakeGraph(LOGO_POSITIONS),
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_1")
        self.assertEqual(target_loc_id, 1)

    def test_larger_novel_cluster_wins_across_graphs(self) -> None:
        # The top graph's novel cluster has two nodes, the second graph's has
        # three (around BASE_POINTS[3]), so the latter should be targeted.
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": logo_graph(
                    np.array([[0.05, 0.0, 0.0], [0.054, 0.0, 0.0]])
                ),
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_1": logo_graph(
                    np.array(
                        [[-0.004, 0.05, 0.0], [0.0, 0.05, 0.0], [0.004, 0.05, 0.0]]
                    )
                ),
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, SENSOR_CHANNEL)
        self.assertEqual(target_loc_id, 3)

    def test_shared_object_id_mismatch_prioritized_over_novel_channel(self) -> None:
        shared_positions = np.array([[0.0, 0.0, 0.0]])
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": FakeGraph(
                    shared_positions, {"object_id": np.array([1])}
                ),
                "learning_module_1": logo_graph(),
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": FakeGraph(
                    shared_positions, {"object_id": np.array([2])}
                ),
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, _ = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_0")

    def test_novel_channel_prioritized_over_hue_mismatch(self) -> None:
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(hues=np.array([0.0, 0.5, 0.0, 0.0])),
                "learning_module_1": logo_graph(),
            },
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph(hues=np.zeros(4))},
        }
        gsg = gsg_with_graphs(graphs)

        channel, _ = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_1")


class ObjectIdLoggingTest(unittest.TestCase):
    def assert_logged(self, gsg, expected_message) -> None:
        with self.assertLogs(
            "tbp.monty.frameworks.models.goal_generation", level="DEBUG"
        ) as logs:
            gsg._compute_graph_mismatch(make_ctx())

        self.assertIn(expected_message, "\n".join(logs.output))

    def test_shared_channel_mismatch_logs_names_of_compared_object_ids(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0]])
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_1": FakeGraph(positions, {"object_id": [1]}),
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_1": FakeGraph(positions, {"object_id": [2]}),
            },
        }
        gsg = gsg_with_graphs(graphs)
        gsg.parent_lm.object_id_feature_names = {1: "tbp_logo", 2: "numenta_logo"}

        self.assert_logged(
            gsg,
            "the top hypothesis stores tbp_logo and the second hypothesis stores "
            "numenta_logo",
        )

    def test_mismatch_beyond_max_match_distance_logs_nearest_object_id(self) -> None:
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": FakeGraph(
                    np.array([[0.0, 0.0, 0.0]]), {"object_id": [1]}
                ),
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": FakeGraph(
                    np.array([[0.0, 0.02, 0.0]]), {"object_id": [1]}
                ),
            },
        }
        gsg = gsg_with_graphs(graphs)
        gsg.parent_lm.object_id_feature_names = {1: "mug"}

        self.assert_logged(
            gsg,
            "the top hypothesis stores mug and the second hypothesis stores "
            "nothing within max_match_distance (nearest: mug)",
        )

    def test_novel_channel_mismatch_logs_name_of_novel_object_id(self) -> None:
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph()},
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_1": logo_graph(),
            },
        }
        gsg = gsg_with_graphs(graphs)
        gsg.parent_lm.object_id_feature_names = {1: "numenta_logo"}

        self.assert_logged(gsg, "only the second hypothesis stores numenta_logo")

    def test_unknown_object_ids_are_logged_as_feature_values(self) -> None:
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(), "learning_module_1": logo_graph()},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)

        self.assert_logged(gsg, "only the top hypothesis stores 1")


class ContinuousFeaturePathTest(unittest.TestCase):
    def test_target_is_node_with_maximal_circular_hue_distance(self) -> None:
        # Node 1 has the largest hue difference (0.5 vs 0.9 -> 0.4); node 0's
        # difference wraps around the hue circle (0.95 vs 0.05 -> 0.1).
        top_hues = np.array([0.95, 0.5, 0.2, 0.3])
        second_hues = np.array([0.05, 0.9, 0.2, 0.3])
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(hues=top_hues)},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph(hues=second_hues)},
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, SENSOR_CHANNEL)
        self.assertEqual(target_loc_id, 1)

    def test_tied_hue_distances_resolved_to_one_of_the_tied_nodes(self) -> None:
        top_hues = np.array([0.5, 0.5, 0.0, 0.0])
        second_hues = np.array([0.9, 0.9, 0.0, 0.0])
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(hues=top_hues)},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph(hues=second_hues)},
        }
        gsg = gsg_with_graphs(graphs)

        _, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertIn(target_loc_id, [0, 1])

    def test_no_goal_when_hue_difference_below_threshold(self) -> None:
        top_hues = np.array([0.5, 0.5, 0.0, 0.0])
        second_hues = np.array([0.55, 0.5, 0.0, 0.0])
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(hues=top_hues)},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph(hues=second_hues)},
        }
        gsg = gsg_with_graphs(graphs, min_hue_mismatch=0.1)

        self.assertIsNone(gsg._compute_graph_mismatch(make_ctx()))

    def test_no_goal_when_hue_mismatch_is_between_achromatic_nodes(self) -> None:
        # Node 0 has a large hue difference, but is near-grey in the top graph
        # (low saturation) and near-black in the second graph (low value).
        top_hues = np.array([0.7, 0.0, 0.0, 0.0])
        second_hues = np.array([0.0, 0.0, 0.0, 0.0])
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(
                    hues=top_hues, saturations=np.array([0.02, 1.0, 1.0, 1.0])
                )
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(
                    hues=second_hues, values=np.array([0.01, 1.0, 1.0, 1.0])
                )
            },
        }
        gsg = gsg_with_graphs(graphs, min_hue_saturation=0.1, min_hue_value=0.1)

        self.assertIsNone(gsg._compute_graph_mismatch(make_ctx()))

    def test_chromatic_hue_mismatch_wins_over_larger_achromatic_one(self) -> None:
        # Node 0's hue difference (0.4) is largest but its top node is grey;
        # node 2's smaller difference (0.2) is between chromatic nodes.
        top_hues = np.array([0.5, 0.0, 0.3, 0.0])
        second_hues = np.array([0.9, 0.0, 0.1, 0.0])
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(
                    hues=top_hues, saturations=np.array([0.02, 1.0, 1.0, 1.0])
                )
            },
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph(hues=second_hues)},
        }
        gsg = gsg_with_graphs(graphs, min_hue_saturation=0.1)

        _, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(target_loc_id, 2)

    def test_no_goal_when_no_valid_feature_channels(self) -> None:
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph()},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)

        self.assertIsNone(gsg._compute_graph_mismatch(make_ctx()))

    def test_generate_goal_returns_none_goal_when_no_mismatch(self) -> None:
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph()},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)

        self.assertIsNone(gsg._generate_goal(make_ctx(), observations=[]))


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


class GeneratedGoalInfoTest(unittest.TestCase):
    @unittest.skip(SKIP_ID_CHANNEL_HYPOTHESIS_TESTING)
    def test_goal_info_carries_hypothesis_identity_and_predicted_displacement(
        self,
    ) -> None:
        # The top graph has an extra point 10cm away (like a mug handle) that
        # the spatial mismatch path will propose as the target location.
        extra_point = np.array([0.025, 0.1, 0.0])
        top_points = np.vstack([BASE_POINTS, extra_point])
        pose_vectors = np.column_stack(
            [np.tile([0.0, 0.0, 1.0], (len(top_points), 1)), np.zeros((5, 6))]
        )
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: FakeGraph(top_points, {"pose_vectors": pose_vectors})
            },
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)
        gsg.parent_lm.get_output.return_value = MagicMock(confidence=1.0)
        sensed_location = np.array([0.01, 0.0, 0.0])
        sensory_input = MagicMock(sender_id=SENSOR_CHANNEL, location=sensed_location)

        goal = gsg._generate_goal(make_ctx(), observations=[sensory_input])

        # The identity of the hypothesis behind the goal is snapshotted so the
        # LM can attribute a failed jump to it later.
        self.assertEqual(goal.info["hypothesis_to_test_graph_id"], TOP_ID)
        self.assertEqual(goal.info["hypothesis_to_test_mlh_id"], 0)
        # The MLH is at the model origin with identity rotation, so a
        # successful jump displaces the sensor by the model-frame vector from
        # the MLH location to the target point.
        nptest.assert_allclose(goal.info["predicted_displacement"], extra_point)
        nptest.assert_allclose(
            goal.info["proposed_surface_loc"],
            sensed_location + goal.info["predicted_displacement"],
        )


class TraceGoalGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = GridObjectModel(
            TOP_ID, max_nodes=10, max_size=1, num_voxels_per_dim=50
        )
        pv = np.eye(3).flatten()  # surface normal is the first row: [1, 0, 0]
        self.model.build_model(
            BASE_POINTS,
            {
                "pose_vectors": np.tile(pv, (len(BASE_POINTS), 1)),
                "pose_fully_defined": np.ones(len(BASE_POINTS), dtype=bool),
            },
        )
        self.surface_normal = np.array([1.0, 0.0, 0.0])
        self.sensed_location = np.array([0.2, 0.3, 0.4])
        self.observations = [
            MagicMock(sender_id=SENSOR_CHANNEL, location=self.sensed_location)
        ]
        self.mlh = identity_mlh(TOP_ID)

        lm = MagicMock()
        lm.learning_module_id = "learning_module_2"
        lm.buffer.get_first_sensory_input_channel.return_value = SENSOR_CHANNEL
        lm.buffer.get_previous_input_percepts.return_value = None
        lm.buffer.get_num_matching_steps.return_value = 20
        lm.buffer.get_num_steps_post_output_goal_generated.return_value = 20
        lm.get_all_known_object_ids.return_value = [TOP_ID]
        lm.get_graph.side_effect = lambda *_args, **_kwargs: {
            SENSOR_CHANNEL: self.model
        }
        lm._get_current_mlh.side_effect = lambda: self.mlh
        lm.get_output.return_value = MagicMock(confidence=1.0)
        self.gsg = TraceGoalGenerator(
            desired_object_distance=0.03, min_steps_between_goals=10
        )
        self.gsg.parent_lm = lm

    def node_at(self, location) -> int:
        return int(np.argmin(np.linalg.norm(self.model.pos - location, axis=1)))

    def test_goal_targets_maximal_hotspot(self) -> None:
        self.model.tag_hotspots([[0.05, 0.05, 0.0], [0.0, 0.05, 0.0]], [0.2, 1.5])
        hotspot_node = self.node_at([0.0, 0.05, 0.0])

        goal = self.gsg._generate_goal(make_ctx(), self.observations)

        # Identity MLH at the model origin: the jump displaces the sensor by the
        # model-frame vector to the hot spot, then backs off along the normal.
        nptest.assert_allclose(goal.info["predicted_displacement"], [0.0, 0.05, 0.0])
        nptest.assert_allclose(
            goal.location,
            self.sensed_location + [0.0, 0.05, 0.0] + self.surface_normal * 0.045,
        )
        nptest.assert_allclose(
            goal.morphological_features["pose_vectors"][0], -self.surface_normal
        )
        self.assertEqual(goal.info["hotspot_node_id"], hotspot_node)
        self.assertAlmostEqual(goal.info["hotspot_value"], 1.5)
        self.assertEqual(goal.info["hypothesis_to_test_graph_id"], TOP_ID)

    def test_goal_is_transformed_by_mlh_pose(self) -> None:
        self.model.tag_hotspots([[0.05, 0.0, 0.0]], [1.0])
        rotation = Rotation.from_euler("xyz", [0, 0, 90], degrees=True)
        mlh_location = np.array([0.0, 0.05, 0.0])
        self.mlh = {
            "graph_id": TOP_ID,
            "mlh_id": 3,
            "location": mlh_location,
            "rotation": rotation,
        }

        goal = self.gsg._generate_goal(make_ctx(), self.observations)

        # The MLH rotation maps body displacements into the model's frame, so
        # model-frame displacements map back into the body frame by its inverse.
        body_displacement = rotation.inv().apply(
            np.array([0.05, 0.0, 0.0]) - mlh_location
        )
        body_normal = rotation.inv().apply(self.surface_normal)
        nptest.assert_allclose(goal.info["predicted_displacement"], body_displacement)
        nptest.assert_allclose(
            goal.location,
            self.sensed_location + body_displacement + body_normal * 0.045,
            atol=1e-12,
        )

    def test_repeatedly_selects_the_maximal_hotspot(self) -> None:
        self.model.tag_hotspots([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0]], [2.0, 1.0])

        goals = [
            self.gsg._generate_goal(make_ctx(), self.observations) for _ in range(3)
        ]

        node = self.node_at([0.05, 0.0, 0.0])
        self.assertEqual([g.info["hotspot_node_id"] for g in goals], [node] * 3)

    def test_no_goal_without_hotspots(self) -> None:
        self.assertIsNone(self.gsg._generate_goal(make_ctx(), self.observations))

    def test_no_goal_when_hotspot_not_above_min_value(self) -> None:
        self.model.tag_hotspots([[0.05, 0.0, 0.0]], [0.0])

        self.assertIsNone(self.gsg._generate_goal(make_ctx(), self.observations))

    def test_no_goal_when_mlh_is_not_a_known_object(self) -> None:
        self.model.tag_hotspots([[0.05, 0.0, 0.0]], [1.0])
        self.mlh = identity_mlh("no_observations_yet")

        self.assertIsNone(self.gsg._generate_goal(make_ctx(), self.observations))

    def test_step_outputs_goal_only_after_min_steps_between_goals(self) -> None:
        self.model.tag_hotspots([[0.05, 0.0, 0.0]], [1.0])
        steps_since_goal = (
            self.gsg.parent_lm.buffer.get_num_steps_post_output_goal_generated
        )

        steps_since_goal.return_value = 10
        self.gsg.step(make_ctx(), self.observations)
        self.assertEqual(self.gsg.output_goals(), [])

        steps_since_goal.return_value = 11
        self.gsg.step(make_ctx(), self.observations)
        (goal,) = self.gsg.output_goals()
        self.assertEqual(goal.info["hotspot_node_id"], self.node_at([0.05, 0.0, 0.0]))

        # Goals are one-off attempts: the next step outputs no goal.
        steps_since_goal.return_value = 1
        self.gsg.step(make_ctx(), self.observations)
        self.assertEqual(self.gsg.output_goals(), [])


class CubeViewGoalGeneratorTest(unittest.TestCase):
    CENTER = np.array([0.1, 0.2, 0.3])

    def setUp(self) -> None:
        # Face centers stick out furthest along the face directions, and the
        # (closer) corners furthest along the corner directions.
        faces = np.vstack([np.eye(3), -np.eye(3)]) * 0.05
        corners = np.array(list(itertools.product([1, -1], repeat=3))) * 0.04
        self.points = np.vstack([faces, corners]) + self.CENTER
        self.graph = FakeGraph(self.points)
        self.sensed_location = np.array([0.5, 0.5, 0.5])
        self.observations = [
            MagicMock(sender_id=SENSOR_CHANNEL, location=self.sensed_location)
        ]
        self.mlh = identity_mlh(TOP_ID)

        lm = MagicMock()
        lm.learning_module_id = "learning_module_2"
        lm.buffer.get_first_sensory_input_channel.return_value = SENSOR_CHANNEL
        lm.buffer.get_previous_input_percepts.return_value = None
        lm.buffer.get_num_matching_steps.return_value = 1
        lm.get_all_known_object_ids.return_value = [TOP_ID]
        lm.get_graph.side_effect = lambda *_args, **_kwargs: {
            SENSOR_CHANNEL: self.graph
        }
        lm._get_current_mlh.side_effect = lambda: self.mlh
        self.gsg = CubeViewGoalGenerator(
            desired_object_distance=0.02, steps_per_view=50
        )
        self.gsg.parent_lm = lm

    def step_at(self, matching_step) -> list:
        self.gsg.parent_lm.buffer.get_num_matching_steps.return_value = matching_step
        self.gsg.step(make_ctx(), self.observations)
        return self.gsg.output_goals()

    def test_there_are_six_face_and_eight_corner_views(self) -> None:
        directions = CubeViewGoalGenerator.VIEW_DIRECTIONS

        self.assertEqual(directions.shape, (14, 3))
        nptest.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        self.assertEqual(int(np.sum(np.count_nonzero(directions, axis=1) == 1)), 6)
        self.assertEqual(int(np.sum(np.count_nonzero(directions, axis=1) == 3)), 8)

    def test_face_view_looks_at_the_face_from_outside(self) -> None:
        (goal,) = self.step_at(51)

        direction = np.array([1.0, 0.0, 0.0])
        face_center = self.CENTER + direction * 0.05
        nptest.assert_allclose(goal.info["view_direction"], direction)
        nptest.assert_allclose(goal.info["model_frame_target_loc"], face_center)
        # Identity MLH at the model origin, backed off along the view direction.
        nptest.assert_allclose(
            goal.location, self.sensed_location + face_center + direction * 0.03
        )
        nptest.assert_allclose(
            goal.morphological_features["pose_vectors"][0], -direction
        )

    def test_corner_view_looks_at_the_corner(self) -> None:
        self.gsg._next_view_index = 6

        (goal,) = self.step_at(351)

        nptest.assert_allclose(
            goal.info["model_frame_target_loc"], self.CENTER + 0.04 * np.ones(3)
        )
        nptest.assert_allclose(
            goal.morphological_features["pose_vectors"][0], -np.ones(3) / np.sqrt(3)
        )

    def test_view_direction_is_transformed_by_mlh_pose(self) -> None:
        rotation = Rotation.from_euler("xyz", [0, 0, 90], degrees=True)
        self.mlh = {
            "graph_id": TOP_ID,
            "mlh_id": 3,
            "location": self.CENTER,
            "rotation": rotation,
        }

        (goal,) = self.step_at(51)

        body_direction = rotation.inv().apply([1.0, 0.0, 0.0])
        nptest.assert_allclose(
            goal.morphological_features["pose_vectors"][0], -body_direction
        )
        nptest.assert_allclose(
            goal.location,
            self.sensed_location + body_direction * (0.05 + 0.03),
            atol=1e-12,
        )

    def test_each_view_is_held_for_steps_per_view_matching_steps(self) -> None:
        view_indices = []
        for matching_step in range(1, 801):
            for goal in self.step_at(matching_step):
                view_indices.append((matching_step, goal.info["view_index"]))

        # The initial view is explored before the first jump.
        self.assertEqual(view_indices, [(51 + 50 * view, view) for view in range(14)])

    def test_view_is_retried_while_mlh_is_not_a_known_object(self) -> None:
        self.mlh = identity_mlh("no_observations_yet")
        self.assertEqual(self.step_at(51), [])

        self.mlh = identity_mlh(TOP_ID)
        (goal,) = self.step_at(52)
        self.assertEqual(goal.info["view_index"], 0)

    def test_failed_view_jumps_are_not_attributed_to_the_mlh(self) -> None:
        (goal,) = self.step_at(51)

        self.assertIsNone(goal.info["hypothesis_to_test_graph_id"])
        self.assertIsNone(goal.info["hypothesis_to_test_mlh_id"])

    def test_reset_restarts_from_the_first_view(self) -> None:
        self.step_at(51)
        self.step_at(101)

        self.gsg.reset()

        self.assertEqual(self.step_at(1), [])
        (goal,) = self.step_at(51)
        self.assertEqual(goal.info["view_index"], 0)


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


class ChildObjectsGoalGeneratorTest(unittest.TestCase):
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
        lm.object_id_feature_names = {}
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
        self.gsg = ChildObjectsGoalGenerator(
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


def surface_graph(
    positions,
    normals,
    hues=None,
    saturations=None,
    curvatures=None,
    curvature_directions=None,
):
    """A sensor-channel graph storing hue and surface geometry.

    Args:
        positions: The node locations.
        normals: The surface normal of every node (or one for all).
        hues: The hue of every node; 0 if None.
        saturations: The saturation of every node; 1 if None.
        curvatures: The two signed principal curvatures (in 1/m) of every node (or
            one pair for all); flat if None.
        curvature_directions: The two principal curvature directions of every node
            (or one pair for all); only used with curvatures.

    Returns:
        The graph.
    """
    num_nodes = len(positions)
    pose_vectors = np.zeros((num_nodes, 3, 3))
    pose_vectors[:, 0] = np.broadcast_to(
        np.asarray(normals, dtype=float), (num_nodes, 3)
    )
    log_curvatures = np.zeros((num_nodes, 2))
    if curvatures is not None:
        log_curvatures[:] = log_sign(
            np.broadcast_to(np.asarray(curvatures, dtype=float), (num_nodes, 2))
        )
        pose_vectors[:, 1:] = np.broadcast_to(
            np.asarray(curvature_directions, dtype=float), (num_nodes, 2, 3)
        )
    hsv = np.column_stack(
        [
            np.zeros(num_nodes) if hues is None else hues,
            np.ones(num_nodes) if saturations is None else saturations,
            np.ones(num_nodes),
        ]
    )
    return FakeGraph(
        positions,
        {
            "pose_vectors": pose_vectors.reshape(num_nodes, 9),
            "hsv": hsv,
            "principal_curvatures_log": log_curvatures,
        },
    )


def cylinder_surface(radius, num_around, num_along):
    """Points around a cylinder along z, with outward normals.

    Returns:
        The points, their normals, and the unit tangents around the cylinder.
    """
    angles = np.arange(num_around) * 2 * np.pi / num_around
    heights = np.arange(num_along) * SURFACE_SPACING
    angle, height = np.meshgrid(angles, heights, indexing="ij")
    angle, height = angle.ravel(), height.ravel()
    normals = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(angle.size)])
    tangents = np.column_stack([-np.sin(angle), np.cos(angle), np.zeros(angle.size)])
    points = np.column_stack([radius * normals[:, :2], height])
    return points, normals, tangents


class SensoryFeatureSpreadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = None
        self.mlh = None
        lm = MagicMock()
        lm.learning_module_id = "learning_module_0"
        lm.object_id_feature_names = {}
        lm.max_match_distance = 0.01
        lm.get_all_known_object_ids.side_effect = lambda: [TOP_ID]
        lm._get_current_mlh.side_effect = lambda: self.mlh
        lm.get_graph.side_effect = lambda _graph_id, input_channel=None: (
            {SENSOR_CHANNEL: self.graph} if input_channel is None else self.graph
        )
        lm.get_input_channels_in_graph.side_effect = lambda _graph_id: [SENSOR_CHANNEL]
        self.gsg = ChildObjectsGoalGenerator()
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
            "principal_curvatures_log": features["principal_curvatures_log"],
            **sensed_overrides,
        }
        percept = MagicMock(
            sender_type="SM",
            sender_id=SENSOR_CHANNEL,
            morphological_features={"pose_vectors": sensed["pose_vectors"]},
            non_morphological_features={
                "hsv": sensed["hsv"],
                "principal_curvatures_log": sensed["principal_curvatures_log"],
            },
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
        side, normals, tangents = cylinder_surface(radius, 60, 8)
        top = side[:, 2].max()
        cap = plane_points([-0.019, -0.019, top], [1, 0, 0], [0, 1, 0], 20, 20)
        cap = cap[np.linalg.norm(cap[:, :2], axis=1) < radius - SURFACE_SPACING]
        side_directions = np.stack(
            [np.tile([0.0, 0.0, 1.0], (len(side), 1)), tangents], axis=1
        )
        cap_directions = np.tile([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (len(cap), 1, 1))
        self.graph = surface_graph(
            np.vstack([side, cap]),
            np.vstack([normals, np.tile([0, 0, 1], (len(cap), 1))]),
            curvatures=np.vstack(
                [np.tile([0.0, -1 / radius], (len(side), 1)), np.zeros((len(cap), 2))]
            ),
            curvature_directions=np.vstack([side_directions, cap_directions]),
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

    def test_curvature_predicts_normals_around_a_tightly_curved_surface(self) -> None:
        # Around a cylinder of radius 3.5mm, neighboring normals differ by more than
        # max_continuity_normal_error, but are as its curvature predicts.
        radius = 0.0035
        points, normals, tangents = cylinder_surface(radius, 7, 8)
        directions = np.stack(
            [np.tile([0.0, 0.0, 1.0], (len(points), 1)), tangents], axis=1
        )
        self.graph = surface_graph(
            points,
            normals,
            curvatures=[0.0, -1 / radius],
            curvature_directions=directions,
        )

        self.assertTrue(np.all(self.spread_from(4)))

        self.graph = surface_graph(points, normals)
        self.assertLess(
            self.spread_from(4).mean(),
            0.5,
            "Without curvature, the surface is modeled as flat, so the normals of "
            "neighbors around the cylinder are not predicted.",
        )

    def test_stops_at_a_step_between_parallel_surfaces(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 20, 10)
        right = points[:, 0] >= 0.02
        points[right, 2] += 0.004
        self.graph = surface_graph(points, [0, 0, 1])

        inhibited = self.spread_from(25)

        nptest.assert_array_equal(
            inhibited,
            ~right,
            "The normals agree across the step, but the locations beyond it are not "
            "on the surface the nodes before it predict.",
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

    def test_nodes_with_unreliable_normals_are_inhibited_but_do_not_stop_it(
        self,
    ) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        normals = np.tile([0.0, 0.0, 1.0], (100, 1))
        # Averaging both sides of a thin wall shortens the stored normal.
        unreliable = [33, 34, 66]
        normals[unreliable] = [0.1, 0.05, 0.0]
        self.graph = surface_graph(points, normals)

        inhibited = self.spread_from(55)

        self.assertTrue(np.all(inhibited))

    def test_noisy_neighbors_do_not_stop_spreading(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        normals = np.tile([0.0, 0.0, 1.0], (100, 1))
        noisy = [44, 45]
        normals[noisy] = [np.sin(np.radians(60)), 0.0, np.cos(np.radians(60))]
        self.graph = surface_graph(points, normals)
        self.gsg.min_inhibited_neighbor_fraction = None
        self.gsg.min_region_fraction = None

        inhibited = self.spread_from(55)

        self.assertFalse(np.any(inhibited[noisy]), "The noisy nodes are not spread to.")
        self.assertTrue(np.all(np.delete(inhibited, noisy)))

    def test_a_small_region_is_inhibited_with_the_region_it_merges_into(
        self,
    ) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 20, 10)
        x, y = np.round(points[:, :2] / SURFACE_SPACING).astype(int).T
        patch = (x >= 8) & (x < 11) & (y >= 3) & (y < 6)
        self.graph = surface_graph(points, [0, 0, 1], hues=np.where(patch, 0.5, 0.0))
        self.gsg.min_inhibited_neighbor_fraction = None

        self.gsg.min_region_fraction = None
        nptest.assert_array_equal(self.spread_from(0), ~patch)
        nptest.assert_array_equal(self.spread_from(np.flatnonzero(patch)[4]), patch)

        self.gsg.min_region_fraction = 0.05
        self.assertTrue(
            np.all(self.spread_from(0)),
            "The patch, of 9 of 200 nodes, is merged into the plane around it.",
        )
        self.assertTrue(np.all(self.spread_from(np.flatnonzero(patch)[4])))

    def test_small_regions_merge_into_the_smallest_neighboring_region(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 20, 10)
        x, y = np.round(points[:, :2] / SURFACE_SPACING).astype(int).T
        medium = (x >= 17) & (y < 4)
        small = (x >= 14) & (x < 17) & (y < 3)
        hues = np.select([medium, small], [0.5, 0.25], default=0.0)
        self.graph = surface_graph(points, [0, 0, 1], hues=hues)
        self.gsg.min_inhibited_neighbor_fraction = None
        self.gsg.min_region_fraction = 0.1

        inhibited = self.spread_from(np.flatnonzero(small)[4])

        nptest.assert_array_equal(
            inhibited,
            medium | small,
            "The small region (9 nodes) merges into its smallest neighbor (12 "
            "nodes), which is then large enough (of at least 20 nodes) to keep.",
        )

    def test_small_regions_do_not_merge_through_a_thin_wall(self) -> None:
        outside = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        inside = plane_points([0.008, 0.008, -0.001], [1, 0, 0], [0, 1, 0], 3, 3)
        self.graph = surface_graph(
            np.vstack([outside, inside]),
            np.vstack([np.tile([0, 0, 1], (100, 1)), np.tile([0, 0, -1], (9, 1))]),
        )
        self.gsg.min_region_fraction = 0.1

        inhibited = self.spread_from(55)

        self.assertTrue(np.all(inhibited[:100]))
        self.assertFalse(
            np.any(inhibited[100:]),
            "The inside patch's only neighboring region is on the far side of the "
            "wall, so it is not merged into it.",
        )

    def test_noisy_nodes_surrounded_by_the_spread_are_inhibited(self) -> None:
        points = plane_points([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 10)
        normals = np.tile([0.0, 0.0, 1.0], (100, 1))
        noisy = [44, 45]
        normals[noisy] = [np.sin(np.radians(60)), 0.0, np.cos(np.radians(60))]
        self.graph = surface_graph(points, normals)

        inhibited = self.spread_from(55)

        self.assertTrue(np.all(inhibited))
        (record,) = self.gsg.spread_records
        self.assertCountEqual(
            record.node_order[-2:], noisy, "Surrounded nodes are inhibited last."
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
