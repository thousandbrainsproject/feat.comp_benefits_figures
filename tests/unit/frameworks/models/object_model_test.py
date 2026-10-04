# Copyright 2025-2026 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import unittest

import numpy as np

from tbp.monty.frameworks.models.object_model import (
    GraphObjectModel,
    GridObjectModel,
    GridTooSmallError,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import check_orthonormal
from tbp.monty.geometry import Rotation


class ObjectModelTest(unittest.TestCase):
    def setUp(self):
        self.dummy_locs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        self.dummy_pv = np.eye(3)
        pv = self.dummy_pv.flatten()
        self.dummy_features = {
            "pose_vectors": np.vstack([pv, pv, pv, pv]),
            "pose_fully_defined": np.array([True, True, True, True]),
            "hsv": np.array(
                [
                    [0.1, 1, 1],
                    [0.0, 1, 1],
                    [0.9, 1, 1],
                    [0.8, 1, 1],
                ]
            ),
            "curvature": np.array([0, 1, 2, 1]),
        }
        self.dummy_delta_thresholds = {
            "distance": 0.5,
            "pose_vectors": np.ones(3),
            "hsv": [0.1, 1, 1],
            "curvature": 0.1,
        }

    def test_create_graph_object_model(self):
        model = GraphObjectModel("test_model")
        self.assertIsNotNone(model)
        self.assertIsNone(model._graph)
        # make sure __repr__ works
        self.assertIsInstance(repr(model), str)

    def test_create_grid_object_model(self):
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=10
        )
        self.assertIsNotNone(model)
        self.assertIsNone(model._graph)
        # make sure __repr__ works
        self.assertIsInstance(repr(model), str)

    def test_can_build_graph_object_model(self):
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds=self.dummy_delta_thresholds,
        )
        self.assertEqual(model.num_nodes, 4, "graph model should have 4 nodes.")
        for feature in self.dummy_features:
            self.assertIn(
                feature, model.feature_mapping.keys(), f"{feature} not stored in model."
            )

    def test_can_build_grid_object_model(self):
        max_nodes, max_size, num_voxels_per_dim = 10, 10, 10
        model = GridObjectModel("test_model", max_nodes, max_size, num_voxels_per_dim)
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(model.num_nodes, 4, "graph model should have 4 nodes.")
        for feature in self.dummy_features:
            self.assertIn(
                feature, model.feature_mapping.keys(), f"{feature} not stored in model."
            )

    def test_apply_delta_thresholds_correctly(self):
        # Test distance.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
            },
        )
        self.assertEqual(
            model.num_nodes,
            1,
            "Graph model should have 1 node since no locations are more that 2cm away "
            f"from the first one. Model has {model.num_nodes} nodes.",
        )
        # Test pose_vectors.
        model = GraphObjectModel("test_model")
        features = copy.deepcopy(self.dummy_features)
        features["pose_vectors"][1] = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1])
        model.build_model(
            self.dummy_locs,
            features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "pose_vectors": np.ones(3),
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but pose vectors 0 and 1 differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )
        # Test hsv.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "hsv": [0.1, 1, 1],
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but hsv 0 (0.1) and 3 (0.8) differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )
        # Test curvature.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "curvature": 1,
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but curvature 0 (0) and 2 (2) differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )

    def test_grid_locations_binned_correctly(self):
        # In test_can_build_grid_object_model each voxel is of size 1cm^3 so every
        # location gets its own voxel. Here, each voxel is of size 2cm^3 which means
        # that all locations should be binned into the same voxel and the location value
        # should be averaged.
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(
            model.num_nodes,
            1,
            f"graph model should have 1 node but has {model.num_nodes}.",
        )
        self.assertListEqual(
            list(model.pos[0]),
            [0.5, 0.5, 0],
            "location stored in model should be average of all locations "
            f"([0.5, 0.5, 0]) but is {model.pos}.",
        )

    def test_grid_features_are_averaged_correctly(self):
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertListEqual(
            list(model.get_values_for_feature("pose_vectors")[0]),
            list(self.dummy_pv.flatten()),
            "pose_vectors not averaged correctly. "
            "Since all PVs are the same, average should be same as dummy_pv.",
        )
        self.assertEqual(
            model.get_values_for_feature("pose_fully_defined")[0],
            True,
            "Average of pose_fully_defined should be True (majority vote).",
        )
        avg_hsv = model.get_values_for_feature("hsv")[0]
        self.assertListEqual(
            list(avg_hsv),
            [0.95, 1, 1],
            "hsv not averaged correctly. "
            "(keep in mind we want the circular average where 0==1).",
        )
        self.assertEqual(
            model.get_values_for_feature("curvature")[0],
            np.mean(self.dummy_features["curvature"]),
            "Curvature not averaged correctly.",
        )
        # Check different cases to make sure pose_fully_defined is averaged correctly.
        features = copy.deepcopy(self.dummy_features)
        features["pose_fully_defined"] = np.array([False, False, False, True])
        model2 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model2.build_model(
            self.dummy_locs,
            features,
        )
        self.assertEqual(
            model2.get_values_for_feature("pose_fully_defined")[0],
            False,
            "Average of pose_fully_defined should be False (majority vote).",
        )

        features["pose_fully_defined"][0] = True
        model2.build_model(
            self.dummy_locs,
            features,
        )
        self.assertEqual(
            model2.get_values_for_feature("pose_fully_defined")[0],
            True,
            "Average of pose_fully_defined should be True (True in case of tie).",
        )
        # Check different cases to make sure pose_vectors are averaged correctly.
        #     Check opposite surface side vectors are sorted out.
        features = copy.deepcopy(self.dummy_features)
        rot = Rotation.from_euler("z", 180, degrees=True)
        opposite_pv = rot.apply(np.eye(3))
        features["pose_vectors"][0] = opposite_pv.flatten()
        model3 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model3.build_model(
            self.dummy_locs,
            features,
        )
        self.assertListEqual(
            list(np.round(model3.get_values_for_feature("pose_vectors")[0], 3)),
            list(self.dummy_pv.flatten()),
            "Average of pose_vectors should still be the same. Changed pv is ignored "
            "because its angle is too far from the other ones (other surface side)",
        )
        features["pose_vectors"][1] = opposite_pv.flatten()
        features["pose_vectors"][2] = opposite_pv.flatten()
        model3.build_model(
            self.dummy_locs,
            features,
        )
        self.assertListEqual(
            list(np.round(model3.get_values_for_feature("pose_vectors")[0], 3)),
            list(np.round(opposite_pv.flatten(), 3)),
            "If majority of pose_vectors are on the opposite side of the surface (here)"
            " opposite_pv), those should become the average.",
        )
        #    Check averaged pose vector are still unit vectors.
        features = copy.deepcopy(self.dummy_features)
        rot = Rotation.from_euler("z", 20, degrees=True)
        rotated_pv = rot.apply(np.eye(3))
        features["pose_vectors"][0] = rotated_pv.flatten()
        model4 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model4.build_model(
            self.dummy_locs,
            features,
        )
        avg_pvs = model4.get_values_for_feature("pose_vectors")[0].reshape((3, 3))
        for pv in avg_pvs:
            self.assertAlmostEqual(
                np.linalg.norm(pv), 1.0, 3, "Average PVs are not unit vectors anymore"
            )
        self.assertTrue(check_orthonormal(avg_pvs), "Average PVs are not orthonormal")

    def test_max_nodes_applied_correctly(self):
        model = GridObjectModel(
            "test_model", max_nodes=3, max_size=10, num_voxels_per_dim=10
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(model.num_nodes, 3, "Max nodes not applied correctly.")

    def test_max_size_applied_correctly(self):
        """Test that GridTooSmallError is raised if locations are outside of grid."""
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=1, num_voxels_per_dim=10
        )
        with self.assertRaises(GridTooSmallError):
            model.build_model(
                self.dummy_locs,
                self.dummy_features,
            )


class GridObjectModelHotspotTest(unittest.TestCase):
    def setUp(self):
        self.locs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
        pv = np.eye(3).flatten()
        self.features = {
            "pose_vectors": np.vstack([pv, pv, pv, pv]),
            "pose_fully_defined": np.array([True, True, True, True]),
        }
        self.model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=10
        )
        self.model.build_model(self.locs, self.features)

    def node_at(self, location) -> int:
        return int(np.argmin(np.linalg.norm(self.model.pos - location, axis=1)))

    def test_untagged_model_has_no_hotspots(self):
        self.assertIsNone(self.model.get_max_hotspot_node())
        np.testing.assert_array_equal(self.model.hotspot_values, np.zeros(4))
        np.testing.assert_array_equal(self.model.hotspot_counts, np.zeros(4))

    def test_tags_go_to_nearest_node(self):
        self.model.tag_hotspots([[0.9, 0.1, 0.0]], [2.0])

        node = self.node_at([1, 0, 0])
        expected = np.zeros(4)
        expected[node] = 2.0
        np.testing.assert_array_equal(self.model.hotspot_values, expected)
        self.assertEqual(self.model.hotspot_counts[node], 1)

    def test_values_are_running_average_across_calls(self):
        self.model.tag_hotspots([[0, 1, 0]], [3.0])
        self.model.tag_hotspots([[0, 1, 0], [0, 1, 0]], [1.0, 2.0])

        node = self.node_at([0, 1, 0])
        self.assertAlmostEqual(self.model.hotspot_values[node], 2.0)
        self.assertEqual(self.model.hotspot_counts[node], 3)

    def test_max_hotspot_is_node_with_highest_average(self):
        # Node at (1, 1, 0) is tagged most often, but node at (0, 1, 0) has the
        # highest average value.
        self.model.tag_hotspots(
            [[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0]], [1.0, 1.0, 1.0, 1.5]
        )

        node_id, value = self.model.get_max_hotspot_node()

        self.assertEqual(node_id, self.node_at([0, 1, 0]))
        self.assertAlmostEqual(value, 1.5)

    def test_max_hotspot_ignores_untagged_nodes_when_all_tags_are_zero(self):
        self.model.tag_hotspots([[1, 0, 0]], [0.0])

        node_id, value = self.model.get_max_hotspot_node()

        self.assertEqual(node_id, self.node_at([1, 0, 0]))
        self.assertEqual(value, 0.0)

    def test_hotspots_survive_copy_round_trip(self):
        self.model.tag_hotspots([[1, 1, 0]], [4.0])

        restored = copy.deepcopy(self.model)

        np.testing.assert_array_equal(
            restored.hotspot_values, self.model.hotspot_values
        )
        np.testing.assert_array_equal(
            restored.hotspot_counts, self.model.hotspot_counts
        )

    def test_model_pickled_before_hotspots_existed_still_works(self):
        del self.model._hotspot_sum
        del self.model._hotspot_count

        self.assertIsNone(self.model.get_max_hotspot_node())
        self.model.tag_hotspots([[0, 0, 0]], [1.0])
        self.assertEqual(self.model.get_max_hotspot_node()[0], self.node_at([0, 0, 0]))

    def test_hotspots_are_carried_over_when_model_is_updated(self):
        self.model.tag_hotspots([[1, 1, 0]], [5.0])
        new_locs = np.array([[2, 2, 0], [2, 3, 0]], dtype=float)
        pv = np.eye(3).flatten()

        self.model.update_model(
            locations=new_locs,
            features={
                "pose_vectors": np.vstack([pv, pv]),
                "pose_fully_defined": np.array([True, True]),
            },
            location_rel_model=np.zeros(3),
            object_location_rel_body=np.zeros(3),
            object_rotation=Rotation.identity(),
        )

        self.assertEqual(self.model.num_nodes, 6)
        node_id, value = self.model.get_max_hotspot_node()
        self.assertEqual(node_id, self.node_at([1, 1, 0]))
        self.assertAlmostEqual(value, 5.0)
        self.assertEqual(self.model.hotspot_counts.sum(), 1)
