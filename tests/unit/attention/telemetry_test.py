# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import unittest

import numpy as np

from tbp.monty.attention.attention_system import (
    DEFAULT_VOXEL_SIZE,
    AttentionSystem,
)
from tbp.monty.attention.telemetry import (
    AttentionSystemTelemetry,
    NoopAttentionSystemTelemetry,
)
from tbp.monty.cmp import AttentionRegion
from tbp.monty.frameworks.models.buffer import BufferEncoder

from .attention_system_test import goal_at, point_in, region

NEAR_VOXEL = (0, 0, 0)
FAR_VOXEL = (50, 0, 0)
NEAR_POINT = point_in(NEAR_VOXEL)
FAR_POINT = point_in(FAR_VOXEL)


class AttentionSystemTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = AttentionSystemTelemetry()
        self.system = AttentionSystem(telemetry=self.telemetry)

    def test_each_step_records_a_snapshot(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [region(FAR_POINT)])
        self.assertEqual(len(self.system.state_dict()["voxel_grids"]), 2)

    def test_a_snapshot_is_unaffected_by_later_steps(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        first = self.system.state_dict()["voxel_grids"][0]
        weight_then = first["weight"].to_numpy().copy()

        # The next step decays the live grid in place and grows it.
        self.system.step([], [region(FAR_POINT)])

        grids = self.system.state_dict()["voxel_grids"]
        np.testing.assert_array_equal(grids[0]["weight"].to_numpy(), weight_then)
        self.assertEqual(len(grids[0]), 1)
        self.assertEqual(len(grids[1]), 2)

    def test_reset_discards_the_snapshots(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.reset()
        self.assertEqual(self.system.state_dict()["voxel_grids"], [])

    def test_the_proposed_grid_holds_only_this_steps_regions(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [region(FAR_POINT)])

        state = self.system.state_dict()
        self.assertEqual(len(state["proposed"][1]), 1)
        self.assertEqual(len(state["voxel_grids"][1]), 2)

    def test_snapshots_encode_into_arrays(self) -> None:
        self.system.step([], [region(NEAR_POINT, weight=2)])
        self.system.step([], [region(NEAR_POINT, FAR_POINT, weight=2)])
        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        snapshot = encoded["voxel_grids"][1]
        self.assertEqual(snapshot["voxels"], [list(NEAR_VOXEL), list(FAR_VOXEL)])
        # Both voxels take this step's freshly proposed weight outright.
        self.assertEqual(snapshot["weight"], [2, 2])

    def test_an_empty_grid_snapshot_is_exported_empty(self) -> None:
        self.system.step([], [region(NEAR_POINT, weight=0.15)])
        # The first empty step decays 0.15 to within the rate of zero,
        # clamping it and expiring the voxel.
        self.system.step([], [])
        self.system.step([], [])
        snapshot = self.system.state_dict()["voxel_grids"][2]
        self.assertEqual(len(snapshot), 0)

    def test_state_dict_is_json_encodable(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [])
        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        self.assertEqual(encoded["voxel_grids"][0]["voxels"], [list(NEAR_VOXEL)])

    def test_a_default_telemetry_is_created_when_none_is_supplied(self) -> None:
        system = AttentionSystem()
        system.step([], [region(NEAR_POINT)])
        self.assertEqual(len(system.state_dict()["voxel_grids"]), 1)

    def test_state_dict_carries_the_grid_geometry(self) -> None:
        state = self.system.state_dict()
        self.assertEqual(state["voxel_size"], DEFAULT_VOXEL_SIZE)


class NoopAttentionSystemTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = NoopAttentionSystemTelemetry()
        self.system = AttentionSystem(telemetry=self.telemetry)

    def test_steps_record_nothing(self) -> None:
        self.system.step([goal_at(NEAR_POINT)], [region(NEAR_POINT)])

        state = self.system.state_dict()
        self.assertEqual(
            {k: state[k] for k in ("voxel_grids", "goals", "regions", "proposed")},
            {"voxel_grids": [], "goals": [], "regions": [], "proposed": []},
        )


class AttentionSystemGoalTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = AttentionSystemTelemetry()
        self.system = AttentionSystem(telemetry=self.telemetry)

    def test_each_step_records_every_goal_tagged_with_the_filter_decision(
        self,
    ) -> None:
        inside = goal_at(NEAR_POINT)
        outside = goal_at([9.0, 9, 9])
        self.system.step([inside, outside], [region(NEAR_POINT)])

        state = self.system.state_dict()
        self.assertEqual(state["goals"], [[inside, outside]])
        self.assertTrue(inside.info["passed_attention_filter"])
        self.assertFalse(outside.info["passed_attention_filter"])

    def test_a_pass_through_step_tags_every_goal_as_passed(self) -> None:
        goal = goal_at(NEAR_POINT)
        self.system.step([goal], [])

        self.assertEqual(self.system.state_dict()["goals"], [[goal]])
        self.assertTrue(goal.info["passed_attention_filter"])

    def test_the_tag_is_json_encoded_with_the_goal(self) -> None:
        inside = goal_at(NEAR_POINT)
        outside = goal_at([9.0, 9, 9])
        self.system.step([inside, outside], [region(NEAR_POINT)])

        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        self.assertEqual(
            [g["info"]["passed_attention_filter"] for g in encoded["goals"][0]],
            [True, False],
        )

    def test_reset_discards_the_goal_records(self) -> None:
        self.system.step([goal_at(NEAR_POINT)], [region(NEAR_POINT)])
        self.system.reset()

        self.assertEqual(self.system.state_dict()["goals"], [])


class AttentionSystemRegionTelemetryTest(unittest.TestCase):
    """Each step's proposals are kept as received, one region per module."""

    def setUp(self) -> None:
        self.system = AttentionSystem(telemetry=AttentionSystemTelemetry())

    def test_each_step_records_the_proposed_regions_unmerged(self) -> None:
        near = AttentionRegion.uniform([NEAR_POINT], 1.0, sender_id="SM_3")
        far = AttentionRegion.uniform([FAR_POINT], -1.0, sender_id="learning_module_2")
        self.system.step([], [near, far])
        self.system.step([], [])

        self.assertEqual(self.system.state_dict()["regions"], [[near, far], []])

    def test_records_the_regions_before_they_are_merged(self) -> None:
        # The grid pools co-voxel points and drops senders; the record does not.
        region_a = AttentionRegion.uniform([NEAR_POINT], 1.0, sender_id="a")
        region_b = AttentionRegion.uniform([NEAR_POINT], -1.0, sender_id="b")
        self.system.step([], [region_a, region_b])

        state = self.system.state_dict()
        self.assertEqual([r.sender_id for r in state["regions"][0]], ["a", "b"])
        self.assertEqual(len(state["voxel_grids"][0]), 1)

    def test_regions_are_json_encodable(self) -> None:
        self.system.step([], [AttentionRegion.uniform([NEAR_POINT], 1.0, "SM_3")])

        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        self.assertEqual(
            encoded["regions"],
            [
                [
                    {
                        "locations": [list(NEAR_POINT)],
                        "weights": [1.0],
                        "sender_id": "SM_3",
                    }
                ]
            ],
        )

    def test_reset_discards_the_region_records(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.reset()

        state = self.system.state_dict()
        self.assertEqual(state["regions"], [])
        self.assertEqual(state["proposed"], [])
