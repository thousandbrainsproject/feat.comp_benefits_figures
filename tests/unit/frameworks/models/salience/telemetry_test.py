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

from tbp.monty.cmp import AttentionRegion
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.salience.telemetry import SalienceSMTelemetry


def region_at(*locations) -> AttentionRegion:
    """Build a region at the given locations.

    Returns:
        A region whose only meaningful property here is its locations.

    """
    return AttentionRegion.uniform(np.asarray(locations, dtype=float), 12)


class SalienceSMTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry()
        self.mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        self.region = region_at([0.0, 0, 1], [1.0, 1, 1])

    def test_state_dict_includes_the_snapshot_telemetry(self) -> None:
        state = self.telemetry.state_dict()
        self.assertIn("raw_observations", state)
        self.assertIn("sm_properties", state)

    def test_records_one_entry_per_step(self) -> None:
        self.telemetry.segmentation(self.mask, self.region)
        self.telemetry.segmentation(None, AttentionRegion.empty())
        state = self.telemetry.state_dict()
        self.assertEqual(len(state["segmentation_maps"]), 2)
        self.assertEqual(len(state["regions"]), 2)

    def test_state_dict_holds_what_was_recorded(self) -> None:
        self.telemetry.segmentation(self.mask, self.region)
        state = self.telemetry.state_dict()
        np.testing.assert_array_equal(state["segmentation_maps"][0], self.mask)
        self.assertIs(state["regions"][0], self.region)

    def test_a_step_without_segmentation_records_none(self) -> None:
        self.telemetry.segmentation(None, AttentionRegion.empty())
        state = self.telemetry.state_dict()
        self.assertIsNone(state["segmentation_maps"][0])
        self.assertEqual(len(state["regions"][0]), 0)

    def test_reset_discards_the_recordings(self) -> None:
        self.telemetry.segmentation(self.mask, self.region)
        self.telemetry.reset()
        state = self.telemetry.state_dict()
        self.assertEqual(state["segmentation_maps"], [])
        self.assertEqual(state["regions"], [])

    def test_state_dict_is_json_encodable(self) -> None:
        self.telemetry.segmentation(self.mask, self.region)
        self.telemetry.segmentation(None, AttentionRegion.empty())
        encoded = json.loads(json.dumps(self.telemetry.state_dict(), cls=BufferEncoder))
        self.assertEqual(encoded["segmentation_maps"][0], [[1, 0], [0, 1]])
        self.assertIsNone(encoded["segmentation_maps"][1])
        self.assertEqual(
            encoded["regions"][0],
            {
                "locations": [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                "weights": [12.0, 12.0],
                "sender_id": "",
            },
        )
        self.assertEqual(
            encoded["regions"][1], {"locations": [], "weights": [], "sender_id": ""}
        )


class SalienceSMTelemetrySalienceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry()
        self.salience_map = np.array([[0.1, 0.9], [0.5, 0.0]])

    def test_each_call_records_a_map(self) -> None:
        self.telemetry.salience(self.salience_map)
        state = self.telemetry.state_dict()
        np.testing.assert_array_equal(state["salience_maps"][0], self.salience_map)

    def test_reset_discards_the_maps(self) -> None:
        self.telemetry.salience(self.salience_map)
        self.telemetry.reset()
        self.assertEqual(self.telemetry.state_dict()["salience_maps"], [])
