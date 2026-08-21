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
import quaternion as qt

from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.salience.telemetry import (
    NoopSalienceSMTelemetry,
    SalienceSMTelemetry,
)


class SalienceSMTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry()
        self.mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    def test_state_dict_includes_the_snapshot_telemetry(self) -> None:
        state = self.telemetry.state_dict()
        self.assertIn("raw_observations", state)
        self.assertIn("sm_properties", state)

    def test_records_one_entry_per_step(self) -> None:
        self.telemetry.segmentation_map(self.mask)
        self.telemetry.segmentation_map(None)
        state = self.telemetry.state_dict()
        self.assertEqual(len(state["segmentation_maps"]), 2)

    def test_state_dict_holds_what_was_recorded(self) -> None:
        self.telemetry.segmentation_map(self.mask)
        state = self.telemetry.state_dict()
        np.testing.assert_array_equal(state["segmentation_maps"][0], self.mask)

    def test_a_step_without_segmentation_records_none(self) -> None:
        self.telemetry.segmentation_map(None)
        state = self.telemetry.state_dict()
        self.assertIsNone(state["segmentation_maps"][0])

    def test_reset_discards_the_recordings(self) -> None:
        self.telemetry.segmentation_map(self.mask)
        self.telemetry.reset()
        state = self.telemetry.state_dict()
        self.assertEqual(state["segmentation_maps"], [])

    def test_state_dict_is_json_encodable(self) -> None:
        self.telemetry.segmentation_map(self.mask)
        self.telemetry.segmentation_map(None)
        encoded = json.loads(json.dumps(self.telemetry.state_dict(), cls=BufferEncoder))
        self.assertEqual(encoded["segmentation_maps"][0], [[1, 0], [0, 1]])
        self.assertIsNone(encoded["segmentation_maps"][1])


class SalienceSMTelemetrySalienceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry()
        self.salience_map = np.array([[0.1, 0.9], [0.5, 0.0]])

    def test_each_call_records_a_map(self) -> None:
        self.telemetry.salience_map(self.salience_map)
        state = self.telemetry.state_dict()
        np.testing.assert_array_equal(state["salience_maps"][0], self.salience_map)

    def test_reset_discards_the_maps(self) -> None:
        self.telemetry.salience_map(self.salience_map)
        self.telemetry.reset()
        self.assertEqual(self.telemetry.state_dict()["salience_maps"], [])


class NoopSalienceSMTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = NoopSalienceSMTelemetry()

    def test_records_nothing(self) -> None:
        self.telemetry.raw_observation(
            {"rgba": np.zeros((2, 2, 4))}, qt.quaternion(1, 0, 0, 0), np.zeros(3)
        )
        self.telemetry.salience_map(np.zeros((2, 2)))
        self.telemetry.segmentation_map(np.zeros((2, 2), dtype=np.uint8))

        self.assertEqual(
            self.telemetry.state_dict(),
            {
                "raw_observations": [],
                "sm_properties": [],
                "salience_maps": [],
                "segmentation_maps": [],
            },
        )

    def test_exports_the_same_keys_as_the_recording_telemetry(self) -> None:
        self.assertEqual(
            set(self.telemetry.state_dict()), set(SalienceSMTelemetry().state_dict())
        )
