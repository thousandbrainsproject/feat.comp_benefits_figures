# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.sensor_modules import (
    CameraSM,
    FeatureChangeFilter,
)


def create_percept(
    location: npt.NDArray[np.float64],
    on_object: bool,
    process_features_in_lm: bool,
) -> Message:
    """Create a percept for testing percept filters.

    Args:
        location: 3D location array.
        on_object: Whether the percept is on the object.
        process_features_in_lm: Whether the observation processor marked the percept as
            carrying valid features (on object and valid).

    Returns:
        A percept Message object
    """
    return Message(
        location=location,
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
            "on_object": on_object,
        },
        non_morphological_features={},
        confidence=1.0,
        pass_message=True,
        process_features_in_lm=process_features_in_lm,
        sender_id="SM_0",
        sender_type="SM",
    )


class FeatureChangeFilterTest(unittest.TestCase):
    @given(valid=st.booleans())
    def test_first_step_process_features_in_lm_iff_valid(self, valid: bool):
        feature_filter = FeatureChangeFilter(delta_thresholds={"distance": 0.5})
        result = feature_filter(
            create_percept(
                location=np.array([0.0, 0.0, 0.0]),
                on_object=valid,
                process_features_in_lm=valid,
            )
        )
        self.assertEqual(result.process_features_in_lm, valid)

    @given(valid=st.booleans(), feature_changed=st.booleans())
    def test_later_step_process_features_in_lm_iff_valid_and_changed(
        self, valid: bool, feature_changed: bool
    ):
        feature_filter = FeatureChangeFilter(delta_thresholds={"distance": 0.5})
        feature_filter(
            create_percept(
                location=np.array([0.0, 0.0, 0.0]),
                on_object=True,
                process_features_in_lm=True,
            )
        )
        location = (
            np.array([10.0, 0.0, 0.0]) if feature_changed else np.array([0.0, 0.0, 0.0])
        )
        result = feature_filter(
            create_percept(
                location=location, on_object=True, process_features_in_lm=valid
            )
        )
        self.assertEqual(result.process_features_in_lm, valid and feature_changed)


class CameraSMReorientationTest(unittest.TestCase):
    """CameraSM proposes whatever its reorientation component proposes."""

    def setUp(self) -> None:
        self.reorientation = Mock()
        self.sm = CameraSM(
            sensor_module_id="patch_0",
            features=["pose_vectors", "on_object"],
            reorientation=self.reorientation,
        )

    def test_proposes_the_components_goals_and_region(self) -> None:
        self.assertIs(self.sm.propose_goals(), self.reorientation.propose_goals())
        self.assertIs(self.sm.propose_region(), self.reorientation.propose_region())

    def test_reset_resets_the_component(self) -> None:
        self.sm.reset()
        self.reorientation.reset.assert_called_once_with()

    def test_state_dict_carries_the_components_telemetry(self) -> None:
        self.reorientation.state_dict.return_value = {"view_angle": [1.0]}
        self.assertEqual(self.sm.state_dict()["reorientation"], {"view_angle": [1.0]})

    def test_proposes_nothing_by_default(self) -> None:
        sm = CameraSM(
            sensor_module_id="patch_0", features=["pose_vectors", "on_object"]
        )
        self.assertEqual(sm.propose_goals(), [])
        self.assertIsNone(sm.propose_region())
