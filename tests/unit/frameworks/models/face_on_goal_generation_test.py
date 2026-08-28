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
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as nptest

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.face_on_goal_generation import FaceOnGoalGenerator

CHANNEL = "patch_2"


def percept(normal, view, depth: float = 0.08, location=(0.0, 1.5, 0.05)) -> Message:
    # A camera SM percept: the surface normal is the first pose vector, the
    # viewing direction and depth are non-morphological features.
    return Message(
        location=np.asarray(location, dtype=float),
        morphological_features={
            "pose_vectors": np.array([normal, [1.0, 0, 0], [0, 1.0, 0]], dtype=float),
            "pose_fully_defined": True,
            "on_object": 1.0,
        },
        non_morphological_features={
            "view_direction": np.asarray(view, dtype=float),
            "mean_depth": depth,
        },
        confidence=1.0,
        pass_message=True,
        sender_id=CHANNEL,
        sender_type="SM",
        process_features_in_lm=True,
    )


def gsg_with_lm(steps_since_goal: int = 100, **kwargs) -> FaceOnGoalGenerator:
    lm = MagicMock()
    lm.learning_module_id = "learning_module_2"
    lm.buffer.get_first_sensory_input_channel.return_value = CHANNEL
    lm.buffer.get_num_steps_post_output_goal_generated.return_value = steps_since_goal
    lm.buffer.get_num_matching_steps.return_value = steps_since_goal
    gsg = FaceOnGoalGenerator(**kwargs)
    gsg.parent_lm = lm
    return gsg


def view_at(angle_deg: float):
    # A camera looking down onto a +Z surface, tilted by the angle about Y.
    a = np.radians(angle_deg)
    return [np.sin(a), 0.0, -np.cos(a)]


class ViewAngleTest(unittest.TestCase):
    def test_is_zero_when_looking_along_the_negated_normal(self) -> None:
        gsg = gsg_with_lm()
        angle = gsg._measure_view_angle([percept([0, 0, 1], [0, 0, -1])])
        self.assertAlmostEqual(angle, 0.0)

    def test_matches_the_tilt_of_the_camera(self) -> None:
        gsg = gsg_with_lm()
        for tilt in (10.0, 45.0, 80.0):
            angle = gsg._measure_view_angle([percept([0, 0, 1], view_at(tilt))])
            self.assertAlmostEqual(angle, tilt, places=5)

    def test_is_none_without_a_view_direction_or_features(self) -> None:
        gsg = gsg_with_lm()
        p = percept([0, 0, 1], [0, 0, -1])
        del p.non_morphological_features["view_direction"]
        self.assertIsNone(gsg._measure_view_angle([p]))
        p = percept([0, 0, 1], [0, 0, -1])
        p.process_features_in_lm = False
        self.assertIsNone(gsg._measure_view_angle([p]))


class TriggerTest(unittest.TestCase):
    def test_a_single_steep_view_does_not_trigger(self) -> None:
        gsg = gsg_with_lm(max_view_angle=45.0, window=5)
        for _ in range(4):
            gsg.step(MagicMock(), [percept([0, 0, 1], view_at(10.0))])
        gsg.step(MagicMock(), [percept([0, 0, 1], view_at(80.0))])
        self.assertEqual(gsg.output_goals(), [])

    def test_a_persistently_steep_view_triggers_a_face_on_goal(self) -> None:
        gsg = gsg_with_lm(max_view_angle=45.0, window=3)
        for _ in range(3):
            gsg.step(MagicMock(), [percept([0, 0, 1], view_at(70.0))])
        goals = gsg.output_goals()
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].sender_type, "GSG")
        self.assertEqual(goals[0].sender_id, "learning_module_2")

    def test_waits_min_steps_between_goals(self) -> None:
        gsg = gsg_with_lm(
            steps_since_goal=5,
            max_view_angle=45.0,
            window=1,
            min_steps_between_goals=20,
        )
        gsg.step(MagicMock(), [percept([0, 0, 1], view_at(70.0))])
        self.assertEqual(gsg.output_goals(), [])

    def test_does_not_trigger_when_this_step_is_face_on(self) -> None:
        # A steep history but a face-on current view: the jump would be
        # computed from a good view, so wait.
        gsg = gsg_with_lm(max_view_angle=45.0, window=3)
        for _ in range(2):
            gsg.step(MagicMock(), [percept([0, 0, 1], view_at(80.0))])
        gsg.step(MagicMock(), [percept([0, 0, 1], view_at(5.0))])
        self.assertEqual(gsg.output_goals(), [])

    def test_the_goal_is_one_off(self) -> None:
        gsg = gsg_with_lm(max_view_angle=45.0, window=1)
        gsg.step(MagicMock(), [percept([0, 0, 1], view_at(70.0))])
        self.assertEqual(len(gsg.output_goals()), 1)
        gsg.parent_lm.buffer.get_num_steps_post_output_goal_generated.return_value = 1
        gsg.step(MagicMock(), [percept([0, 0, 1], view_at(70.0))])
        self.assertEqual(gsg.output_goals(), [])

    def test_achieved_is_logged_from_the_first_view_after_the_goal(self) -> None:
        gsg = gsg_with_lm(max_view_angle=45.0, window=1)
        gsg.step(MagicMock(), [percept([0, 0, 1], view_at(70.0))])
        goal = gsg.output_goals()[0]
        gsg.parent_lm.buffer.get_num_steps_post_output_goal_generated.return_value = 1
        gsg.step(MagicMock(), [percept([0, 0, 1], view_at(2.0))])
        self.assertTrue(goal.info["achieved"])


class GoalGeometryTest(unittest.TestCase):
    def test_places_the_agent_on_the_normal_at_the_patch_depth_looking_back(
        self,
    ) -> None:
        gsg = gsg_with_lm(max_view_angle=45.0, window=1)
        gsg.step(
            MagicMock(),
            [percept([0, 0, 1], view_at(70.0), depth=0.08, location=(0.1, 1.5, 0.05))],
        )
        goal = gsg.output_goals()[0]
        nptest.assert_allclose(goal.location, [0.1, 1.5, 0.13])
        nptest.assert_allclose(
            goal.morphological_features["pose_vectors"][0], [0, 0, -1]
        )
        nptest.assert_allclose(goal.info["proposed_surface_loc"], [0.1, 1.5, 0.05])

    def test_a_configured_standoff_overrides_the_depth(self) -> None:
        gsg = gsg_with_lm(max_view_angle=45.0, window=1, standoff=0.2)
        gsg.step(MagicMock(), [percept([0, 0, 1], view_at(70.0), depth=0.08)])
        nptest.assert_allclose(gsg.output_goals()[0].location, [0.0, 1.5, 0.25])

    def test_uses_a_unit_normal(self) -> None:
        gsg = gsg_with_lm(max_view_angle=45.0, window=1)
        gsg.step(MagicMock(), [percept([0, 0, 2.0], view_at(70.0), depth=0.1)])
        nptest.assert_allclose(gsg.output_goals()[0].location, [0.0, 1.5, 0.15])
