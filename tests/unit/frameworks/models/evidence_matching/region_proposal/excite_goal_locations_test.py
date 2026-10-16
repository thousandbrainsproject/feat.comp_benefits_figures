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
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as nptest
import quaternion as qt

from tbp.monty.attention.attention_system import AttentionSystem
from tbp.monty.cmp import MAX_ATTENTION_WEIGHT, MIN_ATTENTION_WEIGHT, AttentionRegion
from tbp.monty.frameworks.actions.actions import SetAgentPose
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.evidence_matching.region_proposal.excite_goal_locations import (  # noqa: E501
    ExciteGoalLocations,
    sample_ball,
)
from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
    RegionContext,
)
from tbp.monty.frameworks.models.goal_generation import EvidenceGoalGenerator
from tbp.monty.frameworks.models.motor_policies import (
    JUMP_DISTANCE_FACTOR,
    JumpToGoal,
    MotorPolicyResult,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.geometry import Rotation
from tbp.monty.math import DEFAULT_TOLERANCE, VectorXYZ


def goal_at(location: np.ndarray | None) -> MagicMock:
    goal = MagicMock()
    goal.location = location
    return goal


def context_with_goals(*goals: MagicMock) -> MagicMock:
    # The proposer reads nothing else off the context.
    context = MagicMock(spec=RegionContext)
    context.goals = goals
    return context


class SampleBallTest(unittest.TestCase):
    def test_contains_the_center(self) -> None:
        center = np.array([1.0, -2.0, 3.0])

        points = sample_ball(center, radius=0.01, lattice_steps=4)

        self.assertTrue((points == center).all(axis=1).any())

    def test_every_point_lies_within_the_radius(self) -> None:
        center = np.array([1.0, -2.0, 3.0])

        points = sample_ball(center, radius=0.01, lattice_steps=4)

        nptest.assert_array_less(np.linalg.norm(points - center, axis=1), 0.01 + 1e-12)

    def test_spans_the_full_diameter(self) -> None:
        points = sample_ball(np.zeros(3), radius=0.01, lattice_steps=4)

        self.assertAlmostEqual(points[:, 0].min(), -0.01)
        self.assertAlmostEqual(points[:, 0].max(), 0.01)


class ExciteGoalLocationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proposer = ExciteGoalLocations()

    def test_proposes_nothing_without_goals(self) -> None:
        self.assertIsNone(self.proposer(context_with_goals()))

    def test_proposes_nothing_when_no_goal_has_a_location(self) -> None:
        context = context_with_goals(goal_at(None), goal_at(None))

        self.assertIsNone(self.proposer(context))

    def test_excites_a_ball_around_a_goal_location(self) -> None:
        location = np.array([0.1, 0.2, 0.3])

        region = self.proposer(context_with_goals(goal_at(location)))

        distances = np.linalg.norm(region.locations - location, axis=1)
        nptest.assert_array_less(distances, 0.01 + 1e-12)
        nptest.assert_array_equal(
            region.weights, np.full(len(region), MAX_ATTENTION_WEIGHT)
        )

    def test_skips_goals_without_a_location(self) -> None:
        location = np.array([0.1, 0.2, 0.3])
        context = context_with_goals(goal_at(None), goal_at(location))

        region = self.proposer(context)

        distances = np.linalg.norm(region.locations - location, axis=1)
        nptest.assert_array_less(distances, 0.01 + 1e-12)

    def test_proposes_one_ball_per_located_goal(self) -> None:
        near = np.zeros(3)
        far = np.array([1.0, 0.0, 0.0])

        region = self.proposer(context_with_goals(goal_at(near), goal_at(far)))

        to_near = np.linalg.norm(region.locations - near, axis=1) <= 0.01 + 1e-12
        to_far = np.linalg.norm(region.locations - far, axis=1) <= 0.01 + 1e-12
        self.assertTrue(to_near.any())
        self.assertTrue(to_far.any())
        self.assertTrue((to_near | to_far).all())

    def test_radius_and_weight_are_parameterized(self) -> None:
        proposer = ExciteGoalLocations(radius=0.05, weight=0.5)

        region = proposer(context_with_goals(goal_at(np.zeros(3))))

        self.assertAlmostEqual(np.linalg.norm(region.locations, axis=1).max(), 0.05)
        nptest.assert_array_equal(region.weights, np.full(len(region), 0.5))


AGENT_ID = AgentID("agent_id_0")
# The sensed location on the object's surface, in the body frame.
SENSED_LOCATION = np.array([0.0, 1.5, 0.0])
# The node of the MLH object the goal targets, 3 cm along the surface from the
# MLH location (the model's origin), whose surface normal is +Z.
TARGET_IN_MODEL = np.array([0.03, 0.0, 0.0])
TARGET_NORMAL = np.array([0.0, 0.0, 1.0])


def model_based_goal():
    """A goal from an EvidenceGoalGenerator for the target node of the MLH.

    Returns:
        The goal, for an MLH with identity rotation at the sensed location.
    """
    lm = MagicMock()
    lm.learning_module_id = "learning_module_0"
    lm.buffer.get_first_sensory_input_channel.return_value = "patch"
    gsg = EvidenceGoalGenerator()
    gsg.parent_lm = lm
    return gsg._compute_goal_for_target_loc(
        observations=[MagicMock(sender_id="patch", location=SENSED_LOCATION)],
        target_info={
            "target_loc": TARGET_IN_MODEL,
            "target_surface_normal": TARGET_NORMAL,
            "hypothesis_to_test": {
                "graph_id": "mug",
                "mlh_id": 0,
                "location": np.zeros(3),
                "rotation": Rotation.identity(),
            },
            "input_channel": "patch",
        },
    )


class ModelBasedGoalTest(unittest.TestCase):
    """A model-based goal survives the attention system and is jumped to."""

    def setUp(self) -> None:
        self.goal = model_based_goal()
        self.attention = AttentionSystem()
        # Another module attends to the patch under fixation, so the grid is not
        # empty and goals elsewhere must be attended to in order to pass.
        self.fixated = AttentionRegion.uniform(
            sample_ball(SENSED_LOCATION, radius=0.005, lattice_steps=2),
            MAX_ATTENTION_WEIGHT,
        )
        self.excited = ExciteGoalLocations()(context_with_goals(self.goal))

    def test_the_goal_is_on_the_surface_with_its_normal(self) -> None:
        nptest.assert_allclose(self.goal.location, SENSED_LOCATION + TARGET_IN_MODEL)
        nptest.assert_allclose(
            self.goal.morphological_features["pose_vectors"][0], TARGET_NORMAL
        )

    def test_the_goal_is_filtered_out_without_its_excitation(self) -> None:
        self.assertEqual(self.attention.step([self.goal], [self.fixated]), [])

    def test_the_excited_goal_passes_and_is_jumped_to(self) -> None:
        (goal,) = self.attention.step([self.goal], [self.fixated, self.excited])
        self.assertIs(goal, self.goal)

        policy = JumpToGoal(AGENT_ID, SensorID("view_finder"))
        state = MotorSystemState(
            {
                AGENT_ID: AgentState(
                    sensors={
                        SensorID("view_finder"): SensorState(
                            position=cast("VectorXYZ", (0, 0, 0)), rotation=qt.one
                        )
                    },
                    position=cast("VectorXYZ", (0, 0, 0)),
                    rotation=qt.one,
                )
            }
        )
        result = policy(
            ctx=MagicMock(),
            observations=MagicMock(),
            state=state,
            percept=MagicMock(),
            goal=goal,
        )

        assert isinstance(result, MotorPolicyResult)
        self.assertIs(result.attempted_goal, goal)
        set_agent_pose = result.actions[0]
        assert isinstance(set_agent_pose, SetAgentPose)
        # Off the surface along the normal, looking back down it at the goal.
        nptest.assert_allclose(
            set_agent_pose.location,
            goal.location
            + TARGET_NORMAL * policy.desired_object_distance * JUMP_DISTANCE_FACTOR,
        )
        rotation = Rotation.from_quat(qt.as_float_array(set_agent_pose.rotation_quat))
        nptest.assert_allclose(
            -rotation.as_matrix()[:, 2], -TARGET_NORMAL, atol=DEFAULT_TOLERANCE
        )

    def test_an_excited_goal_on_an_inhibited_surface_is_filtered_out(self) -> None:
        # E.g. InhibitRecognizedObject inhibits a shell around a recognized
        # object's surface, which the goal now lies within; inhibition dominates.
        inhibited = AttentionRegion.uniform(
            sample_ball(self.goal.location, radius=0.01, lattice_steps=4),
            MIN_ATTENTION_WEIGHT,
        )

        filtered = self.attention.step(
            [self.goal], [self.fixated, self.excited, inhibited]
        )

        self.assertEqual(filtered, [])
