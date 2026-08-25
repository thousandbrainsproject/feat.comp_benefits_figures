# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import ANY, MagicMock, sentinel

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.monty_base import MontyBase


class MontyBasePrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sm1 = MagicMock()
        self.sm1.sensor_module_id = "sm1"
        self.sm2 = MagicMock()
        self.sm2.sensor_module_id = "sm2"
        self.lm1 = MagicMock()
        self.lm2 = MagicMock()
        self.lm3 = MagicMock()
        self.monty_base = MontyBase(
            sensor_modules=[self.sm1, self.sm2],
            learning_modules=[self.lm1, self.lm2, self.lm3],
            motor_system=MagicMock(),
            sm_to_agent_dict={
                "sm1": AgentID("agent_id_0"),
                "sm2": AgentID("agent_id_0"),
            },
            sm_to_lm_matrix=[[], [], []],
            lm_to_lm_matrix=[[], [], []],
            lm_to_lm_vote_matrix=[[], [], []],
            min_eval_steps=10,
            min_train_steps=10,
            num_exploratory_steps=10,
            max_total_steps=100,
        )

    def test_pass_goals_collects_all_goals_from_learning_and_sensor_modules(
        self,
    ) -> None:
        self.monty_base.step_type = "matching_step"
        self.lm1.propose_goals.return_value = []
        self.lm2.propose_goals.return_value = [sentinel.lm2_goal]
        self.lm3.propose_goals.return_value = [
            sentinel.lm3_goal_1,
            sentinel.lm3_goal_2,
        ]
        self.sm1.propose_goals.return_value = []
        self.sm2.propose_goals.return_value = [
            sentinel.sm2_goal_1,
            sentinel.sm2_goal_2,
        ]
        self.monty_base._pass_goals()

        expected = set(
            {
                sentinel.lm2_goal,
                sentinel.lm3_goal_1,
                sentinel.lm3_goal_2,
                sentinel.sm2_goal_1,
                sentinel.sm2_goal_2,
            }
        )
        self.assertEqual(set(self.monty_base._goals), expected)

    def test_pass_goals_collects_one_region_per_module_lms_first(self) -> None:
        self.monty_base.step_type = "matching_step"
        for module, region in (
            (self.lm1, sentinel.lm1_region),
            (self.lm2, sentinel.lm2_region),
            (self.lm3, sentinel.lm3_region),
            (self.sm1, sentinel.sm1_region),
            (self.sm2, sentinel.sm2_region),
        ):
            module.propose_region.return_value = region
        self.monty_base._attention_system = MagicMock()

        self.monty_base._pass_goals()

        self.assertEqual(
            self.monty_base._regions,
            [
                sentinel.lm1_region,
                sentinel.lm2_region,
                sentinel.lm3_region,
                sentinel.sm1_region,
                sentinel.sm2_region,
            ],
        )

    def test_pass_goals_asks_each_module_for_a_region_once(self) -> None:
        self.monty_base.step_type = "matching_step"
        self.lm2.propose_goals.return_value = [sentinel.lm2_goal]
        self.sm1.propose_goals.return_value = [sentinel.sm1_goal]
        self.monty_base._attention_system = MagicMock()

        self.monty_base._pass_goals()

        # Goals are never handed over: an LM reads its own goals off itself,
        # and SM-generated goals never reach region proposal.
        for module in (self.lm1, self.lm2, self.lm3, self.sm1, self.sm2):
            module.propose_region.assert_called_once_with()

    def test_pass_goals_hands_the_collected_regions_to_the_attention_system(
        self,
    ) -> None:
        self.monty_base.step_type = "matching_step"
        attention_system = MagicMock()
        attention_system.step.return_value = sentinel.filtered_goals
        self.monty_base._attention_system = attention_system

        self.monty_base._pass_goals()

        attention_system.step.assert_called_once_with(
            ANY,
            self.monty_base._regions,
        )
        self.assertIs(self.monty_base._goals, sentinel.filtered_goals)

    def test_reset_clears_the_regions(self) -> None:
        self.monty_base._regions = [sentinel.region]

        self.monty_base.reset()

        self.assertEqual(self.monty_base._regions, [])
