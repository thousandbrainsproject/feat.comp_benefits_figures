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

import numpy as np
import numpy.testing as nptest

from tbp.monty.frameworks.models.model_salience import ModelSalience

MUG = ("mug", "patch")
BOWL = ("bowl", "patch")
NUM_NODES = 10


class ModelSalienceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.salience = ModelSalience(recovery_steps=50)

    def mug(self) -> np.ndarray:
        return self.salience.get(*MUG, NUM_NODES)

    def test_untracked_graphs_have_full_salience(self) -> None:
        nptest.assert_array_equal(self.mug(), np.ones(NUM_NODES))
        self.assertEqual(self.salience.tracked(), [MUG])

    def test_get_returns_the_stored_salience(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.array([3]), np.array([0.2]))

        self.assertIs(self.mug(), self.mug())
        self.assertAlmostEqual(self.mug()[3], 0.8)

    def test_salience_of_a_rebuilt_graph_is_recreated(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.arange(5), np.full(5, 0.2))

        nptest.assert_array_equal(self.salience.get(*MUG, NUM_NODES + 1), 1.0)

    def test_reductions_are_clamped_at_exactly_zero(self) -> None:
        nodes = np.arange(5)
        for _ in range(5):
            self.salience.reduce(*MUG, NUM_NODES, nodes, np.full(5, 0.2))
        self.assertTrue(
            np.all(self.mug()[nodes] == 0), "Residue of repeated reductions is 0."
        )

        self.salience.reduce(*MUG, NUM_NODES, nodes, np.full(5, 0.2))
        self.assertTrue(np.all(self.mug()[nodes] == 0))
        nptest.assert_array_equal(self.mug()[5:], 1.0)

    def test_salience_recovers_linearly_to_one(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.array([0]), np.array([0.2]))

        for _ in range(5):
            self.salience.recover(mlh_graph_id="mug")
        self.assertAlmostEqual(self.mug()[0], 0.9)

        for _ in range(10):
            self.salience.recover(mlh_graph_id="mug")
        self.assertEqual(self.mug()[0], 1.0, "Salience is capped at 1.")

    def test_excepted_nodes_do_not_recover(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.arange(NUM_NODES), np.full(10, 0.5))

        self.salience.recover(except_nodes={MUG: np.arange(3)}, mlh_graph_id="mug")

        nptest.assert_allclose(self.mug()[:3], 0.5)
        nptest.assert_allclose(self.mug()[3:], 0.52)

    def test_no_recovery_with_zero_recovery_steps(self) -> None:
        salience = ModelSalience(recovery_steps=0)
        salience.reduce(*MUG, NUM_NODES, np.array([0]), np.array([0.2]))

        salience.recover(mlh_graph_id="mug")

        self.assertAlmostEqual(salience.get(*MUG, NUM_NODES)[0], 0.8)

    def test_other_objects_recover_and_are_dropped_once_fully_recovered(
        self,
    ) -> None:
        self.salience.reduce(*BOWL, NUM_NODES, np.array([0]), np.array([0.2]))

        for _ in range(9):
            self.salience.recover(mlh_graph_id="mug")
        self.assertIn(BOWL, self.salience.tracked())
        self.assertAlmostEqual(self.salience.get(*BOWL, NUM_NODES)[0], 0.98)

        self.salience.recover(mlh_graph_id="mug")
        self.assertNotIn(BOWL, self.salience.tracked())

    def test_fully_recovered_mlh_stays_tracked(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.array([0]), np.array([0.2]))

        for _ in range(10):
            self.salience.recover(mlh_graph_id="mug")

        self.assertIn(MUG, self.salience.tracked())

    def test_fully_recovered_graphs_are_dropped_without_recovery(self) -> None:
        salience = ModelSalience(recovery_steps=0)
        salience.get(*BOWL, NUM_NODES)

        salience.recover(mlh_graph_id="mug")

        self.assertEqual(salience.tracked(), [])

    def test_reset_restores_full_salience(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.arange(5), np.full(5, 0.2))

        self.salience.reset()

        self.assertEqual(self.salience.tracked(), [])
        nptest.assert_array_equal(self.mug(), 1.0)

    def test_negative_recovery_steps_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelSalience(recovery_steps=-1)


if __name__ == "__main__":
    unittest.main()
