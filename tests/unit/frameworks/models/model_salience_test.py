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

from tbp.monty.frameworks.models.model_salience import (
    MAX_SALIENCE,
    RESTING_SALIENCE,
    ModelSalience,
    relaxed,
)

MUG = ("mug", "patch")
BOWL = ("bowl", "patch")
NUM_NODES = 10


class ModelSalienceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.salience = ModelSalience(relaxation_steps=50)

    def mug(self) -> np.ndarray:
        return self.salience.get(*MUG, NUM_NODES)

    def relax(self, num_steps: int, **kwargs) -> None:
        for _ in range(num_steps):
            self.salience.relax(mlh_graph_id="mug", **kwargs)

    def test_untracked_graphs_are_at_rest(self) -> None:
        nptest.assert_array_equal(self.mug(), np.full(NUM_NODES, RESTING_SALIENCE))
        self.assertEqual(self.salience.tracked(), [MUG])

    def test_get_returns_the_stored_salience(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.array([3]), np.array([0.2]))

        self.assertIs(self.mug(), self.mug())
        self.assertAlmostEqual(self.mug()[3], 0.3)

    def test_salience_of_a_rebuilt_graph_is_recreated_at_rest(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.arange(5), np.full(5, 0.2))

        nptest.assert_array_equal(
            self.salience.get(*MUG, NUM_NODES + 1), RESTING_SALIENCE
        )

    def test_reductions_are_clamped_at_exactly_zero(self) -> None:
        nodes = np.arange(5)
        for _ in range(5):
            self.salience.reduce(*MUG, NUM_NODES, nodes, np.full(5, 0.1))
        self.assertTrue(
            np.all(self.mug()[nodes] == 0), "Residue of repeated reductions is 0."
        )

        self.salience.reduce(*MUG, NUM_NODES, nodes, np.full(5, 0.2))
        self.assertTrue(np.all(self.mug()[nodes] == 0))
        nptest.assert_array_equal(self.mug()[5:], RESTING_SALIENCE)

    def test_excite_raises_nodes_to_max_salience_from_any_salience(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.arange(3), np.full(3, 1.0))

        self.salience.excite(*MUG, NUM_NODES, np.arange(5))

        nptest.assert_array_equal(self.mug()[:5], MAX_SALIENCE)
        nptest.assert_array_equal(self.mug()[5:], RESTING_SALIENCE)

    def test_reductions_from_max_salience_are_not_stopped_at_rest(self) -> None:
        self.salience.excite(*MUG, NUM_NODES, np.array([0]))

        for visit in range(1, 6):
            self.salience.reduce(*MUG, NUM_NODES, np.array([0]), np.array([0.2]))
            self.assertAlmostEqual(self.mug()[0], 1 - 0.2 * visit)
        self.assertEqual(self.mug()[0], 0.0)

    def test_salience_relaxes_linearly_up_to_rest(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.array([0]), np.array([0.2]))

        self.relax(10)
        self.assertAlmostEqual(self.mug()[0], 0.4)

        self.relax(20)
        self.assertEqual(self.mug()[0], RESTING_SALIENCE, "Does not overshoot rest.")

    def test_salience_relaxes_linearly_down_to_rest(self) -> None:
        self.salience.excite(*MUG, NUM_NODES, np.array([0]))

        self.relax(10)
        self.assertAlmostEqual(self.mug()[0], 0.9)

        self.relax(50)
        self.assertEqual(self.mug()[0], RESTING_SALIENCE, "Does not overshoot rest.")

    def test_relaxation_steps_is_the_time_to_rest_from_either_extreme(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.array([0]), np.array([1.0]))
        self.salience.excite(*MUG, NUM_NODES, np.array([1]))

        self.relax(49)
        self.assertLess(self.mug()[0], RESTING_SALIENCE)
        self.assertGreater(self.mug()[1], RESTING_SALIENCE)

        self.relax(1)
        nptest.assert_allclose(self.mug()[:2], RESTING_SALIENCE)

    def test_excepted_nodes_do_not_relax(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.arange(NUM_NODES), np.full(10, 0.2))

        self.relax(1, except_nodes={MUG: np.arange(3)})

        nptest.assert_allclose(self.mug()[:3], 0.3)
        nptest.assert_allclose(self.mug()[3:], 0.31)

    def test_no_relaxation_with_zero_relaxation_steps(self) -> None:
        salience = ModelSalience(relaxation_steps=0)
        salience.reduce(*MUG, NUM_NODES, np.array([0]), np.array([0.2]))
        salience.excite(*MUG, NUM_NODES, np.array([1]))

        salience.relax(mlh_graph_id="mug")

        self.assertAlmostEqual(salience.get(*MUG, NUM_NODES)[0], 0.3)
        self.assertEqual(salience.get(*MUG, NUM_NODES)[1], MAX_SALIENCE)

    def test_other_objects_relax_and_are_dropped_once_at_rest(self) -> None:
        self.salience.reduce(*BOWL, NUM_NODES, np.array([0]), np.array([0.1]))
        self.salience.excite(*BOWL, NUM_NODES, np.array([1]))
        self.salience.relax(mlh_graph_id="mug")

        self.relax(48)
        self.assertIn(BOWL, self.salience.tracked())
        self.assertAlmostEqual(self.salience.get(*BOWL, NUM_NODES)[1], 0.51)

        self.relax(1)
        self.assertNotIn(BOWL, self.salience.tracked())

    def test_mlh_stays_tracked_at_rest(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.array([0]), np.array([0.2]))

        self.relax(20)

        self.assertIn(MUG, self.salience.tracked())
        nptest.assert_array_equal(self.mug(), RESTING_SALIENCE)

    def test_graphs_at_rest_are_dropped_without_relaxation(self) -> None:
        salience = ModelSalience(relaxation_steps=0)
        salience.get(*BOWL, NUM_NODES)

        salience.relax(mlh_graph_id="mug")

        self.assertEqual(salience.tracked(), [])

    def test_reset_restores_resting_salience(self) -> None:
        self.salience.reduce(*MUG, NUM_NODES, np.arange(5), np.full(5, 0.2))
        self.salience.excite(*MUG, NUM_NODES, np.arange(5, 10))

        self.salience.reset()

        self.assertEqual(self.salience.tracked(), [])
        nptest.assert_array_equal(self.mug(), RESTING_SALIENCE)

    def test_negative_relaxation_steps_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelSalience(relaxation_steps=-1)


class RelaxedTest(unittest.TestCase):
    def test_moves_towards_rest_from_either_side_without_overshooting(self) -> None:
        nptest.assert_allclose(
            relaxed(np.array([0.0, 0.45, 0.5, 0.55, 1.0]), 0.1),
            [0.1, 0.5, 0.5, 0.5, 0.9],
        )


if __name__ == "__main__":
    unittest.main()
