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
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from analysis.scripts.compare_reorientation import (
    expected_logo,
    paired_table,
    rotation_label,
    rotation_magnitude,
    summary,
)

angles = st.lists(st.integers(min_value=0, max_value=359), min_size=3, max_size=3)


class ExpectedLogoTest(unittest.TestCase):
    def test_reads_the_logo_off_a_compositional_object_name(self) -> None:
        self.assertEqual(expected_logo("002_cube_tbp"), "021_logo_tbp")
        self.assertEqual(expected_logo("014_cylinder_numenta_horz"), "022_logo_numenta")

    def test_is_none_for_a_plain_parent(self) -> None:
        self.assertIsNone(expected_logo("001_cube"))


class RotationTest(unittest.TestCase):
    @given(angles)
    def test_label_lists_the_angles_as_the_csv_prints_them(self, euler: list) -> None:
        printed = str(np.array(euler))
        self.assertEqual(rotation_label(printed), "/".join(str(a) for a in euler))

    @given(angles)
    def test_magnitude_is_the_rotation_angle(self, euler: list) -> None:
        expected = np.degrees(
            Rotation.from_euler("xyz", euler, degrees=True).magnitude()
        )
        self.assertAlmostEqual(rotation_magnitude(str(np.array(euler))), expected)

    def test_a_single_axis_turn_has_its_own_angle(self) -> None:
        self.assertAlmostEqual(rotation_magnitude("[0 60 0]"), 60.0)


def episode_frame(outcomes: list[str], goals: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            object=["002_cube_tbp"] * len(outcomes),
            rotation=["0/0/0"] * len(outcomes),
            rotation_deg=[0.0] * len(outcomes),
            outcome=outcomes,
            right=[o in ("correct", "correct_mlh") for o in outcomes],
            converged=[o == "correct" for o in outcomes],
            steps=[10] * len(outcomes),
            first_right_mlh_step=[0.0] * len(outcomes),
            final_mlh_right=[True] * len(outcomes),
            goals=goals,
            goals_achieved=goals,
            jumps=goals,
        )
    ).rename_axis("episode")


class TablesTest(unittest.TestCase):
    def test_summary_counts_right_and_converged_episodes(self) -> None:
        tables = {"a": episode_frame(["correct", "correct_mlh", "confused"], [0, 1, 2])}
        row = summary(tables).loc["a"]
        self.assertEqual(row.parent_right, 2)
        self.assertEqual(row.parent_converged, 1)
        self.assertEqual(row.goals, 3)

    def test_paired_table_puts_every_run_beside_the_trial(self) -> None:
        tables = {
            "a": episode_frame(["correct"], [0]),
            "b": episode_frame(["confused"], [2]),
        }
        paired = paired_table(tables)
        self.assertEqual(paired.loc[0, "outcome [a]"], "correct")
        self.assertEqual(paired.loc[0, "outcome [b]"], "confused")
        self.assertEqual(paired.loc[0, "goals [b]"], 2)
