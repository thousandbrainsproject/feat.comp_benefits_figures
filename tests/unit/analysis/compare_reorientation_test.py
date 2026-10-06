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
    accuracy_by_tilt,
    by_rotation,
    expected_logo,
    paired_table,
    rotation_label,
    rotation_magnitude,
    steps_by_axis,
    summary,
    tilt_axis,
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
            correct=[o in ("correct", "correct_mlh") for o in outcomes],
            converged=[o == "correct" for o in outcomes],
            steps=[10] * len(outcomes),
            first_correct_mlh_step=[0.0] * len(outcomes),
            final_mlh_correct=[True] * len(outcomes),
            goals=goals,
            goals_achieved=goals,
            jumps=goals,
        )
    ).rename_axis("episode")


class TablesTest(unittest.TestCase):
    def test_summary_counts_correct_and_converged_episodes(self) -> None:
        tables = {"a": episode_frame(["correct", "correct_mlh", "confused"], [0, 1, 2])}
        row = summary(tables).loc["a"]
        self.assertEqual(row.parent_correct, 2)
        self.assertEqual(row.parent_converged, 1)
        self.assertEqual(row.goals, 3)

    def test_by_rotation_sums_the_trials_at_each_rotation(self) -> None:
        table = episode_frame(["correct", "correct_mlh", "confused"], [0, 1, 2])
        table["rotation"] = ["0/60/0", "0/60/0", "0/75/0"]
        rows = by_rotation({"a": table})
        self.assertEqual(list(rows.index), ["0/60/0", "0/75/0"])
        self.assertEqual(rows.loc["0/60/0", "correct [a]"], 2)
        self.assertEqual(rows.loc["0/60/0", "converged [a]"], 1)
        self.assertEqual(rows.loc["0/75/0", "goals [a]"], 2)

    def test_baseline_rotations_lead_the_table_but_skip_the_step_mean(self) -> None:
        table = episode_frame(["correct", "correct", "correct"], [0, 0, 0])
        table["rotation"] = ["0/60/0", "0/75/0", "0/0/0"]
        table["steps"] = [40, 60, 10]
        self.assertEqual(
            list(by_rotation({"a": table}, baseline=["0/0/0"]).index),
            ["0/0/0", "0/60/0", "0/75/0"],
        )
        self.assertEqual(
            summary({"a": table}, baseline=["0/0/0"]).loc["a", "mean_steps"], 50
        )
        self.assertEqual(summary({"a": table}).loc["a", "mean_steps"], 110 / 3)

    def test_tilt_axis_reads_single_axis_rotations_only(self) -> None:
        self.assertEqual(tilt_axis("0/45/0"), ("y", 45.0))
        self.assertEqual(tilt_axis("85/0/0"), ("x", 85.0))
        self.assertIsNone(tilt_axis("0/0/0"))
        self.assertIsNone(tilt_axis("45/45/0"))

    def test_steps_by_axis_averages_per_axis_and_pooled(self) -> None:
        table = episode_frame(["correct"] * 4, [0] * 4)
        table["rotation"] = ["0/45/0", "45/0/0", "0/45/0", "0/0/0"]
        table["steps"] = [10, 30, 20, 99]
        rows = steps_by_axis({"a": table}).set_index(["axis", "tilt"])["a"]
        self.assertEqual(rows[("y", 45.0)], 15)
        self.assertEqual(rows[("x", 45.0)], 30)
        self.assertEqual(rows[("pooled", 45.0)], 20)
        self.assertNotIn(0.0, rows.index.get_level_values("tilt"))

    def test_accuracy_by_tilt_pools_axes_and_counts_wrong_trials(self) -> None:
        table = episode_frame(["correct", "confused", "correct_mlh"], [0, 0, 0])
        table["rotation_deg"] = [45.0, 45.0, 60.0]
        rows = accuracy_by_tilt({"a": table}).set_index("tilt")
        self.assertEqual(rows.loc[45.0, "correct [a]"], 1)
        self.assertEqual(rows.loc[45.0, "incorrect [a]"], 1)
        self.assertEqual(rows.loc[60.0, "correct [a]"], 1)
        self.assertEqual(rows.loc[60.0, "incorrect [a]"], 0)

    def test_accuracy_by_tilt_drops_baseline_rotations(self) -> None:
        table = episode_frame(["correct", "correct"], [0, 0])
        table["rotation"] = ["0/0/0", "0/45/0"]
        table["rotation_deg"] = [0.0, 45.0]
        rows = accuracy_by_tilt({"a": table}, baseline=["0/0/0"])
        self.assertEqual(list(rows.tilt), [45.0])

    def test_paired_table_puts_every_run_beside_the_trial(self) -> None:
        tables = {
            "a": episode_frame(["correct"], [0]),
            "b": episode_frame(["confused"], [2]),
        }
        paired = paired_table(tables)
        self.assertEqual(paired.loc[0, "outcome [a]"], "correct")
        self.assertEqual(paired.loc[0, "outcome [b]"], "confused")
        self.assertEqual(paired.loc[0, "goals [b]"], 2)
