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
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nptest

from tbp.monty.frameworks.models.evidence_matching.hypothesis_trace import (
    HypothesisTracer,
)
from tbp.monty.geometry import Rotation

MUG = "mug_tbp"
MUG_NUMENTA = "mug_numenta"
CYLINDER = "cylinder_tbp"


def mlh(graph_id, location, rotation=None):
    return {
        "graph_id": graph_id,
        "location": np.asarray(location, dtype=float),
        "rotation": Rotation.identity() if rotation is None else rotation,
    }


class TracerStepper:
    """Feeds a tracer evidence updates for a fixed set of objects."""

    def __init__(self, tracer: HypothesisTracer):
        self.tracer = tracer
        self.evidence = {MUG: 0.0, MUG_NUMENTA: 0.0, CYLINDER: 0.0}

    def step(self, gains, hypothesis, body_location):
        prev = dict(self.evidence)
        for graph_id, gain in gains.items():
            self.evidence[graph_id] += gain
        self.tracer.update(prev, dict(self.evidence), hypothesis, body_location)


class DifferentialTest(unittest.TestCase):
    def test_differential_is_relative_to_strongest_competitor(self):
        tracer = HypothesisTracer()
        stepper = TracerStepper(tracer)

        stepper.step(
            {MUG: 1.0, MUG_NUMENTA: 0.4, CYLINDER: 0.1},
            mlh(MUG, [0, 0, 0]),
            body_location=[0, 0, 0],
        )

        (step,) = tracer.trace
        self.assertAlmostEqual(step.mlh_gain, 1.0)
        self.assertAlmostEqual(step.competitor_gain, 0.4)
        self.assertAlmostEqual(step.differential, 0.6)

    def test_shared_part_gives_small_differential(self):
        tracer = HypothesisTracer()
        stepper = TracerStepper(tracer)

        stepper.step(
            {MUG: 1.0, MUG_NUMENTA: 1.0, CYLINDER: 1.0},
            mlh(MUG, [0, 0, 0]),
            body_location=[0, 0, 0],
        )

        self.assertAlmostEqual(tracer.trace[0].differential, 0.0)

    def test_competitor_gain_is_zero_with_a_single_object(self):
        tracer = HypothesisTracer()

        tracer.update({MUG: 0.0}, {MUG: 1.0}, mlh(MUG, [0, 0, 0]), [0, 0, 0])

        self.assertAlmostEqual(tracer.trace[0].differential, 1.0)

    def test_step_skipped_when_no_prior_evidence(self):
        tracer = HypothesisTracer()

        tracer.update(
            {"patch_off_object": 0}, {MUG: 1.0}, mlh(MUG, [0, 0, 0]), [0, 0, 0]
        )

        self.assertEqual(tracer.trace, [])


class ContinuityTest(unittest.TestCase):
    def setUp(self):
        self.tracer = HypothesisTracer(location_eta=0.01, rotation_eta_degrees=10)
        self.stepper = TracerStepper(self.tracer)
        self.gains = {MUG: 1.0, MUG_NUMENTA: 0.1, CYLINDER: 0.1}

    def test_trace_continues_for_consistently_displaced_hypothesis(self):
        # The object is rotated 90 degrees about z, so a body displacement along x
        # moves the hypothesis along y in the object's reference frame.
        rotation = Rotation.from_euler("xyz", [0, 0, 90], degrees=True)
        self.stepper.step(
            self.gains, mlh(MUG, [0, 0, 0], rotation), body_location=[0, 0, 0]
        )
        self.stepper.step(
            self.gains,
            mlh(MUG, [0, 0.02, 0], rotation),
            body_location=[0.02, 0, 0],
        )

        self.assertEqual(len(self.tracer.trace), 2)

    def test_new_trace_when_object_changes(self):
        self.stepper.step(self.gains, mlh(MUG, [0, 0, 0]), body_location=[0, 0, 0])
        self.stepper.step(
            {MUG: 0.1, MUG_NUMENTA: 0.1, CYLINDER: 1.0},
            mlh(CYLINDER, [0.02, 0, 0]),
            body_location=[0.02, 0, 0],
        )

        (step,) = self.tracer.trace
        self.assertEqual(step.graph_id, CYLINDER)

    def test_new_trace_when_location_jumps(self):
        self.stepper.step(self.gains, mlh(MUG, [0, 0, 0]), body_location=[0, 0, 0])
        # The sensor moved 2cm, but the MLH is now 5cm from where the traced
        # hypothesis is expected to be.
        self.stepper.step(
            self.gains, mlh(MUG, [0.07, 0, 0]), body_location=[0.02, 0, 0]
        )

        (step,) = self.tracer.trace
        nptest.assert_allclose(step.location, [0.07, 0, 0])

    def test_trace_continues_for_small_location_noise(self):
        self.stepper.step(self.gains, mlh(MUG, [0, 0, 0]), body_location=[0, 0, 0])
        self.stepper.step(
            self.gains, mlh(MUG, [0.025, 0, 0]), body_location=[0.02, 0, 0]
        )

        self.assertEqual(len(self.tracer.trace), 2)

    def test_new_trace_when_rotation_exceeds_eta(self):
        self.stepper.step(self.gains, mlh(MUG, [0, 0, 0]), body_location=[0, 0, 0])
        self.stepper.step(
            self.gains,
            mlh(MUG, [0, 0, 0], Rotation.from_euler("xyz", [0, 0, 15], degrees=True)),
            body_location=[0, 0, 0],
        )

        self.assertEqual(len(self.tracer.trace), 1)

    def test_trace_continues_for_rotation_within_eta(self):
        self.stepper.step(self.gains, mlh(MUG, [0, 0, 0]), body_location=[0, 0, 0])
        self.stepper.step(
            self.gains,
            mlh(MUG, [0, 0, 0], Rotation.from_euler("xyz", [0, 0, 5], degrees=True)),
            body_location=[0, 0, 0],
        )

        self.assertEqual(len(self.tracer.trace), 2)


class RecentStepsTest(unittest.TestCase):
    def setUp(self):
        self.tracer = HypothesisTracer(num_recent_locations=3)
        self.stepper = TracerStepper(self.tracer)

    def walk(self, num_steps, start=0):
        for i in range(start, start + num_steps):
            self.stepper.step(
                {MUG: 1.0, MUG_NUMENTA: 0.0, CYLINDER: 0.0},
                mlh(MUG, [0.01 * i, 0, 0]),
                body_location=[0.01 * i, 0, 0],
            )

    def test_recent_steps_are_the_last_num_recent_locations(self):
        self.walk(5)

        locations = [step.location[0] for step in self.tracer.recent_steps()]

        nptest.assert_allclose(locations, [0.02, 0.03, 0.04])

    def test_freeze_keeps_steps_up_to_convergence(self):
        self.walk(4)
        self.tracer.freeze()
        self.walk(3, start=4)

        locations = [step.location[0] for step in self.tracer.frozen_recent_steps()]

        nptest.assert_allclose(locations, [0.01, 0.02, 0.03])
        self.assertEqual(len(self.tracer.trace), 4)

    def test_frozen_steps_empty_when_not_frozen(self):
        self.walk(2)

        self.assertFalse(self.tracer.is_frozen)
        self.assertEqual(self.tracer.frozen_recent_steps(), [])

    def test_reset_clears_trace_and_freeze(self):
        self.walk(2)
        self.tracer.freeze()

        self.tracer.reset()

        self.assertEqual(self.tracer.trace, [])
        self.assertFalse(self.tracer.is_frozen)


class HistoryTest(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.history_dir = Path(temp_dir.name)
        self.tracer = HypothesisTracer(history_dir=str(self.history_dir))
        self.stepper = TracerStepper(self.tracer)

    def step(self, i, inputs=None):
        prev = dict(self.stepper.evidence)
        self.stepper.evidence[MUG] += 1.0
        self.stepper.evidence[CYLINDER] += 0.25
        self.tracer.update(
            prev,
            dict(self.stepper.evidence),
            mlh(MUG, [0.01 * i, 0, 0]),
            body_location=[0.01 * i, 0, 0],
            inputs=inputs,
        )

    def test_history_records_every_step_including_after_freeze(self):
        self.step(0, inputs={"patch": None, "learning_module_1": "tbp_logo"})
        self.step(1)
        self.tracer.freeze()
        self.step(2)

        history = self.tracer.history
        self.assertEqual(len(history), 3)
        self.assertEqual([s["frozen"] for s in history], [False, False, True])
        self.assertEqual([s["new_trace"] for s in history], [True, False, False])
        self.assertEqual(history[0]["competitor_id"], CYLINDER)
        self.assertAlmostEqual(history[0]["differential"], 0.75)
        self.assertEqual(
            history[0]["gains"], {MUG: 1.0, MUG_NUMENTA: 0.0, CYLINDER: 0.25}
        )
        self.assertEqual(history[0]["inputs"]["learning_module_1"], "tbp_logo")
        self.assertEqual(len(self.tracer.trace), 2)

    def test_save_history_writes_one_file_per_episode(self):
        self.step(0)
        first = self.tracer.save_history("lm_2", {"primary_target": MUG})
        self.tracer.reset()
        self.step(0)
        self.step(1)
        second = self.tracer.save_history("lm_2", {"primary_target": MUG})

        self.assertEqual(first.name, "lm_2_episode_000.json")
        self.assertEqual(second.name, "lm_2_episode_001.json")
        saved = json.loads(second.read_text())
        self.assertEqual(saved["episode"], 1)
        self.assertEqual(saved["primary_target"], MUG)
        self.assertEqual(len(saved["steps"]), 2)
        nptest.assert_allclose(saved["steps"][1]["rotation_matrix"], np.eye(3))

    def test_no_history_kept_or_saved_without_history_dir(self):
        tracer = HypothesisTracer()
        tracer.update({MUG: 0.0}, {MUG: 1.0}, mlh(MUG, [0, 0, 0]), [0, 0, 0])

        self.assertEqual(tracer.history, [])
        self.assertIsNone(tracer.save_history("lm_2", {}))
