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
from unittest.mock import MagicMock, call

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.models.monty_base import MontyBase


class UnsetUpExperiment(MontyExperiment):
    """MontyExperiment whose logger arguments do not need counters to be set up."""

    @property
    def logger_args(self):
        return {}


def make_experiment(
    do_train=False, do_post_training=False, do_eval=False, n_post_training_epochs=1
) -> MontyExperiment:
    """Build a MontyExperiment without setup, with mocked model and loggers.

    Returns:
        The experiment.
    """
    experiment = UnsetUpExperiment.__new__(UnsetUpExperiment)
    experiment.do_train = do_train
    experiment.do_post_training_unsupervised_learning = do_post_training
    experiment.do_eval = do_eval
    experiment.n_post_training_unsupervised_epochs = n_post_training_epochs
    experiment.n_train_epochs = 1
    experiment.n_eval_epochs = 1
    experiment._hotspot_learning = False
    experiment.experiment_mode = ExperimentMode.TRAIN
    experiment.model = MagicMock()
    experiment.logger_handler = MagicMock()
    return experiment


class PostTrainingUnsupervisedLearningTest(unittest.TestCase):
    def test_run_order_is_train_then_post_training_then_eval(self):
        experiment = make_experiment(do_train=True, do_post_training=True, do_eval=True)
        order = MagicMock()
        experiment.train = order.train
        experiment.post_training_unsupervised_learning = order.post_training
        experiment.evaluate = order.evaluate

        experiment.run()

        self.assertEqual(
            order.mock_calls, [call.train(), call.post_training(), call.evaluate()]
        )

    def test_post_training_skipped_by_default(self):
        experiment = make_experiment(do_eval=True)
        experiment.post_training_unsupervised_learning = MagicMock()
        experiment.evaluate = MagicMock()

        experiment.run()

        experiment.post_training_unsupervised_learning.assert_not_called()
        experiment.evaluate.assert_called_once()

    def test_hotspot_learning_enabled_in_eval_mode_during_epochs(self):
        experiment = make_experiment(n_post_training_epochs=3)
        states = []

        def run_epoch():
            states.append(
                (
                    experiment.experiment_mode,
                    experiment._hotspot_learning,
                    experiment.model.set_hotspot_learning.call_args,
                )
            )

        experiment.run_epoch = run_epoch

        experiment.post_training_unsupervised_learning()

        self.assertEqual(states, [(ExperimentMode.EVAL, True, call(enabled=True))] * 3)
        experiment.model.set_experiment_mode.assert_called_with(ExperimentMode.EVAL)
        self.assertFalse(experiment._hotspot_learning)
        self.assertEqual(
            experiment.model.set_hotspot_learning.call_args, call(enabled=False)
        )

    def test_hotspot_learning_reapplied_when_monty_is_restored(self):
        experiment = make_experiment()
        experiment._recreation_mode = False
        experiment._hotspot_learning = True

        experiment._restore_monty()

        experiment.model.set_hotspot_learning.assert_called_once_with(enabled=True)


class MontyBaseHotspotLearningTest(unittest.TestCase):
    def test_forwards_to_lms_that_support_hotspot_learning(self):
        supporting_lm = MagicMock()
        other_lm = MagicMock(spec=["set_experiment_mode"])
        model = MontyBase.__new__(MontyBase)
        model.learning_modules = [supporting_lm, other_lm]

        model.set_hotspot_learning(enabled=True)

        supporting_lm.set_hotspot_learning.assert_called_once_with(enabled=True)
