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
from unittest.mock import MagicMock, patch, sentinel

import numpy.testing as nptest

from tbp.monty.cmp import AttentionRegion
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)

CONTEXT_PATCH = (
    "tbp.monty.frameworks.models.evidence_matching.learning_module"
    ".EvidenceLMRegionContext"
)


def lm_with_proposers(*proposers: MagicMock) -> MagicMock:
    # An EvidenceGraphLM stand-in carrying only the region proposers; the
    # real methods under test are bound to it explicitly.
    lm = MagicMock(spec=EvidenceGraphLM)
    lm._region_proposers = proposers
    lm.learning_module_id = "learning_module_7"
    return lm


def region_at(*locations: float) -> AttentionRegion:
    # One location per value, so concatenation order is visible.
    return AttentionRegion.uniform([[x, 0.0, 0.0] for x in locations], 1.0)


class ProposeRegionTest(unittest.TestCase):
    def test_concatenates_every_proposer_output_in_order(self) -> None:
        first = MagicMock(return_value=region_at(1.0, 2.0))
        second = MagicMock(return_value=region_at(3.0))
        lm = lm_with_proposers(first, second)

        with patch(CONTEXT_PATCH, return_value=sentinel.context):
            region = EvidenceGraphLM.propose_region(lm)

        nptest.assert_array_equal(region.locations[:, 0], [1.0, 2.0, 3.0])

    def test_names_this_lm_as_sender(self) -> None:
        lm = lm_with_proposers(MagicMock(return_value=region_at(1.0)))

        with patch(CONTEXT_PATCH, return_value=sentinel.context):
            region = EvidenceGraphLM.propose_region(lm)

        self.assertEqual(region.sender_id, "learning_module_7")

    def test_calls_each_proposer_with_a_context_over_this_lm(self) -> None:
        proposer = MagicMock(return_value=AttentionRegion.empty())
        lm = lm_with_proposers(proposer)

        with patch(CONTEXT_PATCH, return_value=sentinel.context) as context_mock:
            EvidenceGraphLM.propose_region(lm)

        context_mock.assert_called_once_with(lm)
        proposer.assert_called_once_with(sentinel.context)

    def test_proposes_nothing_without_proposers(self) -> None:
        lm = lm_with_proposers()

        with patch(CONTEXT_PATCH, return_value=sentinel.context):
            region = EvidenceGraphLM.propose_region(lm)

        self.assertIsInstance(region, AttentionRegion)
        self.assertEqual(len(region), 0)


class ResetStmTest(unittest.TestCase):
    def test_resets_every_proposer(self) -> None:
        first, second = MagicMock(), MagicMock()
        lm = lm_with_proposers(first, second)

        with patch(
            "tbp.monty.frameworks.models.evidence_matching.learning_module"
            ".GraphLM.reset_stm"
        ):
            EvidenceGraphLM.reset_stm(lm)

        first.reset.assert_called_once_with()
        second.reset.assert_called_once_with()
