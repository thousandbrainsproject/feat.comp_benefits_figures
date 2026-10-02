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
from types import SimpleNamespace

from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_object_id_feature_names,
    object_id_to_features,
)


class ObjectIdFeatureNamesTest(unittest.TestCase):
    def test_feature_values_map_back_to_object_ids(self) -> None:
        names = get_object_id_feature_names(["023_mug", "022_logo_numenta"])

        self.assertEqual(
            names,
            {
                object_id_to_features("023_mug"): "023_mug",
                object_id_to_features("022_logo_numenta"): "022_logo_numenta",
            },
        )

    def test_object_ids_sharing_a_feature_value_are_joined(self) -> None:
        # Anagrams have the same sum of character codes
        names = get_object_id_feature_names(["mug", "gum", "mug"])

        self.assertEqual(names, {object_id_to_features("mug"): "gum/mug"})

    def test_monty_shares_object_names_across_learning_modules(self) -> None:
        # The parent LM only knows the parent objects, while the child LMs know
        # the child objects whose IDs the parent LM receives as features.
        lms = [
            SimpleNamespace(get_all_known_object_ids=lambda: ["023_mug"]),
            SimpleNamespace(get_all_known_object_ids=lambda: ["022_logo_numenta"]),
            SimpleNamespace(get_all_known_object_ids=lambda: ["026_mug_numenta"]),
        ]
        monty = SimpleNamespace(learning_modules=lms)

        MontyForGraphMatching._share_object_id_feature_names(monty)

        expected = get_object_id_feature_names(
            ["023_mug", "022_logo_numenta", "026_mug_numenta"]
        )
        for lm in lms:
            self.assertEqual(lm.object_id_feature_names, expected)


if __name__ == "__main__":
    unittest.main()
