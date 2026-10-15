# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np

__all__ = ["ModelSalience"]

# Salience left below this by repeated reductions is floating-point residue.
_ZERO_SALIENCE = 1e-9


class ModelSalience:
    """The salience of the nodes of an LM's object models during an episode.

    A model-based form of salience: how worthwhile each node of a learned object
    model is to test, from 1 (nothing is known about it) down to 0 (fully
    explained, never to be tested), as opposed to the model-free salience map
    that a `SalienceSM` computes from what its sensor observes (see
    `tbp.monty.frameworks.models.salience`). The
    `EvidenceGoalGenerator` reduces the salience of the parts of the most likely
    hypothesis (MLH) that its input explains, and samples the nodes to test in
    proportion to their salience.

    Salience is tracked separately for the graph of each input channel of each
    object, keyed by (graph_id, input_channel), so a location explained in one
    channel's graph can still be tested in another's. It is indexed by node, i.e.
    in the object's reference frame, so it remains valid when the MLH pose
    changes. It is short-term, episodic state, and so is kept here rather than in
    the object models, which are long-term memory (rebuilt, saved and restored
    independently of any episode):
    - Each node starts at 1, and recovers linearly back to 1 over
      `recovery_steps` steps, except on the steps on which its salience is
      reduced, so that a part explained repeatedly is driven to 0.
    - The salience of every tracked graph recovers on every step, not only the
      MLH's: if the MLH switches to another object and back, the explained parts
      of the first are remembered, less the recovery of the steps in between.
    - Once every node of a graph other than the MLH's has fully recovered, the
      graph is no longer tracked, as its salience is the same as if it had
      never been reduced; it is recreated (at 1) when needed again.
    - Everything is cleared by `reset`, at the start of each episode. A graph
      whose number of nodes has changed (i.e. that was rebuilt) is also
      recreated at 1.
    """

    def __init__(self, recovery_steps: int = 50) -> None:
        """Initialize the salience, with no graph tracked.

        Args:
            recovery_steps: Number of steps over which the salience of a node
                linearly recovers from 0 to 1. 0 to not recover salience.

        Raises:
            ValueError: If recovery_steps is negative.
        """
        if recovery_steps < 0:
            raise ValueError("recovery_steps must be positive or 0")
        self.recovery_steps = recovery_steps
        self._salience: dict[tuple[str, str], np.ndarray] = {}

    def reset(self) -> None:
        """Restore the salience of every node to 1, tracking no graph."""
        self._salience = {}

    def tracked(self) -> list[tuple[str, str]]:
        """Get the graphs whose salience is tracked.

        Returns:
            The (graph_id, input_channel) of each tracked graph.
        """
        return list(self._salience)

    def get(self, graph_id: str, input_channel: str, num_nodes: int) -> np.ndarray:
        """Get the salience of each node of a graph.

        Args:
            graph_id: The object whose graph it is.
            input_channel: The input channel of the graph.
            num_nodes: The number of nodes of the graph.

        Returns:
            The stored array of the salience (in [0, 1]) of each node, which is
            1 for every node of a graph not tracked (or rebuilt); change it only
            through `reduce` and `recover`.
        """
        key = (graph_id, input_channel)
        salience = self._salience.get(key)
        if salience is None or len(salience) != num_nodes:
            salience = np.ones(num_nodes, dtype=float)
            self._salience[key] = salience
        return salience

    def reduce(
        self,
        graph_id: str,
        input_channel: str,
        num_nodes: int,
        nodes: np.ndarray,
        amounts: np.ndarray,
    ) -> None:
        """Reduce the salience of some nodes of a graph, down to at most 0.

        Args:
            graph_id: The object whose graph it is.
            input_channel: The input channel of the graph.
            num_nodes: The number of nodes of the graph.
            nodes: The nodes whose salience to reduce.
            amounts: How much to reduce the salience of each of them by.
        """
        salience = self.get(graph_id, input_channel, num_nodes)
        reduced = salience[nodes] - amounts
        # Snap residue to 0, so that fully explained nodes are never sampled.
        salience[nodes] = np.where(reduced > _ZERO_SALIENCE, reduced, 0.0)

    def recover(
        self,
        except_nodes: dict[tuple[str, str], np.ndarray] | None = None,
        mlh_graph_id: str | None = None,
    ) -> None:
        """Recover the salience of every tracked node by one step, towards 1.

        Then stops tracking the graphs of objects other than the MLH whose
        salience has fully recovered.

        Args:
            except_nodes: For each (graph_id, input_channel), the nodes whose
                salience does not recover on this step (those reduced on it).
            mlh_graph_id: The MLH object, whose graphs stay tracked.
        """
        if self.recovery_steps:
            recovery = 1.0 / self.recovery_steps
            for key, salience in self._salience.items():
                recovers = np.ones(len(salience), dtype=bool)
                if except_nodes and key in except_nodes:
                    recovers[except_nodes[key]] = False
                salience[recovers] = np.minimum(salience[recovers] + recovery, 1.0)
        recovered = [
            key
            for key, salience in self._salience.items()
            if key[0] != mlh_graph_id and np.all(salience >= 1.0)
        ]
        for key in recovered:
            del self._salience[key]
