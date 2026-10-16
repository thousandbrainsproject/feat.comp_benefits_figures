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

__all__ = ["MAX_SALIENCE", "RESTING_SALIENCE", "ModelSalience", "relaxed"]

# The salience every node starts at, and relaxes back to.
RESTING_SALIENCE = 0.5
# The salience of the nodes a goal is about to test.
MAX_SALIENCE = 1.0
# Salience left below this by repeated reductions is floating-point residue.
_ZERO_SALIENCE = 1e-9


def relaxed(salience: np.ndarray, amount: float) -> np.ndarray:
    """Move salience towards RESTING_SALIENCE, without overshooting it.

    Args:
        salience: The salience of some nodes.
        amount: How far to move each of them.

    Returns:
        The relaxed salience (a new array): salience below RESTING_SALIENCE
        raised by amount, salience above it lowered by amount, either of them
        stopping at RESTING_SALIENCE.
    """
    return np.where(
        salience < RESTING_SALIENCE,
        np.minimum(salience + amount, RESTING_SALIENCE),
        np.maximum(salience - amount, RESTING_SALIENCE),
    )


class ModelSalience:
    """The salience of the nodes of an LM's object models during an episode.

    A model-based form of salience: how worthwhile each node of a learned object
    model is to test, in [0, 1], as opposed to the model-free salience map that
    a `SalienceSM` computes from what its sensor observes (see
    `tbp.monty.frameworks.models.salience`). The `EvidenceGoalGenerator` samples
    the nodes to test in proportion to their salience, raises the salience of
    the part of the most likely hypothesis (MLH) a goal is about to test, and
    reduces the salience of the parts its input explains.

    Salience is tracked separately for the graph of each input channel of each
    object, keyed by (graph_id, input_channel), so a location explained in one
    channel's graph can still be tested in another's. It is indexed by node, i.e.
    in the object's reference frame, so it remains valid when the MLH pose
    changes. It is short-term, episodic state, and so is kept here rather than in
    the object models, which are long-term memory (rebuilt, saved and restored
    independently of any episode):
    - Each node starts at the resting salience, RESTING_SALIENCE (0.5): nothing
      makes it more or less worthwhile to test than any other node.
    - `excite` raises nodes to MAX_SALIENCE (1), and `reduce` lowers them, down
      to at most 0, regardless of the resting salience (a node raised to 1 can
      be reduced all the way to 0).
    - On every step, each node relaxes linearly towards the resting salience,
      from above or below, taking `relaxation_steps` steps to get there from
      either 0 or 1, except on the steps on which its salience is reduced, so
      that a part explained repeatedly is driven to 0.
    - The salience of every tracked graph relaxes on every step, not only the
      MLH's: if the MLH switches to another object and back, the explained parts
      of the first are remembered, less the relaxation of the steps in between.
    - Once every node of a graph other than the MLH's is at rest, the graph is no
      longer tracked, as its salience is the same as if it had never changed; it
      is recreated (at rest) when needed again.
    - Everything is cleared by `reset`, at the start of each episode. A graph
      whose number of nodes has changed (i.e. that was rebuilt) is also
      recreated at rest.
    """

    def __init__(self, relaxation_steps: int = 50) -> None:
        """Initialize the salience, with no graph tracked.

        Args:
            relaxation_steps: Number of steps over which the salience of a node
                linearly relaxes to the resting salience from either 0 or 1. 0 to
                not relax salience.

        Raises:
            ValueError: If relaxation_steps is negative.
        """
        if relaxation_steps < 0:
            raise ValueError("relaxation_steps must be positive or 0")
        self.relaxation_steps = relaxation_steps
        self._salience: dict[tuple[str, str], np.ndarray] = {}

    @property
    def relaxation_rate(self) -> float:
        """How far a node's salience relaxes towards rest on each step."""
        if not self.relaxation_steps:
            return 0.0
        return RESTING_SALIENCE / self.relaxation_steps

    def reset(self) -> None:
        """Restore every node to the resting salience, tracking no graph."""
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
            the resting salience for every node of a graph not tracked (or
            rebuilt); change it only through `excite`, `reduce` and `relax`.
        """
        key = (graph_id, input_channel)
        salience = self._salience.get(key)
        if salience is None or len(salience) != num_nodes:
            salience = np.full(num_nodes, RESTING_SALIENCE, dtype=float)
            self._salience[key] = salience
        return salience

    def excite(
        self,
        graph_id: str,
        input_channel: str,
        num_nodes: int,
        nodes: np.ndarray,
    ) -> None:
        """Raise the salience of some nodes of a graph to MAX_SALIENCE.

        Args:
            graph_id: The object whose graph it is.
            input_channel: The input channel of the graph.
            num_nodes: The number of nodes of the graph.
            nodes: The nodes whose salience to raise.
        """
        self.get(graph_id, input_channel, num_nodes)[nodes] = MAX_SALIENCE

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

    def relax(
        self,
        except_nodes: dict[tuple[str, str], np.ndarray] | None = None,
        mlh_graph_id: str | None = None,
    ) -> None:
        """Relax the salience of every tracked node by one step, towards rest.

        Then stops tracking the graphs of objects other than the MLH whose
        salience is entirely at rest.

        Args:
            except_nodes: For each (graph_id, input_channel), the nodes whose
                salience does not relax on this step (those reduced on it).
            mlh_graph_id: The MLH object, whose graphs stay tracked.
        """
        rate = self.relaxation_rate
        if rate:
            for key, salience in self._salience.items():
                relaxes = np.ones(len(salience), dtype=bool)
                if except_nodes and key in except_nodes:
                    relaxes[except_nodes[key]] = False
                salience[relaxes] = relaxed(salience[relaxes], rate)
        at_rest = [
            key
            for key, salience in self._salience.items()
            if key[0] != mlh_graph_id and np.all(salience == RESTING_SALIENCE)
        ]
        for key in at_rest:
            del self._salience[key]
