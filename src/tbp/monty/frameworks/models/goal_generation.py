# Copyright 2025-2026 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    GoalGenerator,
)
from tbp.monty.frameworks.models.model_salience import ModelSalience
from tbp.monty.frameworks.utils.communication_utils import get_percept_from_channel

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.graph_matching import GraphLM

__all__ = [
    "EvidenceGoalGenerator",
    "GraphGoalGenerator",
    "ParentLMNotProvided",
    "PropagationRecord",
]

logger = logging.getLogger(__name__)


class ParentLMNotProvided(AttributeError):
    """Parent LM wasn't provided to a GoalGenerator.

    Error raised when a parent learning module is accessed before it is provided to
    a goal generator.
    """


class GraphGoalGenerator(GoalGenerator):
    """Generate sub-Goals until the received Goal is achieved.

    A component (embedded in a learning module) that receives a high-level Goal and
    generates sub-Goals until that Goal is achieved.

    Generated Goals are received by either:
        i) other learning modules, which may model world objects (e.g., a mug) or
        internal systems (e.g., the agent's robotic limb)
        ii) motor actuators, in which case they represent simpler, primitive Goals
        for the actuator to achieve (e.g., the location and orientation of an
        actuator-sensor pair)

    Alongside the high-level driving Goal, generated sub-Goals can also be
    conditioned on other information such as the LM's current most-likely hypothesis
    and the structure of known object models (i.e., information local to the LM).

    Note that all Goals conform to the Cortical Messaging Protocol (CMP).
    """

    def __init__(self, goal_tolerances=None, **_kwargs) -> None:
        """Initialize the GSG.

        Note: the GSG is not fully initialized until the `parent_lm` is set by the owner
        of the GSG. This step is separated out to allow for dependency injection.

        Args:
            goal_tolerances: The tolerances for each attribute of the Goal that can be
                used by the GSG when determining whether a Goal is achieved. These are
                not necessarily the same as an LM's tolerances used for matching, as
                here we are evaluating whether a Goal is achieved.
            **kwargs: Additional keyword arguments. Unused.
        """
        # Do not access directly, use the property defined below.
        self._parent_lm: GraphLM | None = None

        if goal_tolerances is None:
            self.goal_tolerances = dict(
                location=0.015,  # distance in meters
            )
        else:
            self.goal_tolerances = goal_tolerances

    # =============== Public Interface Functions ===============

    # ------------------ Getters & Setters ---------------------

    @property
    def parent_lm(self) -> GraphLM:
        if not self._parent_lm:
            raise ParentLMNotProvided("Parent learning module has not been provided.")
        return self._parent_lm

    @parent_lm.setter
    def parent_lm(self, parent_lm: GraphLM) -> None:
        """Sets the parent learning module for this GSG.

        After setting the LM, it resets the GSG.
        """
        self._parent_lm = parent_lm
        self.reset()

    def reset(self):
        """Reset any stored attributes of the GSG."""
        self.set_driving_goal(self._generate_none_goal())
        self._set_output_goal(self._generate_none_goal())
        self.parent_lm.buffer.update_stats(
            dict(
                goal_states=[],
                matching_step_when_output_goal_set=[],
                goal_state_achieved=[],
            ),
            update_time=False,
            append=False,
            init_list=False,
        )

    def set_driving_goal(self, goal):
        """Receive a new high-level Goal to drive this Goal Generator (GSG).

        If none is provided, the GSG should default to pursuing a high confidence
        Goal, with no other attributes of the message specified; in
        other words, it attempts to reduce uncertainty about the LM's output
        (object ID and pose, whatever these may be).

        TODO M: Currently GSGs always use the default, however future work will
        implement hierarchical action policies/GSGs, as well as the ability to
        specify a top Goal by the experimenter.

        TODO M: we currently just use "None" as a placeholder for the default Goal.
        > plan : set the default driving Goal to a meaningful, non-None value
        that is compatible with the current method for checking convergence of an
        LM, such that achieving the driving Goal can be used as a test for Monty
        convergence.
        """
        self.driving_goal = goal

    def output_goals(self) -> list[Goal]:
        """Retrieve the output Goals of the GSG.

        This is the Goal projected to other LMs' GSGs and/or motor actuators.

        Returns:
            Output Goals of the GSG if it exists, otherwise empty list.
        """
        return [self.output_goal] if self.output_goal else []

    # ------------------- Main Algorithm -----------------------

    def step(self, ctx: RuntimeContext, observations):
        """Step the GSG.

        Check whether the GSG's output and driving Goals are achieved, and
        generate a new output Goal if necessary.
        """
        output_goal_achieved = self._check_output_goal_achieved(observations)

        self._update_gsg_logging(output_goal_achieved)

        if self._check_need_new_output_goal(ctx, output_goal_achieved):
            self._set_output_goal(new_goal=self._generate_goal(ctx, observations))
        elif self._check_keep_current_output_goal():
            pass
        else:
            self._set_output_goal(new_goal=self._generate_none_goal())

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _generate_none_goal(self):
        """Return a None-type Goal.

        A None-type Goal specifies nothing other than high confidence.

        NOTE: currently we just use a None value, however in the future we might
        specify a GoalState object with a None value for the location, morphological
        features, etc, or some variation of this.
        """
        return

    def _generate_goal(self, _ctx: RuntimeContext, _observations):
        """Generate a new Goal to send out to other LMs and/or motor actuators.

        Given the driving Goal, and information from the parent LM of the GSG
        (including the current observations), generate a new Goal to send out to
        other LMs and/or motor actuators.

        Note the output Goal is in a common, body-centered frame of reference, as
        for voting, such that different modules can mutually communicate.

        This version is a base placeholder method that just returns a None Goal,
        and does not actually make use of observations or the driving Goal.

        Returns:
            A None Goal.
        """
        return self._generate_none_goal()

    def _check_messages_different(
        self,
        msg_a,
        msg_b,
        diff_tolerances,
    ) -> bool:
        """Check whether two messages are different.

        Messages need to be different only by one feature/dimension to be considered
        different.

        When checking whether the messages are different, a dictionary of tolerances
        must be passed; in the GSG-class, this is typically the GSG's own default
        goal-tolerances that are used, but specific tolerances can also be passed
        along with a Goal itself (i.e. achieve this Goal within these tolerance bounds).

        Note:
            If a message is undefined (None), we define a difference as unmeaningful and
            therefore return False. Similarly, for any feature of a message (or
            dimension of a feature) that is undefined (None or NaN), we do not return
            any difference along that dimension.

        TODO M consider making this a utility function, as might be useful in e.g. the
        LM itself as well. However, the significant presence of None/NaN values in
        Goals may mean we want to take a different approach.

        Returns:
            Whether the states are different.
        """
        if msg_a is None or msg_b is None:
            return False

        states_different = False
        for tolerance_key, tolerance_val in diff_tolerances.items():
            # TODO M implement feature comparisons for other Message features (e.g.
            # confidence)
            # TODO M consider using the LM's lm.tolerances for the default values of
            # diff_tolerances

            if (
                tolerance_key == "location"
                and msg_a.location is not None
                and msg_b.location is not None
            ):
                distance = np.linalg.norm(msg_a.location - msg_b.location)

            elif (
                tolerance_key == "pose_vectors"
                and msg_a.morphological_features is not None
                and msg_b.morphological_features is not None
            ):
                raise NotImplementedError(
                    "TODO M implement pose-vector comparisons that handle "
                    "symmetry of objects"
                )
                # TODO M consider using an angular distance instead of Euclidean
                # when we actually begin making use of this feature; try to ensure
                # this handles symmetry conditions e.g. flipped principal curvature
                # directions.
                distance = self._euc_dist_ignoring_nan(
                    msg_a.morphological_features["pose_vectors"],
                    msg_b.morphological_features["pose_vectors"],
                )

            states_different = distance > tolerance_val
            if states_different:
                return states_different

        return states_different

    def _check_driving_goal_achieved(self) -> bool:
        """Check if parent LM's output percept is close enough to driving Goal.

        TODO M Move some of the checks for convergence here

        Returns:
            Whether the parent LM's output percept is close enough to the driving Goal.
        """
        if self.driving_goal.goal_tolerances is None:
            # When not specified by the incoming driving Goal, use the GSG's own
            # default matching tolerances
            diff_tolerances = self.goal_tolerances

        return self._check_messages_different(
            self.parent_lm.get_output(), self.driving_goal, diff_tolerances
        )

    def _check_output_goal_achieved(self, observations) -> bool:
        """Check if the output Goal was achieved.

        Check whether the information entering the LM suggests that the output Goal
        of the GSG was achieved. Recall that the output Goal is the one sent by this
        GSG to other LMs and motor-actuators to be achieved.

        Note:
            In the future we might use feedback from a receiving system that is not
            "sensory input" (i.e. does not inform the graph building of this parent LM);
            Instead, such feedback could include the percept of an LM that controls a
            motor system (such as a hand model LM), or the percept of a motor-actuator
            (akin to proprioceptive feedback); in this case, we could directly compare
            the output Goal to the percept received by this feedback to determine
            whether the Goal was likely achieved. This input would likely come
            from a separate channel (similar to voting). Finally, note that this
            information could be complementary to feedback from the sensory input and
            our sensory predictions, as in some cases we might have no proprioceptive
            feedback, while in other cases we might have no sensory input (e.g.
            blindfolded); alignment or mismatch between these two could form useful
            signals for learning policies and object behaviors.

        Returns:
            Whether the output Goal was achieved.
        """
        if self.output_goal is not None:
            return self._check_input_matches_sensory_prediction(observations)

        return False

    def _check_input_matches_sensory_prediction(self, percepts: list[Message]) -> bool:
        """Check whether the input matches the sensory prediction.

        Here the sensory prediction is simply that the input percept has changed, as
        when the motor-system attempts to achieve a Goal and fails (e.g. due to
        collision with another object), it moves back to the original position.

        Note that there can still be some difference even when a Goal fails, as the
        feature-change-SM and motor-only steps can result in the agent moving after it
        has returned to its original position. Furthermore, there may not always be a
        difference if the agent did "succeed", if the Goal it wanted to achieve
        happened to be very close to its original position. Thus this is an
        approximate method.

        TODO M: Implement also using the target Goal and internal model to predict a
        specific percept, and then compare to that to determine not just whether
        a movement took place, but whether the agent moved to a particular point on a
        particular object.

        Returns:
            Whether the input matches the sensory prediction.
        """
        sensor_channel_name = self.parent_lm.buffer.get_first_sensory_input_channel()

        current_sensory_input = get_percept_from_channel(
            percepts=percepts, channel_name=sensor_channel_name
        )

        prev_input_percepts = self.parent_lm.buffer.get_previous_input_percepts()
        if prev_input_percepts is not None:
            previous_sensory_input = get_percept_from_channel(
                percepts=prev_input_percepts,
                channel_name=sensor_channel_name,
            )
        else:
            previous_sensory_input = None
        # NB if no history of inputs, get_previous_input_percepts returns None, in which
        # case _check_states_different will return False, and we return goal_achieved as
        # False, as we cannot meaningfully evaluate whether this occurred

        return self._check_messages_different(
            current_sensory_input,
            previous_sensory_input,
            diff_tolerances=self.goal_tolerances,
        )

    def _check_need_new_output_goal(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        output_goal_achieved,
    ) -> bool:
        """Determine whether the GSG should generate a new output Goal.

        In the base version, this is True if the output-goal was achieved, suggesting
        we should move on to the next goal.

        Returns:
            Whether the GSG should generate a new output Goal.
        """
        return bool(output_goal_achieved)

    def _check_keep_current_output_goal(self) -> bool:
        """Should we keep our current goal?

        If we don't need a new goal, determine whether we should keep our current
        goal (as opposed to output no goal at all).

        Returns:
            Whether we should keep our current goal.
        """
        return True

    def _euc_dist_ignoring_nan(self, a, b):
        """Euclidean distance between two arrays, ignoring NaN values.

        Take the Euclidean distance between two arrays, but only measuring the
        distance where both arrays have non-NaN values.

        Args:
            a: First array
            b: Second array

        Returns:
            Euclidean distance between the two arrays, ignoring NaN values; if all
            values are NaN, return 0

        TODO M consider making a general utility function
        """
        assert a.shape == b.shape, "Arrays must be of the same shape"

        mask = ~np.isnan(a) & ~np.isnan(b)

        # If the mask is empty, return 0 (i.e. there is no meaningful distance
        # between the two vectors)
        if len(mask) == 0:
            return 0

        return np.linalg.norm(a[mask] - b[mask])

    # ------------------ Getters, Setters & Logging ---------------------

    def _set_output_goal(self, new_goal):
        """Set the output Goal of the GSG."""
        self.output_goal = new_goal

    def _update_gsg_logging(self, output_goal_achieved: bool):
        """Update any logging information (stored in the parent LM's buffer).

        Update any logging information (stored in the parent LM's buffer), such as
        the matching step on which an output Goal was output.
        """
        # Only consider output achieved for the purpose of logging when the
        # output Goal is meaningful (i.e. not None)
        if self.output_goal is not None:
            # Subtract 1 as the Goal was actually set (and potentially achieved)
            # on the previous step, we are simply first checking it now

            match_step = self.parent_lm.buffer.get_num_matching_steps() - 1
            self.output_goal.info["achieved"] = output_goal_achieved
            self.output_goal.info["matching_step_when_output_goal_set"] = match_step
            self.parent_lm.buffer.update_stats(
                dict(
                    goal_states=self.output_goal,
                    matching_step_when_output_goal_set=match_step,
                    goal_state_achieved=output_goal_achieved,
                ),
                update_time=False,
                append=True,
                init_list=True,
            )


@dataclass(frozen=True)
class PropagationRecord:
    """One propagation of salience by an `EvidenceGoalGenerator`.

    Either a propagation of sensed input, which reduces the salience of the nodes
    it reaches, or a hypothetical one, from the target of a Goal as if its
    location were sensed, which raises their salience to MAX_SALIENCE.

    Attributes:
        graph_id: The graph (the MLH object) the propagation went through.
        input_channel: The input channel whose graph the propagation went through,
            i.e. the channel the input was received on (or the Goal's target was
            selected from).
        object_id: The "object_id" feature value that was received (or that the
            Goal's target stores), or None for a propagation through similar
            sensory features.
        node_order: The nodes whose salience was changed, in the order the
            propagation reached them: by increasing number of hops from the seeds
            (see `_get_propagation_order`).
        seeds: The nodes the propagation began from.
        hops: For each node of `node_order`, the number of hops from the nodes its
            stage of the propagation began from (0 for the seeds, and for the
            nodes a push-through began from).
        stage: For each node of `node_order`, the stage of the propagation that
            reached it: 0 for the propagation from the seeds, and n for the n-th
            push-through.
        predecessor: For each node of `node_order`, the node it was reached from,
            or -1 for the nodes a stage began from.
        sinks: The reached nodes that do not propagate (e.g. because too many of
            their neighbors store other features or object IDs), where the
            propagation came to a stop.
        min_size: The number of nodes the propagation had to reach before
            stopping, or 0 if it was never pushed through.
        reduction: For each node of `node_order`, the salience reduction applied
            to it (before clamping its salience at 0), see `_salience_reductions`;
            0 for a hypothetical propagation.
        hypothetical: Whether the propagation was from the target of a Goal,
            raising the salience of the nodes it reached to MAX_SALIENCE.
    """

    graph_id: str
    input_channel: str
    object_id: int | None
    node_order: np.ndarray
    seeds: np.ndarray
    hops: np.ndarray
    stage: np.ndarray
    predecessor: np.ndarray
    sinks: np.ndarray
    min_size: float
    reduction: np.ndarray
    hypothetical: bool = False


class EvidenceGoalGenerator(GraphGoalGenerator):
    """Generator of Goals for an evidence-based graph LM.

    GSG specifically set up for generating Goals for an evidence-based graph LM,
    which can therefore leverage the compositional inference policy. This policy
    uses the most likely object hypothesis (MLH), as well as knowledge of its
    structure from recent memory, to propose test-points that visit the
    unexplained parts of the object the agent is currently observing.

    Goals are directed at the parts of the model of the most likely object
    hypothesis (MLH) that the LM's input has not yet explained. Every node of
    every channel's graph of the MLH has a salience in [0, 1]: how worthwhile it
    is to test, at rest 0.5 (`RESTING_SALIENCE`: no more or less worthwhile than
    any other node), up to 1 (about to be tested by a Goal), and down to 0
    (fully explained, cannot be tested). The location to test is sampled from the
    nodes of all of the channels together, with probability proportional to
    salience, so that initially (salience uniformly at rest) a channel with more
    nodes (e.g. the sensory channel) is more likely to be sampled than one with
    fewer (e.g. a channel of object IDs from a lower-level LM, i.e. a
    compositional model).

    Input that the MLH predicts at its current location "explains" part of the
    model, whose salience is reduced by `salience_reduction` (falling off along
    the propagation by `salience_reduction_falloff`), by propagating through the
    graph of the channel the input was received on, starting from the nodes
    nearest the MLH location, through neighboring nodes with similar features.

    When a Goal is generated, the part of the model its target belongs to is
    highlighted: a hypothetical propagation, as if the target's location were
    sensed (with the features the target stores), goes through the graph of the
    channel the target was selected from, raising the salience of the nodes it
    reaches to 1 rather than reducing it. If the Goal is achieved, the input
    sensed there then reduces that salience on each step the part is explained
    (e.g. from 1 to 0.8, 0.6, ...), so that the sensor dwells on the part until
    its salience falls below that of the rest of the object, and the next Goal
    is likely to be elsewhere. Only the target's channel is highlighted, so e.g.
    a target on a logo highlights the logo, not the rest of the object it is
    on.

    Object IDs and sensory features are propagated alike, differing only in the
    features compared:
    - On a channel receiving object IDs from a lower-level LM, the object ID,
      which is similar only if it is the same, so the propagation goes through
      the nodes storing the received ID.
    - On a channel receiving sensory input, the hue and surface normal, which
      are similar within the parent LM's hue and surface normal ("pose_vectors")
      tolerances for the channel. Propagation stops wherever the hue or the
      surface normal changes abruptly, so e.g. on a mug it covers the side of
      the cylinder (whose normal changes gradually from node to node), but not
      its rim, bottom or handle, and on a cube a single face, but not the other
      faces.
    A propagation that stops before reaching `min_propagation_fraction` of the
    graph's nodes is pushed through to the neighbors whose features differ.

    A node's neighbors are its `num_propagation_neighbors` nearest nodes
    (regardless of their distance), together with the nodes that have it among
    their own nearest nodes, and it propagates to those with features similar to
    its own, as long as at most `max_dissimilar_neighbors` of them are not
    (allowing for noisy neighbors, which it does not propagate to).
    A propagation therefore covers a contiguous region, including outlying nodes,
    without jumping to disjoint regions (e.g. separate wheels on a car).

    Input only propagates once the MLH estbalishes credibility: when the parent LM's
    hypotheses updater deletes hypotheses (e.g. `BurstSamplingHypothesesUpdater`,
    whose `EvidenceSlopeTracker` makes a hypothesis eligible for deletion once its
    age reaches `min_age`), the MLH must have survived its first chance of being
    deleted, i.e. its age must exceed `min_age`. Until then, a newly sampled
    hypothesis that happens to have the most evidence does not reduce the salience
    of its object (salience still relaxes), nor is a Goal generated to test it.
    With an updater that never deletes hypotheses, input always propagates, and
    Goals are generated as soon as the MLH is a known object.

    The salience of the nodes is kept in a `ModelSalience` (`model_salience`),
    which documents how it relaxes back to rest from above or below (over
    `salience_relaxation_steps` steps, except on the steps a node's salience is
    reduced), and how long it is kept: per
    object, for the whole episode, including while the MLH is another object. A
    change of the MLH object or rotation does not reset it; it only prompts a new
    Goal (see `_check_need_new_output_goal`). As salience is indexed by node, in
    the object's reference frame, it remains valid when the MLH rotation changes;
    however, the nodes past propagations reached were found using the MLH pose at
    the time (the seeds nearest the MLH location, and sensed surface normals
    rotated by the MLH rotation), so if that pose was wrong, the wrong nodes were
    reduced, and they are not restored until they relax.
    """

    def __init__(
        self,
        goal_tolerances=None,
        elapsed_steps_trigger=20,
        num_propagation_neighbors=10,
        max_dissimilar_neighbors=2,
        min_propagation_fraction: float | None = 0.05,
        salience_reduction=0.2,
        salience_reduction_falloff=0.5,
        salience_relaxation_steps=50,
        **kwargs,
    ) -> None:
        """Initialize the GSG.

        Args:
            goal_tolerances: The tolerances for each attribute of the Goal that can be
                used by the GSG when determining whether a Goal is achieved.
            elapsed_steps_trigger: A Goal is generated every elapsed_steps_trigger
                steps, even if the MLH has not changed. Defaults to 20.
            num_propagation_neighbors: Number of nearest neighbors of each node
                that, together with the nodes that have it among their own nearest
                neighbors, must have similar features (e.g. store the same object
                ID), up to max_dissimilar_neighbors, for a propagation to continue
                from the node. Also the number of nodes
                nearest the MLH location (or a Goal's target) that a propagation
                begins from. On a wall
                thinner than the spacing of nodes (e.g. a bowl's), about half of a
                node's nearest neighbors are on the far side of the wall (see
                `_far_side_pairs`), and so are ignored. Defaults to 10.
            max_dissimilar_neighbors: The number of a node's neighbors whose
                features (or object IDs) may differ from its own, without stopping
                it from propagating to the others (which have similar features).
                The features and object IDs stored in learned models are noisy, so
                with 0 a single noisy neighbor stops a node from propagating; more
                lets propagation leak across boundaries (e.g. onto a mug's
                handle). Defaults to 2.
            min_propagation_fraction: The fraction of the
                graph's nodes a propagation must reach: when it stops before
                reaching them, it is pushed through to the neighbors whose
                features differ (see `_get_propagation_order`). This keeps
                fragments of noisy features, and small parts like a mug's handle,
                from being explained a few nodes at a time. None to not push
                propagation through. Defaults to 0.05.
            salience_reduction: How much the salience of the first node a
                propagation reaches is reduced by (minimum salience is clamped at 0).
                With 0.2, a part highlighted by a Goal (at salience 1) has to be
                explained on five visits before it will not be tested at all.
                Defaults to 0.2.
            salience_reduction_falloff: Fraction by which the reduction falls off
                along a propagation, linearly in the order the nodes are reached:
                the last node reached is reduced by salience_reduction * (1 -
                salience_reduction_falloff). 0 reduces every reached node by the
                same amount, wherever it is relative to the first node. Defaults to 0.5.
            salience_relaxation_steps: Number of steps over which the salience of
                a node linearly relaxes to rest (0.5) from either 0 or 1 (see
                `ModelSalience`). 0 to not relax salience. Defaults to 50 steps.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If a salience or propagation parameter is out of range.
        """
        super().__init__(goal_tolerances, **kwargs)
        self.elapsed_steps_trigger = elapsed_steps_trigger
        self.num_propagation_neighbors = num_propagation_neighbors
        self.max_dissimilar_neighbors = max_dissimilar_neighbors
        self.min_propagation_fraction = min_propagation_fraction
        self.salience_reduction = salience_reduction
        self.salience_reduction_falloff = salience_reduction_falloff
        self.model_salience = ModelSalience(salience_relaxation_steps)

        if num_propagation_neighbors <= 0:
            raise ValueError("num_propagation_neighbors must be positive")
        if max_dissimilar_neighbors < 0:
            raise ValueError("max_dissimilar_neighbors must be positive or 0")
        if min_propagation_fraction is not None and (
            min_propagation_fraction > 1 or min_propagation_fraction < 0
        ):
            raise ValueError("min_propagation_fraction must be in [0, 1] or None")
        if salience_reduction > 1 or salience_reduction < 0:
            raise ValueError("salience_reduction must be in [0, 1]")
        if salience_reduction_falloff > 1 or salience_reduction_falloff < 0:
            raise ValueError("salience_reduction_falloff must be in [0, 1]")

    # ======================= Public ==========================

    def reset(self):
        """Reset the salience of all nodes, and the MLH at the last Goal."""
        super().reset()
        self.model_salience.reset()
        # Keyed by (graph_id, input_channel, number of nodes).
        self._neighbor_cache: dict[tuple[str, str, int], csr_matrix] = {}
        # For propagating sensory input (see `_get_similar_neighbors`), keyed by
        # (graph_id, input_channel, number of nodes).
        self._similar_neighbors_cache: dict[
            tuple[str, str, int], tuple[csr_matrix, csr_matrix]
        ] = {}
        self._prev_goal_mlh: dict | None = None
        self.propagation_records: list[PropagationRecord] = []
        self.goal_propagation_record: PropagationRecord | None = None

    def step(self, ctx: RuntimeContext, observations):
        """Update the salience of the parts of the MLH, then step the GSG.

        A new `propagation_records` list is created on every step, holding the
        propagations of sensed input that took place on that step, and
        `goal_propagation_record` holds the hypothetical propagation from the
        target of the Goal generated on that step, if any.
        """
        self.goal_propagation_record = None
        self._update_salience(observations)
        super().step(ctx, observations)

    def get_salience(self, graph_id, input_channel) -> np.ndarray:
        """Get the salience of each node of a graph.

        TODO make this private, only public for visualization purposes.

        Returns:
            Array with one salience in [0, 1] per node of the graph of the given
            input channel: how likely the node is to be selected for testing,
            relative to the others (never, at 0; RESTING_SALIENCE when nothing is
            known about it).
        """
        num_nodes = len(self.parent_lm.get_graph(graph_id, input_channel).pos)
        return self.model_salience.get(graph_id, input_channel, num_nodes)

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _update_salience(self, observations) -> None:
        """Reduce the salience of the parts of the MLH explained by this step's input.

        The input propagates through the MLH's graphs (see
        `_propagate_from_observations`), which reduces the salience of the nodes
        it reaches; the salience of every other node relaxes by one step.
        """
        self.propagation_records = []
        self._propagate_from_observations(observations)
        reached: dict[tuple[str, str], list[np.ndarray]] = {}
        for record in self.propagation_records:
            key = (record.graph_id, record.input_channel)
            reached.setdefault(key, []).append(record.node_order)
        self.model_salience.relax(
            except_nodes={key: np.concatenate(nodes) for key, nodes in reached.items()},
            mlh_graph_id=self._get_mlh_graph_id(),
        )

    def _generate_goal(self, ctx: RuntimeContext, observations) -> Goal | None:
        """Generate a Goal that moves the sensor to an unexplained part of the MLH.

        Returns:
            A Goal for the motor system, or a None-type Goal if no node of the
            MLH object's graphs has any salience.
        """
        graph_id = self._get_mlh_graph_id()
        if graph_id is None:
            return self._generate_none_goal()

        mlh = self.parent_lm._get_current_mlh()
        self._prev_goal_mlh = {
            "graph_id": mlh["graph_id"],
            "rotation": mlh["rotation"],
        }

        target = self._propose_target(ctx, graph_id)
        if target is None:
            logger.debug(f"No node of {graph_id} has any salience; no goal")
            return self._generate_none_goal()

        input_channel, node_id, object_id = target
        target_info = self._get_target_loc_info(node_id, input_channel)
        goal = self._compute_goal_for_target_loc(
            observations,
            target_info,
            goal_confidence=self.parent_lm.get_output().confidence,
        )
        goal.info["target_node_id"] = node_id
        goal.info["target_child_object_id"] = object_id
        tested = "sensory features" if object_id is None else f"object ID {object_id}"
        logger.debug(
            f"Child objects goal: testing {tested} on channel {input_channel} "
            f"(node {node_id}) of {graph_id}"
        )
        return goal

    def _propose_target(self, ctx: RuntimeContext, graph_id):
        """Sample a node to test, and highlight the part of the model it is on.

        Returns:
            The target, as returned by `_select_target`, whose part of the model
            has had its salience raised (see `_excite_target_region`); or None if
            no node has any salience.
        """
        target = self._select_target(ctx, graph_id)
        if target is not None:
            self._excite_target_region(graph_id, *target)
        return target

    def _select_target(self, ctx: RuntimeContext, graph_id):
        """Sample a node of the graph's channels to test, weighted by salience.

        The nodes of all of the graph's channels are sampled from together, each
        with probability proportional to its salience, so nodes with no salience
        are never sampled, and a channel's share of the samples is in proportion
        to its number of (salient) nodes.

        Returns:
            A tuple of (input_channel, node_id, object_id), where object_id is the
            ID the node stores on an object-ID channel, or None on a sensory
            channel; or None if no node has any salience.
        """
        channels = self.parent_lm.get_input_channels_in_graph(graph_id)
        graphs = [
            self.parent_lm.get_graph(graph_id, input_channel=channel)
            for channel in channels
        ]
        salience = [
            self.model_salience.get(graph_id, channel, len(graph.pos))
            for channel, graph in zip(channels, graphs)
        ]
        weights = np.concatenate(salience) if salience else np.empty(0)
        total = weights.sum()
        if total <= 0:
            return None

        sample = int(ctx.rng.choice(len(weights), p=weights / total))
        first_node = np.cumsum([0] + [len(s) for s in salience])
        index = int(np.searchsorted(first_node, sample, side="right") - 1)
        channel, graph = channels[index], graphs[index]
        node_id = sample - int(first_node[index])
        object_id = None
        if channel in self._get_object_id_channels(graph_id):
            object_id = int(self._get_feature_values(graph, "object_id")[node_id, 0])
        return channel, node_id, object_id

    def _propagate_from_observations(self, observations) -> None:
        """Reduce the salience of the parts of the MLH explained by this step's input.

        The input received on each channel of the MLH graph (an object ID from a
        lower-level LM, or sensory features) propagates through the nodes with
        similar features (see `_propagate`). Nothing propagates until the MLH has
        survived its first chance of being deleted (see `_mlh_survived_deletion`).
        """
        graph_id = self._get_mlh_graph_id()
        if graph_id is None:
            return
        if not self._mlh_survived_deletion(graph_id):
            logger.debug(
                f"MLH of {graph_id} has not yet survived its first chance of "
                "deletion; not propagating"
            )
            return

        mlh = self.parent_lm._get_current_mlh()
        channels = self.parent_lm.get_input_channels_in_graph(graph_id)
        for percept in observations:
            if percept.sender_id in channels:
                self._propagate(graph_id, percept.sender_id, percept, mlh)

    def _propagate(self, graph_id, input_channel, percept: Message, mlh) -> None:
        """Propagate from the MLH location through nodes with similar features.

        The features compared are those the channel's graph stores (see
        `_get_propagation_features`): the object ID on a channel receiving object
        IDs from a lower-level LM, and the hue and surface normal on a channel
        receiving sensory input. The propagation only begins if the received
        features are predicted by the MLH, i.e. they are similar to the features
        of one of the nodes nearest the MLH location that lie within the parent
        LM's max_match_distance (with a sensed surface normal rotated into the
        model's frame by the MLH rotation). It then begins from those of the nodes
        nearest the MLH location that are similar to the nearest such node, and
        continues through the nodes with features similar to their neighbors'
        (see `_get_similar_neighbors`). Nodes on the far side of a thin wall (see
        `_far_side_pairs`) are ignored throughout.
        """
        graph = self.parent_lm.get_graph(graph_id, input_channel=input_channel)
        features = self._get_propagation_features(graph, input_channel)
        sensed = {
            name: value
            for name, value in self._get_sensed_propagation_features(
                percept, mlh["rotation"]
            ).items()
            if name in features
        }
        if not sensed:
            return

        seeds, seed_distances = self._get_nearest_nodes(graph, mlh["location"])
        nearby = seeds[seed_distances <= self.parent_lm.max_match_distance]
        predicted = nearby[
            self._similar_features(input_channel, features, nearby, sensed)
        ]
        if len(predicted) == 0:
            return
        reference = {name: values[predicted[0]] for name, values in features.items()}
        seeds = self._seeds_like(input_channel, features, seeds, reference)

        propagation, surface = self._get_similar_neighbors(
            graph_id, input_channel, graph, features
        )
        self._propagate_from_seeds(
            graph_id,
            input_channel,
            seeds,
            propagation,
            surface,
            object_id=sensed.get("object_id"),
        )

    def _excite_target_region(
        self, graph_id, input_channel, node_id, object_id=None
    ) -> None:
        """Raise the salience of the part of the model a Goal's target is on.

        A hypothetical propagation, as if the target's location were sensed, with
        the features the target stores: it begins from those of the nodes nearest
        the target that are similar to it (as `_propagate` begins from those
        nearest the MLH location that are similar to the sensed input), and
        continues through the graph of the target's channel as a propagation of
        sensed input would, but raises the salience of the nodes it reaches to
        MAX_SALIENCE rather than reducing it. It is recorded in
        `goal_propagation_record`. Only the graph of the target's channel is
        propagated through.

        Args:
            graph_id: The graph (the MLH object) the target is a node of.
            input_channel: The input channel whose graph the target is a node of.
            node_id: The target node.
            object_id: The object ID the target stores, or None on a sensory
                channel.
        """
        graph = self.parent_lm.get_graph(graph_id, input_channel=input_channel)
        num_nodes = len(graph.pos)
        features = self._get_propagation_features(graph, input_channel)
        if not features:
            # Nothing to compare, so sensing the target would not propagate.
            self.model_salience.excite(graph_id, input_channel, num_nodes, [node_id])
            return
        reference = {name: values[node_id] for name, values in features.items()}
        seeds, _ = self._get_nearest_nodes(graph, graph.pos[node_id])
        seeds = self._seeds_like(input_channel, features, seeds, reference)
        if node_id not in seeds:
            seeds = np.concatenate([[node_id], seeds])

        propagation, surface = self._get_similar_neighbors(
            graph_id, input_channel, graph, features
        )
        self._propagate_from_seeds(
            graph_id,
            input_channel,
            seeds,
            propagation,
            surface,
            object_id=object_id,
            hypothetical=True,
        )

    def _seeds_like(self, input_channel, features, candidates, reference):
        """Get the candidate seeds with features similar to a reference node's.

        Args:
            input_channel: The input channel of the graph the candidates belong to.
            features: The compared features of every node (see
                `_get_propagation_features`).
            candidates: The nodes that may be seeds.
            reference: The features of the reference node.

        Returns:
            The candidates that are not on the far side of a thin wall from the
            reference (see `_far_side_pairs`), and whose features are similar to
            its features, in the order given.
        """
        return candidates[
            ~self._far_side(features, candidates, reference)
            & self._similar_features(input_channel, features, candidates, reference)
        ]

    def _propagate_from_seeds(
        self,
        graph_id,
        input_channel,
        seeds,
        propagation,
        surface,
        object_id,
        hypothetical=False,
    ) -> None:
        """Propagate from seeds through the graph, changing the reached salience.

        The propagation (see `_get_propagation_order`) is recorded, with the
        reached nodes in the order they were reached: for input that was sensed,
        it is appended to `propagation_records`, and the salience of each reached
        node is reduced (see `_salience_reductions`); for a hypothetical
        propagation from a Goal's target, it is kept in `goal_propagation_record`,
        and the salience of each reached node is raised to MAX_SALIENCE.

        Args:
            graph_id: The graph to propagate through.
            input_channel: The input channel whose graph to propagate through.
            seeds: The nodes to begin propagating from.
            propagation: The propagation matrix of the graph: a sparse
                (num_nodes x num_nodes) matrix with an edge at (node, neighbor)
                wherever the node propagates to the neighbor.
            surface: The surface matrix of the graph (see
                `_get_propagation_order`).
            object_id: The received object ID (or the one a Goal's target
                stores), or None for sensory features.
            hypothetical: Whether the propagation is from a Goal's target rather
                than sensed input.
        """
        order, hops, stage, predecessor, min_size = self._get_propagation_order(
            seeds, propagation, surface
        )
        num_nodes = propagation.shape[0]
        if hypothetical:
            reduction = np.zeros(len(order))
            self.model_salience.excite(graph_id, input_channel, num_nodes, order)
        else:
            reduction = self._salience_reductions(len(order))
            self.model_salience.reduce(
                graph_id, input_channel, num_nodes, order, reduction
            )
        record = PropagationRecord(
            graph_id=graph_id,
            input_channel=input_channel,
            object_id=object_id,
            node_order=order,
            seeds=np.asarray(seeds, dtype=int).reshape(-1),
            hops=hops,
            stage=stage,
            predecessor=predecessor,
            sinks=order[propagation.getnnz(axis=1)[order] == 0],
            min_size=min_size,
            reduction=reduction,
            hypothetical=hypothetical,
        )
        if hypothetical:
            self.goal_propagation_record = record
        else:
            self.propagation_records.append(record)
        nodes = (
            "with similar features"
            if object_id is None
            else f"storing object ID {object_id}"
        )
        change = "Raised the salience of" if hypothetical else "Reduced the salience of"
        logger.debug(
            f"{change} {len(order)} nodes {nodes} on channel {input_channel} of "
            f"{graph_id}"
        )

    def _salience_reductions(self, num_reached) -> np.ndarray:
        """Get the salience reduction of each node a propagation reached.

        The first node reached is reduced by salience_reduction, falling off
        linearly in the order the nodes were reached to salience_reduction * (1 -
        salience_reduction_falloff) at the last.

        Args:
            num_reached: The number of nodes the propagation reached.

        Returns:
            The reduction of each reached node, in the order they were reached.
        """
        scale = np.ones(num_reached)
        if num_reached > 1:
            scale -= (
                self.salience_reduction_falloff
                * np.arange(num_reached)
                / (num_reached - 1)
            )
        return self.salience_reduction * scale

    def _get_propagation_order(
        self, seeds, propagation, surface=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Propagate from seeds through a graph, without changing any salience.

        A breadth-first search from the seeds along the edges of the propagation
        matrix: every reached node reaches the nodes it has edges to, and the
        propagation then continues from each of them. Whenever the propagation
        stops having reached fewer than min_propagation_fraction of the graph's
        nodes (e.g. on a fragment of noisy features, or a small part like a mug's
        handle), it is pushed through: every unreached node that
        a reached node has an edge to in the surface matrix (i.e. on the same side
        of the surface, whether or not their features are similar) is reached, and
        the propagation continues from them, but only as many hops as it takes to
        reach min_propagation_fraction of the nodes (so e.g. a propagation pushed
        from the handle onto the side of a mug does not go on to cover the side).

        Args:
            seeds: The nodes to begin propagating from.
            propagation: The propagation matrix of the graph (see
                `_propagate_from_seeds`).
            surface: The surface matrix of the graph: a sparse (num_nodes x
                num_nodes) matrix with an edge at (node, neighbor) for every
                neighbor on the same side of the surface, or None to not push
                the propagation through.

        Returns:
            The reached nodes, in the order they were reached: by increasing
            number of hops from the seeds, then, after each push, from the nodes
            pushed through to (nodes the same number of hops away are in index
            order); for each of them, its number of hops from the nodes its stage
            began from, its stage (0 for the propagation from the seeds, n for
            the n-th push-through), and the node it was reached from (-1 for the
            nodes a stage began from); and the number of nodes the propagation
            had to reach (0 if it was never pushed through).
        """
        num_nodes = propagation.shape[0]
        sources = np.asarray(seeds, dtype=int).reshape(-1)
        empty = np.empty(0, dtype=int)
        if len(sources) == 0:
            return empty, empty, empty, empty, 0.0
        min_size = 0
        if surface is not None and self.min_propagation_fraction:
            min_size = self.min_propagation_fraction * num_nodes
        reached = np.zeros(num_nodes, dtype=bool)
        order, hops, stages, predecessors = [], [], [], []
        stage = 0
        while True:
            # The number of hops from the sources to each node the propagation reaches,
            # by a breadth-first search (inf for the nodes it does not reach), and
            # the node each is reached from (-9999 for the sources).
            stage_hops, predecessor, _ = dijkstra(
                propagation,
                directed=True,
                indices=sources,
                min_only=True,
                unweighted=True,
                return_predecessors=True,
            )
            stage_hops[reached] = np.inf
            new = np.flatnonzero(np.isfinite(stage_hops))
            new = new[np.argsort(stage_hops[new], kind="stable")]
            if stage > 0:
                # Only propagate as many hops as reaching min_size nodes takes,
                # i.e. the hops of the node that reaches it.
                needed = int(np.ceil(min_size)) - np.count_nonzero(reached)
                if needed < len(new):
                    new = new[stage_hops[new] <= stage_hops[new[needed - 1]]]
            reached[new] = True
            order.append(new)
            hops.append(stage_hops[new].astype(int))
            stages.append(np.full(len(new), stage))
            predecessors.append(np.maximum(predecessor[new], -1))
            if np.count_nonzero(reached) >= min_size:
                break
            # Push through: the unreached neighbors (on the same side of the
            # surface) of the reached nodes become the next stage's sources.
            sources = np.flatnonzero((surface.T @ reached.astype(float) > 0) & ~reached)
            if len(sources) == 0:
                break
            stage += 1
        return (
            np.concatenate(order),
            np.concatenate(hops),
            np.concatenate(stages),
            np.concatenate(predecessors),
            min_size if stage > 0 else 0.0,
        )

    def _get_nearest_nodes(self, graph, location) -> tuple[np.ndarray, np.ndarray]:
        """Get the num_propagation_neighbors nodes of a graph nearest a location.

        The location is in the model's frame, e.g. the MLH location, or a Goal's
        target.

        Returns:
            The nodes, nearest first, and their distances from the location.
        """
        num_seeds = min(self.num_propagation_neighbors, len(graph.pos))
        query = np.atleast_2d(np.asarray(location, dtype=float))
        seeds = np.asarray(
            graph.find_nearest_neighbors(query, num_neighbors=num_seeds)
        ).reshape(-1)
        distances = np.asarray(
            graph.find_nearest_neighbors(
                query, num_neighbors=num_seeds, return_distance=True
            )
        ).reshape(-1)
        return seeds, distances

    def _get_max_hue_difference(self, input_channel) -> float | None:
        """Get the largest hue difference between similar features on a channel.

        This is the parent LM's hue tolerance for the channel (the first entry of
        its "hsv" tolerances): the circular hue distance (in [0, 0.5]) at which
        the LM's feature evidence for hue falls to 0.

        Returns:
            The largest hue difference, or None if the LM has no hue tolerance
            for the channel (it does not compare hue there), in which case
            propagation does not compare hue either.
        """
        tolerances = getattr(self.parent_lm, "tolerances", None) or {}
        hsv = tolerances.get(input_channel, {}).get("hsv")
        if hsv is None:
            return None
        return float(np.asarray(hsv, dtype=float).reshape(-1)[0])

    def _get_max_normal_angle(self, input_channel) -> float | None:
        """Get the largest angle between similar surface normals on a channel.

        This is the parent LM's tolerance for the first pose vector (the surface
        normal) on the channel, which the LM stores in radians (configured in
        degrees, 30 by default; see `add_pose_features_to_tolerances`).

        Returns:
            The largest angle, in radians, or None if the LM has no pose vector
            tolerance for the channel, in which case propagation does not compare
            surface normals.
        """
        tolerances = getattr(self.parent_lm, "tolerances", None) or {}
        pose_vectors = tolerances.get(input_channel, {}).get("pose_vectors")
        if pose_vectors is None:
            return None
        return float(np.asarray(pose_vectors, dtype=float).reshape(-1)[0])

    def _get_propagation_features(self, graph, input_channel) -> dict[str, np.ndarray]:
        """Get the features of every node of a graph that propagation compares.

        On a channel receiving object IDs from a lower-level LM, only the object
        ID is compared: the pose vectors stored there are the orientation of the
        child object, not a surface normal. On any other channel, the hue and the
        surface normal are compared.

        Returns:
            Arrays of each compared feature the graph stores, with one row per node:
            "object_id", or "hsv" (hue, saturation and value) and "normal" (the
            unit surface normal, which also identifies the far side of thin walls,
            even if its similarity is not compared; see `_get_max_normal_angle`).
            Hue is omitted when not compared on the channel (see
            `_get_max_hue_difference`).
        """
        feature_mapping = graph.feature_mapping or {}
        if "object_id" in feature_mapping:
            return {"object_id": self._get_feature_values(graph, "object_id")[:, 0]}
        features = {}
        if (
            self._get_max_hue_difference(input_channel) is not None
            and "hsv" in feature_mapping
        ):
            features["hsv"] = self._get_feature_values(graph, "hsv")[:, :3]
        if "pose_vectors" in feature_mapping:
            normals = self._get_feature_values(graph, "pose_vectors")[:, :3]
            features["normal"] = normals / np.maximum(
                np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
            )
        return features

    def _get_sensed_propagation_features(self, percept: Message, rotation) -> dict:
        """Get the received features that propagation compares.

        Args:
            percept: The input: sensory features, or an object ID from a
                lower-level LM.
            rotation: The MLH rotation, which rotates the sensed (body-frame)
                surface normal into the model's frame.

        Returns:
            The received features, in the format of `_get_propagation_features`
            (with one value per feature, and the normal in the model's frame), for
            those the percept carries. Only the features also present in the
            stored ones are compared (e.g. not the pose of a child object).
        """
        non_morphological = percept.non_morphological_features or {}
        morphological = percept.morphological_features or {}
        sensed = {}
        if non_morphological.get("object_id") is not None:
            sensed["object_id"] = int(
                np.asarray(non_morphological["object_id"]).reshape(-1)[0]
            )
        if non_morphological.get("hsv") is not None:
            sensed["hsv"] = np.asarray(non_morphological["hsv"], dtype=float)[:3]
        if morphological.get("pose_vectors") is not None:
            normal = rotation.apply(
                np.asarray(morphological["pose_vectors"], dtype=float).reshape(3, 3)[0]
            )
            sensed["normal"] = normal / np.linalg.norm(normal)
        return sensed

    def _similar_features(
        self, input_channel, features, nodes, reference
    ) -> np.ndarray:
        """Check which of a set of nodes have features similar to a reference.

        Only the features present in both `features` and `reference` are compared.

        Args:
            input_channel: The input channel of the graph the nodes belong to.
            features: The compared features of every node (see
                `_get_propagation_features`).
            nodes: The nodes to check.
            reference: The features to compare them to, in the same format (one
                value per feature).

        Returns:
            Whether each node's features are similar to the reference.
        """
        return self._similar_feature_pairs(
            input_channel,
            {name: values[nodes] for name, values in features.items()},
            {
                name: np.broadcast_to(value, (len(nodes), *np.shape(value)))
                for name, value in reference.items()
            },
        )

    def _similar_feature_pairs(
        self, input_channel, features_a, features_b
    ) -> np.ndarray:
        """Check which pairs of features are similar.

        Object IDs are similar only if they are the same. Hues are similar if they
        differ by at most the parent LM's hue tolerance for the channel (see
        `_get_max_hue_difference`), and surface normals if they differ by at most
        its surface normal tolerance (see `_get_max_normal_angle`).

        Args:
            input_channel: The input channel the features were received on.
            features_a: Arrays of features, with one row per pair.
            features_b: Arrays of the features to compare them to, likewise.

        Returns:
            Whether the features of each pair are similar, comparing only the
            features present in both.
        """
        num_pairs = len(next(iter(features_a.values()), []))
        similar = np.ones(num_pairs, dtype=bool)
        if "object_id" in features_a and "object_id" in features_b:
            similar &= features_a["object_id"] == features_b["object_id"]
        max_hue_difference = self._get_max_hue_difference(input_channel)
        if (
            max_hue_difference is not None
            and "hsv" in features_a
            and "hsv" in features_b
        ):
            hsv_a, hsv_b = features_a["hsv"], features_b["hsv"]
            hue_difference = np.abs(hsv_a[:, 0] - hsv_b[:, 0])
            hue_difference = np.minimum(hue_difference, 1.0 - hue_difference)
            similar &= ~(self._chromatic(hsv_a) & self._chromatic(hsv_b)) | (
                hue_difference <= max_hue_difference
            )
        max_normal_angle = self._get_max_normal_angle(input_channel)
        if (
            max_normal_angle is not None
            and "normal" in features_a
            and "normal" in features_b
        ):
            cosine = np.sum(features_a["normal"] * features_b["normal"], axis=1)
            similar &= cosine >= np.cos(max_normal_angle)
        return similar

    @staticmethod
    def _chromatic(hsv, min_saturation=0.1, min_value=0.1) -> np.ndarray:
        """Check which colors have a meaningful hue.

        Hue is only meaningful for a color with enough chroma: a difference
        between its brightest and darkest RGB channels that is large compared to
        sensor noise. In HSV, chroma is saturation * value, where value is the
        brightest channel and saturation is the difference relative to it, so
        both are needed. Saturation alone does not do: being a ratio, for a dark
        color it is the ratio of two small, noisy numbers, so a near-black color
        can have any saturation (e.g. RGB (0.02, 0.01, 0.0) has saturation 1) and
        an essentially random hue. The minimum value rules out such dark colors,
        and the minimum saturation rules out greys of any brightness, including
        near-white: unlike lightness in HSL, a value near 1 does not mean white
        (e.g. pure red has value 1 and a well-defined hue); white has a value near
        1 and a saturation near 0, so it is excluded by the minimum saturation, as
        are washed-out specular highlights.

        Args:
            hsv: Colors, with one row of (hue, saturation, value) per color.
            min_saturation: Minimum saturation for a hue to be meaningful.
            min_value: Minimum value for a hue to be meaningful.

        Returns:
            Whether each color's hue is meaningful.
        """
        return (hsv[:, 1] >= min_saturation) & (hsv[:, 2] >= min_value)

    @staticmethod
    def _far_side_pairs(normals_a, normals_b, min_far_side_angle=135.0) -> np.ndarray:
        """Check which pairs of nodes are on opposite sides of a thin wall.

        Two neighboring nodes whose surface normals face away from each other by
        at least min_far_side_angle (in degrees) are on the far side of a thin
        wall from each other (e.g. the inside and outside of a mug), so the one
        neither stops a propagation from the other nor is propagated to.

        Args:
            normals_a: Unit surface normals, with one row per pair.
            normals_b: The unit surface normals to compare them to, likewise.
            min_far_side_angle: Smallest angle between the normals of two nodes
                on opposite sides of a wall.

        Returns:
            Whether the nodes of each pair are on opposite sides of a wall.
        """
        cosine = np.sum(normals_a * normals_b, axis=-1)
        return cosine <= np.cos(np.radians(min_far_side_angle))

    def _far_side(self, features, nodes, reference) -> np.ndarray:
        """Check which of a set of nodes are on the far side of a thin wall.

        Args:
            features: The compared features of every node (see
                `_get_propagation_features`).
            nodes: The nodes to check.
            reference: The features of the node on the near side.

        Returns:
            Whether each node is on the far side of a wall from the reference
            (see `_far_side_pairs`; never, if normals are not compared).
        """
        if "normal" not in features or "normal" not in reference:
            return np.zeros(len(nodes), dtype=bool)
        return self._far_side_pairs(features["normal"][nodes], reference["normal"])

    def _get_similar_neighbors(
        self, graph_id, input_channel, graph, features
    ) -> tuple[csr_matrix, csr_matrix]:
        """Get the edges of a graph for propagating through similar features.

        Neighbors on the far side of a thin wall from a node (see
        `_far_side_pairs`) are not its neighbors here, as they are not part of the
        same surface. A node propagates to the neighbors with features similar to its
        own (e.g. storing the same object ID) if at most max_dissimilar_neighbors
        of its neighbors are not.

        Returns:
            The propagation matrix (see `_propagate_from_seeds`), with an edge from
            each node to the neighbors it propagates to, and the surface matrix
            (see `_get_propagation_order`), with an edge from each node to every
            neighbor on the same side of the surface, whether or not their
            features are similar.
        """
        key = (graph_id, input_channel, len(graph.pos))
        if key not in self._similar_neighbors_cache:
            num_nodes = len(graph.pos)
            source, target = self._get_neighbor_matrix(
                graph_id, input_channel, graph
            ).nonzero()
            if "normal" in features:
                near_side = ~self._far_side_pairs(
                    features["normal"][source], features["normal"][target]
                )
                source, target = source[near_side], target[near_side]
            surface = self._edge_matrix(source, target, num_nodes)
            similar = self._similar_feature_pairs(
                input_channel,
                {name: values[source] for name, values in features.items()},
                {name: values[target] for name, values in features.items()},
            )
            num_dissimilar = np.bincount(source[~similar], minlength=num_nodes)
            propagates = (surface.getnnz(axis=1) > 0) & (
                num_dissimilar <= self.max_dissimilar_neighbors
            )
            keep = similar & propagates[source]
            propagation = self._edge_matrix(source[keep], target[keep], num_nodes)
            self._similar_neighbors_cache[key] = (propagation, surface)
        return self._similar_neighbors_cache[key]

    @staticmethod
    def _edge_matrix(source, target, num_nodes) -> csr_matrix:
        """Build the sparse adjacency matrix of a set of directed edges.

        Args:
            source: The node each edge starts from.
            target: The node each edge leads to.
            num_nodes: The number of nodes in the graph.

        Returns:
            Sparse (num_nodes x num_nodes) matrix with a 1 at (source, target) for
            every edge.
        """
        return csr_matrix(
            (np.ones(len(source)), (source, target)), shape=(num_nodes, num_nodes)
        )

    def _get_neighbor_matrix(self, graph_id, input_channel, graph) -> csr_matrix:
        """Get the two-way nearest-neighbor relations of every node of a graph.

        A node's neighbors are its num_propagation_neighbors nearest nodes,
        together with every node that has it among their own
        num_propagation_neighbors nearest nodes. Nearest-neighbor relations are
        not mutual, so without the latter, an outlying node (whose nearest nodes
        all have closer neighbors of their own) could never be reached.

        Args:
            graph_id: The graph whose nodes to get the neighbors of.
            input_channel: The input channel of the graph.
            graph: The graph.

        Returns:
            Sparse (num_nodes x num_nodes) matrix with a 1 at (node, neighbor) for
            every neighbor of every node (excluding the node itself).
        """
        pos = np.asarray(graph.pos, dtype=float)
        num_nodes = len(pos)
        key = (graph_id, input_channel, num_nodes)
        if key not in self._neighbor_cache:
            k = min(self.num_propagation_neighbors, num_nodes - 1)
            if k < 1:
                neighbors = csr_matrix((num_nodes, num_nodes), dtype=float)
            else:
                # Nodes are excluded from their own nearest neighbors by index, so
                # that nodes sharing a location are still each other's neighbors.
                nearest = kneighbors_graph(pos, n_neighbors=k, include_self=False)
                neighbors = ((nearest + nearest.T) > 0).astype(float).tocsr()
                neighbors.sort_indices()
            self._neighbor_cache[key] = neighbors
        return self._neighbor_cache[key]

    def _check_need_new_output_goal(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        output_goal_achieved,
    ) -> bool:
        """Determine whether the GSG should generate a new output Goal.

        Success in achieving the Goal is not an indication to need a new one, as
        the sensor should explore the tested location for a while. No Goal is
        generated until the MLH has survived its first chance of being deleted
        (see `_mlh_survived_deletion`), as a newly sampled hypothesis is not yet
        credible enough to be tested. After that, a new Goal is generated if the MLH
        (object or rotation) has changed since the last Goal, or every
        elapsed_steps_trigger steps.

        Returns:
            Whether the GSG should generate a new output Goal.
        """
        if output_goal_achieved:
            return False

        graph_id = self._get_mlh_graph_id()
        if graph_id is None:
            return False

        if not self._mlh_survived_deletion(graph_id):
            logger.debug(
                f"MLH of {graph_id} has not yet survived its first chance of "
                "deletion; no goal"
            )
            return False

        if self._mlh_changed_since_last_goal():
            logger.debug("Child objects goal indicated: MLH changed")
            return True

        num_elapsed_steps = self._get_num_steps_post_output_goal_generated()
        if num_elapsed_steps % self.elapsed_steps_trigger == 0:
            logger.debug("Child objects goal indicated: sufficient steps elapsed")
            return True

        return False

    def _mlh_changed_since_last_goal(self) -> bool:
        """Check whether the MLH object or rotation changed since the last Goal.

        Returns:
            Whether the MLH changed; True if no Goal has been generated yet.
        """
        if self._prev_goal_mlh is None:
            return True
        mlh = self.parent_lm._get_current_mlh()
        if mlh["graph_id"] != self._prev_goal_mlh["graph_id"]:
            return True
        return not np.allclose(
            mlh["rotation"].as_matrix(), self._prev_goal_mlh["rotation"].as_matrix()
        )

    # ------------------ Getters & Setters ---------------------

    def _get_mlh_graph_id(self) -> str | None:
        """Get the graph ID of the MLH, if it is a known object.

        Returns:
            The graph ID, or None before the MLH has been determined.
        """
        graph_id = self.parent_lm._get_current_mlh()["graph_id"]
        if graph_id not in self.parent_lm.get_all_known_object_ids():
            return None
        return graph_id

    def _mlh_survived_deletion(self, graph_id) -> bool:
        """Whether the MLH hypothesis has survived its first chance of deletion.

        The parent LM's hypotheses updater (e.g. `BurstSamplingHypothesesUpdater`)
        tracks the age of each hypothesis of each object in an
        `EvidenceSlopeTracker`, in the order of the LM's hypotheses (which the
        MLH's mlh_id indexes). A hypothesis becomes eligible for deletion once its
        age reaches the tracker's `min_age`; it is considered for deletion at the
        start of the next update, which ages it to `min_age + 1` if it survives.
        The parent LM updates its hypotheses before stepping the GSG, so the age
        read here is the one after this step's update.

        Returns:
            Whether the MLH's age exceeds `min_age`; True if the hypotheses
            updater does not track ages (it never deletes hypotheses), and False
            if it does not track the MLH's.
        """
        updater = getattr(self.parent_lm, "hypotheses_updater", None)
        trackers = getattr(updater, "evidence_slope_trackers", None)
        if trackers is None:
            logger.debug("No hypothesis trackers configured for propagation to check.")
            return True
        tracker = trackers.get(graph_id)
        mlh_id = self.parent_lm._get_current_mlh()["mlh_id"]
        if tracker is None or mlh_id is None or mlh_id >= tracker.total_size():
            return False
        return bool(tracker.hyp_ages()[mlh_id] > tracker.min_age)

    def _get_object_id_channels(self, graph_id) -> list[str]:
        """Get the input channels of a graph that store "object_id" features.

        Returns:
            The input channels of the graph that receive object IDs from other LMs.
        """
        channels = []
        for channel in self.parent_lm.get_input_channels_in_graph(graph_id):
            graph = self.parent_lm.get_graph(graph_id, input_channel=channel)
            if graph.feature_mapping and "object_id" in graph.feature_mapping:
                channels.append(channel)
        return channels

    # ------------- Goals for a target location on the MLH -------------

    def _get_target_loc_info(self, target_loc_id, input_channel):
        """Given a target location ID and channel, get the location and pose vectors.

        Note:
            Currently assumes we are computing with the MLH graph.

        Args:
            target_loc_id: Index of the target node in the graph of the given
                input channel.
            input_channel: The input channel whose graph the target node belongs to.

        Returns:
            A dictionary containing the hypothesis to test, the target location and
            surface normal of the target point on the object.
        """
        mlh = self.parent_lm._get_current_mlh()
        mlh_id = mlh["graph_id"]

        target_object = self.parent_lm.get_graph(mlh_id)
        sensor_channel_name = self.parent_lm.buffer.get_first_sensory_input_channel()
        target_graph = target_object[input_channel]
        target_loc = target_graph.pos[target_loc_id]

        if input_channel == sensor_channel_name:
            surface_normal_mapping = target_graph.feature_mapping["pose_vectors"]
            target_surface_normal = target_graph.x[
                target_loc_id,
                surface_normal_mapping[0] : surface_normal_mapping[0] + 3,
            ]
        else:
            # The pose vectors stored on nodes of an LM input channel describe the
            # child object's pose, not the surface of the parent object, so we
            # retrieve the surface normal from the nearest node in the sensory
            # channel's graph instead
            sensor_graph = target_object[sensor_channel_name]
            nearest_sensor_node = sensor_graph.find_nearest_neighbors(
                np.atleast_2d(np.asarray(target_loc)),
                num_neighbors=1,
                return_distance=False,
            )[0]
            surface_normal_mapping = sensor_graph.feature_mapping["pose_vectors"]
            target_surface_normal = sensor_graph.x[
                nearest_sensor_node,
                surface_normal_mapping[0] : surface_normal_mapping[0] + 3,
            ]

        return {
            "hypothesis_to_test": mlh,
            "target_loc": target_loc,
            "target_surface_normal": target_surface_normal,
            "input_channel": input_channel,
        }

    def _compute_goal_for_target_loc(
        self, observations, target_info, goal_confidence=1.0
    ) -> Goal:
        """Specify a Goal for the motor-actuator.

        Based on a target location (in object-centric coordinates) and the associated
        surface normal of that location, specify a Goal for the motor-actuator: the
        target location on the object's surface, and the surface normal there, both
        in body-centric coordinates. It is up to the motor system to position the
        sensors associated with the motor-actuator so that they observe the target
        location, e.g. by looking down the surface normal from some distance.

        For the movement to have a high probability of arriving at the desired location,
        the current hypothesis of the object ID and pose used to inform the movement
        should be correct, although subsequent observations may still provide useful
        information to the agent, i.e. even if we are wrong about the object ID and
        pose.

        Args:
            observations: The current observations, which should include the sensory
                input.
            target_info: A dictionary containing the target location and surface normal
                of the target point on the object.
            goal_confidence: The confidence of the Goal, which should be in the
                range [0, 1]. This is used by receiving modules to weigh the
                importance of the Goal relative to other Goals.

        Returns:
            A Goal for the motor-actuator.
        """
        # Determine the displacement, and therefore the environmental target location,
        # that we will use
        sensor_channel_name = self.parent_lm.buffer.get_first_sensory_input_channel()
        sensory_input = get_percept_from_channel(
            percepts=observations, channel_name=sensor_channel_name
        )
        displacement = (
            target_info["target_loc"] - target_info["hypothesis_to_test"]["location"]
        )

        object_rot = target_info["hypothesis_to_test"]["rotation"].inv()  # MLH rotation
        # is stored as the rotation needed to convert a displacement to the object pose,
        # so the *object pose* is given by its inverse

        # Rotate the displacement; note we're converting from an *internal* object frame
        # of reference, to the global frame of reference; thus, we rotate not by the
        # inverse, but by the actual object orientation.
        rotated_disp = object_rot.apply(displacement)

        # The target location on the object's surface in global/body-centric coordinates
        target_loc = sensory_input.location + rotated_disp

        # Rotate the learned surface normal (which was committed to memory assuming a
        # default 0,0,0 orientation of the object)
        target_surface_normal_rotated = object_rot.apply(
            target_info["target_surface_normal"]
        )

        # Extra metadata for logging. 'achieved' and
        # 'matching_step_when_output_goal_set' should be updated at the next step.
        # We initialize them as `None` to indicate that no valid values have been set.
        # The model-frame target location and graph id are snapshotted here because
        # they cannot be reconstructed later (the sensor location used to derive the
        # world-frame goal is not stored) and `hypothesis_to_test` is a live dict
        # whose graph_id may change after the goal is created; visualizers use them
        # to mark the goal on the hypothesized object model. The input channel whose
        # graph the target was selected from (None when the target was not a node of
        # a stored graph) lets them draw the goal in that channel's model.
        # The hypothesis identity ('hypothesis_to_test_graph_id' / '..._mlh_id') is
        # snapshotted for the same reason: if the motor system attempts this goal,
        # the parent LM uses these to decrement the evidence of the hypothesis that
        # proposed the jump when the jump is judged to have failed.
        # 'predicted_displacement' is the sensory displacement the parent LM should
        # experience if the jump succeeds; the LM compares it against the actually
        # sensed displacement to judge success.
        info = {
            "model_frame_target_loc": np.array(target_info["target_loc"]),
            "model_frame_graph_id": target_info["hypothesis_to_test"]["graph_id"],
            "model_frame_input_channel": target_info.get("input_channel"),
            "hypothesis_to_test": target_info["hypothesis_to_test"],
            "hypothesis_to_test_graph_id": target_info["hypothesis_to_test"][
                "graph_id"
            ],
            "hypothesis_to_test_mlh_id": target_info["hypothesis_to_test"]["mlh_id"],
            "predicted_displacement": np.array(rotated_disp),
            "achieved": None,
            "matching_step_when_output_goal_set": None,
        }

        return Goal(
            location=np.array(target_loc),
            morphological_features={
                # Only the surface normal is specified; the curvature directions are
                # not relevant to observing the target location
                "pose_vectors": np.array(
                    [
                        target_surface_normal_rotated,
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                "pose_fully_defined": None,
                "on_object": 1,
            },
            non_morphological_features=None,
            confidence=goal_confidence,
            pass_message=True,
            sender_id=self.parent_lm.learning_module_id,
            sender_type="GSG",
            process_features_in_lm=True,
            goal_tolerances=None,
            info=info,
        )

    def _check_keep_current_output_goal(self) -> bool:
        """Determine whether the GSG should keep the current Goal.

        Jumps to a target location should be executed as one-off attempts, lest we get
        stuck in a loop of trying to achieve the same goal that is impossible (e.g.
        due to collision with objects).

        Returns:
            Whether the GSG should keep the current Goal. Always returns False.
        """
        return False

    def _get_num_steps_post_output_goal_generated(self):
        """Number of steps since last output Goal.

        Returns:
            The number of Monty-matching steps that have elapsed since the last time
            an output Goal was generated.
        """
        return self.parent_lm.buffer.get_num_steps_post_output_goal_generated()

    @staticmethod
    def _get_feature_values(graph, feature) -> np.ndarray:
        """Get the values of a feature for all nodes of a graph.

        Returns:
            Array of shape (num_nodes, feature_dim) of the feature's values.
        """
        feature_idx = graph.feature_mapping[feature]
        return np.asarray(graph.x[:, feature_idx[0] : feature_idx[1]])
