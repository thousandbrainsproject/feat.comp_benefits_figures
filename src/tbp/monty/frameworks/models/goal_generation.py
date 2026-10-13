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
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    GoalGenerator,
)
from tbp.monty.frameworks.utils.communication_utils import get_percept_from_channel

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.graph_matching import GraphLM

__all__ = [
    "EvidenceGoalGenerator",
    "GraphGoalGenerator",
    "ParentLMNotProvided",
    "SpreadRecord",
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
class SpreadRecord:
    """One spread of inhibition by an `EvidenceGoalGenerator`.

    Attributes:
        graph_id: The graph (the MLH object) the inhibition spread through.
        input_channel: The input channel whose graph the inhibition spread through,
            i.e. the channel the input was received on.
        object_id: The "object_id" feature value that was received, or None for a
            spread of similar features from sensory input.
        node_order: The inhibited nodes, in the order the spread reached them.
    """

    graph_id: str
    input_channel: str
    object_id: int | None
    node_order: np.ndarray


class EvidenceGoalGenerator(GraphGoalGenerator):
    """Generator of Goals for an evidence-based graph LM.

    GSG specifically set up for generating Goals for an evidence-based graph LM,
    which can therefore leverage the compositional inference policy. This policy
    uses the most likely object hypothesis (MLH), as well as knowledge of its
    structure from recent memory, to propose test-points that visit the
    unexplained parts of the object the agent is currently observing.

    Goals are directed at the parts of the model of the most likely
    object hypothesis (MLH) that the LM's input has not yet explained. Within the
    MLH graph, each channel storing "object_id" features (input from lower-level
    LMs, i.e. a compositional model) contributes one candidate per child object (the
    nodes storing that object ID), and every other (sensory) channel contributes
    one candidate made up of all of its nodes. Candidates are ranked by their
    number of uninhibited nodes, and a random uninhibited node of the highest-ranked
    candidate with any uninhibited node is selected as the location to test.

    Input that the MLH predicts at its current location "explains" part of the
    model, which is then inhibited from being tested for a while, by spreading
    through the graph of the channel the input was received on, starting from the
    nodes nearest the MLH location:
    - An object ID from a lower-level LM spreads through the nodes storing it.
    - Sensory input spreads through neighboring nodes with a similar hue and
      surface normal. Spreading stops wherever the hue or the surface normal
      changes abruptly, so e.g. on a mug the side of the cylinder (whose normal
      changes gradually from node to node) is inhibited, but not its rim, bottom
      or handle, and on a cube a single face is inhibited, but not the other
      faces. A spread that stops
      before reaching `min_spread_fraction` of the graph's nodes is pushed
      through to the neighbors whose features differ.

    A node's neighbors are its `num_spread_neighbors` (or, for sensory input,
    `num_feature_spread_neighbors`) nearest nodes (regardless of their distance),
    together with the nodes that have it among their own nearest
    nodes, and it spreads to them only if all of them store the received object ID
    or, for sensory input, are similar to it (allowing for
    `max_dissimilar_neighbors` noisy neighbors, which it does not spread to).
    Spreading therefore covers a contiguous region, including outlying nodes,
    without jumping to disjoint regions (e.g. separate wheels on a car).
    Inhibition is tracked separately for the graph of each channel, so a location
    inhibited in one channel's graph can still be tested in another's. Inhibited
    nodes are not selected for testing until their inhibition has linearly decayed
    to 0 over `inhibition_decay_steps` steps.
    """

    def __init__(
        self,
        goal_tolerances=None,
        desired_object_distance=0.03,
        elapsed_steps_factor=10,
        min_post_goal_success_steps=np.inf,
        num_spread_neighbors=6,
        num_feature_spread_neighbors=10,
        inhibition_decay_steps=50,
        max_hue_difference: float | None = 0.05,
        max_normal_angle: float | None = 40.0,
        min_far_side_angle=135.0,
        max_dissimilar_neighbors=2,
        min_spread_fraction: float | None = 0.05,
        min_hue_saturation=0.1,
        min_hue_value=0.1,
        **kwargs,
    ) -> None:
        """Initialize the GSG.

        Args:
            goal_tolerances: The tolerances for each attribute of the Goal that can be
                used by the GSG when determining whether a Goal is achieved.
            desired_object_distance: The desired distance between the agent and the
                object, which is used to determine whether the agent is close enough to
                the object to consider it "achieved". Note this need not be the same as
                the one specified for the motor-system (e.g. the surface-policy), as we
                may want to aim for an initially farther distance, while the
                surface-policy may want to stay quite close to the object. Defaults to
                0.03.
            elapsed_steps_factor: Once min_post_goal_success_steps have elapsed, a
                Goal is generated every elapsed_steps_factor steps, even if the MLH
                has not changed. Defaults to 10.
            min_post_goal_success_steps: Number of steps that must elapse since the
                last Goal was generated before a new one is considered. Infinity by
                default, resulting in no Goals being generated.
            num_spread_neighbors: Number of nearest neighbors of each node that,
                together with the nodes that have it among their own nearest
                neighbors, must all store the received object ID for inhibition
                to spread from the node. Also the number of nodes nearest the MLH
                location that a spread begins from. Defaults to 6.
            num_feature_spread_neighbors: Like num_spread_neighbors, but for
                spreading sensory input through similar features. Larger, as on a
                wall thinner than the spacing of nodes (e.g. a bowl's), about half
                of a node's nearest neighbors are on the far side of the wall (see
                min_far_side_angle), and so are ignored. Defaults to 10.
            inhibition_decay_steps: Number of steps over which the inhibition of a
                node linearly decays from 1 to 0. Defaults to 50.
            max_hue_difference: Largest circular distance in hue (in [0, 0.5]) for
                two features to be similar. Hue is only compared between two
                chromatic features (see min_hue_saturation and min_hue_value), as
                it is undefined for achromatic ones; shading can make parts of a
                dark, colored surface appear achromatic, so a chromatic and an
                achromatic feature are not treated as different. None to not
                compare hue. Defaults to 0.05.
            max_normal_angle: Largest angle (in degrees) between the surface
                normals of two neighboring nodes for their features to be similar.
                None to not compare normals. Defaults to 40.
            min_far_side_angle: Smallest angle (in degrees) between the surface
                normals of two neighboring nodes for one to be on the far side of
                a thin wall from the other (e.g. the inside and outside of a mug),
                in which case it neither stops spreading nor is spread to.
                Defaults to 135.
            max_dissimilar_neighbors: For sensory input, the number of a node's
                neighbors whose features may differ from its own, without stopping
                it from spreading to the others (which have similar features). The
                features stored in learned models are noisy, so with 0 a single
                noisy neighbor stops a node from spreading; more lets spreading
                leak across boundaries (e.g. onto a mug's handle). Spreading
                through object IDs always requires all neighbors to store the ID.
                Defaults to 2.
            min_spread_fraction: For sensory input, the fraction of the graph's
                nodes a spread must reach: when spreading stops before reaching
                them, it is pushed through to the neighbors whose features differ
                (see `_get_spread_order`). This keeps fragments of noisy features,
                and small parts like a mug's handle, from being inhibited a few
                nodes at a time. None to not push spreading through. Defaults to
                0.05.
            min_hue_saturation: Minimum HSV saturation for a hue to be chromatic.
                Defaults to 0.1.
            min_hue_value: Minimum HSV value for a hue to be chromatic. Defaults to
                0.1.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(goal_tolerances, **kwargs)
        self.desired_object_distance = desired_object_distance
        self.elapsed_steps_factor = elapsed_steps_factor
        self.min_post_goal_success_steps = min_post_goal_success_steps
        self.num_spread_neighbors = num_spread_neighbors
        self.num_feature_spread_neighbors = num_feature_spread_neighbors
        self.inhibition_decay_steps = inhibition_decay_steps
        self.max_hue_difference = max_hue_difference
        self.max_normal_angle = max_normal_angle
        self.min_far_side_angle = min_far_side_angle
        self.max_dissimilar_neighbors = max_dissimilar_neighbors
        self.min_spread_fraction = min_spread_fraction
        self.min_hue_saturation = min_hue_saturation
        self.min_hue_value = min_hue_value

    # ======================= Public ==========================

    def reset(self):
        """Reset the inhibition of all nodes, and the MLH at the last Goal."""
        super().reset()
        # Number of steps of inhibition remaining for each node, keyed by
        # (graph_id, input_channel).
        self._inhibition_steps: dict[tuple[str, str], np.ndarray] = {}
        # Keyed by (graph_id, input_channel, number of nodes, number of neighbors).
        self._neighbor_cache: dict[tuple[str, str, int, int], list[np.ndarray]] = {}
        # For sensory spreading (see `_get_similar_neighbors`), keyed by
        # (graph_id, input_channel, number of nodes).
        self._similar_neighbors_cache: dict[
            tuple[str, str, int],
            tuple[np.ndarray, list[np.ndarray], tuple[np.ndarray, np.ndarray]],
        ] = {}
        self._prev_goal_mlh: dict | None = None
        self.spread_records: list[SpreadRecord] = []

    def step(self, ctx: RuntimeContext, observations):
        """Update the inhibition of explained parts of the MLH, then step the GSG.

        A new `spread_records` list is created on every step, holding the spreads
        that took place on that step.
        """
        self.spread_records = []
        self._decay_inhibition()
        self._spread_from_observations(observations)
        super().step(ctx, observations)

    def get_inhibition_weights(self, graph_id, input_channel) -> np.ndarray:
        """Get the inhibition weight of each node of a graph.

        Returns:
            Array with one weight in [0, 1] per node of the graph of the given input
            channel; nodes with a weight of 0 can be selected for testing.
        """
        num_nodes = len(self.parent_lm.get_graph(graph_id, input_channel).pos)
        steps = self._get_inhibition_steps(graph_id, input_channel, num_nodes)
        return steps / self.inhibition_decay_steps

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _generate_goal(self, ctx: RuntimeContext, observations) -> Goal | None:
        """Generate a Goal that moves the sensor to an unexplained part of the MLH.

        Returns:
            A Goal for the motor system, or a None-type Goal if every node of the
            MLH object's graphs is inhibited.
        """
        graph_id = self._get_mlh_graph_id()
        if graph_id is None:
            return self._generate_none_goal()

        mlh = self.parent_lm._get_current_mlh()
        self._prev_goal_mlh = {
            "graph_id": mlh["graph_id"],
            "rotation": mlh["rotation"],
        }

        target = self._select_target(ctx, graph_id)
        if target is None:
            logger.debug(f"All nodes of {graph_id} are inhibited; no goal")
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

    def _select_target(self, ctx: RuntimeContext, graph_id):
        """Select a random uninhibited node of the largest candidate of the graph.

        Each object ID stored on an object-ID channel of the graph is one candidate
        (the nodes storing it), and every other channel is one candidate (all of its
        nodes). Candidates are ranked by their number of nodes; the highest-ranked
        one with any uninhibited node is selected.

        Returns:
            A tuple of (input_channel, node_id, object_id), where object_id is None
            for a sensory channel, or None if every node is inhibited.
        """
        candidates = []
        candidate_nodes = {}
        object_id_channels = self._get_object_id_channels(graph_id)
        for channel in self.parent_lm.get_input_channels_in_graph(graph_id):
            graph = self.parent_lm.get_graph(graph_id, input_channel=channel)
            num_nodes = len(graph.pos)
            if channel in object_id_channels:
                object_ids = self._get_feature_values(graph, "object_id")[:, 0]
                for object_id in np.unique(object_ids):
                    nodes = object_ids == object_id
                    candidate_nodes[(channel, int(object_id))] = nodes
                    candidates.append(
                        (np.count_nonzero(nodes), channel, int(object_id))
                    )
            else:
                candidate_nodes[(channel, None)] = np.ones(num_nodes, dtype=bool)
                candidates.append((num_nodes, channel, None))

        candidates.sort(key=lambda candidate: -candidate[0])

        for _, channel, object_id in candidates:
            nodes = candidate_nodes[(channel, object_id)]
            inhibition = self._get_inhibition_steps(graph_id, channel, len(nodes))
            eligible = np.nonzero(nodes & (inhibition == 0))[0]
            if len(eligible) > 0:
                return channel, int(ctx.rng.choice(eligible)), object_id

        return None

    def _spread_from_observations(self, observations) -> None:
        """Inhibit the parts of the MLH graph explained by this step's input.

        Object IDs received from lower-level LMs on an object-ID channel of the MLH
        graph spread through the nodes storing them, and sensory input received on
        any other channel of the MLH graph spreads through nodes with similar
        features.
        """
        graph_id = self._get_mlh_graph_id()
        if graph_id is None:
            return

        mlh = self.parent_lm._get_current_mlh()
        channels = self.parent_lm.get_input_channels_in_graph(graph_id)
        object_id_channels = self._get_object_id_channels(graph_id)
        for percept in observations:
            channel = percept.sender_id
            if channel not in channels:
                continue
            if channel in object_id_channels:
                object_id = (percept.non_morphological_features or {}).get("object_id")
                if percept.sender_type != "LM" or object_id is None:
                    continue
                self._spread_object_id(
                    graph_id,
                    channel,
                    int(np.asarray(object_id).flatten()[0]),
                    mlh["location"],
                )
            elif percept.sender_type == "SM":
                self._spread_similar_features(graph_id, channel, percept, mlh)

    def _spread_object_id(
        self, graph_id, input_channel, received_id, mlh_location
    ) -> None:
        """Spread inhibition from the MLH location through nodes storing an ID.

        Spreading only begins if the received ID is predicted by the MLH, i.e. it
        is stored by one of the nodes nearest the MLH location that lie within the
        parent LM's max_match_distance, and all of the nodes nearest the MLH
        location store it.
        """
        graph = self.parent_lm.get_graph(graph_id, input_channel=input_channel)
        stores_id = self._get_feature_values(graph, "object_id")[:, 0] == received_id

        seeds, seed_distances = self._get_nodes_nearest_mlh(graph, mlh_location)
        nearby = seed_distances <= self.parent_lm.max_match_distance
        if not np.any(stores_id[seeds[nearby]]) or not np.all(stores_id[seeds]):
            return

        neighbors = self._get_node_neighbors(graph_id, input_channel, graph)
        source, target = self._get_neighbor_edges(neighbors)
        spreads = self._all_neighbors_satisfy(stores_id[target], source, len(stores_id))
        self._spread_from_seeds(
            graph_id, input_channel, seeds, neighbors, spreads, object_id=received_id
        )

    def _spread_similar_features(
        self, graph_id, input_channel, percept: Message, mlh
    ) -> None:
        """Spread inhibition from the MLH location through nodes with similar features.

        Spreading only begins if the sensed features are predicted by the MLH, i.e.
        they are similar to the features of one of the nodes nearest the MLH
        location that lie within the parent LM's max_match_distance (with the
        sensed surface normal rotated into the model's frame by the MLH
        rotation).
        Unlike for object IDs, stored features vary from node to node, so the
        spread begins from those of the nodes nearest the MLH location that are
        similar to the nearest such node, rather than requiring all of them to be.
        Nodes on the far side of a thin wall (see min_far_side_angle) are ignored
        throughout.
        """
        graph = self.parent_lm.get_graph(graph_id, input_channel=input_channel)
        features = self._get_spread_features(graph)
        if not features:
            return
        sensed = self._get_sensed_spread_features(percept, mlh["rotation"])

        seeds, seed_distances = self._get_nodes_nearest_mlh(graph, mlh["location"])
        nearby = seeds[seed_distances <= self.parent_lm.max_match_distance]
        predicted = nearby[self._similar_features(features, nearby, sensed)]
        if len(predicted) == 0:
            return
        reference = {name: values[predicted[0]] for name, values in features.items()}
        seeds = seeds[
            ~self._far_side(features, seeds, reference)
            & self._similar_features(features, seeds, reference)
        ]

        spreads, neighbors, surface_edges = self._get_similar_neighbors(
            graph_id, input_channel, graph, features
        )
        self._spread_from_seeds(
            graph_id,
            input_channel,
            seeds,
            neighbors,
            spreads,
            object_id=None,
            surface_edges=surface_edges,
        )

    def _spread_from_seeds(
        self,
        graph_id,
        input_channel,
        seeds,
        neighbors,
        spreads,
        object_id,
        surface_edges=None,
    ) -> None:
        """Inhibit the seeds and spread inhibition from them through the graph.

        The spread (see `_get_spread_order`) is appended to `spread_records`, with
        the inhibited nodes in the order they were reached.

        Args:
            graph_id: The graph to spread through.
            input_channel: The input channel whose graph to spread through.
            seeds: The nodes to begin spreading from.
            neighbors: The neighbors of each node of the graph (see
                `_get_node_neighbors`).
            spreads: Whether each node of the graph spreads to its neighbors.
            object_id: The received object ID, or None for sensory input.
            surface_edges: For sensory input, every (node, neighbor) pair on the
                same side of the surface (see `_get_spread_order`); None for
                object IDs.
        """
        order = self._get_spread_order(seeds, neighbors, spreads, surface_edges)
        inhibition = self._get_inhibition_steps(graph_id, input_channel, len(spreads))
        inhibition[order] = self.inhibition_decay_steps
        self.spread_records.append(
            SpreadRecord(
                graph_id=graph_id,
                input_channel=input_channel,
                object_id=object_id,
                node_order=order,
            )
        )
        explained = (
            "with similar features"
            if object_id is None
            else f"storing object ID {object_id}"
        )
        logger.debug(
            f"Inhibited {len(order)} nodes {explained} on channel "
            f"{input_channel} of {graph_id}"
        )

    def _get_spread_order(
        self, seeds, neighbors, spreads, surface_edges=None
    ) -> np.ndarray:
        """Spread from seeds through a graph, without inhibiting the reached nodes.

        Beginning with the seeds, every reached node that spreads reaches all of
        its neighbors, and spreading then continues from each of them. For
        sensory input, whenever spreading stops having reached fewer than
        min_spread_fraction of the graph's nodes (e.g. on a fragment of noisy
        features, or a small part like a mug's handle), it is pushed through: every
        reached node reaches all of its neighbors on the same side of the surface
        (`surface_edges`), whether or not their features are similar, and
        spreading continues from them, but only until it has reached
        min_spread_fraction of the nodes (so e.g. a spread pushed from the handle
        onto the side of a mug does not go on to cover the side).

        Args:
            seeds: The nodes to begin spreading from.
            neighbors: The neighbors of each node of the graph (see
                `_get_node_neighbors`).
            spreads: Whether each node of the graph spreads to its neighbors.
            surface_edges: The (node, neighbor) pairs on the same side of the
                surface, or None to not push spreading through.

        Returns:
            The reached nodes, in the order they were reached.
        """
        num_nodes = len(spreads)
        reached = np.zeros(num_nodes, dtype=bool)
        reached[seeds] = True
        order = seeds.tolist()
        queue = deque(order)
        min_size = 0
        if surface_edges is not None and self.min_spread_fraction:
            min_size = self.min_spread_fraction * num_nodes
        pushed = False
        while True:
            while queue and not (pushed and len(order) >= min_size):
                node = queue.popleft()
                if not spreads[node]:
                    continue
                for neighbor in neighbors[node].tolist():
                    if not reached[neighbor]:
                        reached[neighbor] = True
                        order.append(neighbor)
                        queue.append(neighbor)
            if len(order) >= min_size:
                break
            pushed = True
            source, target = surface_edges
            frontier = np.unique(target[reached[source] & ~reached[target]])
            if len(frontier) == 0:
                break
            reached[frontier] = True
            order.extend(frontier.tolist())
            queue.extend(frontier.tolist())
        return np.array(order, dtype=int)

    def _get_nodes_nearest_mlh(
        self, graph, mlh_location
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the num_spread_neighbors nodes of a graph nearest the MLH location.

        Returns:
            The nodes, nearest first, and their distances from the MLH location.
        """
        num_seeds = min(self.num_spread_neighbors, len(graph.pos))
        query = np.atleast_2d(np.asarray(mlh_location, dtype=float))
        seeds = np.asarray(
            graph.find_nearest_neighbors(query, num_neighbors=num_seeds)
        ).reshape(-1)
        distances = np.asarray(
            graph.find_nearest_neighbors(
                query, num_neighbors=num_seeds, return_distance=True
            )
        ).reshape(-1)
        return seeds, distances

    def _get_spread_features(self, graph) -> dict[str, np.ndarray]:
        """Get the features of every node of a graph that sensory spreading compares.

        Returns:
            Arrays of each compared feature the graph stores, with one row per node:
            "hsv" (hue, saturation and value) and "normal" (the unit surface
            normal). Hue is omitted when not compared.
        """
        feature_mapping = graph.feature_mapping or {}
        features = {}
        if self.max_hue_difference is not None and "hsv" in feature_mapping:
            features["hsv"] = self._get_feature_values(graph, "hsv")[:, :3]
        if "pose_vectors" in feature_mapping:
            normals = self._get_feature_values(graph, "pose_vectors")[:, :3]
            features["normal"] = normals / np.maximum(
                np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
            )
        return features

    def _get_sensed_spread_features(self, percept: Message, rotation) -> dict:
        """Get the sensed features that sensory spreading compares.

        Args:
            percept: The sensory input.
            rotation: The MLH rotation, which rotates the sensed (body-frame)
                surface normal into the model's frame.

        Returns:
            The sensed features, in the format of `_get_spread_features` (with one
            value per feature, and the normal in the model's frame), for those the
            percept carries.
        """
        non_morphological = percept.non_morphological_features or {}
        morphological = percept.morphological_features or {}
        sensed = {}
        if (
            self.max_hue_difference is not None
            and non_morphological.get("hsv") is not None
        ):
            sensed["hsv"] = np.asarray(non_morphological["hsv"], dtype=float)[:3]
        if morphological.get("pose_vectors") is not None:
            normal = rotation.apply(
                np.asarray(morphological["pose_vectors"], dtype=float).reshape(3, 3)[0]
            )
            sensed["normal"] = normal / np.linalg.norm(normal)
        return sensed

    def _similar_features(self, features, nodes, reference) -> np.ndarray:
        """Check which of a set of nodes have features similar to a reference.

        Only the features present in both `features` and `reference` are compared.

        Args:
            features: The compared features of every node (see
                `_get_spread_features`).
            nodes: The nodes to check.
            reference: The features to compare them to, in the same format (one
                value per feature).

        Returns:
            Whether each node's features are similar to the reference.
        """
        return self._similar_feature_pairs(
            {name: values[nodes] for name, values in features.items()},
            {
                name: np.broadcast_to(value, (len(nodes), *np.shape(value)))
                for name, value in reference.items()
            },
        )

    def _similar_feature_pairs(self, features_a, features_b) -> np.ndarray:
        """Check which pairs of features are similar.

        Args:
            features_a: Arrays of features, with one row per pair.
            features_b: Arrays of the features to compare them to, likewise.

        Returns:
            Whether the features of each pair are similar in hue and surface
            normal, comparing only the features present in both.
        """
        num_pairs = len(next(iter(features_a.values()), []))
        similar = np.ones(num_pairs, dtype=bool)
        if "hsv" in features_a and "hsv" in features_b:
            hsv_a, hsv_b = features_a["hsv"], features_b["hsv"]
            chromatic_a = (hsv_a[:, 1] >= self.min_hue_saturation) & (
                hsv_a[:, 2] >= self.min_hue_value
            )
            chromatic_b = (hsv_b[:, 1] >= self.min_hue_saturation) & (
                hsv_b[:, 2] >= self.min_hue_value
            )
            hue_difference = np.abs(hsv_a[:, 0] - hsv_b[:, 0])
            hue_difference = np.minimum(hue_difference, 1.0 - hue_difference)
            similar &= ~(chromatic_a & chromatic_b) | (
                hue_difference <= self.max_hue_difference
            )
        if (
            self.max_normal_angle is not None
            and "normal" in features_a
            and "normal" in features_b
        ):
            cosine = np.sum(features_a["normal"] * features_b["normal"], axis=1)
            similar &= cosine >= np.cos(np.radians(self.max_normal_angle))
        return similar

    def _far_side(self, features, nodes, reference) -> np.ndarray:
        """Check which of a set of nodes are on the far side of a thin wall.

        Args:
            features: The compared features of every node (see
                `_get_spread_features`).
            nodes: The nodes to check.
            reference: The features of the node on the near side.

        Returns:
            Whether each node's surface normal faces away from the reference's by
            at least min_far_side_angle (never, if normals are not compared).
        """
        if "normal" not in features or "normal" not in reference:
            return np.zeros(len(nodes), dtype=bool)
        cosine = features["normal"][nodes] @ reference["normal"]
        return cosine <= np.cos(np.radians(self.min_far_side_angle))

    def _get_similar_neighbors(
        self, graph_id, input_channel, graph, features
    ) -> tuple[np.ndarray, list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Get the neighbors of each node for spreading through similar features.

        Neighbors on the far side of a thin wall from a node (see
        min_far_side_angle) are not its neighbors here, as they are not part of the
        same surface. A node spreads to the neighbors with features similar to its
        own if at most max_dissimilar_neighbors of its neighbors are not.

        Returns:
            For each node of the graph, whether it spreads, and the neighbors it
            spreads to; and every (node, neighbor) pair on the same side of the
            surface, whether or not their features are similar.
        """
        key = (graph_id, input_channel, len(graph.pos))
        if key not in self._similar_neighbors_cache:
            num_nodes = len(graph.pos)
            all_neighbors = self._get_node_neighbors(
                graph_id,
                input_channel,
                graph,
                num_neighbors=self.num_feature_spread_neighbors,
            )
            source, target = self._get_neighbor_edges(all_neighbors)
            near_side = np.ones(len(source), dtype=bool)
            if "normal" in features:
                cosine = np.sum(
                    features["normal"][source] * features["normal"][target], axis=1
                )
                near_side = cosine > np.cos(np.radians(self.min_far_side_angle))
            source, target = source[near_side], target[near_side]
            surface_edges = (source, target)
            similar = self._similar_feature_pairs(
                {name: values[source] for name, values in features.items()},
                {name: values[target] for name, values in features.items()},
            )
            num_dissimilar = np.bincount(source[~similar], minlength=num_nodes)
            spreads = (np.bincount(source, minlength=num_nodes) > 0) & (
                num_dissimilar <= self.max_dissimilar_neighbors
            )
            source, target = source[similar], target[similar]
            boundaries = np.searchsorted(source, np.arange(num_nodes + 1))
            neighbors = [
                target[boundaries[node] : boundaries[node + 1]]
                for node in range(num_nodes)
            ]
            self._similar_neighbors_cache[key] = (spreads, neighbors, surface_edges)
        return self._similar_neighbors_cache[key]

    @staticmethod
    def _all_neighbors_satisfy(satisfied, source, num_nodes) -> np.ndarray:
        """Get whether each node has neighbors, and all of them satisfy a condition.

        Args:
            satisfied: Whether the condition is satisfied, for each neighbor edge.
            source: The node each neighbor edge belongs to.
            num_nodes: The number of nodes in the graph.

        Returns:
            For each node, whether it has neighbors that all satisfy the condition.
        """
        has_neighbors = np.bincount(source, minlength=num_nodes) > 0
        unsatisfied = np.bincount(source[~satisfied], minlength=num_nodes) > 0
        return has_neighbors & ~unsatisfied

    @staticmethod
    def _get_neighbor_edges(neighbors) -> tuple[np.ndarray, np.ndarray]:
        """Get every (node, neighbor) pair of a graph.

        Args:
            neighbors: The neighbors of each node (see `_get_node_neighbors`).

        Returns:
            The node and the neighbor of each pair, ordered by node.
        """
        source = np.repeat(
            np.arange(len(neighbors)),
            [len(node_neighbors) for node_neighbors in neighbors],
        )
        target = np.concatenate(neighbors) if len(neighbors) else np.empty(0, dtype=int)
        return source, target.astype(int)

    def _get_node_neighbors(
        self, graph_id, input_channel, graph, num_neighbors=None
    ) -> list[np.ndarray]:
        """Get the two-way nearest-neighbor relations of every node of a graph.

        A node's neighbors are its num_neighbors nearest nodes, together with every
        node that has it among their own num_neighbors nearest nodes.
        Nearest-neighbor relations are not mutual, so without the latter, an
        outlying node (whose nearest nodes all have closer neighbors of their own)
        could never be reached.

        Args:
            graph_id: The graph whose nodes to get the neighbors of.
            input_channel: The input channel of the graph.
            graph: The graph.
            num_neighbors: The number of nearest nodes of each node. Defaults to
                num_spread_neighbors.

        Returns:
            For each node, the indices of its neighbors (excluding the node itself):
            its own nearest neighbors first, nearest first, then the nodes that
            have it among their nearest neighbors.
        """
        if num_neighbors is None:
            num_neighbors = self.num_spread_neighbors
        pos = np.asarray(graph.pos)
        num_nodes = len(pos)
        key = (graph_id, input_channel, num_nodes, num_neighbors)
        if key in self._neighbor_cache:
            return self._neighbor_cache[key]

        k = min(num_neighbors, num_nodes - 1)
        if k < 1:
            neighbors = [np.empty(0, dtype=int) for _ in range(num_nodes)]
        else:
            nearest = np.asarray(
                graph.find_nearest_neighbors(pos, num_neighbors=k + 1)
            ).reshape(num_nodes, k + 1)
            # A node is usually its own nearest neighbor, but not necessarily
            # when several nodes share a location.
            forward = [
                [n for n in row if n != node][:k]
                for node, row in enumerate(nearest.tolist())
            ]
            reverse = [[] for _ in range(num_nodes)]
            for node, node_neighbors in enumerate(forward):
                for neighbor in node_neighbors:
                    reverse[neighbor].append(node)
            neighbors = [
                np.array(list(dict.fromkeys(forward[node] + reverse[node])), dtype=int)
                for node in range(num_nodes)
            ]
        self._neighbor_cache[key] = neighbors
        return neighbors

    def _decay_inhibition(self) -> None:
        for inhibition in self._inhibition_steps.values():
            np.maximum(inhibition - 1, 0, out=inhibition)

    def _check_need_new_output_goal(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        output_goal_achieved,
    ) -> bool:
        """Determine whether the GSG should generate a new output Goal.

        Success in achieving the Goal is not an indication to need a new one, as
        the sensor should explore the tested location for a while. Once
        min_post_goal_success_steps have elapsed, a new Goal is generated if the
        MLH (object or rotation) has changed since the last Goal, or every
        elapsed_steps_factor steps.

        Returns:
            Whether the GSG should generate a new output Goal.
        """
        if output_goal_achieved:
            return False

        num_elapsed_steps = self._get_num_steps_post_output_goal_generated()
        if num_elapsed_steps <= self.min_post_goal_success_steps:
            return False

        if self._get_mlh_graph_id() is None:
            return False

        if self._mlh_changed_since_last_goal():
            logger.debug("Child objects goal indicated: MLH changed")
            return True

        if num_elapsed_steps % self.elapsed_steps_factor == 0:
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

    def _get_inhibition_steps(self, graph_id, input_channel, num_nodes) -> np.ndarray:
        """Get the remaining inhibition steps of each node of a graph.

        Returns:
            Array (stored, so modifiable in place) with one entry per node.
        """
        key = (graph_id, input_channel)
        inhibition = self._inhibition_steps.get(key)
        if inhibition is None or len(inhibition) != num_nodes:
            inhibition = np.zeros(num_nodes, dtype=int)
            self._inhibition_steps[key] = inhibition
        return inhibition

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
        surface normal of that location, specify a Goal for the motor-actuator,
        such that any sensors associated with the motor-actuator should be pointed down
        at and observing the target location (i.e. parallel to the surface normal).

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
        proposed_surface_loc = sensory_input.location + rotated_disp

        # Rotate the learned surface normal (which was committed to memory assuming a
        # default 0,0,0 orientation of the object)
        target_surface_normal_rotated = object_rot.apply(
            target_info["target_surface_normal"]
        )

        # Scale the surface normal by the desired distance x1.5 (i.e. so that we start
        # a bit further away from the object; we will separately move forward if we
        # are indeed facing it)
        surface_displacement = (
            target_surface_normal_rotated * self.desired_object_distance * 1.5
        )

        target_loc = proposed_surface_loc + surface_displacement

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
            "proposed_surface_loc": proposed_surface_loc,
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
                # Note the hypothesis-testing policy does not specify the roll of the
                # agent, because this is not relevant to the task
                "pose_vectors": np.array(
                    [
                        (-1) * target_surface_normal_rotated,
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
