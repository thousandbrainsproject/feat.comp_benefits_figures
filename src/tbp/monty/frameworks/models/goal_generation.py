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

import itertools
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    GoalGenerator,
)
from tbp.monty.frameworks.utils.communication_utils import get_percept_from_channel

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.graph_matching import GraphLM

__all__ = [
    "ChildObjectsGoalGenerator",
    "CubeViewGoalGenerator",
    "EvidenceGoalGenerator",
    "GraphGoalGenerator",
    "ModelTargetGoalGenerator",
    "ParentLMNotProvided",
    "SpreadRecord",
    "TraceGoalGenerator",
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


class ModelTargetGoalGenerator(GraphGoalGenerator):
    """Base class for GSGs that move the sensor to a location on the MLH's model.

    Provides the transform of a target location in the reference frame of the most
    likely object hypothesis (MLH) into a Goal in the body-centric frame of
    reference for the motor-actuator.
    """

    def __init__(
        self, goal_tolerances=None, desired_object_distance=0.03, **kwargs
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
            **kwargs: Additional keyword arguments.
        """
        super().__init__(goal_tolerances, **kwargs)
        self.desired_object_distance = desired_object_distance

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

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

    def _get_feature_object_id_names(self, object_id_features) -> list:
        """Get the names of the objects encoded by "object_id" feature values.

        Returns:
            The name of each object, or the feature value itself if its name is
            not known to the parent LM.
        """
        names = self.parent_lm.object_id_feature_names
        return [names.get(int(feature), int(feature)) for feature in object_id_features]

    @staticmethod
    def _get_feature_values(graph, feature) -> np.ndarray:
        """Get the values of a feature for all nodes of a graph.

        Returns:
            Array of shape (num_nodes, feature_dim) of the feature's values.
        """
        feature_idx = graph.feature_mapping[feature]
        return np.asarray(graph.x[:, feature_idx[0] : feature_idx[1]])


class EvidenceGoalGenerator(ModelTargetGoalGenerator):
    """Generator of Goals for an evidence-based graph LM.

    GSG specifically set up for generating Goals for an evidence-based graph LM,
    which can therefore leverage the hypothesis-testing action policy. This policy uses
    hypotheses about the most likely objects, as well as knowledge of their structure
    from long-term memory, to propose test-points that should efficiently disambiguate
    the ID or pose of the object the agent is currently observing.

    TODO M separate out the hypothesis-testing policy (which is one example of a
    model-based policy), from the GSG, which is the system that is capable of leveraging
    a variety of model-based policies.
    """

    def __init__(
        self,
        goal_tolerances=None,
        elapsed_steps_factor=5,
        min_post_goal_success_steps=np.inf,
        x_percent_scale_factor=0.75,
        desired_object_distance=0.03,
        wait_growth_multiplier=1,
        *,
        feature_mismatch_distance_threshold=0.02,
        cluster_distance_threshold=0.005,
        min_hue_mismatch=0.1,
        min_hue_saturation=0.1,
        min_hue_value=0.1,
        **kwargs,
    ) -> None:
        """Initialize the Evidence GSG.

        Args:
            parent_lm: ?
            goal_tolerances: ?
            elapsed_steps_factor: Factor that considers the number of elapsed
                steps as a possible condition for initiating a hypothesis-testing Goal;
                should be set to an integer reflecting a number of steps. In general,
                when we have taken number of non-Goal-driven steps
                greater than elapsed_steps_factor, then this is an indication to
                initiate a hypothesis-testing Goal. In addition however, we can
                multiply elapsed_steps_factor by an exponentially increasing
                wait-factor, such that we use longer and longer intervals as the
                experiment
                continues. Defaults to 10.
            min_post_goal_success_steps: Number of necessary steps for a hypothesis
                Goal to be considered. Unlike elapsed_steps_factor, this is a
                *necessary* criteria for us to generate a new hypothesis-testing Goal.
                For example, if set to 5, then the agent must take 5
                non-hypothesis-testing steps before it can even consider generating a
                new hypothesis-testing Goal. Infinity by default, resulting in no use
                of the hypothesis-testing policy (desirable for unit tests etc.).
                Defaults to np.infty.
            x_percent_scale_factor: Scale x-percent threshold to decide when to focus
                on pose rather than determining object ID; in particular, this is used
                to determine whether the top object is sufficiently more likely (based
                on MLH evidence) than the second MLH object to warrant focusing on
                disambiguating the pose of the first; should be bounded between 0:1.0.
                If x_percent_scale_factor=1.0, then will wait until the standard
                x-percent threshold is exceeded, equivalent to the LM converging to a
                single object, but not a pose. If it is <1.0, then we will start testing
                pose of the MLH object even before we are entirely certain about its ID.
                Defaults to 0.75.
            desired_object_distance: The desired distance between the agent and the
                object, which is used to determine whether the agent is close enough to
                the object to consider it "achieved". Note this need not be the same as
                the one specified for the motor-system (e.g. the surface-policy), as we
                may want to aim for an initially farther distance, while the
                surface-policy may want to stay quite close to the object. Defaults to
                0.03.
            wait_growth_multiplier: Multiplier used to increase the `wait_factor`, which
                in turn controls how long to wait before the next jump attempt.
            feature_mismatch_distance_threshold: Spatial distance (in meters) below
                which the two candidate graphs are considered too similar in shape
                for Euclidean distance to be a useful discriminator. When the largest
                nearest-neighbor separation between the two graphs falls below this
                threshold, the GSG instead looks for mismatches in the features
                stored at graph nodes (object IDs from LM input channels, or hue from
                sensor channels). Defaults to 0.02 (2cm).
            cluster_distance_threshold: Distance (in meters) used when spatially
                clustering nodes with mismatching discrete features; any node farther
                than this from all other members of a cluster is treated as an
                outlier. Defaults to 0.005 (0.5cm).
            min_hue_mismatch: Minimum circular distance in hue space (hue is in the
                range [0, 1]) between two nearest-neighbor nodes for a color-based
                mismatch to be considered meaningful. Defaults to 0.1.
            min_hue_saturation: Minimum HSV saturation (in the range [0, 1]) both
                nodes of a nearest-neighbor pair must have for their hues to be
                compared. Hue is undefined for achromatic (grey, white, or black)
                colors, so it varies arbitrarily between neighboring achromatic
                nodes. Defaults to 0.1.
            min_hue_value: Minimum HSV value (in the range [0, 1]) both nodes of a
                nearest-neighbor pair must have for their hues to be compared, as hue
                is similarly unreliable for near-black colors. Defaults to 0.1.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            goal_tolerances, desired_object_distance=desired_object_distance, **kwargs
        )

        self.elapsed_steps_factor = elapsed_steps_factor
        self.min_post_goal_success_steps = min_post_goal_success_steps
        self.x_percent_scale_factor = x_percent_scale_factor
        self.wait_growth_multiplier = wait_growth_multiplier
        self.feature_mismatch_distance_threshold = feature_mismatch_distance_threshold
        self.cluster_distance_threshold = cluster_distance_threshold
        self.min_hue_mismatch = min_hue_mismatch
        self.min_hue_saturation = min_hue_saturation
        self.min_hue_value = min_hue_value

    # ======================= Public ==========================

    # ------------------- Main Algorithm -----------------------

    def reset(self):
        """Reset additional parameters specific to the Evidence GSG."""
        super().reset()

        self.focus_on_pose = False  # Whether the jump should be executed to focus on
        # distinguishing between possible poses of the current MLH object, rather
        # than trying to distinguish different possible object IDs.
        self.wait_factor = 1  # Initial value; scales how long to wait before the
        # next jump attempt
        self.prev_top_mlhs = None  # Store the top two object hypothesis IDs from
        # previous hypothesis-testing actions; used to track when these have changed,
        # and therefore a possible reason to initiate another hypothesis-testing action.
        # TODO M consider moving to buffer.

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _generate_goal(self, ctx: RuntimeContext, observations) -> Goal:
        """Use the hypothesis-testing policy to generate a Goal.

        The Goal will rapidly disambiguate the pose and/or ID of the object the
        LM is currently observing.

        Returns:
            A Goal for the motor system, or a None-type Goal if no informative
            location to test could be found.
        """
        # Determine where we want to test in the MLH graph
        mismatch = self._compute_graph_mismatch(ctx)

        if mismatch is None:
            # The two most likely graphs could not be meaningfully distinguished
            # either spatially or by node features, so there is no informative
            # location to propose.
            return self._generate_none_goal()

        input_channel, target_loc_id = mismatch

        # Get pose information for the target point
        target_info = self._get_target_loc_info(target_loc_id, input_channel)

        # Estimate how important this Goal will be for the Monty-system as a
        # whole
        goal_confidence = self.parent_lm.get_output().confidence

        # Compute the Goal (for the motor-actuator)
        return self._compute_goal_for_target_loc(
            observations,
            target_info,
            goal_confidence=goal_confidence,
        )

    def _compute_graph_mismatch(self, ctx: RuntimeContext):
        """Propose a point for the model to test.

        The aim is to propose a point for the model to test, with the aim of performing
        object pose and ID recognition, by looking at the graph of the most likely
        object, and comparing it to the graph of the second most likely object. If there
        is a local mismatch in the graphs (e.g. the presence of a handle in one and not
        the other), this should return the necessary coordinates to move there.

        If the two graphs are spatially near-identical (i.e. the largest
        nearest-neighbor separation between the two point clouds is below
        `feature_mismatch_distance_threshold`), Euclidean distance is not a useful
        discriminator. In that case, we instead compare the features stored at the
        graphs' nodes, in the following order of priority:
        - Discrete features: object IDs provided by input from other LMs, on the
          input channels present in both graphs.
        - Novel channels: input channels storing object IDs in only one of the two
          graphs (e.g. a logo present on one object but absent from the other).
        - Continuous features: hue (from HSV) provided by input from sensor modules,
          on the input channels present in both graphs.

        --- Some Details ---
        Part of this method transforms the graph of the most likely object into
        the reference frame in which the 2nd most-likely object was *learned*.
        We perform this operation because when comparing the two point-clouds (using
        the points of the most-likely object as queries), we can then re-use the KDTree
        already constructed for the 2nd most-likely object, hence the importance of
        being in that reference frame

        TODO M eventually can try looking at more objects, or flipping the MLH
        object - e.g. if the two most likely are the mug and one of the handle-less
        cups, then depending on the order in which we compare them, we may not
        actually identify the handle as a good candidate to test (i.e. if one graph is
        entirely a subset of the other graph). We could use the returned separation
        distance to estimate which of these approaches would be better.
        - i.e. imagine 2nd MLH is a mug with a handle, and MLH has no handle; when
        checking all the points for the handleless mug, there will always be nearby
        points.
        - Re. implementing this: could start with the MLH as the query points, looking
        for points with minimal neighbors with the 2nd most likely graph; if found
        too many neighbors in a given radius (threshold dependent), this suggests
        the 1st MLH graph is a sub-graph of the 2nd MLH; therefore, check whether
        the 2nd graph has any points with few neighbors with the first; if still many
        neighbors, this could then serve as a learning signal to merge the graphs?
        (at least at some levels of hierarchy) --> NB merging should use "picky" graph-
        building method to ensure we don't just double the number of points in the graph
        unnecessarily.

        TODO M consider adding a factor so that we ensure our testing spot is also far
        away from any previously visited locations (at least according to MLH path),
        including our current location.

        Returns:
            A tuple of (input_channel, target_loc_id), where target_loc_id is the
            index of the point to test in the top MLH graph of the given input
            channel, or None if no informative location could be found.
        """
        logger.debug("Proposing an evaluation location based on graph mismatch")

        top_id, second_id = self.parent_lm.get_top_two_mlh_ids()

        top_mlh = self.parent_lm.get_mlh_for_object(top_id)
        # Determine the second most-likely object for saving to history, even if we
        # are going to focus on pose mismatch
        second_mlh_object = self.parent_lm.get_mlh_for_object(second_id)

        sensor_channel_name = self.parent_lm.buffer.get_first_sensory_input_channel()

        top_mlh_graph = np.asarray(
            self.parent_lm.get_graph(top_id, input_channel=sensor_channel_name).pos
        )

        if self.focus_on_pose:
            # Overwrite the second most likely hypothesis with the second most likely
            # *pose* of the most-likely object
            second_id = top_id
            _, second_mlh = self.parent_lm.get_top_two_pose_hypotheses_for_graph_id(
                top_id
            )
            print("Focusing on pose mismatch")
            print(
                f"Pose of first MLH object: {top_mlh['rotation']}, {top_mlh['location']}"
            )
            print(
                f"Pose of second MLH object: {second_mlh['rotation']}, {second_mlh['location']}"
            )

        else:
            second_mlh = second_mlh_object

        top_mlh_graph = self._transform_to_second_mlh_rf(
            top_mlh_graph, top_mlh, second_mlh
        )

        # Perform the same transformation to the estimated location (sanity check)
        transformed_current_loc = self._transform_to_second_mlh_rf(
            top_mlh["location"], top_mlh, second_mlh
        )
        assert np.all(transformed_current_loc == second_mlh["location"]), (
            "Graph transformation to 2nd object reference frame not returning correct "
            "transformed location"
        )

        # Perform kdtree search to identify the point with the most distant
        # nearest-neighbor
        # Note we ultimately want the target location to be one on the most likely
        # graph, so we pass the top-MLH graph in as the query points
        second_mlh_graph = self.parent_lm.get_graph(
            second_id, input_channel=sensor_channel_name
        )
        radius_node_dists = second_mlh_graph.find_nearest_neighbors(
            top_mlh_graph,
            num_neighbors=1,
            return_distance=True,
        )

        target_loc_id = np.argmax(radius_node_dists)

        self.prev_top_mlhs = [top_mlh, second_mlh_object]

        if radius_node_dists[target_loc_id] >= self.feature_mismatch_distance_threshold:
            # The graphs are sufficiently different spatially, so the most separated
            # point is an informative location to test
            return sensor_channel_name, target_loc_id

        # The point clouds are near-identical in shape; fall back to comparing the
        # features stored at the graphs' nodes, over the channels present in both
        logger.debug("Graphs spatially near-identical; comparing node features instead")

        top_channels = self.parent_lm.get_input_channels_in_graph(top_id)
        second_channels = self.parent_lm.get_input_channels_in_graph(second_id)
        shared_channels = [
            channel for channel in top_channels if channel in second_channels
        ]

        discrete_mismatch = self._compute_discrete_feature_mismatch(
            shared_channels,
            top_id=top_id,
            second_id=second_id,
            top_mlh=top_mlh,
            second_mlh=second_mlh,
        )
        if discrete_mismatch is not None:
            return discrete_mismatch

        novel_channel_mismatch = self._compute_novel_channel_mismatch(
            top_id=top_id,
            second_id=second_id,
            top_mlh=top_mlh,
            second_mlh=second_mlh,
        )
        if novel_channel_mismatch is not None:
            return novel_channel_mismatch

        return self._compute_continuous_feature_mismatch(
            ctx,
            shared_channels,
            top_id=top_id,
            second_id=second_id,
            top_mlh=top_mlh,
            second_mlh=second_mlh,
        )

    def _transform_to_second_mlh_rf(self, points, top_mlh, second_mlh):
        """Transform points from the top MLH graph's frame to the second MLH's frame.

        Fully correct the origin and rotation of points defined in the reference frame
        in which the top object's graph was learned, so they are in the reference
        frame in which the second object's graph was learned. Note the graph of the
        second most likely object is already in the reference frame from learning that
        was used when constructing its KDTree, hence transforming into that frame
        allows re-using the KDTree for nearest-neighbor queries.

        Args:
            points: Array of points (or a single point) in the top MLH graph's
                learned reference frame.
            top_mlh: The most-likely hypothesis (dict with "rotation" and "location").
            second_mlh: The second most-likely hypothesis.

        Returns:
            The points transformed into the second MLH graph's learned reference frame.
        """
        # Convert to environmental coordinates, and normalize by current MLH location
        # Note the MLH rotation is the rotation required to match a displacement to
        # a model, so it is the *inverse* of e.g. the ground-truth rotation
        # TODO M: See if apply_rf_transform_to_points could be used here
        rotated_points = top_mlh["rotation"].inv().apply(points)
        current_mlh_location = top_mlh["rotation"].inv().apply(top_mlh["location"])
        env_points = rotated_points - current_mlh_location
        # Convert from environmental coordinates to the learned coordinate of 2nd
        # object. Thus we don't need to invert the stored rotation, as we would like
        # to actually apply the inverse form.
        return second_mlh["rotation"].apply(env_points) + second_mlh["location"]

    def _compute_discrete_feature_mismatch(
        self, shared_channels, *, top_id, second_id, top_mlh, second_mlh
    ):
        """Find a target location based on mismatching discrete (object ID) features.

        For every input channel (present in both graphs) that stores an "object_id"
        feature (i.e. input from another LM), compare the object ID stored at each
        node of the top MLH graph to the object ID stored at its nearest neighbor in
        the second MLH graph. A node mismatches if the object IDs differ, or if its
        nearest neighbor is further than the parent LM's `max_match_distance` away
        (i.e. the second MLH graph stores no object ID at that location, even if
        the nearest one it does store is the same). Mismatching nodes are spatially
        clustered (nodes more than `cluster_distance_threshold` from all other
        members are outliers), and the largest cluster is retained for each channel.

        When several channels contain mismatching nodes, the channel whose largest
        cluster contains the most points wins, as object IDs are discrete (same or
        different), so cluster size is the best available proxy for how informative
        the region is.

        Returns:
            A tuple of (input_channel, target_loc_id) where target_loc_id is the
            node of the winning cluster that sits closest to the cluster's center,
            or None if no channel has mismatching object IDs.
        """
        best_channel = None
        best_cluster_size = 0
        best_target_loc_id = None
        best_object_ids = None

        top_graph_name, second_graph_name = self._get_graph_id_names(
            [top_id, second_id]
        )

        for channel in shared_channels:
            top_graph = self.parent_lm.get_graph(top_id, input_channel=channel)
            second_graph = self.parent_lm.get_graph(second_id, input_channel=channel)

            print(
                f"Comparing object-ID channel {channel} between top and second graphs"
            )
            print(f"Top ID: {top_graph_name}, Second ID: {second_graph_name}")

            if (
                "object_id" not in top_graph.feature_mapping
                or "object_id" not in second_graph.feature_mapping
            ):
                logger.debug(
                    f"Object-ID channel {channel} not present in both graphs; skipping"
                )
                continue

            top_pos = np.asarray(top_graph.pos)
            nearest_node_ids, nearest_node_dists = self._nearest_second_graph_nodes(
                top_pos, second_graph, top_mlh, second_mlh
            )

            top_object_ids = self._get_feature_values(top_graph, "object_id").flatten()
            second_object_ids = self._get_feature_values(second_graph, "object_id")[
                nearest_node_ids
            ].flatten()
            too_far = nearest_node_dists > self.parent_lm.max_match_distance

            print(
                "Unique top object IDs: "
                f"{self._get_feature_object_id_names(np.unique(top_object_ids))}"
            )
            print(
                "Unique second object IDs: "
                f"{self._get_feature_object_id_names(np.unique(second_object_ids))}"
            )
            print(f"Minimum nearest node distance: {np.min(nearest_node_dists)}")

            mismatching_nodes = np.nonzero(
                (top_object_ids != second_object_ids) | too_far
            )[0]

            if len(mismatching_nodes) == 0:
                logger.debug(
                    f"No mismatching nodes found on channel {channel}; skipping"
                )
                if np.nonzero(too_far) == 0:
                    logger.debug(
                        f"No nodes within max_match_distance found on channel {channel}"
                    )
                if np.nonzero(top_object_ids != second_object_ids) == 0:
                    logger.debug(f"No object-ID mismatch found on channel {channel}")
                continue

            cluster_members = self._largest_spatial_cluster(top_pos[mismatching_nodes])
            cluster_node_ids = mismatching_nodes[cluster_members]

            if len(cluster_node_ids) > best_cluster_size:
                closest_member = self._closest_to_center(top_pos[cluster_node_ids])
                best_channel = channel
                best_cluster_size = len(cluster_node_ids)
                best_target_loc_id = cluster_node_ids[closest_member]
                best_object_ids = (
                    top_object_ids[best_target_loc_id],
                    second_object_ids[best_target_loc_id],
                )
                best_target_too_far = too_far[best_target_loc_id]

        if best_channel is None:
            return None

        top_name, second_name = self._get_feature_object_id_names(best_object_ids)
        if best_target_too_far:
            second_stores = (
                f"nothing within max_match_distance (nearest: {second_name})"
            )
        else:
            second_stores = second_name
        logger.debug(
            f"Object-ID mismatch found on channel {best_channel} "
            f"(cluster of {best_cluster_size} nodes); at the target, the top "
            f"hypothesis stores {top_name} and the second hypothesis stores "
            f"{second_stores}"
        )
        return best_channel, best_target_loc_id

    def _compute_novel_channel_mismatch(
        self, *, top_id, second_id, top_mlh, second_mlh
    ):
        """Find a target location on an input channel present in only one graph.

        A channel is novel to a graph if that graph stores "object_id" features for
        it (i.e. input from another LM), while the other graph either has no data
        stored for the channel, or stores it without object IDs. For example, if one
        candidate mug has a logo (a child object recognized by another LM) and the
        other candidate mug has no logo, the logo's input channel is only present
        in the graph of the former, so there are no nearest-neighbor object IDs to
        compare against. Instead, all nodes of each novel channel are spatially
        clustered (as in `_compute_discrete_feature_mismatch`), and the channel
        whose largest cluster contains the most points wins, regardless of which
        of the two graphs it belongs to.

        If the winning channel belongs to the top MLH graph, the target is the node
        of the cluster closest to the cluster's center. If it belongs to the second
        MLH graph, the cluster is first transformed into the reference frame of the
        top MLH graph (in which Goals are computed), and the target is the node of
        the top MLH graph's sensory channel nearest to the cluster's center, i.e.
        a learned surface point (with a surface normal) where the top MLH graph
        predicts the novel child object would be, if the second MLH were correct.

        Returns:
            A tuple of (input_channel, target_loc_id), where target_loc_id is a node
            in the top MLH graph of the given input channel, or None if neither
            graph has a novel channel.
        """
        candidates = [
            (top_id, channel)
            for channel in self._get_novel_object_id_channels(top_id, second_id)
        ] + [
            (second_id, channel)
            for channel in self._get_novel_object_id_channels(second_id, top_id)
        ]

        best_graph_id = None
        best_channel = None
        best_cluster_size = 0
        best_target_loc_id = None
        best_target_pos = None

        for graph_id, channel in candidates:
            pos = np.asarray(
                self.parent_lm.get_graph(graph_id, input_channel=channel).pos
            )
            if graph_id != top_id:
                pos = self._transform_to_second_mlh_rf(pos, second_mlh, top_mlh)

            cluster_node_ids = self._largest_spatial_cluster(pos)

            if len(cluster_node_ids) > best_cluster_size:
                closest_member = self._closest_to_center(pos[cluster_node_ids])
                best_graph_id = graph_id
                best_channel = channel
                best_cluster_size = len(cluster_node_ids)
                best_target_loc_id = cluster_node_ids[closest_member]
                best_target_pos = pos[best_target_loc_id]

        if best_channel is None:
            return None

        novel_graph = self.parent_lm.get_graph(
            best_graph_id, input_channel=best_channel
        )
        (novel_name,) = self._get_feature_object_id_names(
            self._get_feature_values(novel_graph, "object_id")[best_target_loc_id]
        )
        hypothesis = "top" if best_graph_id == top_id else "second"
        logger.debug(
            f"Novel channel {best_channel} found in graph of {best_graph_id} "
            f"(cluster of {best_cluster_size} nodes); at the target, only the "
            f"{hypothesis} hypothesis stores {novel_name}"
        )

        if best_graph_id == top_id:
            return best_channel, best_target_loc_id

        sensor_channel_name = self.parent_lm.buffer.get_first_sensory_input_channel()
        sensor_graph = self.parent_lm.get_graph(
            top_id, input_channel=sensor_channel_name
        )
        nearest_sensor_node = sensor_graph.find_nearest_neighbors(
            np.atleast_2d(best_target_pos),
            num_neighbors=1,
            return_distance=False,
        )[0]
        return sensor_channel_name, nearest_sensor_node

    def _get_novel_object_id_channels(self, graph_id, other_id):
        """Get channels storing object IDs in one graph, but not in the other.

        Returns:
            The input channels of graph_id that store "object_id" features, and
            for which other_id either has no data stored, or stores no object IDs.
        """
        other_channels = self.parent_lm.get_input_channels_in_graph(other_id)
        novel_channels = []
        for channel in self.parent_lm.get_input_channels_in_graph(graph_id):
            graph = self.parent_lm.get_graph(graph_id, input_channel=channel)
            if "object_id" not in graph.feature_mapping:
                continue
            if channel in other_channels:
                other_graph = self.parent_lm.get_graph(other_id, input_channel=channel)
                if "object_id" in other_graph.feature_mapping:
                    continue
            novel_channels.append(channel)
        return novel_channels

    def _get_graph_id_names(self, graph_ids) -> list:
        """Get the names of the objects modeled by the parent LM's graphs.

        As a graph may have been built from several target objects, their names
        are joined with "/".

        Returns:
            The name of the object(s) each graph was built from, or the graph ID
            itself if the parent LM has no record of its target objects.
        """
        targets = self.parent_lm.graph_id_to_target
        return [
            "/".join(sorted(targets[graph_id])) if graph_id in targets else graph_id
            for graph_id in graph_ids
        ]

    @staticmethod
    def _closest_to_center(points) -> int:
        """Find the point closest to the geometric mean of a set of points.

        Used to select an actual learned model point approximately at the center
        of a cluster.

        Returns:
            Index (into `points`) of the point closest to their geometric mean.
        """
        return int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))

    def _compute_continuous_feature_mismatch(
        self,
        ctx: RuntimeContext,
        shared_channels,
        *,
        top_id,
        second_id,
        top_mlh,
        second_mlh,
    ):
        """Find a target location based on mismatching continuous (hue) features.

        For every input channel (present in both graphs) that stores an "hsv"
        feature (i.e. input from a sensor module), compute the circular distance in
        hue space between each node of the top MLH graph and its nearest neighbor in
        the second MLH graph. Pairs in which either node is achromatic (saturation
        below `min_hue_saturation`, or value below `min_hue_value`) are ignored, as
        their hue is meaningless. The target is the node with the maximally
        different hue; in the event of a tie, one of the tied nodes is chosen at
        random.

        Returns:
            A tuple of (input_channel, target_loc_id), or None if no channel has a
            hue difference exceeding `min_hue_mismatch`.
        """
        best_channel = None
        best_hue_dist = self.min_hue_mismatch
        best_target_loc_id = None

        for channel in shared_channels:
            top_graph = self.parent_lm.get_graph(top_id, input_channel=channel)
            second_graph = self.parent_lm.get_graph(second_id, input_channel=channel)

            if (
                "hsv" not in top_graph.feature_mapping
                or "hsv" not in second_graph.feature_mapping
            ):
                continue

            top_pos = np.asarray(top_graph.pos)
            nearest_node_ids, _ = self._nearest_second_graph_nodes(
                top_pos, second_graph, top_mlh, second_mlh
            )

            top_hsv = self._get_feature_values(top_graph, "hsv")
            second_hsv = self._get_feature_values(second_graph, "hsv")[nearest_node_ids]

            # Circular distance in hue space (hue lives on a circle in [0, 1])
            abs_diff = np.abs(top_hsv[:, 0] - second_hsv[:, 0])
            hue_dists = np.minimum(abs_diff, 1.0 - abs_diff)

            chromatic = (
                (top_hsv[:, 1] >= self.min_hue_saturation)
                & (second_hsv[:, 1] >= self.min_hue_saturation)
                & (top_hsv[:, 2] >= self.min_hue_value)
                & (second_hsv[:, 2] >= self.min_hue_value)
            )
            hue_dists[~chromatic] = 0.0

            max_hue_dist = hue_dists.max()
            if max_hue_dist <= best_hue_dist:
                continue

            tied_nodes = np.nonzero(hue_dists == max_hue_dist)[0]
            best_channel = channel
            best_hue_dist = max_hue_dist
            best_target_loc_id = (
                tied_nodes[0] if len(tied_nodes) == 1 else ctx.rng.choice(tied_nodes)
            )

        if best_channel is None:
            logger.debug("No feature mismatch found; not proposing a target location")
            return None

        logger.debug(
            f"Hue mismatch found on channel {best_channel} "
            f"(circular hue distance {best_hue_dist:.3f})"
        )
        return best_channel, best_target_loc_id

    def _nearest_second_graph_nodes(self, top_pos, second_graph, top_mlh, second_mlh):
        """For each top-graph point, find its nearest neighbor in the second graph.

        Transforms the top MLH graph's points into the second MLH graph's learned
        reference frame (so the second graph's KDTree can be re-used), then queries
        for each point's single nearest neighbor.

        Returns:
            Tuple of (node indices into the second graph, distances to those
            nodes), each with one entry per top-graph point.
        """
        transformed_pos = self._transform_to_second_mlh_rf(top_pos, top_mlh, second_mlh)
        nearest_node_ids = second_graph.find_nearest_neighbors(
            transformed_pos,
            num_neighbors=1,
            return_distance=False,
        )
        distances = second_graph.find_nearest_neighbors(
            transformed_pos,
            num_neighbors=1,
            return_distance=True,
        )
        return nearest_node_ids, distances

    def _largest_spatial_cluster(self, points):
        """Find the largest spatially-contiguous cluster among points.

        Uses single-linkage hierarchical clustering: points are in the same cluster
        if they can be chained together via steps no larger than
        `cluster_distance_threshold`; any point farther than this from all members
        of a cluster is therefore excluded from it (i.e. treated as an outlier with
        respect to that cluster).

        Args:
            points: Array of shape (N, 3) of point locations.

        Returns:
            Indices (into `points`) of the members of the largest cluster.
        """
        if len(points) == 1:
            return np.array([0])

        cluster_labels = fcluster(
            linkage(points, method="single"),
            t=self.cluster_distance_threshold,
            criterion="distance",
        )
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        return np.nonzero(cluster_labels == largest_cluster_label)[0]

    def _check_need_new_output_goal(
        self, ctx: RuntimeContext, output_goal_achieved
    ) -> bool:
        """Determine whether the GSG should generate a new output Goal.

        Unlike the base version, success in achieving the Goal is not an
        indication to need a new goal, because we should now be exploring a new
        part of the hypothesis-space, and so want to stay there for some time.

        Returns:
            Whether the GSG should generate a new output Goal.
        """
        if output_goal_achieved:
            return False

        return self._check_conditions_for_hypothesis_test(ctx)

    def _check_conditions_for_hypothesis_test(self, ctx: RuntimeContext):
        """Check if good chance to discriminate between conflicting object IDs or poses.

        Evaluates possible conditions for performing a hypothesis-guided action for
        pose and object ID determination, i.e. determines whether there is a good chance
        of discriminating between conflicting object IDs or poses.

        The schedule is designed to balance discriminating the pose and objects as
        efficiently as possible; TODO M future work can use the schedule conditions as
        primitives and use RL or evolutionary algorithms to optimize the relevant
        parameters.

        TODO M each of the below conditions could be their own method; could then pass
        a set of keys which we iterate through, and thereby quickly test as a
        hyper-parameter which of these are worth keeping, and which of these
        we should get rid of.

        Returns:
            Whether there's a good chance to discriminate between conflicting object IDs
            or poses.
        """
        num_elapsed_steps = self._get_num_steps_post_output_goal_generated()

        self.focus_on_pose = False  # Default

        if num_elapsed_steps <= self.min_post_goal_success_steps:
            # Exceeding this threshold is necessary to consider a jump
            return False

        # === Collect additional information that will be used to check conditions
        # for initializing a jump ===

        top_id, second_id = self.parent_lm.get_top_two_mlh_ids()

        # This happens when all hypothesis spaces are empty
        if top_id is None and second_id is None:
            return False

        if second_id is None:
            # If we only have one object with a single hypothesis, we should not
            # attempt to generate a Goal.
            if len(self.parent_lm._hypotheses[top_id].evidence) == 1:
                return False

            # If the LM's hypothesis space only contains one object, focus on pose.
            self.focus_on_pose = True
            return True

        # Used to check if pose for top MLH has changed
        top_mlh = self.parent_lm._get_current_mlh()

        # If the MLH evidence is significantly above the second MLH (where "significant"
        # is determined by x_percent_scale_factor below), then focus on discriminating
        # its pose on some (random) occasions; always focus on pose if we've converged
        # to one object
        # TODO M update so that not accessing private methods here; part of 2nd phase
        # of refactoring
        pm_base_thresh = self.parent_lm._threshold_possible_matches()
        pm_smaller_thresh = self.parent_lm._threshold_possible_matches(
            x_percent_scale_factor=self.x_percent_scale_factor
        )

        if (len(pm_smaller_thresh) == 1 and (ctx.rng.uniform() <= 0.5)) or len(
            pm_base_thresh
        ) == 1:
            # If we only have one object with a single hypothesis, we should not
            # attempt to generate a Goal.
            if len(self.parent_lm._hypotheses[top_id].evidence) == 1:
                return False

            # We always focus on pose if there is just 1 possible match - if we are part
            # of the way towards being certain about the ID
            # (len(pm_smaller_thresh) == 1), then we sometimes (hence the randomness)
            # focus on pose.
            logger.debug(
                "Hypothesis jump indicated: One object more likely, focusing on pose"
            )
            self.focus_on_pose = True
            return True

        # If the identities or *order* (i.e. which one is most likely)
        # of the top two MLH changes, perform a new jump test, as this
        # is a reasonable heuristic for us having a new interesting
        # place to test
        # TODO when optimizing, consider using np.any rather than np.all, i.e. as long
        # as there is any change in the top two MLH
        if self.prev_top_mlhs is not None and np.all(
            [
                self.prev_top_mlhs[0]["graph_id"],
                self.prev_top_mlhs[1]["graph_id"],
            ]
            != [top_id, second_id]
        ):
            logger.debug(
                "Hypothesis jump indicated: change or shuffle in top-two MLH IDs"
            )
            return True

        # If the most-likely pose of the top object has changed (e.g. we didn't find
        # the mug handle following a previous jump, and have therefore eliminated a
        # pose), then we are likely to gain new information by performing another jump
        # TODO expand this to handle change in translationm/location pose as well
        # TODO add a parameter that specifies the angle between the two poses above
        # which we consider it a new pose (rather than it needing to be identical)
        if self.prev_top_mlhs is not None and np.all(
            top_mlh["rotation"].as_euler("xyz")
            != self.prev_top_mlhs[0]["rotation"].as_euler("xyz")
        ):
            logger.debug(
                "Hypothesis jump indicated: change in most-likely rotation of MLH"
            )
            return True

        # Otherwise, if a sufficient number of steps have elapsed,
        # still perform a jump; note however that this threshold exponentially
        # increases, so that we avoid continuously returning to the same location
        if num_elapsed_steps % (self.wait_factor * self.elapsed_steps_factor) == 0:
            logger.debug(
                "Hypothesis jump indicated: sufficient steps elapsed with no jump"
            )

            self.wait_factor *= self.wait_growth_multiplier
            return True

        return False


@dataclass(frozen=True)
class SpreadRecord:
    """One spread of inhibition by a `ChildObjectsGoalGenerator`.

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


class ChildObjectsGoalGenerator(ModelTargetGoalGenerator):
    """Generator of Goals that visit unexplained parts of the most likely object.

    Hypothesis tests are directed at the parts of the model of the most likely
    object hypothesis (MLH) that the LM's input has not yet explained. Within the
    MLH graph, each channel storing "object_id" features (input from lower-level
    LMs, i.e. a compositional model) contributes one candidate per child object (the
    nodes storing that object ID), and every other (sensory) channel contributes
    one candidate made up of all of its nodes. Candidates are ranked by their
    number of nodes, and a random uninhibited node of the highest-ranked candidate
    with any uninhibited node is selected as the location to test.

    Input that the MLH predicts at its current location "explains" part of the
    model, which is then inhibited from being tested for a while, by spreading
    through the graph of the channel the input was received on, starting from the
    nodes nearest the MLH location:
    - An object ID from a lower-level LM spreads through the nodes storing it.
    - Sensory input spreads through nodes with a similar hue on a continuous
      surface. Two neighboring nodes are on a continuous surface if either one's
      local model of the surface (see `_surface_continuity_errors`) predicts the
      other's surface normal and location. Spreading stops wherever the hue
      differs or the surface is discontinuous, so e.g. on a mug the side of the
      cylinder is inhibited, but not its rim, bottom or handle, and on a cube a
      single face is inhibited, but not the other faces. As the features stored
      in learned models are noisy, a spread also inhibits the whole region of
      the graph it begins in, with small regions (fragments of noisy features)
      merged into their neighbors (see `min_region_fraction`).

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

    # Stored surface normals are averaged over the observations in a voxel, so a
    # voxel spanning both sides of a wall thinner than it (e.g. a mug's) stores a
    # shortened normal of arbitrary direction, along with mixed curvatures.
    MIN_RELIABLE_NORMAL_LENGTH = 0.9

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
        max_continuity_normal_error: float | None = 40.0,
        max_continuity_location_error: float | None = 30.0,
        min_far_side_angle=135.0,
        max_dissimilar_neighbors=2,
        min_inhibited_neighbor_fraction: float | None = 0.75,
        min_region_fraction: float | None = 0.03,
        min_hue_saturation=0.1,
        min_hue_value=0.1,
        **kwargs,
    ) -> None:
        """Initialize the Child Objects GSG.

        Args:
            goal_tolerances: The tolerances for each attribute of the Goal that can be
                used by the GSG when determining whether a Goal is achieved.
            desired_object_distance: The desired distance between the agent and the
                object surface at the target. Defaults to 0.03.
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
            max_continuity_normal_error: Largest angle (in degrees) between a
                node's surface normal and the normal predicted for it by its
                neighbor's local model of the surface, for the two to be on a
                continuous surface (see `_surface_continuity_errors`). None to not
                compare normals. Defaults to 40.
            max_continuity_location_error: Largest error (as an angle in degrees,
                see `_surface_continuity_errors`) in a node's location predicted
                by its neighbor's local model of the surface, for the two to be on
                a continuous surface. None to not compare locations. Defaults to
                30.
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
            min_inhibited_neighbor_fraction: For sensory input, once a spread has
                finished, a node it did not reach (e.g. because the node's own
                stored features are noisy) is inhibited anyway if at least this
                fraction of its neighbors were, as it is surrounded by the region
                the spread inhibited. This is done once, so the region does not
                creep along its boundary. None to not do this. Defaults to 0.75.
            min_region_fraction: For sensory input, the graph is divided into
                regions that spreads from each of their nodes reach, and each
                region with fewer than this fraction of the graph's nodes (e.g. a
                fragment of noisy features) is merged into the smallest
                neighboring region on the same side of the surface (see
                `_merge_small_regions`). A spread then also inhibits the whole
                region of the node at which the sensed features are predicted.
                None to not divide graphs into regions. Defaults to 0.03.
            min_hue_saturation: Minimum HSV saturation for a hue to be chromatic.
                Defaults to 0.1.
            min_hue_value: Minimum HSV value for a hue to be chromatic. Defaults to
                0.1.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            goal_tolerances, desired_object_distance=desired_object_distance, **kwargs
        )
        self.elapsed_steps_factor = elapsed_steps_factor
        self.min_post_goal_success_steps = min_post_goal_success_steps
        self.num_spread_neighbors = num_spread_neighbors
        self.num_feature_spread_neighbors = num_feature_spread_neighbors
        self.inhibition_decay_steps = inhibition_decay_steps
        self.max_hue_difference = max_hue_difference
        self.max_continuity_normal_error = max_continuity_normal_error
        self.max_continuity_location_error = max_continuity_location_error
        self.min_far_side_angle = min_far_side_angle
        self.max_dissimilar_neighbors = max_dissimilar_neighbors
        self.min_inhibited_neighbor_fraction = min_inhibited_neighbor_fraction
        self.min_region_fraction = min_region_fraction
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
        # The region of each node (see `_get_regions`), keyed like the similar
        # neighbors cache.
        self._region_cache: dict[tuple[str, str, int], np.ndarray] = {}
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
        if object_id is None:
            tested = "sensory features"
        else:
            (tested,) = self._get_feature_object_id_names([object_id])
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
        sensed features located at the MLH location, and the sensed pose vectors
        rotated into the model's frame by the MLH rotation).
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
        sensed = self._get_sensed_spread_features(
            percept, mlh["rotation"], mlh["location"]
        )

        seeds, seed_distances = self._get_nodes_nearest_mlh(graph, mlh["location"])
        nearby = seeds[seed_distances <= self.parent_lm.max_match_distance]
        if "reliable" in features:
            nearby = nearby[features["reliable"][nearby]]
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
        region = None
        if self.min_region_fraction:
            regions = self._get_regions(graph_id, input_channel, graph, features)
            region = regions == regions[predicted[0]]
        self._spread_from_seeds(
            graph_id,
            input_channel,
            seeds,
            neighbors,
            spreads,
            object_id=None,
            surrounding_edges=surface_edges,
            region=region,
        )

    def _spread_from_seeds(
        self,
        graph_id,
        input_channel,
        seeds,
        neighbors,
        spreads,
        object_id,
        surrounding_edges=None,
        region=None,
    ) -> None:
        """Inhibit the seeds and spread inhibition from them through the graph.

        The spread (see `_get_spread_order`) is appended to `spread_records`, with
        the inhibited nodes in the order they were reached, followed by the
        nodes of `region` it did not reach.

        Args:
            graph_id: The graph to spread through.
            input_channel: The input channel whose graph to spread through.
            seeds: The nodes to begin spreading from.
            neighbors: The neighbors of each node of the graph (see
                `_get_node_neighbors`).
            spreads: Whether each node of the graph spreads to its neighbors.
            object_id: The received object ID, or None for sensory input.
            surrounding_edges: The (node, neighbor) pairs by which to find the
                nodes surrounded by the spread, or None to not look for them.
            region: Whether each node is inhibited along with the spread, or None
                for no other nodes.
        """
        num_nodes = len(spreads)
        order = self._get_spread_order(seeds, neighbors, spreads, surrounding_edges)
        inhibited = np.zeros(num_nodes, dtype=bool)
        inhibited[order] = True
        if region is not None:
            order = np.concatenate([order, np.flatnonzero(region & ~inhibited)])
            inhibited |= region

        inhibition = self._get_inhibition_steps(graph_id, input_channel, num_nodes)
        inhibition[inhibited] = self.inhibition_decay_steps
        self.spread_records.append(
            SpreadRecord(
                graph_id=graph_id,
                input_channel=input_channel,
                object_id=object_id,
                node_order=order,
            )
        )
        if object_id is None:
            explained = "similar features"
        else:
            (object_name,) = self._get_feature_object_id_names([object_id])
            explained = f"storing {object_name}"
        logger.debug(
            f"Inhibited {np.count_nonzero(inhibited)} nodes {explained} on channel "
            f"{input_channel} of {graph_id}"
        )

    def _get_spread_order(
        self, seeds, neighbors, spreads, surrounding_edges=None
    ) -> np.ndarray:
        """Spread from seeds through a graph, without inhibiting the reached nodes.

        Beginning with the seeds, every reached node that spreads reaches all of
        its neighbors, and spreading then continues from each of them. Then, if
        `surrounding_edges` are given, every node the spread did not reach but that
        has at least min_inhibited_neighbor_fraction of its neighbors (by these
        edges) reached is reached too.

        Args:
            seeds: The nodes to begin spreading from.
            neighbors: The neighbors of each node of the graph (see
                `_get_node_neighbors`).
            spreads: Whether each node of the graph spreads to its neighbors.
            surrounding_edges: The (node, neighbor) pairs by which to find the
                nodes surrounded by the spread, or None to not look for them.

        Returns:
            The reached nodes, in the order they were reached (the surrounded
            nodes last).
        """
        num_nodes = len(spreads)
        reached = np.zeros(num_nodes, dtype=bool)
        reached[seeds] = True
        queue = deque(seeds.tolist())
        order = seeds.tolist()
        while queue:
            node = queue.popleft()
            if not spreads[node]:
                continue
            for neighbor in neighbors[node].tolist():
                if not reached[neighbor]:
                    reached[neighbor] = True
                    order.append(neighbor)
                    queue.append(neighbor)

        if surrounding_edges is not None and self.min_inhibited_neighbor_fraction:
            source, target = surrounding_edges
            num_neighbors = np.bincount(source, minlength=num_nodes)
            num_reached = np.bincount(source[reached[target]], minlength=num_nodes)
            surrounded = (
                ~reached
                & (num_neighbors > 0)
                & (num_reached >= self.min_inhibited_neighbor_fraction * num_neighbors)
            )
            order.extend(np.flatnonzero(surrounded).tolist())
        return np.array(order, dtype=int)

    def _get_regions(self, graph_id, input_channel, graph, features) -> np.ndarray:
        """Divide a graph into regions of similar features, merging small ones.

        The graph is first divided into regions (see `_get_unmerged_regions`), and
        then each region with fewer than min_region_fraction of the graph's nodes
        is merged into a neighboring region (see `_merge_small_regions`).

        Returns:
            The region of each node of the graph.
        """
        key = (graph_id, input_channel, len(graph.pos))
        if key not in self._region_cache:
            regions = self._get_unmerged_regions(
                graph_id, input_channel, graph, features
            )
            source, target = self._get_neighbor_edges(
                self._get_node_neighbors(
                    graph_id,
                    input_channel,
                    graph,
                    num_neighbors=self.num_feature_spread_neighbors,
                )
            )
            self._region_cache[key] = self._merge_small_regions(
                regions, features, source, target
            )
        return self._region_cache[key]

    def _get_unmerged_regions(
        self, graph_id, input_channel, graph, features
    ) -> np.ndarray:
        """Divide a graph into the regions that spreads from its nodes reach.

        Taking the nodes in turn, each node not yet in a region begins a spread of
        similar features from it alone, and the nodes it reaches that are not yet
        in a region form a new region.

        Returns:
            The region of each node of the graph, numbered from 0.
        """
        spreads, neighbors, surface_edges = self._get_similar_neighbors(
            graph_id, input_channel, graph, features
        )
        regions = np.full(len(spreads), -1)
        num_regions = 0
        for node in range(len(spreads)):
            if regions[node] >= 0:
                continue
            order = self._get_spread_order(
                np.array([node]), neighbors, spreads, surface_edges
            )
            regions[order[regions[order] < 0]] = num_regions
            num_regions += 1
        return regions

    def _merge_small_regions(self, regions, features, source, target) -> np.ndarray:
        """Merge each region with too few nodes into a neighboring region.

        The smallest region with fewer than min_region_fraction of the graph's
        nodes is merged into the smallest neighboring region on the same side of
        the surface as it (see `_on_same_side`), then the next smallest, until
        none is left that has such a neighbor. Small neighboring regions can
        therefore merge into a region large enough to keep (e.g. the parts of a
        mug's handle), rather than each merging into the largest region nearby.

        Args:
            regions: The region of each node.
            features: The compared features of each node (see
                `_get_spread_features`).
            source: The node of each (node, neighbor) pair by which regions
                neighbor one another.
            target: The neighbor of each pair.

        Returns:
            The region of each node after merging, numbered from 0 by the size of
            the region, largest first.
        """
        regions = regions.copy()
        if self.min_region_fraction:
            min_size = self.min_region_fraction * len(regions)
            sizes = np.bincount(regions)
            # Regions that had no neighbor to merge into since the last merge.
            unmergeable = set()
            while True:
                small = [
                    region
                    for region in np.flatnonzero((sizes > 0) & (sizes < min_size))
                    if region not in unmergeable
                ]
                if not small:
                    break
                region = min(small, key=lambda region: sizes[region])
                boundary = (regions[source] == region) & (regions[target] != region)
                parent = self._get_merge_parent(
                    regions, sizes, features, source[boundary], target[boundary]
                )
                if parent is None:
                    unmergeable.add(region)
                    continue
                regions[regions == region] = parent
                sizes[parent] += sizes[region]
                sizes[region] = 0
                unmergeable.clear()
        _, regions, sizes = np.unique(regions, return_inverse=True, return_counts=True)
        rank = np.empty(len(sizes), dtype=int)
        rank[np.argsort(-sizes, kind="stable")] = np.arange(len(sizes))
        return rank[regions]

    def _get_merge_parent(self, regions, sizes, features, source, target):
        """Get the region to merge a region into.

        Args:
            regions: The region of each node.
            sizes: The number of nodes in each region.
            features: The compared features of each node (see
                `_get_spread_features`).
            source: The node in the region of each (node, neighbor) pair that
                crosses its boundary.
            target: The neighbor in another region of each pair.

        Returns:
            The smallest neighboring region on the same side of the surface as the
            region (see `_on_same_side`), or None if there is none.
        """
        neighbor_regions = np.unique(regions[target])
        for parent in sorted(neighbor_regions, key=lambda region: sizes[region]):
            to_parent = regions[target] == parent
            if self._on_same_side(features, source[to_parent], target[to_parent]):
                return int(parent)
        return None

    @staticmethod
    def _on_same_side(features, source, target) -> bool:
        """Get whether a region's boundary nodes are on the same side as a neighbor's.

        On a thin wall (e.g. a bowl's), the nearest nodes of a region on one side
        of the wall include nodes on the far side, so a node of the region's
        boundary is on the same side of the surface as a neighboring region if any
        of its neighbors in that region has a normal in the same hemisphere as its
        own. Nodes with unreliable normals (see MIN_RELIABLE_NORMAL_LENGTH) are
        not compared.

        Args:
            features: The compared features of each node (see
                `_get_spread_features`).
            source: The node in the region of each (node, neighbor) pair between
                the two regions.
            target: The neighbor in the other region of each pair.

        Returns:
            Whether at least half of the region's compared boundary nodes are on
            the same side as the neighboring region (true if none are compared).
        """
        if "normal" not in features:
            return True
        reliable = features["reliable"]
        compared = reliable[source] & reliable[target]
        source, target = source[compared], target[compared]
        if len(source) == 0:
            return True
        same_hemisphere = (
            np.sum(features["normal"][source] * features["normal"][target], axis=1) > 0
        )
        boundary_nodes = np.unique(source)
        same_side_nodes = np.unique(source[same_hemisphere])
        return len(same_side_nodes) >= 0.5 * len(boundary_nodes)

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
            "hsv" (hue, saturation and value), and the surface geometry: "location",
            "normal" (the unit surface normal), "curvature_directions" (the two
            unit principal curvature directions), "curvatures" (the two signed
            principal curvatures, in 1/m; 0, i.e. flat, when not stored) and
            "reliable" (whether the stored normal is reliable, see
            MIN_RELIABLE_NORMAL_LENGTH). Hue is omitted when not compared.
        """
        feature_mapping = graph.feature_mapping or {}
        features = {}
        if self.max_hue_difference is not None and "hsv" in feature_mapping:
            features["hsv"] = self._get_feature_values(graph, "hsv")[:, :3]
        if "pose_vectors" in feature_mapping:
            pose_vectors = self._get_feature_values(graph, "pose_vectors").reshape(
                -1, 3, 3
            )
            lengths = np.linalg.norm(pose_vectors[:, 0], axis=1)
            features["location"] = np.asarray(graph.pos, dtype=float)
            features["normal"] = (
                pose_vectors[:, 0] / np.maximum(lengths, 1e-12)[:, None]
            )
            features["curvature_directions"] = pose_vectors[:, 1:]
            features["curvatures"] = np.zeros((len(lengths), 2))
            if "principal_curvatures_log" in feature_mapping:
                features["curvatures"] = self._curvatures_from_log(
                    self._get_feature_values(graph, "principal_curvatures_log")[:, :2]
                )
            features["reliable"] = lengths >= self.MIN_RELIABLE_NORMAL_LENGTH
        return features

    def _get_sensed_spread_features(self, percept: Message, rotation, location) -> dict:
        """Get the sensed features that sensory spreading compares.

        Args:
            percept: The sensory input.
            rotation: The MLH rotation, which rotates sensed (body-frame) pose
                vectors into the model's frame.
            location: The MLH location, i.e. the sensed location in the model's
                frame.

        Returns:
            The sensed features, in the format of `_get_spread_features` (with one
            value per feature, and pose vectors in the model's frame), for those
            the percept carries.
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
            pose_vectors = rotation.apply(
                np.asarray(morphological["pose_vectors"], dtype=float).reshape(3, 3)
            )
            sensed["location"] = np.asarray(location, dtype=float)
            sensed["normal"] = pose_vectors[0] / np.linalg.norm(pose_vectors[0])
            sensed["curvature_directions"] = pose_vectors[1:]
            sensed["curvatures"] = np.zeros(2)
            if non_morphological.get("principal_curvatures_log") is not None:
                sensed["curvatures"] = self._curvatures_from_log(
                    np.asarray(
                        non_morphological["principal_curvatures_log"], dtype=float
                    )
                )
        return sensed

    @staticmethod
    def _curvatures_from_log(log_curvatures) -> np.ndarray:
        """Invert the sign-preserving log of "principal_curvatures_log" features.

        Returns:
            The signed principal curvatures (in 1/m).
        """
        log_curvatures = np.asarray(log_curvatures, dtype=float)
        return np.sign(log_curvatures) * np.expm1(np.abs(log_curvatures))

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
            Whether the features of each pair are similar in hue and lie on a
            continuous surface (as predicted from at least one of the two),
            comparing only the features present in both.
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
        if "normal" in features_a and "normal" in features_b:
            # A single node's stored normal and curvatures can be noisy, whereas a
            # discontinuity in the surface is not predicted from either side of it.
            similar &= self._predicts_continuous_surface(
                features_a, features_b
            ) | self._predicts_continuous_surface(features_b, features_a)
        return similar

    def _predicts_continuous_surface(self, from_features, to_features) -> np.ndarray:
        """Check which nodes' local surfaces predict other nodes' normals and locations.

        Args:
            from_features: Features of the nodes whose local surface predicts
                (see `_get_spread_features`), with one row per pair.
            to_features: Features of the nodes being predicted, likewise.

        Returns:
            Whether the errors of each prediction (see `_surface_continuity_errors`)
            are within max_continuity_normal_error and
            max_continuity_location_error.
        """
        normal_error, location_error = self._surface_continuity_errors(
            from_features, to_features
        )
        continuous = np.ones(len(normal_error), dtype=bool)
        if self.max_continuity_normal_error is not None:
            continuous &= normal_error <= self.max_continuity_normal_error
        if self.max_continuity_location_error is not None:
            continuous &= location_error <= self.max_continuity_location_error
        return continuous

    @staticmethod
    def _surface_continuity_errors(
        from_features, to_features
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict nodes' surface normals and locations from others' local surfaces.

        The surface around the "from" node is modeled by its normal, principal
        curvatures and their directions, i.e. as the ellipsoid (or, for curvatures
        of different signs, the saddle) with those curvatures that touches the
        surface there. Along the direction of the "to" node in the tangent plane,
        this surface curves with the normal curvature given by Euler's formula,
        and so locally follows a circle with that curvature (or, for a curvature
        of 0, a straight line, in which case the predicted normal is simply the
        "from" node's own). From the "to" node's distance along the tangent plane,
        the circle predicts both its normal and its height above the tangent
        plane.

        Args:
            from_features: Features of the nodes whose local surface predicts
                (see `_get_spread_features`), with one row per pair.
            to_features: Features of the nodes being predicted, likewise.

        Returns:
            For each pair: the angle (in degrees) between the "to" node's normal and
            the predicted one, and the location error as an angle (in degrees),
            namely the angle subtended at the "from" node by the difference
            between the "to" node's height above the tangent plane and the
            predicted height. Both are infinite when the "to" node lies further
            along the tangent plane than the radius of the circle, i.e. beyond the
            modeled surface.
        """
        normal = from_features["normal"]
        displacement = to_features["location"] - from_features["location"]
        height = np.sum(displacement * normal, axis=1)
        tangential = displacement - height[:, None] * normal
        tangential_distance = np.linalg.norm(tangential, axis=1)
        direction = tangential / np.maximum(tangential_distance, 1e-12)[:, None]

        cosines = np.einsum(
            "pj,pij->pi", direction, from_features["curvature_directions"]
        )
        curvature = np.sum(from_features["curvatures"] * cosines**2, axis=1)

        # The sine of the angle the normal turns through along the circle.
        sine = curvature * tangential_distance
        beyond = np.abs(sine) > 1
        sine = np.clip(sine, -1.0, 1.0)
        cosine = np.sqrt(1.0 - sine**2)
        predicted_normal = cosine[:, None] * normal - sine[:, None] * direction
        predicted_height = sine * tangential_distance / (1.0 + cosine)

        normal_error = np.degrees(
            np.arccos(
                np.clip(np.sum(predicted_normal * to_features["normal"], axis=1), -1, 1)
            )
        )
        location_error = np.degrees(
            np.arctan2(
                np.abs(height - predicted_height),
                np.maximum(np.linalg.norm(displacement, axis=1), 1e-12),
            )
        )
        normal_error[beyond] = np.inf
        location_error[beyond] = np.inf
        return normal_error, location_error

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
        own if at most max_dissimilar_neighbors of its neighbors are not. A node
        with an unreliable stored normal (see MIN_RELIABLE_NORMAL_LENGTH) does not
        spread, and does not stop its neighbors from spreading, but is inhibited
        when a spread reaches it.

        Returns:
            For each node of the graph, whether it spreads, and the neighbors it
            spreads to; and every (node, neighbor) pair on the same side of the
            surface, whether or not their features are similar.
        """
        key = (graph_id, input_channel, len(graph.pos))
        if key not in self._similar_neighbors_cache:
            num_nodes = len(graph.pos)
            reliable = features.get("reliable", np.ones(num_nodes, dtype=bool))
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
                near_side = (
                    (cosine > np.cos(np.radians(self.min_far_side_angle)))
                    | ~reliable[source]
                    | ~reliable[target]
                )
            source, target = source[near_side], target[near_side]
            surface_edges = (source, target)
            similar = self._similar_feature_pairs(
                {name: values[source] for name, values in features.items()},
                {name: values[target] for name, values in features.items()},
            )
            reachable = similar | ~reliable[target]
            num_dissimilar = np.bincount(source[~reachable], minlength=num_nodes)
            spreads = (
                reliable
                & (np.bincount(source, minlength=num_nodes) > 0)
                & (num_dissimilar <= self.max_dissimilar_neighbors)
            )
            source, target = source[reachable], target[reachable]
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


class TraceGoalGenerator(ModelTargetGoalGenerator):
    """Generator of Goals that visit the hot spots of the most likely object.

    Hot spots are learned on each object model during post-training unsupervised
    learning (see `HypothesisTracer`), and mark the locations that have proven most
    informative for distinguishing the object from others. This GSG moves the sensor
    to the hot spot with the highest value on the model of the most likely object
    hypothesis (MLH), transformed into the body frame using the MLH's pose.

    The maximal hot spot is always selected, even if this means revisiting it.
    """

    def __init__(
        self,
        goal_tolerances=None,
        desired_object_distance=0.03,
        min_steps_between_goals=10,
        min_hotspot_value=0.0,
        **kwargs,
    ) -> None:
        """Initialize the Trace GSG.

        Args:
            goal_tolerances: The tolerances for each attribute of the Goal that can be
                used by the GSG when determining whether a Goal is achieved.
            desired_object_distance: The desired distance between the agent and the
                object surface at the target. Defaults to 0.03.
            min_steps_between_goals: Number of matching steps that must have elapsed
                since the last Goal was generated before a new one is generated.
                Defaults to 10.
            min_hotspot_value: A hot spot is only visited if its value is above this
                threshold. Defaults to 0.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            goal_tolerances, desired_object_distance=desired_object_distance, **kwargs
        )
        self.min_steps_between_goals = min_steps_between_goals
        self.min_hotspot_value = min_hotspot_value

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _generate_goal(self, _ctx: RuntimeContext, observations) -> Goal | None:
        """Generate a Goal that moves the sensor to the MLH object's maximal hot spot.

        Returns:
            A Goal for the motor system, or a None-type Goal if the MLH object has no
            hot spot above min_hotspot_value.
        """
        mlh = self.parent_lm._get_current_mlh()
        graph_id = mlh["graph_id"]
        if graph_id not in self.parent_lm.get_all_known_object_ids():
            return self._generate_none_goal()

        input_channel = self.parent_lm.buffer.get_first_sensory_input_channel()
        model = self.parent_lm.get_graph(graph_id).get(input_channel)
        hotspot = model.get_max_hotspot_node() if model is not None else None
        if hotspot is None or hotspot[1] <= self.min_hotspot_value:
            return self._generate_none_goal()

        node_id, hotspot_value = hotspot
        target_info = self._get_target_loc_info(node_id, input_channel)
        goal = self._compute_goal_for_target_loc(
            observations,
            target_info,
            goal_confidence=self.parent_lm.get_output().confidence,
        )
        goal.info["hotspot_node_id"] = node_id
        goal.info["hotspot_value"] = hotspot_value
        logger.debug(
            f"Trace goal: visiting hot spot {node_id} (value {hotspot_value:.3f}) "
            f"on {graph_id}"
        )
        return goal

    def _check_need_new_output_goal(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        output_goal_achieved,
    ) -> bool:
        """Determine whether the GSG should generate a new output Goal.

        After reaching a hot spot, the sensor should explore there for a while, so
        success in achieving the Goal is not an indication to need a new one.

        Returns:
            Whether the GSG should generate a new output Goal.
        """
        if output_goal_achieved:
            return False
        return (
            self._get_num_steps_post_output_goal_generated()
            > self.min_steps_between_goals
        )


def _cube_view_directions() -> np.ndarray:
    """Unit directions onto the 6 faces and 8 corners of a cube centered at 0.

    Returns:
        A (14, 3) array of unit vectors, faces first.
    """
    faces = np.vstack([np.eye(3), -np.eye(3)])
    corners = np.array(list(itertools.product([1.0, -1.0], repeat=3))) / np.sqrt(3)
    return np.vstack([faces, corners])


class CubeViewGoalGenerator(ModelTargetGoalGenerator):
    """Generator of Goals that view the most likely object from 14 directions.

    The views look onto the 6 faces and 8 corners of a cube enclosing the object,
    with the cube aligned to the reference frame of the most likely object
    hypothesis (MLH). The initial view, and then each cube view, is held for
    steps_per_view matching steps, during which the motor system's default policy
    explores locally, before the sensor jumps to the next view. This is an
    LM-driven, 3D analogue of a scan policy.

    For each view direction, the sensor looks along the opposite direction at the
    point of the MLH object's model that lies furthest along the view direction,
    so that the view is centered on the object's surface.

    Jumps are made under the current MLH pose, so early views (before the pose is
    known) may land elsewhere on the object, or fail and be undone by the motor
    system. Each view is attempted once.
    """

    VIEW_DIRECTIONS = _cube_view_directions()

    def __init__(
        self,
        goal_tolerances=None,
        desired_object_distance=0.06,
        steps_per_view=50,
        **kwargs,
    ) -> None:
        """Initialize the Cube View GSG.

        Args:
            goal_tolerances: The tolerances for each attribute of the Goal that can be
                used by the GSG when determining whether a Goal is achieved.
            desired_object_distance: The desired distance between the agent and the
                object surface at the center of a view. Defaults to 0.03.
            steps_per_view: Number of matching steps spent on each view before
                moving on to the next one. Defaults to 50.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            goal_tolerances, desired_object_distance=desired_object_distance, **kwargs
        )
        self.steps_per_view = steps_per_view
        assert desired_object_distance > 0.03, (
            f"Desired object distance must be greater than 0.03, but got {desired_object_distance}"
        )

    def reset(self):
        super().reset()
        self._next_view_index = 0

    @property
    def num_views(self) -> int:
        return len(self.VIEW_DIRECTIONS)

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _generate_goal(self, _ctx: RuntimeContext, observations) -> Goal | None:
        """Generate a Goal that moves the sensor to the next view of the MLH object.

        Returns:
            A Goal for the motor system, or a None-type Goal if the MLH is not yet
            a known object, in which case the view is attempted on a later step.
        """
        mlh = self.parent_lm._get_current_mlh()
        graph_id = mlh["graph_id"]
        if graph_id not in self.parent_lm.get_all_known_object_ids():
            return self._generate_none_goal()

        input_channel = self.parent_lm.buffer.get_first_sensory_input_channel()
        model = self.parent_lm.get_graph(graph_id).get(input_channel)
        if model is None:
            return self._generate_none_goal()

        view_index = self._next_view_index
        view_direction = self.VIEW_DIRECTIONS[view_index]
        positions = np.asarray(model.pos)
        center = (positions.min(axis=0) + positions.max(axis=0)) / 2

        goal = self._compute_goal_for_target_loc(
            observations,
            {
                "hypothesis_to_test": mlh,
                "target_loc": center,  # Use the center of the object as the target location
                "target_surface_normal": view_direction,
            },
        )
        # Exploration jumps do not test the MLH, so a failed jump must not be
        # attributed to it (see EvidenceGraphLM.receive_goal_attempt).
        goal.info["hypothesis_to_test_graph_id"] = None
        goal.info["hypothesis_to_test_mlh_id"] = None
        goal.info["view_index"] = view_index
        goal.info["view_direction"] = view_direction
        self._next_view_index += 1
        logger.debug(
            f"Cube view goal: view {view_index} (direction {view_direction}) of "
            f"{graph_id}, centered on the object at {center}"
        )
        print(f"Performing a cube view jump!")

        return goal

    def _check_need_new_output_goal(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        output_goal_achieved,  # noqa: ARG002
    ) -> bool:
        """Determine whether it is time to move on to the next view.

        Returns:
            Whether the GSG should generate a new output Goal.
        """
        if self._next_view_index >= self.num_views:
            return False
        return (
            self.parent_lm.buffer.get_num_matching_steps()
            > (self._next_view_index + 1) * self.steps_per_view
        )
