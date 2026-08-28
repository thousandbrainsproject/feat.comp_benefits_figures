# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.models.goal_generation import GraphGoalGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tbp.monty.cmp import Message
    from tbp.monty.context import RuntimeContext

logger = logging.getLogger(__name__)

# The feature a sensor module must report for this GSG to know how obliquely
# it views the surface (see CameraSM's "view_direction").
VIEW_DIRECTION_FEATURE = "view_direction"


class FaceOnGoalGenerator(GraphGoalGenerator):
    """Re-orient the agent to view its sensor patch's surface face-on.

    Meant for the case where an object's pose leaves the surface the LM is
    sensing at a steep angle to the camera. Each step, the GSG reads its LM's
    first sensory channel: the surface normal at the patch (first pose
    vector) and the direction the camera looks along (the ``view_direction``
    feature). The angle between the two, smoothed over the last
    ``window`` processed steps so a single saccade does not count, is the
    view angle. When it exceeds ``max_view_angle``, the GSG outputs a goal
    placing the agent on the surface normal, at the patch's current depth,
    looking back at the patch, so the motor system's jump policy re-orients
    the view; exploration then carries on as usual from the new vantage.

    Goals are one-off (never kept), count as achieved once the smoothed
    angle is back within ``max_view_angle``, and a new one is not proposed
    until ``min_steps_between_goals`` matching steps have passed since the
    last, so re-orienting stays occasional.
    """

    def __init__(
        self,
        goal_tolerances=None,
        max_view_angle: float = 45.0,
        window: int = 5,
        min_steps_between_goals: int = 20,
        standoff: float | None = None,
        confidence: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the GSG.

        Args:
            goal_tolerances: See ``GraphGoalGenerator``.
            max_view_angle: The largest smoothed angle, in degrees, between the
                viewing direction and the surface normal that still counts as
                face-on. Saccades alone swing the instantaneous angle by tens
                of degrees, so keep this above their range.
            window: How many recent processed steps the view angle is averaged
                over before it is compared with ``max_view_angle``.
            min_steps_between_goals: Matching steps that must pass after a goal
                before another is proposed.
            standoff: Distance from the surface, in meters, to place the agent
                at; the patch's ``mean_depth`` when None, keeping the current
                viewing distance.
            confidence: The confidence of the goals; the motor system executes
                the highest-confidence goal when several LMs propose one.
            **kwargs: Passed on to ``GraphGoalGenerator``.
        """
        super().__init__(goal_tolerances, **kwargs)
        self.max_view_angle = max_view_angle
        self.window = window
        self.min_steps_between_goals = min_steps_between_goals
        self.standoff = standoff
        self.confidence = confidence
        # This step's angle, and the recent angles it is smoothed with.
        self._view_angle: float | None = None
        self._recent_angles: list[float] = []

    def reset(self):
        """Reset per-episode state and start a fresh view-angle log."""
        super().reset()
        self._view_angle = None
        self._recent_angles = []
        self.parent_lm.buffer.update_stats(
            dict(view_angle=[], smoothed_view_angle=[]),
            update_time=False,
            append=False,
            init_list=False,
        )

    @property
    def smoothed_view_angle(self) -> float | None:
        """The view angle averaged over the recent processed steps, or None."""
        if not self._recent_angles:
            return None
        return float(np.mean(self._recent_angles))

    def step(self, ctx: RuntimeContext, observations: Sequence[Message]):
        """Measure this step's view angle, log it, then run the GSG cycle."""
        self._view_angle = self._measure_view_angle(observations)
        if self._view_angle is not None:
            self._recent_angles = (self._recent_angles + [self._view_angle])[
                -self.window :
            ]
        smoothed = self.smoothed_view_angle
        self.parent_lm.buffer.update_stats(
            dict(
                view_angle=np.nan if self._view_angle is None else self._view_angle,
                smoothed_view_angle=np.nan if smoothed is None else smoothed,
            ),
            update_time=False,
            append=True,
            init_list=True,
        )
        super().step(ctx, observations)

    def _sensory_percept(self, observations: Sequence[Message]) -> Message | None:
        channel = self.parent_lm.buffer.get_first_sensory_input_channel()
        for percept in observations:
            if percept.sender_id == channel:
                return percept
        return None

    def _measure_view_angle(self, observations: Sequence[Message]) -> float | None:
        """The angle, in degrees, between the viewing direction and the normal.

        Returns:
            The angle, or None when the patch reported no usable normal or
            no viewing direction.
        """
        percept = self._sensory_percept(observations)
        if percept is None or not percept.process_features_in_lm:
            return None
        view = percept.non_morphological_features.get(VIEW_DIRECTION_FEATURE)
        if view is None:
            return None
        normal = np.asarray(percept.morphological_features["pose_vectors"])[0]
        view = np.asarray(view, dtype=float)
        norms = np.linalg.norm(view) * np.linalg.norm(normal)
        if norms == 0:
            return None
        cosine = -np.dot(view, normal) / norms
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def _check_output_goal_achieved(self, observations) -> bool:  # noqa: ARG002
        """Whether the first view after the goal is within the angle.

        The angle was measured from these observations in ``step``. This
        step's own angle is used, not the smoothed one: the goal is dropped
        right after this check, and the smoothing window still holds the
        steep views that triggered it.

        Returns:
            True once a goal is set and this step's view angle is within
            ``max_view_angle``.
        """
        if self.output_goal is None or self._view_angle is None:
            return False
        return self._view_angle <= self.max_view_angle

    def _check_need_new_output_goal(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        output_goal_achieved,
    ) -> bool:
        """Whether to propose a goal: the view is steep and enough steps passed.

        Returns:
            True when the smoothed view angle exceeds ``max_view_angle``, this
            step's own angle does too (so the jump is computed from a steep
            view, not a passing face-on one), and more than
            ``min_steps_between_goals`` matching steps passed since the last goal.
        """
        smoothed = self.smoothed_view_angle
        if output_goal_achieved or smoothed is None or self._view_angle is None:
            return False
        if smoothed <= self.max_view_angle or self._view_angle <= self.max_view_angle:
            return False
        steps_since = self.parent_lm.buffer.get_num_steps_post_output_goal_generated()
        return steps_since > self.min_steps_between_goals

    def _check_keep_current_output_goal(self) -> bool:
        """Goals are one-off attempts.

        Returns:
            False, always.
        """
        return False

    def _generate_goal(self, observations) -> Goal:
        """An agent pose on the patch's surface normal, looking back at the patch.

        Returns:
            The goal.
        """
        percept = self._sensory_percept(observations)
        normal = np.asarray(percept.morphological_features["pose_vectors"])[0]
        normal = normal / np.linalg.norm(normal)
        standoff = self.standoff
        if standoff is None:
            standoff = float(percept.non_morphological_features.get("mean_depth", 0.05))
        surface_loc = np.asarray(percept.location, dtype=float)
        logger.debug(
            f"Face-on goal from {self.parent_lm.learning_module_id}: view angle "
            f"{self._view_angle:.1f} deg, standoff {standoff:.3f} m"
        )
        return Goal(
            location=surface_loc + normal * standoff,
            morphological_features={
                # Look along the negated normal; roll is left unspecified.
                "pose_vectors": np.array(
                    [-normal, [np.nan] * 3, [np.nan] * 3], dtype=float
                ),
                "pose_fully_defined": None,
                "on_object": 1,
            },
            non_morphological_features=None,
            confidence=self.confidence,
            pass_message=True,
            sender_id=self.parent_lm.learning_module_id,
            sender_type="GSG",
            process_features_in_lm=True,
            goal_tolerances=None,
            info={
                "proposed_surface_loc": surface_loc,
                "view_angle": self._view_angle,
                "achieved": None,
                "matching_step_when_output_goal_set": None,
            },
        )
