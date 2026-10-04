# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """One step of the trace followed by the most likely hypothesis (MLH).

    Attributes:
        graph_id: ID of the object of the MLH.
        location: Location of the MLH in the object's reference frame.
        rotation: Rotation of the MLH (maps body displacements into the object's
            reference frame).
        body_location: Sensed location in the body reference frame.
        mlh_gain: Evidence gained on this step by the MLH object's best hypothesis.
        competitor_gain: Largest evidence gained on this step by the best
            hypothesis of any other object.
        differential: mlh_gain - competitor_gain.
    """

    graph_id: str
    location: npt.NDArray[np.float64]
    rotation: Rotation
    body_location: npt.NDArray[np.float64]
    mlh_gain: float
    competitor_gain: float
    differential: float


class HypothesisTracer:
    """Records the trace of locations visited by the most likely hypothesis.

    At every evidence update, the tracer stores where the MLH is in its object's
    reference frame, and how much more evidence its object gained than the strongest
    competing object. If the MLH switches to a hypothesis that is not consistent
    with the one being traced (a different object, or a pose that differs by more
    than location_eta / rotation_eta_degrees), a new trace is started.

    Once the LM converges, the trace is frozen so that the most recent steps
    leading up to convergence can be used to tag hot spots on the object model.

    For diagnostics, every evidence update of an episode (including those after
    convergence) can also be saved to a JSON file per episode in history_dir.
    """

    def __init__(
        self,
        num_recent_locations: int = 5,
        location_eta: float = 0.01,
        rotation_eta_degrees: float = 10.0,
        history_dir: str | None = None,
    ):
        """Initialize the tracer.

        Args:
            num_recent_locations: Number of most recent trace steps (prior to
                convergence) used to tag hot spots.
            location_eta: Maximum distance (in meters) between the expected location
                of the traced hypothesis and the new MLH location for the trace to
                be continued.
            rotation_eta_degrees: Maximum angle between the rotation of the traced
                hypothesis and the new MLH rotation for the trace to be continued.
            history_dir: If set, directory in which the per-step history of each
                episode is saved by save_history.
        """
        self.num_recent_locations = num_recent_locations
        self.location_eta = location_eta
        self.rotation_eta_degrees = rotation_eta_degrees
        self.history_dir = Path(history_dir).expanduser() if history_dir else None
        self._num_saved_episodes = 0
        self.reset()

    def reset(self) -> None:
        self._trace: list[TraceStep] = []
        self._frozen_steps: list[TraceStep] | None = None
        self._history: list[dict[str, Any]] = []

    @property
    def trace(self) -> list[TraceStep]:
        return list(self._trace)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def is_frozen(self) -> bool:
        return self._frozen_steps is not None

    def update(
        self,
        prev_max_evidence: Mapping[str, float],
        curr_max_evidence: Mapping[str, float],
        mlh: Mapping,
        body_location: npt.NDArray[np.float64],
        inputs: Mapping[str, Any] | None = None,
        episode_step: int | None = None,
    ) -> None:
        """Add the current MLH to the trace.

        Args:
            prev_max_evidence: Max evidence of each object before the update.
            curr_max_evidence: Max evidence of each object after the update.
            mlh: The MLH after the update (with graph_id, location and rotation).
            body_location: Sensed location in the body reference frame.
            inputs: Optional summary of the inputs of this step, only stored in the
                history.
            episode_step: Optional number of Monty steps elapsed in the episode,
                only stored in the history.
        """
        graph_id = mlh["graph_id"]
        if graph_id not in prev_max_evidence or graph_id not in curr_max_evidence:
            # No evidence before this step to compute a gain from (e.g. the
            # hypothesis space was only just initialized).
            return

        gains = {
            other_id: float(curr_max_evidence[other_id] - prev_max_evidence[other_id])
            for other_id in curr_max_evidence
            if other_id in prev_max_evidence
        }
        mlh_gain = gains[graph_id]
        competitor_id = max(
            (other_id for other_id in gains if other_id != graph_id),
            key=gains.get,
            default=None,
        )
        competitor_gain = gains[competitor_id] if competitor_id is not None else 0.0

        step = TraceStep(
            graph_id=graph_id,
            location=np.array(mlh["location"], dtype=float),
            rotation=mlh["rotation"],
            body_location=np.array(body_location, dtype=float),
            mlh_gain=mlh_gain,
            competitor_gain=competitor_gain,
            differential=mlh_gain - competitor_gain,
        )
        new_trace = False
        if not self.is_frozen:
            if self._trace and not self._continues_trace(step):
                logger.debug(
                    f"MLH changed from {self._trace[-1].graph_id} to {graph_id}; "
                    "starting a new trace."
                )
                self._trace = []
            new_trace = len(self._trace) == 0
            self._trace.append(step)

        if self.history_dir is not None:
            self._history.append(
                {
                    "episode_step": episode_step,
                    "graph_id": graph_id,
                    "location": step.location.tolist(),
                    "rotation_matrix": np.asarray(step.rotation.as_matrix()).tolist(),
                    "body_location": step.body_location.tolist(),
                    "evidence": {k: float(v) for k, v in curr_max_evidence.items()},
                    "gains": gains,
                    "competitor_id": competitor_id,
                    "differential": step.differential,
                    "new_trace": new_trace,
                    "frozen": self.is_frozen,
                    "inputs": dict(inputs) if inputs is not None else None,
                }
            )

    def recent_steps(self) -> list[TraceStep]:
        return self._trace[-self.num_recent_locations :]

    def freeze(self) -> None:
        """Freeze the most recent steps of the trace, e.g. on convergence."""
        if not self.is_frozen:
            self._frozen_steps = self.recent_steps()

    def frozen_recent_steps(self) -> list[TraceStep]:
        return list(self._frozen_steps) if self.is_frozen else []

    def save_history(self, prefix: str, metadata: Mapping[str, Any]) -> Path | None:
        """Save the history of this episode to history_dir, if it is set.

        Args:
            prefix: Prefix of the file name, e.g. the ID of the LM.
            metadata: Information about the episode to save with the history.

        Returns:
            The path of the saved file, or None if history_dir is not set.
        """
        if self.history_dir is None:
            return None
        self.history_dir.mkdir(parents=True, exist_ok=True)
        path = (
            self.history_dir / f"{prefix}_episode_{self._num_saved_episodes:03d}.json"
        )
        with path.open("w") as f:
            json.dump(
                {
                    "episode": self._num_saved_episodes,
                    **metadata,
                    "steps": self._history,
                },
                f,
            )
        self._num_saved_episodes += 1
        return path

    def _continues_trace(self, step: TraceStep) -> bool:
        """Whether step follows from the same hypothesis as the last trace step.

        The last traced hypothesis is moved by the body displacement since that
        step, in the same way that the LM displaces its hypotheses, and compared
        to the new MLH.

        Returns:
            True if the object is the same and the pose matches within eta.
        """
        last = self._trace[-1]
        if step.graph_id != last.graph_id:
            return False
        expected_location = last.location + last.rotation.apply(
            step.body_location - last.body_location
        )
        if np.linalg.norm(expected_location - step.location) >= self.location_eta:
            return False
        angle = (last.rotation.inv() * step.rotation).magnitude()
        return bool(np.degrees(angle) < self.rotation_eta_degrees)
