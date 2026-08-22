# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from tbp.monty.attention.voxel_grid import VoxelGrid
from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.memento import Memento

__all__ = [
    "AttentionSystemTelemetry",
    "AttentionSystemTelemetryProtocol",
    "NoopAttentionSystemTelemetry",
    "goals_to_columns",
]


class AttentionSystemTelemetryProtocol(Protocol):
    def reset(self) -> None: ...

    def regions(self, regions: Sequence[AttentionRegion]) -> None: ...

    def proposed(self, grid: VoxelGrid) -> None: ...

    def voxel_grid(self, grid: VoxelGrid) -> None: ...

    def goals(self, goals: Sequence[Goal]) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopAttentionSystemTelemetry(AttentionSystemTelemetryProtocol):
    def reset(self) -> None:
        pass

    def regions(self, regions: Sequence[AttentionRegion]) -> None:
        pass

    def proposed(self, grid: VoxelGrid) -> None:
        pass

    def voxel_grid(self, grid: VoxelGrid) -> None:
        pass

    def goals(self, goals: Sequence[Goal]) -> None:
        pass

    def state_dict(self) -> Memento:
        # The empty schema, so consumers indexing these keys stay simple.
        return dict(voxel_grids=[], goals=[], regions=[], proposed=[])


class AttentionSystemTelemetry(AttentionSystemTelemetryProtocol):
    """Keeps each step's regions, grids and goals for the episode.

    Goals are kept as columns (see :func:`goal_columns`) rather than as the
    tens of thousands of ``Goal`` objects a step can carry.
    """

    def __init__(self) -> None:
        self._voxel_grids: list[VoxelGrid] = []
        self._goals: list[dict[str, object]] = []
        self._regions: list[list[AttentionRegion]] = []
        self._proposed: list[VoxelGrid] = []

    def reset(self) -> None:
        self._voxel_grids = []
        self._goals = []
        self._regions = []
        self._proposed = []

    def voxel_grid(self, grid: VoxelGrid) -> None:
        # Snapshot: decay updates the live grid's frame in place.
        self._voxel_grids.append(grid.copy())

    def goals(self, goals: Sequence[Goal]) -> None:
        self._goals.append(goals_to_columns(goals))

    def regions(self, regions: Sequence[AttentionRegion]) -> None:
        self._regions.append(list(regions))

    def proposed(self, grid: VoxelGrid) -> None:
        self._proposed.append(grid)

    def state_dict(self) -> Memento:
        # The grids ride along as objects; BufferEncoder flattens them at
        # serialization time (see voxel_grid.encode_voxel_grid).
        return dict(
            voxel_grids=list(self._voxel_grids),
            goals=list(self._goals),
            regions=list(self._regions),
            proposed=list(self._proposed),
        )


def goals_to_columns(goals: Sequence[Goal]) -> dict[str, np.ndarray]:
    """One step's goals as columns.

    Args:
        goals: The goals the attention system saw this step, each stamped
            with ``info["passed_attention_filter"]``.

    Returns:
        A dict with the following items:
         - "locations": (N, 3) (NaNs for unlocated goals)
         - "confidences": (N,)
         - "sender_ids": (N,)
         - "passed_attention_filter": (N,) bool
    """
    locations = np.array(
        [
            np.full(3, np.nan) if goal.location is None else goal.location
            for goal in goals
        ],
    ).reshape(-1, 3)
    confidences = np.array([goal.confidence for goal in goals])
    passed_attention_filter = np.array(
        [bool(goal.info.get("passed_attention_filter")) for goal in goals],
        dtype=bool,
    )
    sender_ids = np.array([goal.sender_id for goal in goals])

    return {
        "locations": locations,
        "confidences": confidences,
        "sender_ids": sender_ids,
        "passed_attention_filter": passed_attention_filter,
    }
