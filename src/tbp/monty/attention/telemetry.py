# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol

from tbp.monty.attention.voxel_grid import VoxelGrid
from tbp.monty.memento import Memento

__all__ = [
    "AttentionSystemTelemetry",
    "AttentionSystemTelemetryProtocol",
    "NoopAttentionSystemTelemetry",
]


class AttentionSystemTelemetryProtocol(Protocol):
    def reset(self) -> None: ...

    def proposed(self, grid: VoxelGrid) -> None: ...

    def voxel_grid(self, grid: VoxelGrid) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopAttentionSystemTelemetry(AttentionSystemTelemetryProtocol):
    def reset(self) -> None:
        pass

    def proposed(self, grid: VoxelGrid) -> None:
        pass

    def voxel_grid(self, grid: VoxelGrid) -> None:
        pass

    def state_dict(self) -> Memento:
        # The empty schema, so consumers indexing these keys stay simple.
        return dict(voxel_grids=[], proposed=[])


class AttentionSystemTelemetry(AttentionSystemTelemetryProtocol):
    """Keeps each step's grids for the episode: the proposals and the result."""

    def __init__(self) -> None:
        self._voxel_grids: list[VoxelGrid] = []
        self._proposed: list[VoxelGrid] = []

    def reset(self) -> None:
        self._voxel_grids = []
        self._proposed = []

    def voxel_grid(self, grid: VoxelGrid) -> None:
        # Snapshot: decay updates the live grid's frame in place.
        self._voxel_grids.append(grid.copy())

    def proposed(self, grid: VoxelGrid) -> None:
        self._proposed.append(grid)

    def state_dict(self) -> Memento:
        # The grids ride along as objects; BufferEncoder flattens them at
        # serialization time (see voxel_grid.encode_voxel_grid).
        return dict(
            voxel_grids=list(self._voxel_grids),
            proposed=list(self._proposed),
        )
