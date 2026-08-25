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

import pandas as pd

from tbp.monty.attention.voxel_grid import WEIGHT_FEATURE, VoxelGrid
from tbp.monty.cmp import MIN_ATTENTION_WEIGHT


def _check_can_merge(grid_a: VoxelGrid, grid_b: VoxelGrid) -> None:
    """Check that two grids are compatible for merging.

    Args:
        grid_a: The grid being merged into.
        grid_b: The grid being merged in.
    """
    assert grid_a.voxel_size == grid_b.voxel_size, "Voxel sizes must match for merging."


class VoxelGridMerge(Protocol):
    def __call__(self, grid_a: VoxelGrid, grid_b: VoxelGrid) -> VoxelGrid: ...


class Union(VoxelGridMerge):
    """Merge grid_b into grid_a, taking the union of their voxels.

    Every voxel occupied in either grid is occupied in the result. Where the
    grids overlap, grid_b's weight wins; making that overlap policy
    configurable is left for later.
    """

    def __call__(
        self,
        grid_a: VoxelGrid,
        grid_b: VoxelGrid,
    ) -> VoxelGrid:
        """Merge grid_b into grid_a.

        Args:
            grid_a: The grid being merged into.
            grid_b: The grid being merged in; its weights win on overlap.

        Returns:
            The merged grid.
        """
        _check_can_merge(grid_a, grid_b)

        a_to_merge: pd.DataFrame = grid_a.to_pandas()
        b_to_merge: pd.DataFrame = grid_b.to_pandas()

        overlap: pd.MultiIndex = grid_a.index.intersection(grid_b.index)
        if len(overlap):
            a_to_merge = a_to_merge.drop(overlap, errors="ignore")

        df = pd.concat([a_to_merge, b_to_merge])
        return VoxelGrid(grid_a.voxel_size, df)


class InhibitionFlipsGrid(VoxelGridMerge):
    """Union, unless the proposal inhibits anywhere: then all is inhibited.

    A single negative weight among the proposed voxels (grid_b) flips the
    whole merged grid: the result holds every voxel of either grid, all at
    ``MIN_ATTENTION_WEIGHT``. With no inhibition proposed this is a plain
    :class:`Union`.
    """

    def __init__(self) -> None:
        self._union = Union()

    def __call__(self, grid_a: VoxelGrid, grid_b: VoxelGrid) -> VoxelGrid:
        """Merge grid_b into grid_a.

        Args:
            grid_a: The grid being merged into.
            grid_b: The grid being merged in.

        Returns:
            The union of both grids' voxels: at grid_b's weights on overlap
            when nothing in grid_b is negative, else all at
            ``MIN_ATTENTION_WEIGHT``.
        """
        merged = self._union(grid_a, grid_b)
        if (grid_b.to_pandas()[WEIGHT_FEATURE] < 0).any():
            inhibited = merged.to_pandas().assign(
                **{WEIGHT_FEATURE: MIN_ATTENTION_WEIGHT}
            )
            return VoxelGrid(grid_a.voxel_size, inhibited)
        return merged
