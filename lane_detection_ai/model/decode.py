"""Decoding of the network outputs into lane points (pixel coordinates)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def _window_tables(grid_size: int, local_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid indices of the refinement window around every possible center cell.

    Returns:
        Indices [grid_size, window] (clamped to the grid) and a mask
        [grid_size, window] of the indices that are inside the grid.
    """
    centers = np.arange(grid_size)[:, None]
    start = np.maximum(centers - local_width, 0)
    end = np.minimum(centers + local_width, grid_size - 1)
    indices = start + np.arange(2 * local_width + 1)[None, :]
    valid = indices <= end
    return np.minimum(indices, grid_size - 1), valid


def _decode_axis(
    loc: np.ndarray,
    exist: np.ndarray,
    anchors: np.ndarray,
    scale: float,
    *,
    local_width: int,
    refine: bool,
    row_anchors: bool,
) -> list[np.ndarray]:
    """Decode the lanes of one anchor type.

    Args:
        loc: Location logits [grid, anchors, lanes].
        exist: Existence logits [2, anchors, lanes].
        anchors: Pixel position of every anchor [anchors].
        scale: Pixels per grid cell.
        local_width: Half width of the refinement window (in grid cells).
        refine: Refine the location with a softmax-weighted average over the
            window around the best grid cell (else the best cell is used).
        row_anchors: True for row anchors (point = (location, anchor)), False for
            column anchors (point = (anchor, location)).

    Returns:
        One int32 array [points, 2] of (x, y) pixel coordinates per lane.
    """
    grid_size = loc.shape[0]
    best_cell = loc.argmax(axis=0)  # [anchors, lanes]
    lane_exists = exist[1] > exist[0]  # same as exist.argmax(axis=0) == 1
    window_indices, window_valid = _window_tables(grid_size, local_width)

    lanes = []
    for lane in range(loc.shape[2]):
        anchor_idx = np.flatnonzero(lane_exists[:, lane])
        center = best_cell[anchor_idx, lane]

        if refine:
            indices = window_indices[center]  # [points, window]
            logits = np.take_along_axis(loc[:, anchor_idx, lane].T, indices, axis=1)
            logits = np.where(window_valid[center], logits, np.float32(-np.inf))
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            # float32 arithmetic like the original torch implementation
            position = (weights * indices.astype(np.float32)).sum(axis=1) + 0.5
        else:
            position = center.astype(np.float32) + 0.5

        location_px = np.rint(position * np.float32(scale)).astype(np.int32)
        anchor_px = anchors[anchor_idx]
        points = (location_px, anchor_px) if row_anchors else (anchor_px, location_px)
        lanes.append(np.stack(points, axis=1).astype(np.int32, copy=False))

    return lanes


def decode_lanes(
    pred: dict[str, np.ndarray],
    row_anchor: np.ndarray,
    col_anchor: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    local_width: int = 1,
    refine: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Decode the network outputs of the first image of a batch into lane points.

    Args:
        pred: Network outputs "loc_row" [B, grid, rows, lanes], "exist_row"
            [B, 2, rows, lanes], "loc_col" [B, grid, cols, lanes] and "exist_col"
            [B, 2, cols, lanes].
        row_anchor: Relative y position of the row anchors (0..1).
        col_anchor: Relative x position of the column anchors (0..1).
        image_width: Width of the image the points refer to.
        image_height: Height of the image the points refer to.
        local_width: Half width of the refinement window (in grid cells).
        refine: Refine the locations with a local softmax (more precise),
            otherwise only the best grid cell is used (faster).

    Returns:
        The lanes found on the row anchors and the lanes found on the column
        anchors, one int32 array [points, 2] of (x, y) coordinates per lane.
    """
    lanes = []
    for axis, anchor, anchor_size, location_size in (
        ("row", row_anchor, image_height, image_width),
        ("col", col_anchor, image_width, image_height),
    ):
        loc = np.asarray(pred[f"loc_{axis}"][0], dtype=np.float32)
        exist = np.asarray(pred[f"exist_{axis}"][0], dtype=np.float32)
        grid_size = loc.shape[0]
        scale = location_size / (grid_size - 1) if grid_size > 1 else location_size
        # Truncation (not rounding) of the anchor positions as in the original UFLD
        anchors = (np.asarray(anchor, dtype=np.float32) * anchor_size).astype(np.int32)
        lanes.append(
            _decode_axis(
                loc,
                exist,
                anchors,
                float(scale),
                local_width=local_width,
                refine=refine,
                row_anchors=axis == "row",
            )
        )
    return lanes[0], lanes[1]


def merge_lanes(
    row_lanes: list[np.ndarray], col_lanes: list[np.ndarray]
) -> list[np.ndarray | None]:
    """Combine the points of the row and column anchors of every lane.

    Args:
        row_lanes: Points per lane found on the row anchors.
        col_lanes: Points per lane found on the column anchors.

    Returns:
        Points [n, 2] per lane (row anchor points first), None if a lane has no
        points.
    """
    merged: list[np.ndarray | None] = []
    for lane in range(max(len(row_lanes), len(col_lanes))):
        parts = [
            lanes[lane]
            for lanes in (row_lanes, col_lanes)
            if lane < len(lanes) and len(lanes[lane]) > 0
        ]
        merged.append(np.vstack(parts) if parts else None)
    return merged
