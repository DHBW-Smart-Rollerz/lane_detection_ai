import numpy as np
import pytest
import torch

from lane_detection_ai.model.decode import decode_lanes, merge_lanes


def reference_decode(pred, row_anchor, col_anchor, width, height, local_width=1):
    """Original UFLDv2 decoding (one anchor at a time) as reference."""
    lanes = []
    for axis, anchors, anchor_size, loc_size in (
        ("row", row_anchor, height, width),
        ("col", col_anchor, width, height),
    ):
        loc = torch.from_numpy(pred[f"loc_{axis}"])
        exist = torch.from_numpy(pred[f"exist_{axis}"]).argmax(1)
        grid = loc.shape[1]
        best = loc.argmax(1)
        anchor_px = (torch.as_tensor(anchors, dtype=torch.float32) * anchor_size).int()
        for lane in range(loc.shape[3]):
            points = []
            for k in range(loc.shape[2]):
                if not exist[0, k, lane]:
                    continue
                center = int(best[0, k, lane])
                ind = torch.arange(
                    max(0, center - local_width),
                    min(grid - 1, center + local_width) + 1,
                )
                pos = (loc[0, ind, k, lane].softmax(0) * ind.float()).sum() + 0.5
                pos_px = int(torch.round(pos * (loc_size / (grid - 1))))
                anchor = int(anchor_px[k])
                points.append((pos_px, anchor) if axis == "row" else (anchor, pos_px))
            lanes.append(np.array(points, dtype=np.int32).reshape(-1, 2))
    return lanes


def random_pred(rng, grid_row, rows, grid_col, cols, lanes):
    return {
        "loc_row": rng.normal(size=(1, grid_row, rows, lanes)).astype(np.float32) * 5,
        "exist_row": rng.normal(size=(1, 2, rows, lanes)).astype(np.float32),
        "loc_col": rng.normal(size=(1, grid_col, cols, lanes)).astype(np.float32) * 5,
        "exist_col": rng.normal(size=(1, 2, cols, lanes)).astype(np.float32),
    }


@pytest.mark.parametrize("seed", range(20))
def test_decode_matches_reference(seed):
    rng = np.random.default_rng(seed)
    grid_row, grid_col = rng.choice([2, 3, 20, 100], size=2)
    rows, cols, lanes = rng.integers(1, 30), rng.integers(1, 30), rng.integers(1, 5)
    pred = random_pred(rng, grid_row, rows, grid_col, cols, lanes)
    row_anchor = np.linspace(100, 1540, rows) / 1550
    col_anchor = np.linspace(0, 1, cols)

    row_lanes, col_lanes = decode_lanes(pred, row_anchor, col_anchor, 640, 480)
    expected = reference_decode(pred, row_anchor, col_anchor, 640, 480)

    assert len(row_lanes) == len(col_lanes) == lanes
    for actual, reference in zip(row_lanes + col_lanes, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


def test_decode_argmax_uses_best_cell_center():
    pred = {
        "loc_row": np.zeros((1, 11, 2, 1), np.float32),
        "exist_row": np.array([[[[0.0], [1.0]], [[1.0], [0.0]]]], np.float32),
        "loc_col": np.zeros((1, 11, 1, 1), np.float32),
        "exist_col": np.zeros((1, 2, 1, 1), np.float32),
    }
    pred["loc_row"][0, 4, :, 0] = 10.0  # best cell 4 for both row anchors

    row_lanes, col_lanes = decode_lanes(
        pred, np.array([0.5, 1.0]), np.array([0.5]), 100, 200, refine=False
    )

    # Only the first row anchor exists: x = (4 + 0.5) * 100 / 10, y = 0.5 * 200
    np.testing.assert_array_equal(row_lanes[0], [[45, 100]])
    assert col_lanes[0].shape == (0, 2)


def test_merge_lanes():
    a = np.array([[1, 2]], np.int32)
    b = np.array([[3, 4], [5, 6]], np.int32)
    empty = np.empty((0, 2), np.int32)

    left, center, right = merge_lanes([a, empty, empty], [b, b, empty])

    np.testing.assert_array_equal(left, [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(center, b)
    assert right is None
