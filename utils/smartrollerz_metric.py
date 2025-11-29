from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torchmetrics import Metric


_DATASET_DIMENSIONS: Dict[str, Tuple[float, float]] = {
    "CULane": (1640.0, 590.0),
    "Tusimple": (1280.0, 720.0),
    "CurveLanes": (2560.0, 1440.0),
    "Smartrollerz": (1364.0, 944.0),
}

_DATASET_PIXEL_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "CULane": (30.0, 20.0),
    "Tusimple": (20.0, 20.0),
    "CurveLanes": (25.0, 25.0),
    "Smartrollerz": (20.0, 20.0),
}


@dataclass
class DatasetGeometry:
    width: float
    height: float
    row_anchor: np.ndarray
    col_anchor: np.ndarray
    row_min_points: int
    col_min_points: int
    row_pixel_threshold: float
    col_pixel_threshold: float

    @classmethod
    def from_cfg(cls, cfg) -> "DatasetGeometry":
        width, height = _DATASET_DIMENSIONS.get(
            getattr(cfg, "dataset", ""),
            (float(getattr(cfg, "train_width", 1)), float(getattr(cfg, "train_height", 1))),
        )
        row_anchor = np.asarray(getattr(cfg, "row_anchor", []), dtype=np.float32)
        col_anchor = np.asarray(getattr(cfg, "col_anchor", []), dtype=np.float32)
        row_min_ratio = getattr(cfg, "f1_row_min_ratio", 0.5)
        col_min_ratio = getattr(cfg, "f1_col_min_ratio", 0.25)
        row_min_points = max(3, int(len(row_anchor) * row_min_ratio)) if len(row_anchor) else 0
        col_min_points = max(2, int(len(col_anchor) * col_min_ratio)) if len(col_anchor) else 0

        default_row_thr, default_col_thr = _DATASET_PIXEL_THRESHOLDS.get(
            getattr(cfg, "dataset", ""), (25.0, 25.0)
        )
        row_pixel_threshold = float(getattr(cfg, "f1_row_pixel_threshold", default_row_thr))
        col_pixel_threshold = float(getattr(cfg, "f1_col_pixel_threshold", default_col_thr))

        return cls(
            width=float(width),
            height=float(height),
            row_anchor=row_anchor,
            col_anchor=col_anchor,
            row_min_points=row_min_points,
            col_min_points=col_min_points,
            row_pixel_threshold=row_pixel_threshold,
            col_pixel_threshold=col_pixel_threshold,
        )


@dataclass
class LaneSamples:
    indices: np.ndarray
    values: np.ndarray

    @property
    def is_empty(self) -> bool:
        return self.indices.size == 0


class _LaneDecoder:
    def __init__(self, geometry: DatasetGeometry):
        self.geometry = geometry

    @staticmethod
    def _weighted_position(logits: torch.Tensor, dim: int) -> torch.Tensor:
        grid = torch.arange(logits.shape[dim], device=logits.device, dtype=logits.dtype) + 0.5
        view_shape = [1] * logits.ndim
        view_shape[dim] = -1
        probs = torch.softmax(logits, dim=dim)
        return (probs * grid.view(*view_shape)).sum(dim=dim) / max(logits.shape[dim] - 1, 1)

    def decode_rows(self, logits: torch.Tensor, exist_logits: torch.Tensor) -> List[LaneSamples]:
        if logits is None or exist_logits is None or self.geometry.row_anchor.size == 0:
            return []
        refined = self._weighted_position(logits, dim=0).clamp(0.0, 1.0)
        exist_mask = exist_logits.argmax(dim=0).to(dtype=torch.bool)
        return self._build_samples(refined, exist_mask, self.geometry.row_min_points)

    def decode_cols(self, logits: torch.Tensor, exist_logits: torch.Tensor) -> List[LaneSamples]:
        if logits is None or exist_logits is None or self.geometry.col_anchor.size == 0:
            return []
        refined = self._weighted_position(logits, dim=0).clamp(0.0, 1.0)
        exist_mask = exist_logits.argmax(dim=0).to(dtype=torch.bool)
        return self._build_samples(refined, exist_mask, self.geometry.col_min_points)

    @staticmethod
    def _build_samples(
        refined: torch.Tensor, mask: torch.Tensor, min_points: int
    ) -> List[LaneSamples]:
        refined_np = refined.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        num_cls, num_lane = refined_np.shape[0], refined_np.shape[1]
        lanes: List[LaneSamples] = []
        for lane_idx in range(num_lane):
            lane_mask = mask_np[:, lane_idx]
            indices = np.nonzero(lane_mask)[0]
            if indices.size < min_points:
                continue
            values = refined_np[indices, lane_idx]
            lanes.append(LaneSamples(indices=indices.astype(np.int32), values=values.astype(np.float32)))
        return lanes

    def decode_row_labels(self, labels: torch.Tensor) -> List[LaneSamples]:
        if labels is None or self.geometry.row_anchor.size == 0:
            return []
        labels_np = labels.detach().cpu().numpy()
        lanes: List[LaneSamples] = []
        for lane_idx in range(labels_np.shape[1]):
            lane_vals = labels_np[:, lane_idx]
            mask = lane_vals >= 0
            indices = np.nonzero(mask)[0]
            if indices.size < self.geometry.row_min_points:
                continue
            lanes.append(
                LaneSamples(
                    indices=indices.astype(np.int32),
                    values=lane_vals[indices].astype(np.float32),
                )
            )
        return lanes

    def decode_col_labels(self, labels: torch.Tensor) -> List[LaneSamples]:
        if labels is None or self.geometry.col_anchor.size == 0:
            return []
        labels_np = labels.detach().cpu().numpy()
        lanes: List[LaneSamples] = []
        for lane_idx in range(labels_np.shape[1]):
            lane_vals = labels_np[:, lane_idx]
            mask = lane_vals >= 0
            indices = np.nonzero(mask)[0]
            if indices.size < self.geometry.col_min_points:
                continue
            lanes.append(
                LaneSamples(
                    indices=indices.astype(np.int32),
                    values=lane_vals[indices].astype(np.float32),
                )
            )
        return lanes


def _lane_distance(
    pred: LaneSamples,
    gt: LaneSamples,
    pixel_scale: float,
    threshold: float,
    min_overlap: int,
) -> Optional[float]:
    common, pred_idx, gt_idx = np.intersect1d(pred.indices, gt.indices, return_indices=True)
    if common.size < min_overlap:
        return None
    diff = np.abs(pred.values[pred_idx] - gt.values[gt_idx]) * pixel_scale
    mean_diff = float(diff.mean()) if diff.size else None
    if mean_diff is None or mean_diff > threshold:
        return None
    return mean_diff


def _match_lanes(
    pred_lanes: Sequence[LaneSamples],
    gt_lanes: Sequence[LaneSamples],
    pixel_scale: float,
    threshold: float,
    min_overlap: int,
) -> Tuple[int, int, int]:
    if not gt_lanes and not pred_lanes:
        return 0, 0, 0
    used_gt: Dict[int, bool] = {}
    tp = 0
    fp = 0
    for pred_idx, pred_lane in enumerate(pred_lanes):
        best_gt = -1
        best_dist = float("inf")
        for gt_idx, gt_lane in enumerate(gt_lanes):
            if gt_idx in used_gt:
                continue
            dist = _lane_distance(pred_lane, gt_lane, pixel_scale, threshold, min_overlap)
            if dist is None:
                continue
            if dist < best_dist:
                best_dist = dist
                best_gt = gt_idx
        if best_gt >= 0:
            tp += 1
            used_gt[best_gt] = True
        else:
            fp += 1
    fn = len(gt_lanes) - len(used_gt)
    return tp, fp, fn


class UFLDV2F1Score(Metric):
    """Distributed-safe F1 score that mimics the matching logic from UFLDv2."""

    full_state_update = False

    def __init__(self, cfg, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.geometry = DatasetGeometry.from_cfg(cfg)
        self.decoder = _LaneDecoder(self.geometry)
        device = torch.device("cpu")
        self.add_state("total_tp", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")
        self.add_state("total_fp", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")
        self.add_state("total_fn", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")

    def update(self, outputs: Dict[str, torch.Tensor]) -> None:  # type: ignore[override]
        if outputs is None:
            return
        if "labels_row_float" not in outputs:
            return
        tp, fp, fn = self._accumulate_batch(outputs)
        self.total_tp += float(tp)
        self.total_fp += float(fp)
        self.total_fn += float(fn)

    def _accumulate_batch(self, outputs: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
        cls_out = outputs.get("cls_out")
        cls_out_ext = outputs.get("cls_out_ext")
        cls_out_col = outputs.get("cls_out_col")
        cls_out_col_ext = outputs.get("cls_out_col_ext")
        labels_row = outputs.get("labels_row_float")
        labels_col = outputs.get("labels_col_float")

        if cls_out is None or cls_out_ext is None or labels_row is None:
            return 0, 0, 0

        batch_size = cls_out.shape[0]
        total_tp = total_fp = total_fn = 0
        for batch_idx in range(batch_size):
            row_pred = self.decoder.decode_rows(cls_out[batch_idx], cls_out_ext[batch_idx])
            row_gt = self.decoder.decode_row_labels(labels_row[batch_idx])
            row_tp, row_fp, row_fn = _match_lanes(
                row_pred,
                row_gt,
                pixel_scale=self.geometry.width,
                threshold=self.geometry.row_pixel_threshold,
                min_overlap=self.geometry.row_min_points,
            )

            col_tp = col_fp = col_fn = 0
            if cls_out_col is not None and cls_out_col_ext is not None and labels_col is not None:
                col_pred = self.decoder.decode_cols(cls_out_col[batch_idx], cls_out_col_ext[batch_idx])
                col_gt = self.decoder.decode_col_labels(labels_col[batch_idx])
                col_tp, col_fp, col_fn = _match_lanes(
                    col_pred,
                    col_gt,
                    pixel_scale=self.geometry.height,
                    threshold=self.geometry.col_pixel_threshold,
                    min_overlap=self.geometry.col_min_points,
                )

            total_tp += row_tp + col_tp
            total_fp += row_fp + col_fp
            total_fn += row_fn + col_fn
        return total_tp, total_fp, total_fn

    def compute(self) -> Dict[str, float]:  # type: ignore[override]
        tp = float(self.total_tp)
        fp = float(self.total_fp)
        fn = float(self.total_fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
