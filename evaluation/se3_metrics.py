"""SE(3) metrics, and the clustering behind the object level.

Rotation:
  - ``rotation_error_deg``  RRE: geodesic angle between the predicted and GT rotations, in degrees
  - ``auc_at``              RRE-AUC: area under the accuracy curve up to a threshold, in [0, 1]

ADD (pose error on the object's points):
  - ``add_errors``   ADD of a predicted object pose P against the GT pose G: ``mean_x ||P x - G x||`` over a
                     fixed object-frame point cloud x, in metres; failed fits get inf
  - ``add_summary``  mean ADD over successful fits, and ADD-AUC (percent) with failed fits counted as misses,
                     following YCB-Video / PoseCNN (``vocap_auc``):
                     https://github.com/wenbowen123/iros20-6d-pose-tracking/blob/master/eval_ycb.py

Clustering:
  - ``hdbscan_labels``    HDBSCAN labels of per-point features; -1 = noise
  - ``best_iou_cluster``  the cluster with the highest IoU against the GT mask
"""
from __future__ import annotations

import numpy as np
import torch


def rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between rotations [..., 3, 3], in degrees."""
    trace = (R_gt.transpose(-1, -2) @ R_pred).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)).rad2deg()


def auc_at(errors: torch.Tensor, thresh: float, n_steps: int = 300) -> float:
    """Area under the recall curve recall(x) = mean(errors <= x) on [0, thresh], normalized to [0, 1]."""
    if errors.numel() == 0:
        return float("nan")
    errors = errors.detach().float().flatten()
    steps = torch.linspace(0.0, thresh, n_steps, device=errors.device)
    return float((errors.unsqueeze(-1) <= steps.unsqueeze(0)).float().mean(dim=0).mean().item())


def compose_se3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] rotations and [..., 3] translations -> [..., 4, 4]."""
    T = torch.zeros((*R.shape[:-2], 4, 4), dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def _is_rigid(poses: np.ndarray) -> np.ndarray:
    """[M, 4, 4] -> [M] finite proper rigid transforms, up to roundoff."""
    valid = np.isfinite(poses).all(axis=(1, 2))
    R = poses[valid, :3, :3]
    valid[valid] = (np.isclose(poses[valid, 3], [0, 0, 0, 1], atol=1e-6, rtol=0).all(axis=1)
                    & np.isclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-4, rtol=0).all(axis=(1, 2))
                    & np.isclose(np.linalg.det(R), 1, atol=1e-4, rtol=0))
    return valid


def add_errors(pred_poses: np.ndarray, gt_poses: np.ndarray, points: np.ndarray, pred_ok: np.ndarray) -> np.ndarray:
    """ADD (m) of object poses [M, 4, 4] on object-frame points [P, 3]; failed predictions get inf."""
    pred_poses, gt_poses = pred_poses.astype(np.float64), gt_poses.astype(np.float64)
    if not _is_rigid(gt_poses).all():
        raise ValueError("GT poses must be finite proper rigid transforms")
    add = np.full(len(pred_poses), np.inf)
    for i in np.flatnonzero(_is_rigid(pred_poses) & pred_ok):
        # both posed clouds in the prediction's object frame
        R_inv = pred_poses[i, :3, :3].T
        gt_in_pred = points @ (R_inv @ gt_poses[i, :3, :3]).T + R_inv @ (gt_poses[i, :3, 3] - pred_poses[i, :3, 3])
        add[i] = np.linalg.norm(gt_in_pred - points, axis=1).mean()
    return add


def vocap_auc(errors: np.ndarray, max_m: float) -> float:
    """VOCap AUC (percent) of distance errors up to ``max_m``; non-finite errors count as failures."""
    values = np.sort(np.where(np.isfinite(errors), errors, np.inf))
    if values.size == 0:
        return float("nan")
    accepted = values < max_m
    if not accepted.any():
        return 0.0
    recall = (np.arange(1, len(values) + 1, dtype=np.float64) / len(values))[accepted]
    distances = np.concatenate(([0.0], values[accepted], [max_m]))
    recalls = np.concatenate(([0.0], recall, [recall[-1]]))
    return float(np.sum(np.diff(distances) * recalls[1:]) / max_m * 100.0)


def add_summary(add: np.ndarray, auc_bands_cm, prefix: str = "") -> dict:
    """Mean ADD (m) over successful fits, and ADD-AUC (%) per band with failures kept as misses."""
    ok = np.isfinite(add)
    out = {f"{prefix}add_mean_m": float(add[ok].mean()) if ok.any() else float("nan")}
    for cm in auc_bands_cm:
        out[f"{prefix}add_auc_{cm}cm_pct"] = vocap_auc(add, cm / 100)
    return out


def hdbscan_labels(features: torch.Tensor, min_cluster_size: int = 30) -> np.ndarray:
    """HDBSCAN (EOM selection) labels [N] of per-point features [N, D]; -1 = noise."""
    import hdbscan
    feats = features.detach().cpu().numpy().astype(np.float32)
    if len(feats) < max(4, min_cluster_size):
        return np.full(len(feats), -1, dtype=np.int64)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), cluster_selection_method="eom")
    return clusterer.fit_predict(feats).astype(np.int64)


def best_iou_cluster(labels: np.ndarray, gt_mask: np.ndarray):
    """The cluster with the highest IoU against the GT mask [N]: (iou, label, size); (nan, -1, 0) if none."""
    best_iou, best_label, best_size = float("nan"), -1, 0
    if gt_mask.any():
        for label in np.unique(labels[labels != -1]):
            cluster = labels == label
            iou = (cluster & gt_mask).sum() / (cluster | gt_mask).sum()
            if np.isnan(best_iou) or iou > best_iou:
                best_iou, best_label, best_size = float(iou), int(label), int(cluster.sum())
    return best_iou, best_label, best_size
