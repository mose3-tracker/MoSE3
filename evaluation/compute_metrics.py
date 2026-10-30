"""Score SE(3) predictions on one benchmark cache: one ``summary.csv`` row per (clip, SE(3) path).

    python evaluation/compute_metrics.py --cache_dir data/se3_bench/ho3d \
        --preds_dir outputs/se3_bench/ho3d/preds/mose3 --out_dir outputs/se3_bench/ho3d/metrics/mose3 --clean_gt

Input: one ``<preds_dir>/<clip>.npz`` per cached clip, holding
  - ``pred_tracks_cam0``  [T, N, 3]  3D tracks of the cache's N query pixels, in the first camera's frame
  - ``rigid_emb``         [T, N, D]  rigidity embeddings (optional, needed for the ``emb`` path)

Each clip is scored in four steps:
  1. Alignment: one scale + xyz shift maps the tracks onto the GT, fitted on the visible GT-labelled points.
  2. Per-pixel SE(3) from frame 0 to every frame t, along each SE(3) path:
       - ``knn_K<k>``  uniform Horn fit over the k nearest 3D neighbours at frame 0 (any point tracker)
       - ``emb``       Horn fit over all other queries, weighted by rigidity-embedding similarity
  3. Object SE(3): HDBSCAN clusters the queries (``knn``: on the per-pixel SE(3)s at frames T/2 and T-1;
     ``emb``: on the rigidity embeddings). The cluster with the best IoU against the GT rigid mask is the
     object, and gets one SE(3) per frame.
  4. Metrics on the scored queries (``metric_mask``) at frames t > 0:
       - per pixel:  RRE, RRE-AUC@30, ADD and ADD-AUC
       - object:     IoU, RRE, ADD and ADD-AUC (``cluster_*`` columns)

``make_table.py`` then picks K per clip and averages over clips.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alignment import align_points_scale_xyz_shift
from se3_fits import fit_cluster_uniform, fit_cluster_weighted, fit_emb_se3, fit_knn_se3
from se3_metrics import (add_errors, add_summary, auc_at, best_iou_cluster, compose_se3, hdbscan_labels,
                         rotation_error_deg)

KNN_KS = (4, 8, 16, 32, 64, 128, 256)
ADD_AUC_BANDS_CM = (10, 20)
EMB_TEMPERATURE = 0.01          # per-pixel fit: softmax temperature on embedding similarity
CLUSTER_EMB_TEMPERATURE = 0.1   # object fit: softmax temperature on similarity to the cluster centroid
MIN_CLUSTER_SIZE = 30           # HDBSCAN minimum cluster size for the object-level clustering


def to_tensor(npz, key: str, device) -> torch.Tensor:
    return torch.from_numpy(npz[key]).to(device)


def broken_gt_mask(gt: torch.Tensor, has_gt: torch.Tensor, metric_mask: torch.Tensor, far_mad: float = 5.0):
    """HO3D GT tracks are unprojected depth: drop points from depth holes (at the camera centre) and from
    mask bleed onto the background (> ``far_mad`` MADs from the object's median radius)."""
    q = torch.nan_to_num(gt[0])
    degen = has_gt & (q.norm(dim=-1) < 0.05)
    far = torch.zeros_like(degen)
    good = metric_mask & ~degen
    if int(good.sum()) >= 8:
        rad = (q - q[good].median(dim=0).values).norm(dim=-1)
        med = rad[good].median()
        mad = 1.4826 * (rad[good] - med).abs().median() + 1e-9
        far = metric_mask & ~degen & (rad > med + far_mad * mad)
    return degen | far


def surface_points(gt: torch.Tensor, obj_pose: torch.Tensor, metric_mask: torch.Tensor) -> np.ndarray:
    """ADD reference cloud: the GT frame-0 points of the scored object, in that object's frame."""
    poses = obj_pose[:, metric_mask].double()
    if not torch.allclose(poses, poses[:, :1].expand_as(poses), atol=1e-5, rtol=0):
        raise ValueError("metric_mask must select a single rigid body")
    points = gt[0, metric_mask].double()
    points = points[torch.isfinite(points).all(-1)]
    inv = torch.linalg.inv(poses[0, 0])
    return (points @ inv[:3, :3].T + inv[:3, 3]).cpu().numpy()


def align_to_gt(pred: torch.Tensor, gt: torch.Tensor, use: torch.Tensor, max_points: int = 10000) -> torch.Tensor:
    """Apply the scale + xyz shift that best maps ``pred`` [T, N, 3] onto ``gt`` at the ``use`` [T, N] points."""
    use = use & torch.isfinite(pred).all(-1) & torch.isfinite(gt).all(-1)
    if use.any():
        src, tgt = pred[use], gt[use]
        if len(src) >= 2:
            src, tgt = src[::2], tgt[::2]
    else:
        both = torch.isfinite(pred.reshape(-1, 3)).all(-1) & torch.isfinite(gt.reshape(-1, 3)).all(-1)
        src, tgt = pred.reshape(-1, 3)[both][::4], gt.reshape(-1, 3)[both][::4]
    if len(src) == 0:
        return pred
    if len(src) > max_points:
        step = math.ceil(len(src) / max_points)
        src, tgt = src[::step], tgt[::step]
    scale, shift = align_points_scale_xyz_shift(src, tgt, torch.ones_like(tgt.norm(dim=-1)))
    return scale * pred + shift


def pose_add(R, t, ok, pose0, gt_pose, points):
    """ADD of fitted warps (R, t, ok) at frames 1.. applied to the GT frame-0 object pose ``pose0``."""
    pred_pose = compose_se3(R, t) @ pose0.to(R)
    return add_errors(pred_pose.cpu().numpy().reshape(-1, 4, 4), gt_pose.to(R).cpu().numpy().reshape(-1, 4, 4),
                      points, ok.cpu().numpy().reshape(-1))


def score_per_pixel(R, t, ok, clip) -> dict:
    """RRE, AUC_rot@30 and ADD / ADD-AUC of per-pixel warps [T, N] over the scored queries at t > 0."""
    m = clip["metric_mask"]
    valid = m[None] & ok & clip["after_first"]
    rre = rotation_error_deg(R, clip["R_gt"].to(R.dtype))[valid]
    add = pose_add(R[1:, m], t[1:, m], ok[1:, m], clip["obj_pose"][0, m], clip["obj_pose"][1:, m], clip["points"])
    return {"rre_pp_mean": float(rre.mean()) if valid.any() else float("nan"),
            "auc_rot_pp_30": auc_at(rre, 30.0),
            **add_summary(add, ADD_AUC_BANDS_CM)}


def score_object(feat, clip, emb_weighted: bool) -> dict:
    """Cluster ``feat`` [N, D], match the best-IoU cluster to the GT mask, and score one SE(3) per frame on it."""
    labels = hdbscan_labels(feat, min_cluster_size=clip["min_cluster_size"])
    iou, label, size = best_iou_cluster(labels, clip["gt_object"])
    out = {"cluster_iou": iou, "cluster_rre_mean": float("nan"),
           **add_summary(np.zeros(0), ADD_AUC_BANDS_CM, prefix="cluster_")}
    if label < 0 or size < 3:
        return out
    tracks = clip["tracks"]
    in_cluster = torch.from_numpy(labels == label).to(tracks.device)
    if emb_weighted:  # weight members by similarity to the cluster's mean embedding
        feat_n = F.normalize(feat, dim=-1, eps=1e-6)
        centroid = feat_n[in_cluster].mean(0)
        logits = (feat_n @ (centroid / (centroid.norm() + 1e-8))) / CLUSTER_EMB_TEMPERATURE
        R, t, ok = fit_cluster_weighted(tracks[0], tracks, torch.exp(logits - logits.max()) * in_cluster.float())
    else:
        R, t, ok = fit_cluster_uniform(tracks[0], tracks, in_cluster)

    # the GT pose of the cluster's first scored point stands for the object
    members = in_cluster.nonzero(as_tuple=False).squeeze(-1)
    scored = members[clip["metric_mask"][members]]
    rep = int(scored[0] if len(scored) else members[0])
    valid = ok & clip["after_first"][:, 0]
    if valid.any():
        out["cluster_rre_mean"] = float(rotation_error_deg(R[valid].float(), clip["R_gt"][valid, rep]).mean())
    add = pose_add(R[1:], t[1:], ok[1:], clip["obj_pose"][0, rep], clip["obj_pose"][1:, rep], clip["points"])
    out.update(add_summary(add, ADD_AUC_BANDS_CM, prefix="cluster_"))
    return out


def se3_feature(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-pixel SE(3) [N, 3, 3], [N, 3] -> [N, 12] clustering feature."""
    return torch.cat([R.reshape(len(R), 9), t], dim=-1).float()


@torch.no_grad()
def evaluate_clip(cache_path: Path, pred_path: Path, clean_gt: bool, device, adaptive_min_cluster: bool = False) -> list:
    cache, pred = np.load(cache_path), np.load(pred_path)
    gt = to_tensor(cache, "tracks_cam", device).float()                 # [T, N, 3] GT tracks, first camera's frame
    obj_pose = to_tensor(cache, "obj_se3_per_point", device).float()    # [T, N, 4, 4] GT pose of each query's body
    vis = to_tensor(cache, "vis", device).bool()
    has_gt = to_tensor(cache, "has_gt_se3", device).bool()
    metric_mask = (to_tensor(cache, "metric_mask", device).bool() if "metric_mask" in cache.files
                   else has_gt.clone())
    # GT rigid mask for the IoU: the moving actor (iTACO) or the labelled object (HO3D, YCBInEOAT)
    actor = cache["actor_idx"] if "actor_idx" in cache.files else None
    gt_object = actor == 0 if actor is not None and not np.all(actor == 0) else has_gt.cpu().numpy()
    align_gt = has_gt
    if clean_gt:
        bad = broken_gt_mask(gt, has_gt, metric_mask)
        align_gt, metric_mask = has_gt & ~bad, metric_mask & ~bad
        gt_object = gt_object & ~bad.cpu().numpy()

    tracks = align_to_gt(to_tensor(pred, "pred_tracks_cam0", device).float(), gt, vis & align_gt[None])
    finite = torch.isfinite(tracks).all(-1)
    T = len(tracks)
    pose0_inv = torch.linalg.inv(obj_pose[0].double())
    clip = {
        "tracks": tracks, "metric_mask": metric_mask, "gt_object": gt_object.astype(bool), "obj_pose": obj_pose,
        "R_gt": torch.einsum("tnij,njk->tnik", obj_pose.double(), pose0_inv)[..., :3, :3].float(),  # frame 0 -> t
        "after_first": torch.arange(T, device=device)[:, None] > 0,
        "points": surface_points(gt, obj_pose, metric_mask),
    }
    clip["min_cluster_size"] = (max(2, min(MIN_CLUSTER_SIZE, int(clip["gt_object"].sum()))) if adaptive_min_cluster
                                else MIN_CLUSTER_SIZE)

    rows = []
    for k in KNN_KS:
        R, t, ok = fit_knn_se3(tracks[0], tracks, k, finite)
        feat = torch.cat([se3_feature(R[T // 2], t[T // 2]), se3_feature(R[-1], t[-1])], dim=-1)
        rows.append({"se3_path": f"knn_K{k}", **score_per_pixel(R, t, ok, clip),
                     **score_object(feat, clip, emb_weighted=False)})
    if "rigid_emb" in pred.files and pred["rigid_emb"].size:
        emb = to_tensor(pred, "rigid_emb", device)[0].float()   # query-frame embedding [N, D]
        R, t, ok = fit_emb_se3(tracks[0], tracks, emb, finite, EMB_TEMPERATURE)
        rows.append({"se3_path": "emb", **score_per_pixel(R, t, ok, clip),
                     **score_object(F.normalize(emb, dim=-1, eps=1e-6), clip, emb_weighted=True)})
    return [{"sample_name": cache_path.stem, **row} for row in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True, help="Benchmark cache: one <clip>.npz per clip.")
    p.add_argument("--preds_dir", required=True, help="Predictions: one <clip>.npz per cached clip.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clean_gt", action="store_true",
                   help="Drop broken GT points (depth holes, mask bleed) from the alignment and the scores. "
                        "Used for HO3D and iTACO.")
    p.add_argument("--adaptive_min_cluster", action="store_true",
                   help=f"On clips whose object has fewer than {MIN_CLUSTER_SIZE} scored queries, lower the HDBSCAN "
                        "minimum cluster size to that number so the object can form a cluster. "
                        "Off for the reported results.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    clips = sorted(Path(args.cache_dir).glob("*.npz"))
    if not clips:
        raise SystemExit(f"no .npz clips in {args.cache_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, cache_path in enumerate(clips):
        clip_rows = evaluate_clip(cache_path, Path(args.preds_dir) / cache_path.name, args.clean_gt,
                                  torch.device(args.device), args.adaptive_min_cluster)
        rows += clip_rows
        last = clip_rows[-1]
        print(f"[{i + 1}/{len(clips)}] {cache_path.stem}  {last['se3_path']}: RRE {last['rre_pp_mean']:.2f} deg  "
              f"ADD {100 * last['add_mean_m']:.2f} cm")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
