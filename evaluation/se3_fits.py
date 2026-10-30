"""Closed-form rigid fits from the frame-0 points ``src`` [N, 3] to every frame ``dst`` [T, N, 3], in float64.

Per-pixel fits: one SE(3) per query and frame, from a Horn fit over a group of support points
  - ``fit_knn_se3``  its k nearest 3D neighbours at frame 0, equal weights (any point tracker)
  - ``fit_emb_se3``  all other queries, weighted by a softmax over rigidity-embedding similarity (MoSE3)

Object fits: one SE(3) per frame, over all the points of one cluster
  - ``fit_cluster_uniform``   equal weights, Horn (``knn`` path)
  - ``fit_cluster_weighted``  given point weights, SVD Procrustes (``emb`` path)

Every fit returns R, t and ``ok``. Points missing at a frame (non-finite) are left out of that frame's fit,
and failed fits are left at the identity with ``ok`` = False.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mose3.utils.se3_fit import solve_rotation_horn


def _identity_fits(T: int, N: int, device):
    return (torch.eye(3, device=device, dtype=torch.float64).repeat(T, N, 1, 1),
            torch.zeros(T, N, 3, device=device, dtype=torch.float64),
            torch.zeros(T, N, dtype=torch.bool, device=device))


def _horn_per_query(w_norm, src_mean, dst_mean, src_c, dst_c, enough):
    """One Horn fit per query from row-normalized weights [N, K] and centred supports [N, K, 3]."""
    keep = (w_norm > 0).unsqueeze(-1)  # zero-weight neighbours must not leak into the covariance
    cov = torch.einsum("qk,qki,qkj->qij", w_norm, src_c * keep, dst_c * keep)
    horn = solve_rotation_horn(cov, eig_min_gap=1e-4)
    R, ok = horn["R"], horn["ok"] & enough
    return R, dst_mean - torch.einsum("nij,nj->ni", R, src_mean), ok


def fit_knn_se3(src: torch.Tensor, dst: torch.Tensor, k: int, valid: torch.Tensor):
    """Per query: uniform Horn fit over its k nearest frame-0 neighbours (itself included); points that are
    not ``valid`` [T, N] at frame t are dropped. Returns R [T, N, 3, 3], t [T, N, 3], ok [T, N]."""
    T, N = dst.shape[:2]
    k = max(3, min(k, N))
    idx = torch.cdist(src, src, p=2).topk(k, dim=-1, largest=False).indices    # [N, k]
    src_nb = src[idx].double()
    R_all, t_all, ok_all = _identity_fits(T, N, src.device)
    for t in range(T):
        dst_nb = dst[t][idx].double()
        w = (valid[t][idx] & torch.isfinite(dst_nb).all(dim=-1)).to(torch.float64)
        dst_nb = torch.where(torch.isfinite(dst_nb), dst_nb, torch.zeros_like(dst_nb))
        w_sum = w.sum(dim=-1, keepdim=True)
        w_norm = w / torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
        src_mean = (w_norm.unsqueeze(-1) * src_nb).sum(dim=1)
        dst_mean = (w_norm.unsqueeze(-1) * dst_nb).sum(dim=1)
        R, tr, ok = _horn_per_query(w_norm, src_mean, dst_mean, src_nb - src_mean.unsqueeze(1),
                                    dst_nb - dst_mean.unsqueeze(1), w.sum(dim=-1) >= 3)
        R_all[t] = torch.where(ok.view(-1, 1, 1), R, R_all[t])
        t_all[t] = torch.where(ok.view(-1, 1), tr, t_all[t])
        ok_all[t] = ok
    return R_all, t_all, ok_all


def emb_softmax_weights(emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """[N, N] row-normalized softmax(cos(emb_i, emb_j) / temperature), with the diagonal zeroed."""
    e = F.normalize(emb, dim=-1, eps=1e-6)
    logits = (e @ e.T) / max(temperature, 1e-6)
    self_mask = torch.eye(len(e), dtype=torch.bool, device=e.device)
    w = F.softmax(logits.masked_fill(self_mask, torch.finfo(logits.dtype).min), dim=-1)
    w = w.masked_fill(self_mask, 0.0)
    return w / (w.sum(dim=-1, keepdim=True) + 1e-8)


def fit_emb_se3(src: torch.Tensor, dst: torch.Tensor, emb: torch.Tensor, valid: torch.Tensor,
                temperature: float = 0.01):
    """Per query: Horn fit over all other queries, weighted by rigidity-embedding similarity ``emb`` [N, D]
    (no neighbourhood truncation). Returns R [T, N, 3, 3], t [T, N, 3], ok [T, N]."""
    T, N = dst.shape[:2]
    W = emb_softmax_weights(emb, temperature).double()
    src_d = src.double()
    R_all, t_all, ok_all = _identity_fits(T, N, src.device)
    for t in range(T):
        dst_t = dst[t].double()
        dst_t = torch.where(torch.isfinite(dst_t), dst_t, torch.zeros_like(dst_t))
        col_valid = valid[t].to(torch.float64) * torch.isfinite(dst[t]).all(dim=-1).to(torch.float64)
        w = W * col_valid.unsqueeze(0)
        w_sum = w.sum(dim=-1, keepdim=True)
        w_norm = w / torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
        src_mean = w_norm @ src_d
        dst_mean = w_norm @ dst_t
        R, tr, ok = _horn_per_query(w_norm, src_mean, dst_mean, src_d.unsqueeze(0) - src_mean.unsqueeze(1),
                                    dst_t.unsqueeze(0) - dst_mean.unsqueeze(1), w_sum.squeeze(-1) > 1e-6)
        R_all[t] = torch.where(ok.view(-1, 1, 1), R, R_all[t])
        t_all[t] = torch.where(ok.view(-1, 1), tr, t_all[t])
        ok_all[t] = ok
    return R_all, t_all, ok_all


def fit_cluster_uniform(src: torch.Tensor, dst: torch.Tensor, mask: torch.Tensor):
    """One uniform Horn fit per frame over the ``mask`` [N] points. Returns R [T, 3, 3], t [T, 3], ok [T]."""
    T = dst.shape[0]
    R_all, t_all, ok_all = (x[:, 0] for x in _identity_fits(T, 1, src.device))
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() < 3:
        return R_all, t_all, ok_all
    src, dst = src[idx].double(), dst[:, idx].double()
    for t in range(T):
        finite = torch.isfinite(dst[t]).all(dim=-1)
        if int(finite.sum()) < 3:
            continue
        s, d = src[finite], dst[t][finite]
        w = torch.full((len(s),), 1.0 / len(s), device=s.device, dtype=torch.float64)
        s_mean = (w.unsqueeze(-1) * s).sum(dim=0)
        d_mean = (w.unsqueeze(-1) * d).sum(dim=0)
        cov = torch.einsum("k,ki,kj->ij", w, s - s_mean, d - d_mean).unsqueeze(0)
        horn = solve_rotation_horn(cov)
        R = horn["R"][0]
        if not bool(horn["ok"][0]) or not torch.isfinite(R).all():
            continue
        R_all[t], t_all[t], ok_all[t] = R, d_mean - R @ s_mean, True
    return R_all, t_all, ok_all


def fit_cluster_weighted(src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor):
    """One weighted Procrustes (SVD) per frame, with point ``weights`` [N]. Returns R [T, 3, 3], t [T, 3], ok [T]."""
    T = dst.shape[0]
    src, dst = src.double(), dst.double()
    finite = torch.isfinite(dst).all(-1)                                   # [T, N]
    Wf = weights.double().clamp(min=0)[None, :] * finite.double()
    Wsum = Wf.sum(-1, keepdim=True)
    valid_frames = (Wsum.squeeze(-1) > 1e-6) & (finite.sum(-1) >= 3)
    Wn = Wf / Wsum.clamp_min(1e-12)

    s_mean = torch.einsum("tn,nd->td", Wn, src)
    d_mean = torch.einsum("tn,tnd->td", Wn, dst)
    H = torch.einsum("tn,tni,tnj->tij", Wn, src.unsqueeze(0) - s_mean.unsqueeze(1), dst - d_mean.unsqueeze(1))
    H = torch.where(torch.isfinite(H), H, torch.zeros_like(H))

    R_all, t_all, _ = (x[:, 0] for x in _identity_fits(T, 1, src.device))
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.transpose(-1, -2) @ U.transpose(-1, -2)
    flip = torch.det(R) < 0  # reflection -> rotation
    if flip.any():
        Vt2 = Vt.clone()
        Vt2[flip, -1] = -Vt2[flip, -1]
        R[flip] = Vt2[flip].transpose(-1, -2) @ U[flip].transpose(-1, -2)
    ok = valid_frames & torch.isfinite(R).all(dim=(-1, -2))
    t = d_mean - torch.einsum("tij,tj->ti", R, s_mean)
    R_all[ok], t_all[ok] = R[ok], t[ok]
    return R_all, t_all, ok
