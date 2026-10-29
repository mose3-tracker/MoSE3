"""Weighted rigid (SE(3)) fit: weighted centroids and cross-covariance, then Horn's quaternion solver.

One rotation per row of the weight matrix; the translation is ``dst_mean - R @ src_mean``.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion [w, x, y, z] (..., 4) -> rotation matrix (..., 3, 3)."""
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def weighted_centroids(
    src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor,
]:
    """Weighted centroids and cross-covariance for row-normalized weights w [Q, P].

    src/dst are shared by all rows ([P, 3]) or given per row ([Q, P, 3]).
    Returns (src_mean [Q, 3], dst_mean [Q, 3], None, None, cov [Q, 3, 3]).
    """
    assert w.ndim == 2, f"w must be [Q, P], got {tuple(w.shape)}"
    if src.ndim == 2:
        # shared points: one outer-product table for all rows
        src_mean = w @ src                                              # [Q, 3]
        dst_mean = w @ dst                                              # [Q, 3]
        outer = src.unsqueeze(-1) * dst.unsqueeze(-2)                   # [P, 3, 3]
        weighted_outer = torch.einsum("qk,kab->qab", w, outer)          # [Q, 3, 3]
        mean_outer = src_mean.unsqueeze(-1) * dst_mean.unsqueeze(-2)    # [Q, 3, 3]
        cov = weighted_outer - mean_outer
        return src_mean, dst_mean, None, None, cov

    assert src.ndim == 3 and dst.ndim == 3, (
        f"per-anchor src/dst must be [Q,P,3], got src={tuple(src.shape)} dst={tuple(dst.shape)}"
    )
    assert src.shape[0] == w.shape[0] and src.shape[1] == w.shape[1], (
        f"shape mismatch: w={tuple(w.shape)} src={tuple(src.shape)}"
    )
    # per-row points: covariance via batched matmul
    w_exp = w.unsqueeze(-1)                                             # [Q, P, 1]
    src_mean = (w_exp * src).sum(dim=1)                                 # [Q, 3]
    dst_mean = (w_exp * dst).sum(dim=1)                                 # [Q, 3]
    # weighted_outer[q, a, b] = Σ_p w[q,p] * src[q,p,a] * dst[q,p,b]
    weighted_src = w_exp * src                                          # [Q, P, 3]
    weighted_outer = weighted_src.transpose(-1, -2) @ dst               # [Q, 3, 3]
    mean_outer = src_mean.unsqueeze(-1) * dst_mean.unsqueeze(-2)        # [Q, 3, 3]
    cov = weighted_outer - mean_outer
    return src_mean, dst_mean, None, None, cov


def build_horn_matrix(cov: torch.Tensor) -> torch.Tensor:
    """Build Horn's 4x4 symmetric quaternion matrix from a 3x3 covariance."""
    Q = cov.shape[0]
    dev, dtype = cov.device, cov.dtype

    Sxx = cov[:, 0, 0]; Sxy = cov[:, 0, 1]; Sxz = cov[:, 0, 2]
    Syx = cov[:, 1, 0]; Syy = cov[:, 1, 1]; Syz = cov[:, 1, 2]
    Szx = cov[:, 2, 0]; Szy = cov[:, 2, 1]; Szz = cov[:, 2, 2]

    N = torch.zeros(Q, 4, 4, device=dev, dtype=dtype)
    N[:, 0, 0] = Sxx + Syy + Szz
    N[:, 0, 1] = Syz - Szy;        N[:, 1, 0] = N[:, 0, 1]
    N[:, 0, 2] = Szx - Sxz;        N[:, 2, 0] = N[:, 0, 2]
    N[:, 0, 3] = Sxy - Syx;        N[:, 3, 0] = N[:, 0, 3]
    N[:, 1, 1] = Sxx - Syy - Szz
    N[:, 1, 2] = Sxy + Syx;        N[:, 2, 1] = N[:, 1, 2]
    N[:, 1, 3] = Szx + Sxz;        N[:, 3, 1] = N[:, 1, 3]
    N[:, 2, 2] = -Sxx + Syy - Szz
    N[:, 2, 3] = Syz + Szy;        N[:, 3, 2] = N[:, 2, 3]
    N[:, 3, 3] = -Sxx - Syy + Szz
    return N


def solve_rotation_horn(
    cov: torch.Tensor,
    eig_min_gap: float = 1e-4,
    eig_jitter: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Rotation from Horn's matrix: its top eigenvector is the optimal quaternion (wxyz).

    ``ok`` is False when the top two eigenvalues are too close for a unique rotation; the
    ``raw_*`` / ``used_*`` entries are spectrum diagnostics.
    """
    N_raw = build_horn_matrix(cov)
    raw_eigenvalues = torch.linalg.eigvalsh(N_raw)  # ascending
    raw_scale = raw_eigenvalues.abs().amax(dim=-1).clamp_min(1e-12)
    raw_gaps = torch.diff(raw_eigenvalues, dim=-1)
    raw_min_gap = raw_gaps.min(dim=-1).values
    raw_mean_gap = raw_gaps.mean(dim=-1)
    raw_top_gap = raw_gaps[:, -1]
    raw_min_gap_ratio = raw_min_gap / raw_scale
    raw_mean_gap_ratio = raw_mean_gap / raw_scale
    raw_top_gap_ratio = raw_top_gap / raw_scale

    N_used = N_raw
    if eig_jitter > 0:
        diag_weights = torch.arange(4, device=cov.device, dtype=cov.dtype)
        diag_jitter = torch.diag(diag_weights).unsqueeze(0)
        N_used = N_raw + (eig_jitter * raw_scale)[:, None, None] * diag_jitter

    used_eigenvalues, eigenvectors = torch.linalg.eigh(N_used)  # ascending order

    quat = eigenvectors[:, :, -1]  # [Q, 4] = [w, x, y, z]

    # canonical sign: w >= 0
    # Scalar operands avoid constructing and uploading two CUDA tensors per fit batch.
    sign = torch.where(quat[:, 0:1] < 0, -1.0, 1.0).to(dtype=cov.dtype)
    quat = quat * sign

    quat = F.normalize(quat, dim=-1, eps=1e-12)

    R = quat_to_rotmat(quat)
    projected_ok = torch.isfinite(R).all(dim=(-2, -1))
    ok = projected_ok & (raw_top_gap_ratio > eig_min_gap)

    return {
        "R": R,
        "quat": quat,
        "ok": ok,
        "projected_ok": projected_ok,
        "raw_eigenvalues": raw_eigenvalues,
        "used_eigenvalues": used_eigenvalues,
        "raw_min_gap_ratio": raw_min_gap_ratio,
        "raw_mean_gap_ratio": raw_mean_gap_ratio,
        "raw_top_gap_ratio": raw_top_gap_ratio,
    }
