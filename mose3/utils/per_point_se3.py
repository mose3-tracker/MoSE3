"""Per-pixel SE(3) (query frame -> frame t) from 3D tracks and rigidity embeddings.

Each query pixel gets a closed-form weighted Horn fit over a shared pool of query-frame pixels.
Weights are a softmax over rigidity-embedding similarity (Eq. 1-2 of the paper), times a Gaussian
on pixel distance when ``sigma_px`` > 0. Tracks must be in one common frame, e.g. the first camera's.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .se3_fit import solve_rotation_horn


def _sample_ref_indices(
    n_pixels: int, n_ref: int, generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    n_ref = min(n_ref, n_pixels)
    perm = torch.randperm(n_pixels, generator=generator, device=device)
    return perm[:n_ref]


@torch.no_grad()
def fit_per_pixel_se3(
    tracks: torch.Tensor,
    rigidity_emb: torch.Tensor,
    query_idx: int = 0,
    num_ref_pts: int = 12000,
    temperature: float = 0.01,
    sigma_px: float = 0.0,
    chunk_size: int = 4096,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Per-pixel SE(3) from the query frame to every frame t.

    Args:
        tracks: [T, H, W, 3] 3D tracks of the query pixels, in one common (world) frame.
        rigidity_emb: [T, H, W, D] rigidity embeddings; only the query frame's is used.
        query_idx: the query frame.
        num_ref_pts: number of query-frame pixels sampled at random as the shared pool.
        temperature: softmax temperature on rigidity-embedding cosine similarity.
        sigma_px: std, in pixels, of the Gaussian weight on query-frame pixel distance; 0 = off.
        chunk_size: query pixels fitted at once.
        seed: seed for sampling the pool.
        device: compute device (default: that of ``tracks``).

    Returns a dict of CPU tensors: ``quat`` [T, H, W, 4] (wxyz) and ``trans`` [T, H, W, 3] with
    x_t = R(quat) x_query + trans, and ``ok`` [T, H, W] (the fit was well conditioned).

    Query-to-reference weights depend only on the query frame. Reuse each chunk's
    weights and source centroid across target frames without caching a full P-by-reference matrix.
    """
    assert tracks.shape[:3] == rigidity_emb.shape[:3], (
        f"shape mismatch tracks={tuple(tracks.shape)} emb={tuple(rigidity_emb.shape)}"
    )
    if device is None:
        device = tracks.device

    T, H, W, _ = tracks.shape
    P = H * W
    assert 0 <= query_idx < T

    src_pts = tracks[query_idx].reshape(P, 3).to(device).float()
    src_emb = rigidity_emb[query_idx].reshape(P, -1).to(device).float()
    src_emb_n = F.normalize(src_emb, dim=-1, eps=1e-6)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    ref_idx = _sample_ref_indices(P, num_ref_pts, gen, device)
    ref_src = src_pts.index_select(0, ref_idx)
    ref_emb_n = src_emb_n.index_select(0, ref_idx)
    ref_h, ref_w = (ref_idx // W).float(), (ref_idx % W).float()

    quat = torch.zeros(T, P, 4, device=device, dtype=torch.float32)
    trans = torch.zeros(T, P, 3, device=device, dtype=torch.float32)
    ok = torch.zeros(T, P, device=device, dtype=torch.bool)

    quat[query_idx, :, 0] = 1.0
    ok[query_idx, :] = True

    inv_temp = 1.0 / max(float(temperature), 1e-6)

    if T > 1:
        ref_dst_all = torch.stack([
            tracks[t].reshape(P, 3).to(device).float().index_select(0, ref_idx)
            for t in range(T)
        ])
        ref_outer = ref_src[None, :, :, None] * ref_dst_all[:, :, None, :]
        for s in range(0, P, chunk_size):
            e = min(s + chunk_size, P)
            chunk_emb_n = src_emb_n[s:e]
            logits = chunk_emb_n @ ref_emb_n.T
            logits.mul_(inv_temp)
            if sigma_px > 0:  # spatial Gaussian on query-frame pixel distance
                pix = torch.arange(s, e, device=device)
                dh = (pix // W).float()[:, None] - ref_h[None]
                dw = (pix % W).float()[:, None] - ref_w[None]
                logits = logits - (dh * dh + dw * dw) / (2.0 * sigma_px ** 2)
                del pix, dh, dw
            w = F.softmax(logits, dim=-1)
            del logits
            w.div_(w.sum(dim=-1, keepdim=True).clamp_min(1e-8))
            src_mean = w @ ref_src
            for t in range(T):
                if t == query_idx:
                    continue
                dst_mean = w @ ref_dst_all[t]
                weighted_outer = torch.einsum("qk,kab->qab", w, ref_outer[t])
                cov = weighted_outer - src_mean.unsqueeze(-1) * dst_mean.unsqueeze(-2)
                horn = solve_rotation_horn(cov, eig_min_gap=1e-4, eig_jitter=1e-6)
                R = horn["R"]
                sol_ok = horn["ok"]
                tr = dst_mean - (src_mean.unsqueeze(1) @ R.transpose(-2, -1)).squeeze(1)
                sol_ok = sol_ok & torch.isfinite(tr).all(dim=-1)

                quat[t, s:e] = horn["quat"]
                trans[t, s:e] = tr
                ok[t, s:e] = sol_ok
            del w  # Do not retain the previous chunk's large weight matrix.

    return {
        "quat":  quat.reshape(T, H, W, 4).cpu(),
        "trans": trans.reshape(T, H, W, 3).cpu(),
        "ok":    ok.reshape(T, H, W).cpu(),
    }
