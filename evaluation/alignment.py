"""Scale + xyz-shift alignment of 3D points, adapted from MoGe's ``align_points_scale_xyz_shift``
(https://github.com/microsoft/MoGe, ``moge/utils/alignment.py``, MIT License, see licenses/LICENSE-MoGe).
Only the untruncated (weighted L1) case is kept, and anchors are processed in chunks of ``chunk_size``.
"""
from __future__ import annotations

import math

import torch


def scatter_min(size: int, dim: int, index: torch.LongTensor, src: torch.Tensor) -> torch.return_types.min:
    "Scatter the minimum value along the given dimension of `input` into `src` at the indices specified in `index`."
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device).scatter_reduce(dim=dim, index=index, src=src, reduce='amin', include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1:])] = minimum_where[dim]
    return torch.return_types.min((minimum, indices))


def align(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps: float = 1e-7):
    """Solve ``min_a sum_i w_i * |a * x_i - y_i|`` over the last dim (a weighted median of y / x).

    Returns ``a`` (...), the loss at ``a`` (...), and ``index`` (...) with ``a = y[index] / x[index]``.
    """
    x, y, w = torch.broadcast_tensors(x, y, w)
    sign = torch.sign(x)
    x, y = x * sign, y * sign
    y_div_x = y / x.clamp_min(eps)
    y_div_x, argsort = y_div_x.sort(dim=-1)

    wx = torch.gather(x * w, dim=-1, index=argsort)
    derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
    search = torch.searchsorted(derivatives, torch.zeros_like(derivatives[..., :1]), side='left').clamp_max(derivatives.shape[-1] - 1)

    a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
    index = argsort.gather(dim=-1, index=search).squeeze(-1)
    loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
    return a, loss, index


def align_points_scale_xyz_shift(points_src: torch.Tensor, points_tgt: torch.Tensor, weight: torch.Tensor,
                                 chunk_size: int = 512):
    """Scale (...) and xyz shift (..., 3) minimizing ``sum_i w_i * |scale * src_i + shift - tgt_i|_1``.

    points_src, points_tgt: (..., N, 3); weight: (..., N). Every point with w > 0 is tried as the anchor
    the shift is pinned to; the anchor with the lowest loss wins.
    """
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = points_src.reshape(batch_size, n, 3), points_tgt.reshape(batch_size, n, 3), weight.reshape(batch_size, n)

    anchor_where_batch, anchor_where_n = torch.where(weight > 0)

    with torch.no_grad():
        points_src_anchor = points_src[anchor_where_batch, anchor_where_n]          # (anchors, 3)
        points_tgt_anchor = points_tgt[anchor_where_batch, anchor_where_n]          # (anchors, 3)

        points_src_anchored = points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]    # (anchors, n, 3)
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]    # (anchors, n, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)                         # (anchors, n, 3)

        # optimal scale for each anchor, then the best anchor per batch element
        chunks = [align(*(t[i:i + chunk_size] for t in (points_src_anchored.flatten(-2), points_tgt_anchored.flatten(-2), weight_anchored.flatten(-2))))
                  for i in range(0, len(anchor_where_batch), chunk_size)]
        scale, loss, index = (torch.cat(c, dim=0) for c in zip(*chunks))   # (anchors,)
        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)    # (batch_size,)

    index_2 = index[index_anchor]                               # (batch_size,) [0, 3n)
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3    # (batch_size,) [0, 3n)

    src_1, tgt_1 = torch.gather(points_src.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(points_tgt.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    src_2, tgt_2 = torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(points_tgt, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2) - scale[..., None] * torch.gather(points_src, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2)

    return scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)
