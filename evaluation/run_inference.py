"""Run MoSE3 on an SE(3) benchmark cache and save its predictions at the query pixels.

    python evaluation/run_inference.py --ckpt ckpt --cache_dir data/se3_bench/ho3d \
        --out_dir outputs/se3_bench/ho3d/preds/mose3

Input: one ``<cache_dir>/<clip>.npz`` per clip, holding
  - ``images``    [T, H, W, 3] uint8  RGB frames
  - ``query_uv``  [N, 2]              (x, y) query pixels at frame 0

Each clip is run in three steps:
  1. Resize the frames to ``--resolution`` (default 504 x 378) and run MoSE3 once, with frame 0 as the
     query frame (bfloat16 autocast).
  2. Move the predicted 3D tracks from each frame's camera into the first camera's frame, with the
     predicted camera poses.
  3. Map the query pixels onto the resized grid and read the tracks and rigidity embeddings there
     (nearest pixel).

Output: one ``<out_dir>/<clip>.npz`` per clip, holding
  - ``pred_tracks_cam0``  [T, N, 3]  3D track of every query pixel, in the first camera's frame
  - ``rigid_emb``         [T, N, D]  rigidity embeddings at the query pixel locations in every frame;
                                     ``compute_metrics.py`` uses frame 0's
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mose3.models.mose3 import MoSE3


def sample_at(dense: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Nearest-pixel lookup of dense maps [T, H, W, C] at (x, y) pixels uv [N, 2] -> [T, N, C]."""
    H, W = dense.shape[1:3]
    uv = uv.round().long()
    return dense[:, uv[:, 1].clamp(0, H - 1), uv[:, 0].clamp(0, W - 1)]


@torch.inference_mode()
def run_clip(model, images: np.ndarray, query_uv: np.ndarray, resolution, device: torch.device):
    """images [T, H, W, 3] uint8, query_uv [N, 2] -> tracks [T, N, 3] (first camera's frame), rigidity_emb [T, N, D]."""
    T, H, W, _ = images.shape
    w, h = resolution
    imgs = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    imgs = F.interpolate(imgs, size=(h, w), mode="bilinear", align_corners=False)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(imgs[None], query_frame_idx=0)

    poses = out["camera_poses"][0].float()
    poses = torch.linalg.inv(poses[:1]) @ poses                       # camera-to-first-camera
    tracks = (torch.einsum("tij,thwj->thwi", poses[:, :3, :3], out["tracks_local"][0].float())
              + poses[:, None, None, :3, 3])

    # cache pixels -> model pixels, with pixel centres matching the bilinear resize
    uv = torch.from_numpy(query_uv).to(device).float()
    uv = (uv + 0.5) * uv.new_tensor([w / W, h / H]) - 0.5
    return sample_at(tracks, uv), sample_at(out["rigidity_emb"][0].float(), uv)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="Folder holding model.safetensors + config.json, or a Hugging Face repo id.")
    p.add_argument("--cache_dir", required=True, help="Benchmark cache: one <clip>.npz per clip.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--resolution", type=int, nargs=2, default=(504, 378), metavar=("W", "H"),
                   help="Model input size, multiples of 14 (default 504 378).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    clips = sorted(Path(args.cache_dir).glob("*.npz"))
    if not clips:
        raise SystemExit(f"no .npz clips in {args.cache_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = MoSE3.from_pretrained(args.ckpt, strict=True).to(device).eval()
    for i, path in enumerate(clips):
        cache = np.load(path)
        tracks, emb = run_clip(model, cache["images"], cache["query_uv"], args.resolution, device)
        np.savez_compressed(out_dir / path.name,
                            pred_tracks_cam0=tracks.cpu().numpy(), rigid_emb=emb.cpu().numpy())
        print(f"[{i + 1}/{len(clips)}] {path.stem}")


if __name__ == "__main__":
    main()
