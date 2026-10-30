<div align="center">
<h1>MoSE3: Learning World-Space SE(3) at Every Pixel</h1>
<h3>NeurIPS 2026 (Spotlight)</h3>

<a href="https://arxiv.org/abs/2610.03716"><img src="https://img.shields.io/badge/Paper-MoSE3-b31b1b" height="20" alt="Paper"></a>
<a href="https://mose3-tracker.github.io/"><img src="https://img.shields.io/badge/Project-Website-green" height="20" alt="Project Page"></a>
<a href="https://huggingface.co/mose3-tracker/MoSE3"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" height="20" alt="Model"></a>
<a href="https://huggingface.co/datasets/mose3-tracker/Art-Kubric"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" height="20" alt="Dataset"></a>

[Jiahuan Cheng](https://joannaccjh.github.io/)\*, [Zhiyi Li](https://sky-lzy.github.io/)\*, [Tian Xia](https://tianx-ia.github.io/)\*, [Ruojin Cai](https://ruojincai.github.io/), [Yilun Du](https://yilundu.github.io/), [Qianqian Wang](https://qianqianwang68.github.io/)

\* Equal contribution
</div>

# MoSE3 Evaluation

This branch reproduces the SE(3) estimation results of MoSE3 on [HO3D](https://github.com/shreyashampali/ho3d),
[iTACO](https://github.com/3dlg-hcvc/video2articulation) and
[YCBInEOAT](https://github.com/wenbowen123/iros20-6d-pose-tracking), at the per-pixel
and at the object/part level. To run MoSE3 on your own videos, see the
[main branch](https://github.com/mose3-tracker/MoSE3).

## Installation

```bash
git clone -b evaluation https://github.com/mose3-tracker/MoSE3.git
cd MoSE3
```

Set up the conda environment and PyTorch as in the main branch's
[Installation](https://github.com/mose3-tracker/MoSE3#installation), and download the weights into `ckpt/` as in
its [Model weights](https://github.com/mose3-tracker/MoSE3#model-weights). Then, in the same environment,
install the evaluation dependencies:

```bash
pip install -r requirements_eval.txt
```

## Benchmark data

Download the pre-built benchmark caches (2.1 GB) from
[mose3-tracker/mose3-eval](https://huggingface.co/datasets/mose3-tracker/mose3-eval) into `data/se3_bench/`.
The dataset is gated: accept its terms on the dataset page and log in with `hf auth login`, then download:

```bash
hf download mose3-tracker/mose3-eval --repo-type dataset --local-dir data/se3_bench
```

```
data/se3_bench/
  ho3d/        54 clips
  itaco/       55 clips
  ycbineoat/    9 clips
```

<details>
<summary>Cache format</summary>

Each clip is one `<clip>.npz` with T = 50 frames (49 on YCBInEOAT) at 480 x 640 and N = 3072 query pixels
at frame 0:

| key | shape | |
|---|---|---|
| `images` | [T, H, W, 3] uint8 | RGB frames |
| `query_uv` | [N, 2] | (x, y) query pixels at frame 0 |
| `intrinsics` | [4] | (fx, fy, cx, cy) in pixels, for methods that take them; not used by the evaluation |
| `tracks_cam` | [T, N, 3] | GT 3D tracks, in the first camera's frame |
| `obj_se3_per_point` | [T, N, 4, 4] | GT pose of each query's rigid body at every frame, in the first camera's frame; only valid where `has_gt_se3` is true |
| `vis` | [T, N] bool | GT visibility |
| `has_gt_se3` | [N] bool | the query has a GT pose: it lies on the object (iTACO: on any of the object's parts) |
| `metric_mask` | [N] bool | the scored queries: the object (HO3D, YCBInEOAT) or its moving part (iTACO) |
| `actor_idx` | [N] int8 | 0 = the object (YCBInEOAT) or its moving part (iTACO), 1 = everything else; all 0 on HO3D |

The predicted tracks are aligned to the GT on the `has_gt_se3` queries and scored on the `metric_mask` queries.
On HO3D and YCBInEOAT the two masks are the same. The GT SE(3) of query `n` from frame 0 to frame `t`, which
RRE and ADD are measured against, is `obj_se3_per_point[t, n] @ inv(obj_se3_per_point[0, n])`.

</details>

## Run

All three benchmarks, then the table:

```bash
bash evaluation/eval_se3.sh data/se3_bench ckpt
```

or step by step, here for HO3D:

```bash
# 1. predictions at the query pixels -> outputs/se3_bench/ho3d/preds/mose3/<clip>.npz
python evaluation/run_inference.py --ckpt ckpt --cache_dir data/se3_bench/ho3d \
    --out_dir outputs/se3_bench/ho3d/preds/mose3

# 2. per-clip metrics -> outputs/se3_bench/ho3d/metrics/mose3/summary.csv
python evaluation/compute_metrics.py --cache_dir data/se3_bench/ho3d \
    --preds_dir outputs/se3_bench/ho3d/preds/mose3 --out_dir outputs/se3_bench/ho3d/metrics/mose3 --clean_gt

# 3. averages over clips
python evaluation/make_table.py --root outputs/se3_bench --methods mose3 --benchmarks ho3d
```

`--clean_gt` (HO3D and iTACO) drops GT points with broken depth: unprojected from a depth hole or from
the background through the mask border.

## Protocol

Each clip is scored in four steps:

1. **Scale alignment.** One scale and xyz shift per clip, fitted to the GT on visible labelled points
   ([MoGe](https://github.com/microsoft/MoGe)'s solver).
2. **Per-pixel SE(3).** For each query pixel, fit the SE(3) from frame 0 to frame t on the predicted tracks
   of a group of support points (Horn's method). How the support points are chosen differs between `emb`
   and `knn` (table below).
3. **Object/part-level SE(3).** HDBSCAN (min cluster size 30) clusters the query pixels into rigid parts. The
   cluster that overlaps the GT object best (highest IoU) is taken as the object, and one rigid motion per
   frame is fitted to the tracks of all its pixels.
4. **Metrics**, on the scored queries at frames t > 0:
   - RRE (degrees) and ADD (cm), per pixel and per object
   - RRE-AUC up to 30° (per pixel)
   - ADD-AUC: area under the accuracy curve up to 10 cm (20 cm on iTACO), as in YCB-Video
   - IoU of the chosen group (per object)

The two SE(3) protocols differ only in steps 2 and 3:

| | `emb` (MoSE3) | `knn` (any point tracker) |
|---|---|---|
| Step 2: support points | all query pixels, weighted by rigidity-embedding similarity (softmax, temperature 0.01) | the K nearest 3D neighbours at frame 0, equal weights |
| Step 3: grouped by | rigidity embeddings | per-pixel SE(3)s |
| K | — | K in {4, ..., 256}, chosen per clip by the best z-scored sum of the metrics, separately for the per-pixel and the object-level columns |

**Small objects.** An object with fewer than 30 query pixels cannot form its own cluster in step 3. To allow
it, pass `--adaptive_min_cluster` to `compute_metrics.py`: on those clips (2 in HO3D, 2 in iTACO, 3 in
YCBInEOAT) the minimum cluster size becomes the object's number of query pixels. The expected results below
use the default of 30.

## Expected results

`make_table.py` output with the released weights. `emb` is MoSE3 with its rigidity embeddings; `knn` applies
the k-NN protocol to MoSE3's tracks. ADD is in cm.

| | | Per-pixel | | | | Object/part level | | | |
|---|---|---|---|---|---|---|---|---|---|
| **Benchmark** | **SE(3)** | **RRE ↓** | **RRE-AUC ↑** | **ADD ↓** | **ADD-AUC ↑** | **IoU ↑** | **RRE ↓** | **ADD ↓** | **ADD-AUC ↑** |
| HO3D | `emb` | 18.28 | 0.569 | 2.16 | 78.55 | 0.709 | 17.96 | 2.06 | 80.24 |
| | `knn` | 24.09 | 0.509 | 2.61 | 74.37 | 0.504 | 20.04 | 2.26 | 78.35 |
| iTACO | `emb` | 12.30 | 0.741 | 11.37 | 59.42 | 0.784 | 10.76 | 10.49 | 62.51 |
| | `knn` | 20.78 | 0.570 | 14.47 | 50.13 | 0.505 | 9.87 | 10.27 | 63.06 |
| YCBInEOAT | `emb` | 14.89 | 0.588 | 3.98 | 62.76 | 0.498 | 11.93 | 4.00 | 63.83 |
| | `knn` | 17.97 | 0.546 | 4.50 | 59.45 | 0.171 | 14.02 | 3.67 | 65.85 |

## Evaluating other methods

Any method that predicts 3D point tracks can be scored with the `knn` protocol. Write one
`outputs/se3_bench/<benchmark>/preds/<method>/<clip>.npz` per cached clip with `pred_tracks_cam0`
[T, N, 3], the 3D tracks of the cached query pixels in the first camera's frame. Add `rigid_emb`
[T, N, D] for the `emb` protocol. Then run steps 2 and 3 with `--methods mose3 <method>`.

## Citation

```bibtex
@inproceedings{cheng2026mose3,
  title     = {{MoSE3}: Learning World-Space {SE(3)} at Every Pixel},
  author    = {Cheng, Jiahuan and Li, Zhiyi and Xia, Tian and
               Cai, Ruojin and Du, Yilun and Wang, Qianqian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

The code is released under the BSD 3-Clause license, except for a few third-party files that keep
their own licenses (some non-commercial), listed in [LICENSE](LICENSE). The model weights include
π³'s weights and are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
