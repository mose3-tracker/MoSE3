#!/bin/bash
# MoSE3 on the SE(3) benchmarks: inference -> metrics -> table. Run from the repository root:
#   bash evaluation/eval_se3.sh <cache_root> [ckpt] [out_root]
# <cache_root> holds the ho3d/, itaco/ and ycbineoat/ caches.
set -e
CACHE_ROOT=${1:?usage: bash evaluation/eval_se3.sh <cache_root> [ckpt] [out_root]}
CKPT=${2:-ckpt}
OUT=${3:-outputs/se3_bench}

for bench in ho3d itaco ycbineoat; do
  clean=--clean_gt
  [ "$bench" = ycbineoat ] && clean=   # some YCBInEOAT clips score only ~12 queries: keep them all
  python evaluation/run_inference.py --ckpt "$CKPT" --cache_dir "$CACHE_ROOT/$bench" \
    --out_dir "$OUT/$bench/preds/mose3"
  python evaluation/compute_metrics.py --cache_dir "$CACHE_ROOT/$bench" --preds_dir "$OUT/$bench/preds/mose3" \
    --out_dir "$OUT/$bench/metrics/mose3" $clean
done
python evaluation/make_table.py --root "$OUT" --methods mose3
