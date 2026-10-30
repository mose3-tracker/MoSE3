"""Average ``compute_metrics.py`` results over clips into the SE(3) table.

    python evaluation/make_table.py --root outputs/se3_bench --methods mose3

Input: ``<root>/<benchmark>/metrics/<method>/summary.csv`` for each method on HO3D, iTACO and YCBInEOAT.
ADD-AUC uses a 10 cm band on HO3D and YCBInEOAT and 20 cm on iTACO, whose articulated parts are larger.

Each method gets up to two rows per benchmark:
  1. ``knn``: per clip, K is picked from the K sweep twice, once for the per-pixel and once for the
     object-level columns. The metrics are z-scored over the clip's K values and summed, signed so that
     lower is better; the lowest sum wins:
       - per pixel:  RRE + ADD - RRE-AUC - ADD-AUC
       - object:     RRE + ADD - ADD-AUC
  2. ``emb``: the clip's ``emb`` result as is (methods with rigidity embeddings only).
Each column is then averaged over clips, skipping clips where it is undefined (e.g. no object cluster).

Output: one printed table per benchmark: per-pixel RRE, RRE-AUC, ADD (cm) and ADD-AUC; object IoU, RRE,
ADD (cm) and ADD-AUC; and ``coverage``, the number of clips with an object-level fit.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ADD_AUC_BAND_CM = {"ho3d": 10, "itaco": 20, "ycbineoat": 10}
# K-selection criterion: (column, sign); lower total z-score wins
PP_SCORE = [("rre_pp_mean", 1), ("add_mean_m", 1), ("auc_rot_pp_30", -1), ("add_auc_{band}cm_pct", -1)]
OBJ_SCORE = [("cluster_rre_mean", 1), ("cluster_add_mean_m", 1), ("cluster_add_auc_{band}cm_pct", -1)]
COLUMNS = [  # (header, column, scale, decimals)
    ("RRE", "rre_pp_mean", 1, 2), ("RRE-AUC", "auc_rot_pp_30", 1, 3),
    ("ADD(cm)", "add_mean_m", 100, 2), ("ADD-AUC", "add_auc_{band}cm_pct", 1, 2),
    ("IoU", "cluster_iou", 1, 3), ("oRRE", "cluster_rre_mean", 1, 2), ("oADD(cm)", "cluster_add_mean_m", 100, 2),
    ("oADD-AUC", "cluster_add_auc_{band}cm_pct", 1, 2),
]


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _z(values):
    """z-scores over the finite values; non-finite entries get +inf."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0] * len(values)
    mean = sum(finite) / len(finite)
    sd = (sum((v - mean) ** 2 for v in finite) / len(finite)) ** 0.5
    if sd < 1e-8:
        return [0.0] * len(values)
    return [(v - mean) / sd if math.isfinite(v) else float("inf") for v in values]


def pick_k(clip_rows: dict, score, key: str, band: int):
    """The kNN row of one clip with the lowest total z-score, among those where ``key`` is finite."""
    knn = sorted((p for p in clip_rows if p.startswith("knn_K")), key=lambda p: int(p[len("knn_K"):]))
    cand = [clip_rows[p] for p in knn if math.isfinite(_f(clip_rows[p][key]))]
    if not cand:
        return None
    zs = [_z([_f(r[col.format(band=band)]) for r in cand]) for col, _ in score]
    totals = [sum(sign * z[j] for z, (_, sign) in zip(zs, score)) for j in range(len(cand))]
    totals = [t if math.isfinite(t) else float("inf") for t in totals]
    return cand[min(range(len(cand)), key=totals.__getitem__)]


def table_rows(csv_path: Path, band: int) -> dict:
    """{"knn": {column: clip mean}, "emb": {...}} plus the number of clips with an object-level fit."""
    clips = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            clips.setdefault(row["sample_name"], {})[row["se3_path"]] = row
    # one (per-pixel row, object-level row) pair per clip
    selections = {"knn": [(pick_k(c, PP_SCORE, "rre_pp_mean", band), pick_k(c, OBJ_SCORE, "cluster_rre_mean", band))
                          for c in clips.values()]}
    if all("emb" in c for c in clips.values()):
        selections["emb"] = [(c["emb"], c["emb"]) for c in clips.values()]

    out = {}
    for name, picks in selections.items():
        res = {}
        for _, col, _, _ in COLUMNS:
            col = col.format(band=band)
            rows = [obj if col.startswith("cluster_") else pp for pp, obj in picks]
            vals = [v for v in (_f(r[col]) for r in rows if r is not None) if math.isfinite(v)]
            res[col] = sum(vals) / len(vals) if vals else float("nan")
        n_fit = sum(obj is not None and math.isfinite(_f(obj["cluster_rre_mean"])) for _, obj in picks)
        res["coverage"] = f"{n_fit}/{len(picks)}"
        out[name] = res
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Holds <benchmark>/metrics/<method>/summary.csv.")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--benchmarks", nargs="+", default=list(ADD_AUC_BAND_CM), choices=list(ADD_AUC_BAND_CM))
    args = p.parse_args()

    for bench in args.benchmarks:
        band = ADD_AUC_BAND_CM[bench]
        print(f"\n{bench} (ADD-AUC @ {band} cm)   per-pixel | object level")
        print("\t".join(["method", *(h for h, *_ in COLUMNS), "coverage"]))
        for method in args.methods:
            csv_path = Path(args.root) / bench / "metrics" / method / "summary.csv"
            if not csv_path.is_file():
                print(f"{method}\tmissing {csv_path}")
                continue
            for name, res in table_rows(csv_path, band).items():
                cells = [f"{res[col.format(band=band)] * scale:.{dec}f}" for _, col, scale, dec in COLUMNS]
                print("\t".join([f"{method} ({name})", *cells, res["coverage"]]))


if __name__ == "__main__":
    main()
