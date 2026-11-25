"""score_refs.py - compare Flood Model v1 and the reference flood maps with each other on the kit v4 Amphan grid.

  python score_refs.py [--refs DIR ...] [--cut 0.30]

Maps:
  model        ops ensemble (detect_v3_ops70 + detect_v4_plus70, TTA 8), flood = probability >= --cut (0.30, chosen on
               validation blocks). Read from the improve/ cache made by improve_eval.py.
  published    Mondal map (kit ref_labels: 1 flood, 0 dry, 2 water, -1 unlabelled). The TRAINING-LABEL source.
  independent  NASA ARIA map (kit ref2_labels: 1 flood, 0 dry, 255 none). Bangladesh only, same S1 pair as the input.
  extra        every *.npy in the --refs folders: uint8 on the kit v4 grid, 1 flood, 0 not flood, 255 no data.

For every pair (A judged against B as the truth) it scores flood IoU / precision / recall on the TEST blocks, where
both maps have data (inside the 250 km circle, radar pair valid). "Reference vs reference" rows show how far two
expert maps agree with each other: a rough ceiling for any model trained on one of them.
Report: improve/refs_report.md. Nothing is changed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = Path(r"D:\sar_flood\train_kit_v4\data")
CACHE = HERE / "improve"
OPS = ["detect_v3_ops70", "detect_v4_plus70"]


def stats(a, b):
    tp = int(np.count_nonzero(a & b))
    fp = int(np.count_nonzero(a & ~b))
    fn = int(np.count_nonzero(~a & b))
    iou = tp / max(tp + fp + fn, 1)
    return {"iou": iou, "p": tp / max(tp + fp, 1), "r": tp / max(tp + fn, 1), "n": int(a.size), "flood_b": tp + fn}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--cut", type=float, default=0.30)
    a = ap.parse_args()
    meta = json.loads((KIT / "meta.json").read_text(encoding="utf-8-sig"))
    chans = meta["tasks"]["detect"]["channels"]
    X = np.load(KIT / "inputs_detect.npy", mmap_mode="r")
    valid = np.asarray(X[chans.index("valid")]) > 0
    S, C = np.load(KIT / "split.npy"), np.load(KIT / "circle.npy").astype(bool)
    test = (S == meta["split_code"]["test"]) & C & valid
    area = np.asarray(meta["pixel_area_km2_rows"], np.float64)[:, None]

    P = np.mean([np.load(CACHE / f"prob_{m}_tta8.npy").astype(np.float32) for m in OPS], 0)
    R1, R2 = np.load(KIT / "ref_labels.npy"), np.load(KIT / "ref2_labels.npy")
    maps = {  # name -> (has data, flood)
        "model": (np.ones_like(test), P >= a.cut),
        "published (Mondal)": (R1 >= 0, R1 == 1),
        "independent (ARIA)": (R2 != 255, R2 == 1),
    }
    for d in a.refs:
        for f in sorted(Path(d).glob("*.npy")):
            r = np.load(f, allow_pickle=False)
            if r.shape != test.shape:
                print(f"skip {f.name}: shape {r.shape} is not the kit v4 grid {test.shape}")
                continue
            maps[f"extra: {f.stem}"] = (r != 255, r == 1)

    names = list(maps)
    refs = names[1:]
    pairs = [("model", r) for r in refs] + [(refs[i], refs[j]) for i in range(len(refs)) for j in range(i + 1, len(refs))]
    rows = []
    for a_name, b_name in pairs:
        ha, fa = maps[a_name]
        hb, fb = maps[b_name]
        m = test & ha & hb
        s = stats(fa[m], fb[m])
        s["km2"] = float((m * area).sum())
        rows.append((a_name, b_name, s))

    # the same comparison restricted to cells where ALL reference maps have data (apples to apples)
    common = test.copy()
    for n in refs:
        common &= maps[n][0]
    crow = []
    if common.any():
        for a_name, b_name in pairs:
            crow.append((a_name, b_name, stats(maps[a_name][1][common], maps[b_name][1][common])))

    md = [f"# Flood maps compared on Amphan test blocks ({time.strftime('%Y-%m-%d %H:%M')})", "",
          f"Model = ops ensemble, TTA 8, cutoff {a.cut:.2f}. Each row judges map A against map B as the truth, on test-block "
          "cells where both have data. Reference-vs-reference rows = how much two expert maps agree.", "",
          "| A (judged) | B (truth) | flood IoU | precision | recall | cells compared (km²) | flood in B (cells) |",
          "|---|---|---|---|---|---|---|"]
    for an, bn, s in rows:
        md.append(f"| {an} | {bn} | **{s['iou']:.3f}** | {s['p']:.3f} | {s['r']:.3f} | {s['km2']:,.0f} | {s['flood_b']:,} |")
    if crow:
        md += ["", f"## Same, only where all {len(names) - 1} reference maps have data ({float((common * area).sum()):,.0f} km²)", "",
               "| A (judged) | B (truth) | flood IoU | precision | recall |", "|---|---|---|---|---|"]
        for an, bn, s in crow:
            md.append(f"| {an} | {bn} | **{s['iou']:.3f}** | {s['p']:.3f} | {s['r']:.3f} |")
    md += ["", "Caveat: the model's training labels come from the published Mondal map (Zenodo 4390084)."]
    out = CACHE / "refs_report.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
