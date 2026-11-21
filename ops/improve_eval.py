"""improve_eval.py - choose prediction settings HONESTLY, then score them once.

Phase "val"  (choosing): uses only models trained on TRAIN blocks (v4_B, v4_B_seed1, v4_sarOnly = 3 seeds of the
             same 16-layer recipe). On VALIDATION blocks it compares: plain vs 8-view test-time augmentation (TTA),
             1 vs 2 vs 3 averaged models, and the flood cutoff 0.20-0.70. Score = mean of flood IoU vs the published
             (Mondal) map and vs the independent (ARIA) map. Writes improve/recipe.json (the choice) + val_report.md.
Phase "test" (scoring, once): applies recipe.json to the ops models (trained on train+val blocks) and scores the
             TEST blocks vs both maps, next to the current settings (plain, cutoff 0.5). Writes improve/test_report.md.
The test blocks are never used to choose anything.

  python improve_eval.py --phase val|test [--kit D:/sar_flood/train_kit_v4]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from predict import run_model

HERE = Path(__file__).resolve().parent
OUT = HERE / "improve"
TRAIN_ONLY = ["v4_B", "v4_B_seed1", "v4_sarOnly"]
OPS = ["detect_v3_ops70.pt", "detect_v4_plus70.pt"]
CUTS = [round(0.20 + 0.05 * i, 2) for i in range(11)]


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")
    (OUT / "progress.txt").write_text(s, encoding="utf-8")


def prob(name, ckpt, tta, X, chans, device):
    f = OUT / f"prob_{name}_tta{tta}.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    t0 = time.time()
    p = run_model(str(ckpt), chans, X, device, lambda m: None, tta=tta)[0]
    np.save(f, p.astype(np.float16))
    log(f"{name} tta{tta}: {time.time() - t0:.0f} s")
    return p


def iou(pf, rf):
    tp, fp, fn = int((pf & rf).sum()), int((pf & ~rf).sum()), int((~pf & rf).sum())
    return {"iou": round(tp / max(tp + fp + fn, 1), 4), "p": round(tp / max(tp + fp, 1), 4), "r": round(tp / max(tp + fn, 1), 4)}


def score(p, masks, cut):
    """p: dict mapname -> prob values on that map's pixels; masks: mapname -> flood truth (bool) on same pixels."""
    return {k: iou(p[k] >= cut, masks[k]) for k in masks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["val", "test"], required=True)
    ap.add_argument("--kit", default=r"D:\sar_flood\train_kit_v4")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    kit = Path(a.kit)
    d = kit / "data"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    chans = meta["tasks"]["detect"]["channels"]
    X = np.load(d / "inputs_detect.npy", mmap_mode="r")
    valid = np.asarray(X[chans.index("valid")]) > 0
    S, C = np.load(d / "split.npy"), np.load(d / "circle.npy").astype(bool)
    R1, R2 = np.load(d / "ref_labels.npy"), np.load(d / "ref2_labels.npy")
    code = meta["split_code"]["val" if a.phase == "val" else "test"]
    blk = (S == code) & C & valid
    sel = {"published": blk & (R1 >= 0), "independent": blk & (R2 != 255)}
    truth = {"published": R1[sel["published"]] == 1, "independent": R2[sel["independent"]] == 1}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"phase {a.phase} on {device}; pixels published {sel['published'].sum():,} (flood {truth['published'].sum():,}), "
        f"independent {sel['independent'].sum():,} (flood {truth['independent'].sum():,})")

    def on_maps(p):
        return {k: p[sel[k]] for k in sel}

    if a.phase == "val":
        res = {}
        vals = {}
        for tta in (1, 8):
            for r in TRAIN_ONLY:
                vals[(r, tta)] = on_maps(prob(r, kit / "runs" / r / "best.pt", tta, X, chans, device))
        for tta in (1, 8):
            combos = {f"single {r}": [r] for r in TRAIN_ONLY}
            combos.update({"avg of 2 (v4_B + seed1)": TRAIN_ONLY[:2], "avg of 2 (v4_B + sarOnly)": [TRAIN_ONLY[0], TRAIN_ONLY[2]],
                           "avg of 2 (seed1 + sarOnly)": TRAIN_ONLY[1:], "avg of 3": TRAIN_ONLY})
            for nm, mem in combos.items():
                p = {k: np.mean([vals[(m, tta)][k] for m in mem], 0) for k in sel}
                res[f"{nm} | tta{tta}"] = {str(c): score(p, truth, c) for c in CUTS}
        rows = []
        for k, by in res.items():
            for c, s in by.items():
                rows.append((k, float(c), s["published"]["iou"], s["independent"]["iou"],
                             (s["published"]["iou"] + s["independent"]["iou"]) / 2, s))
        best = {}
        for k, c, i1, i2, m, s in rows:
            if k not in best or m > best[k][4]:
                best[k] = (k, c, i1, i2, m, s)

        def mean_best(prefix, tta):
            ks = [k for k in best if k.startswith(prefix) and k.endswith(f"tta{tta}")]
            return float(np.mean([best[k][4] for k in ks])), ks
        # decision: TTA (on the 2-model averages, the size of the ops ensemble), then the cutoff for that setting
        m1, _ = mean_best("avg of 2", 1)
        m8, _ = mean_best("avg of 2", 8)
        tta = 8 if m8 > m1 else 1
        cut_scores = {c: float(np.mean([next(r[4] for r in rows if r[0] == k and r[1] == c)
                                        for k in best if k.startswith("avg of 2") and k.endswith(f"tta{tta}")])) for c in CUTS}
        cut = max(cut_scores, key=cut_scores.get)
        g2 = mean_best("avg of 2", tta)[0]
        g1 = mean_best("single", tta)[0]
        g3 = best[f"avg of 3 | tta{tta}"][4]
        recipe = {"tta": tta, "threshold": cut, "chosen_on": "validation blocks, train-only models (v4_B, v4_B_seed1, v4_sarOnly)",
                  "criterion": "mean of flood IoU vs published and vs independent map",
                  "val_mean_iou": {"single_best_cut": round(g1, 4), "avg2_best_cut": round(g2, 4), "avg3_best_cut": round(g3, 4),
                                   "avg2_tta1": round(m1, 4), "avg2_tta8": round(m8, 4)},
                  "cut_curve_avg2": {str(c): round(v, 4) for c, v in cut_scores.items()},
                  "third_model_helps": bool(g3 > g2 + 0.003), "created": time.strftime("%Y-%m-%d %H:%M")}
        (OUT / "recipe.json").write_text(json.dumps(recipe, indent=1), encoding="utf-8")
        (OUT / "val_all.json").write_text(json.dumps(res), encoding="utf-8")
        md = ["# Validation-block comparison (train-only models)", "",
              f"Pixels: published map {sel['published'].sum():,}, independent map {sel['independent'].sum():,}.", "",
              "| setting | best cutoff | IoU published | IoU independent | mean |", "|---|---|---|---|---|"]
        md += [f"| {k.replace(' | ', ' · ')} | {c:.2f} | {i1:.3f} | {i2:.3f} | {m:.3f} |" for k, c, i1, i2, m, _ in sorted(best.values(), key=lambda r: -r[4])]
        md += ["", "## Cutoff curve (avg of 2 models, chosen TTA): mean IoU", "",
               " | ".join(f"{c:.2f}: {v:.3f}" for c, v in cut_scores.items()), "",
               f"**Chosen: TTA {tta}, cutoff {cut:.2f}.** 2-model avg vs single: {g2:.3f} vs {g1:.3f}; 3-model avg {g3:.3f}."]
        (OUT / "val_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        log(f"recipe: tta {tta}, cutoff {cut}; val mean IoU single {g1:.3f} / avg2 {g2:.3f} / avg3 {g3:.3f}; tta1 {m1:.3f} vs tta8 {m8:.3f}")
        return

    # ---- test: once, with the recipe fixed beforehand ----
    rec = json.loads((OUT / "recipe.json").read_text(encoding="utf-8"))
    rows, out = [], {"recipe": rec, "results": {}}
    for tta in sorted({1, rec["tta"]}):
        ps = [on_maps(prob(Path(m).stem, HERE / "models" / m, tta, X, chans, device)) for m in OPS]
        ens = {k: np.mean([p[k] for p in ps], 0) for k in sel}
        for cut in sorted({0.5, rec["threshold"]}):
            if (tta, cut) not in ((1, 0.5), (rec["tta"], rec["threshold"])):
                continue
            s = score(ens, truth, cut)
            lab = "CURRENT (plain, 0.50)" if (tta, cut) == (1, 0.5) else f"NEW (tta{tta}, {cut:.2f})"
            out["results"][lab] = s
            rows.append(f"| {lab} | {s['published']['iou']:.3f} | {s['published']['p']:.3f} | {s['published']['r']:.3f} | "
                        f"{s['independent']['iou']:.3f} | {s['independent']['p']:.3f} | {s['independent']['r']:.3f} |")
    (OUT / "test_scores.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# Test blocks, ops ensemble (v3_ops70 + v4_plus70), scored once", "",
          f"Recipe fixed on validation blocks before this run: TTA {rec['tta']}, cutoff {rec['threshold']}.", "",
          "| setting | IoU published | P | R | IoU independent | P | R |", "|---|---|---|---|---|---|---|"] + rows + \
         ["", "Caveat: the published map is the source of the training labels; the independent (ARIA) map is the honest one."]
    (OUT / "test_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("test done: " + "; ".join(f"{k}: pub {v['published']['iou']:.3f} / ind {v['independent']['iou']:.3f}" for k, v in out["results"].items()))


if __name__ == "__main__":
    main()
