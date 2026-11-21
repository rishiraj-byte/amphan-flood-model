"""train_queue.py - try new training recipes HONESTLY (unattended, one GPU job at a time).

1. Train each candidate recipe on TRAIN blocks only (kit v4, 16 storm-agnostic layers).
2. Score candidates + the 3 existing train-only runs (v4_B, v4_B_seed1, v4_sarOnly = same old recipe, 3 seeds) on
   VALIDATION blocks: TTA 8, best cutoff, mean flood IoU of the published + independent maps.
3. A candidate wins only if it beats the old recipe's mean by more than the old recipe's own seed spread.
4. Winner -> retrain on train+val for its best epoch count -> ops/models/detect_<name>_70.pt -> score the TEST
   blocks ONCE: current ensemble vs current ensemble + new model (TTA 8, cutoff 0.30, declared here in advance).
5. Report -> D:/sar_flood/REPORT_train_queue_v5.md, sent to the PC. ops_config.json is NOT changed automatically.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import improve_eval as ie  # noqa: E402

KIT = Path(r"D:\sar_flood\train_kit_v4")
PY = sys.executable
BASE = ["--task", "detect", "--base", "48", "--lr", "0.0005", "--weight-decay", "0.0001", "--flood-frac", "0.65",
        "--seed", "0", "--mmap", "--eval-every", "2"]
CANDS = {  # one change each vs the old recipe (patch 256, batch 12, 300 steps, 60 epochs, patience 12)
    "v5_wide384": ["--patch", "384", "--batch", "6", "--steps", "300", "--epochs", "60", "--patience", "12"],
    "v5_long": ["--patch", "256", "--batch", "12", "--steps", "450", "--epochs", "80", "--patience", "16"],
}
OLD = ["v4_B", "v4_B_seed1", "v4_sarOnly"]
OPS = ["detect_v3_ops70.pt", "detect_v4_plus70.pt"]
CUT, TTA = 0.30, 8
REP = Path(r"D:\sar_flood\REPORT_train_queue_v5.md")


def log(m):
    ie.OUT.mkdir(exist_ok=True)
    ie.log("[queue] " + m)


def train(name, args):
    if (KIT / "runs" / name / "metrics.json").exists():
        return
    log(f"training {name}: {' '.join(args)}")
    with open(KIT / f"runs_{name}.out", "a", encoding="utf-8") as f:
        rc = subprocess.call([PY, "-u", "train.py", *args, "--name", name], cwd=str(KIT), stdout=f, stderr=subprocess.STDOUT)
    log(f"{name} finished, exit {rc}")


def setup(split):
    d = KIT / "data"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    chans = meta["tasks"]["detect"]["channels"]
    X = np.load(d / "inputs_detect.npy", mmap_mode="r")
    valid = np.asarray(X[chans.index("valid")]) > 0
    S, C = np.load(d / "split.npy"), np.load(d / "circle.npy").astype(bool)
    R1, R2 = np.load(d / "ref_labels.npy"), np.load(d / "ref2_labels.npy")
    blk = (S == meta["split_code"][split]) & C & valid
    sel = {"published": blk & (R1 >= 0), "independent": blk & (R2 != 255)}
    truth = {"published": R1[sel["published"]] == 1, "independent": R2[sel["independent"]] == 1}
    return X, chans, sel, truth


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n, a in CANDS.items():
        train(n, BASE + a)
    X, chans, sel, truth = setup("val")
    sc = {}
    for n in OLD + list(CANDS):
        ck = KIT / "runs" / n / "best.pt"
        if not ck.exists():
            continue
        p = ie.prob(n, ck, TTA, X, chans, dev)
        pv = {k: p[sel[k]] for k in sel}
        c, s = max(((c, ie.score(pv, truth, c)) for c in ie.CUTS),
                   key=lambda t: t[1]["published"]["iou"] + t[1]["independent"]["iou"])
        sc[n] = {"cut": c, "pub": s["published"]["iou"], "ind": s["independent"]["iou"],
                 "mean": round((s["published"]["iou"] + s["independent"]["iou"]) / 2, 4),
                 "best_epoch": (json.loads((KIT / "runs" / n / "metrics.json").read_text()) or {}).get("best_epoch")}
    old = [sc[b]["mean"] for b in OLD if b in sc]
    bar = round(float(np.mean(old)) + (max(old) - min(old)), 4)
    cands = {n: sc[n] for n in CANDS if n in sc}
    win = max(cands, key=lambda n: cands[n]["mean"]) if cands else None
    ok = bool(win and cands[win]["mean"] > bar)
    md = [f"# Training queue v5 ({time.strftime('%Y-%m-%d %H:%M')}, {(time.time() - t0) / 3600:.1f} h)", "",
          "Validation blocks, single model, TTA 8, best cutoff. Old recipe = v4_B / v4_B_seed1 / v4_sarOnly (3 seeds).", "",
          "| run | recipe change | best epoch | cutoff | IoU published | IoU independent | mean |", "|---|---|---|---|---|---|---|"]
    for n, v in sc.items():
        md.append(f"| {n} | {' '.join(CANDS.get(n, ['old recipe']))} | {v['best_epoch']} | {v['cut']:.2f} | {v['pub']:.3f} | {v['ind']:.3f} | {v['mean']:.3f} |")
    md += ["", f"Bar to beat = old mean {np.mean(old):.3f} + old seed spread {max(old) - min(old):.3f} = **{bar:.3f}**.",
           f"Best candidate: {win} ({cands[win]['mean']:.3f})." if win else "No candidate finished.",
           f"**Verdict: {'WINNER, retrained on train+val and tested below' if ok else 'no recipe beat the old one by more than luck; nothing changed'}.**"]
    if ok:
        be = sc[win]["best_epoch"] or 40
        args = BASE + CANDS[win]
        args[args.index("--epochs") + 1] = str(be)
        n70 = f"{win}_70"
        train(n70, args + ["--trainval"])
        dst = HERE / "models" / f"detect_{n70}.pt"
        shutil.copy2(KIT / "runs" / n70 / "best.pt", dst)
        ck = torch.load(dst, map_location="cpu", weights_only=False)
        m = json.loads((KIT / "runs" / n70 / "metrics.json").read_text())
        f = (m.get("test_vs_zenodo_reference") or {}).get("flood") or {}
        (dst.with_suffix(".json")).write_text(json.dumps({
            "file": dst.name, "task": "detect", "channels": ck["channels"], "base": ck["base"], "params_M": 17.5,
            "note": f"storm-agnostic, kit v4, recipe {win} ({' '.join(CANDS[win])}), train+val", "run": str(KIT / "runs" / n70),
            "test_vs_published_map": {k: round(f[k], 3) for k in ("iou", "precision", "recall", "f1") if k in f},
            "best_epoch": be}, indent=1), encoding="utf-8")
        X, chans, sel, truth = setup("test")
        P = {m_: ie.prob(Path(m_).stem, HERE / "models" / m_, TTA, X, chans, dev) for m_ in OPS + [dst.name]}
        md += ["", f"## Test blocks, scored once (TTA {TTA}, cutoff {CUT})", "",
               "| ensemble | IoU published | P | R | IoU independent | P | R |", "|---|---|---|---|---|---|---|"]
        for lab, mem in (("CURRENT v3_ops70 + v4_plus70", OPS), (f"CURRENT + {n70}", OPS + [dst.name])):
            p = np.mean([P[x] for x in mem], 0)
            s = ie.score({k: p[sel[k]] for k in sel}, truth, CUT)
            md.append(f"| {lab} | {s['published']['iou']:.3f} | {s['published']['p']:.3f} | {s['published']['r']:.3f} | "
                      f"{s['independent']['iou']:.3f} | {s['independent']['p']:.3f} | {s['independent']['r']:.3f} |")
        md += ["", f"New model saved as ops/models/{dst.name}. Not switched on automatically: tick it on the Control page."]
    md += ["", "Caveat: the published map is the source of the training labels; the independent (ARIA) map is the honest one."]
    REP.write_text("\n".join(md) + "\n", encoding="utf-8")
    log("report written: " + REP.name)
    subprocess.run([PY, str(HERE / "bridge.py"), "send", "Laptop training queue v5 finished. Report attached.",
                    "--subject", "training queue v5 result", "--attach", str(REP)], cwd=str(HERE), timeout=900)


if __name__ == "__main__":
    main()
