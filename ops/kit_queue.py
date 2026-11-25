"""kit_queue.py - test a NEW training kit from the PC honestly (e.g. kit_v5_amphan = kit v4 + rain / terrain / land-cover).

  python kit_queue.py --kit D:\\sar_flood\\train_kit_kit_v5_amphan [--hours 70]

Uses the laptop's own reviewed train.py (train_kit_v4/train.py) with --kit. It never runs code shipped inside a kit.
The question it answers: do the kit's NEW layers make the models better?

DETECT (flood map)
  base = the same 16 radar layers as Flood Model v1 (the new layers dropped). If the kit's split, labels, reference maps and
         radar layers are byte-identical to kit v4, the 3 existing kit v4 runs (v4_B, v4_B_seed1, v4_sarOnly) are the
         base; otherwise 2 base seeds are trained on this kit.
  all  = every layer, 2 seeds.
  Score: VALIDATION blocks, TTA 8, best cutoff, mean flood IoU of the published + independent maps.
  WIN if mean(all) > mean(base) + spread(base). Then: retrain 'all' on train+val for its mean best epoch ->
  ops/models/detect_<kit>_70.pt (+ card) -> TEST blocks scored ONCE: current ensemble / current + new / new alone
  (TTA 8, cutoff 0.30, declared here in advance).
FORECAST (early warning)
  base vs all, 2 seeds each, the risk v4 recipe. Score: validation PR-AUC at each run's best epoch (train.py history).
  WIN rule as above. Then: retrain on train+val -> ops/models/risk_<kit>_70.pt (+ card); test numbers from train.py.
ops_config.json and the models in use are NEVER changed. Report: D:/sar_flood/REPORT_<kit>.md + TRAINING_LOG.md, sent to the PC.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "train_kit_v4"
TRAIN = V4 / "train.py"
PY = sys.executable
TLOG = ROOT / "TRAINING_LOG.md"
DET = ["--task", "detect", "--base", "48", "--lr", "0.0005", "--weight-decay", "0.0001", "--flood-frac", "0.65", "--mmap",
       "--eval-every", "2", "--patch", "256", "--batch", "12", "--steps", "300", "--epochs", "60", "--patience", "12"]
FC = ["--task", "forecast", "--base", "32", "--lr", "0.001", "--weight-decay", "0.0001", "--flood-frac", "0.5", "--mmap",
      "--eval-every", "2", "--patch", "256", "--batch", "8", "--steps", "300", "--epochs", "40", "--patience", "10"]
V4_BASE = ["v4_B", "v4_B_seed1", "v4_sarOnly"]
OPS = ["detect_v3_ops70.pt", "detect_v4_plus70.pt"]
CUT, TTA = 0.30, 8
END = [0.0]
KIT, KITNAME = None, "?"


def tlog(msg):
    with TLOG.open("a", encoding="utf-8") as f:
        f.write(f"\n**{time.strftime('%Y-%m-%d %H:%M')}** [{KITNAME}] {msg}\n")
    print(msg, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train(name, args):
    m = KIT / "runs" / name / "metrics.json"
    if m.exists():
        return json.loads(m.read_text(encoding="utf-8-sig"))
    if time.time() > END[0]:
        tlog(f"time limit reached before training {name}")
        return None
    tlog(f"training {name}: {' '.join(args)}")
    with open(KIT / f"runs_{name}.out", "a", encoding="utf-8") as f:
        rc = subprocess.call([PY, "-u", str(TRAIN), *args, "--kit", str(KIT), "--name", name], cwd=str(V4),
                             stdout=f, stderr=subprocess.STDOUT)
    if rc != 0 or not m.exists():
        tlog(f"{name} FAILED (exit {rc}); see {KIT / f'runs_{name}.out'}")
        return None
    return json.loads(m.read_text(encoding="utf-8-sig"))


def best_pr_auc(mt):
    h = [e for e in (mt.get("history") or []) if isinstance(e, dict) and "val_pr_auc" in e]
    at = [e for e in h if e.get("epoch") == mt.get("best_epoch")]
    return (at or h or [{"val_pr_auc": float("nan")}])[-1]["val_pr_auc"] if h else float("nan")


def decide(base, cand):
    """base/cand: lists of scores. WIN if mean(cand) > mean(base) + spread(base)."""
    if len(base) < 2 or not cand:
        return False, float("nan")
    bar = float(np.mean(base)) + (max(base) - min(base))
    return float(np.mean(cand)) > bar, bar


def main():
    global KIT, KITNAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--hours", type=float, default=70)
    a = ap.parse_args()
    KIT = Path(a.kit).resolve()
    KITNAME = KIT.name.replace("train_kit_", "")
    END[0] = time.time() + a.hours * 3600
    t0 = time.time()
    d = KIT / "data"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8-sig"))
    m4 = json.loads((V4 / "data" / "meta.json").read_text(encoding="utf-8-sig"))
    new = {t: [c for c in meta["tasks"][t]["channels"] if c not in m4["tasks"][t]["channels"]] for t in ("detect", "forecast")}
    tlog(f"kit_queue started. New layers: detect {new['detect']}, forecast {new['forecast']}")
    if not new["detect"] and not new["forecast"]:
        tlog("the kit has no new layers compared with kit v4: nothing to test")
        return 0

    # is everything except the new layers identical to kit v4? Then the kit v4 runs are a fair base.
    same = all((d / f).exists() and sha(d / f) == sha(V4 / "data" / f)
               for f in ("split.npy", "labels.npy", "ref_labels.npy", "ref2_labels.npy", "circle.npy"))
    X = np.load(d / "inputs_detect.npy", mmap_mode="r")
    chans = meta["tasks"]["detect"]["channels"]
    if same:
        X4 = np.load(V4 / "data" / "inputs_detect.npy", mmap_mode="r")
        c4 = m4["tasks"]["detect"]["channels"]
        same = all(np.array_equal(np.asarray(X[chans.index(c)]), np.asarray(X4[i])) for i, c in enumerate(c4))
    tlog(f"split, labels, reference maps and radar layers identical to kit v4: {same}")

    sys.path.insert(0, str(HERE))
    import improve_eval as ie   # noqa: E402
    import torch                # noqa: E402
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = [f"# New kit test: {KITNAME} ({time.strftime('%Y-%m-%d %H:%M')})", "",
          f"New detect layers: {', '.join(new['detect']) or '-'}", f"New forecast layers: {', '.join(new['forecast']) or '-'}",
          f"Identical to kit v4 apart from the new layers: {same}", ""]

    # ---------------- DETECT ----------------
    def setup(split):
        S, C = np.load(d / "split.npy"), np.load(d / "circle.npy").astype(bool)
        R1, R2 = np.load(d / "ref_labels.npy"), np.load(d / "ref2_labels.npy")
        valid = np.asarray(X[chans.index("valid")]) > 0
        blk = (S == meta["split_code"][split]) & C & valid
        sel = {"published": blk & (R1 >= 0), "independent": blk & (R2 != 255)}
        return sel, {"published": R1[sel["published"]] == 1, "independent": R2[sel["independent"]] == 1}

    def val_score(cache_name, ck, sel, truth):
        p = ie.prob(cache_name, ck, TTA, X, chans, dev)
        pv = {k: p[sel[k]] for k in sel}
        c, s = max(((c, ie.score(pv, truth, c)) for c in ie.CUTS),
                   key=lambda t: t[1]["published"]["iou"] + t[1]["independent"]["iou"])
        return {"cut": c, "pub": s["published"]["iou"], "ind": s["independent"]["iou"],
                "mean": round((s["published"]["iou"] + s["independent"]["iou"]) / 2, 4)}

    if new["detect"]:
        drop = ",".join(new["detect"])
        runs = {f"{KITNAME}_det_all_s{s}": DET + ["--seed", str(s)] for s in (0, 1)}
        base_runs = {} if same else {f"{KITNAME}_det_base_s{s}": DET + ["--seed", str(s), "--drop-channels", drop] for s in (0, 1)}
        mts = {n: train(n, args) for n, args in {**runs, **base_runs}.items()}
        sel, truth = setup("val")
        sc = {}
        if same:
            for n in V4_BASE:
                sc[n] = val_score(n, V4 / "runs" / n / "best.pt", sel, truth)
        for n in list(base_runs) + list(runs):
            if mts.get(n):
                sc[n] = val_score(f"{KITNAME}_{n}", KIT / "runs" / n / "best.pt", sel, truth)
        base = [sc[n]["mean"] for n in (V4_BASE if same else base_runs) if n in sc]
        cand = [sc[n]["mean"] for n in runs if n in sc]
        win, bar = decide(base, cand)
        md += ["## Flood map (detect): validation blocks, TTA 8, best cutoff", "",
               "| run | layers | cutoff | IoU published | IoU independent | mean |", "|---|---|---|---|---|---|"]
        for n, v in sc.items():
            md.append(f"| {n} | {'all' if n in runs else 'radar only'} | {v['cut']:.2f} | {v['pub']:.3f} | {v['ind']:.3f} | {v['mean']:.3f} |")
        md += ["", f"Bar = base mean + base spread = **{bar:.3f}**; new layers mean {np.mean(cand) if cand else float('nan'):.3f}. "
               f"**{'WIN' if win else 'No gain beyond luck'}.**", ""]
        tlog(f"detect: base {base}, all {cand}, bar {bar:.3f} -> {'WIN' if win else 'no gain'}")
        if win:
            be = int(round(np.mean([mts[n]["best_epoch"] for n in runs if mts.get(n)])))
            for grp, cols in (("norain", [c for c in new["detect"] if c.startswith("rain")]),
                              ("noland", [c for c in new["detect"] if not c.startswith("rain")])):
                if cols and len(cols) < len(new["detect"]):
                    n = f"{KITNAME}_det_{grp}_s0"
                    if train(n, DET + ["--seed", "0", "--drop-channels", ",".join(cols)]):
                        v = val_score(f"{KITNAME}_{n}", KIT / "runs" / n / "best.pt", sel, truth)
                        md.append(f"Ablation {grp} (dropped {', '.join(cols)}): validation mean {v['mean']:.3f}")
            n70 = f"{KITNAME}_det_all_70"
            args = list(DET)
            args[args.index("--epochs") + 1] = str(be)
            if train(n70, args + ["--seed", "0", "--trainval"]):
                dst = HERE / "models" / f"detect_{n70}.pt"
                shutil.copy2(KIT / "runs" / n70 / "best.pt", dst)
                (dst.with_suffix(".json")).write_text(json.dumps({
                    "file": dst.name, "task": "detect", "channels": chans, "base": 48, "params_M": 17.5,
                    "note": f"kit {KITNAME}, all layers, train+val, {be} epochs. Needs the kit's extra layers in every stack.",
                    "run": str(KIT / "runs" / n70)}, indent=1), encoding="utf-8")
                sel_t, truth_t = setup("test")
                P = {m: ie.prob(f"{KITNAME}_{Path(m).stem}", HERE / "models" / m, TTA, X, chans, dev) for m in OPS + [dst.name]}
                md += ["", f"### Test blocks, scored once (TTA {TTA}, cutoff {CUT})", "",
                       "| models | IoU published | P | R | IoU independent | P | R |", "|---|---|---|---|---|---|---|"]
                for lab, mem in (("CURRENT (v3_ops70 + v4_plus70)", OPS), ("CURRENT + new", OPS + [dst.name]), ("new alone", [dst.name])):
                    p = np.mean([P[x] for x in mem], 0)
                    s = ie.score({k: p[sel_t[k]] for k in sel_t}, truth_t, CUT)
                    md.append(f"| {lab} | {s['published']['iou']:.3f} | {s['published']['p']:.3f} | {s['published']['r']:.3f} | "
                              f"{s['independent']['iou']:.3f} | {s['independent']['p']:.3f} | {s['independent']['r']:.3f} |")
                md += ["", f"Saved ops/models/{dst.name}. NOT switched on: it needs the new layers, which only this kit has."]

    # ---------------- FORECAST ----------------
    if new["forecast"]:
        drop = ",".join(new["forecast"])
        runs = {f"{KITNAME}_fc_all_s{s}": FC + ["--seed", str(s)] for s in (0, 1)}
        base_runs = {f"{KITNAME}_fc_base_s{s}": FC + ["--seed", str(s), "--drop-channels", drop] for s in (0, 1)}
        mts = {n: train(n, args) for n, args in {**base_runs, **runs}.items()}
        base = [best_pr_auc(mts[n]) for n in base_runs if mts.get(n)]
        cand = [best_pr_auc(mts[n]) for n in runs if mts.get(n)]
        win, bar = decide(base, cand)
        md += ["## Early warning (forecast): validation PR-AUC at the best epoch", "", "| run | layers | PR-AUC | best epoch |", "|---|---|---|---|"]
        for n in list(base_runs) + list(runs):
            if mts.get(n):
                md.append(f"| {n} | {'all' if n in runs else 'without new'} | {best_pr_auc(mts[n]):.3f} | {mts[n].get('best_epoch')} |")
        md += ["", f"Bar = **{bar:.3f}**. **{'WIN' if win else 'No gain beyond luck'}.**", ""]
        tlog(f"forecast: base {base}, all {cand}, bar {bar:.3f} -> {'WIN' if win else 'no gain'}")
        if win:
            be = int(round(np.mean([mts[n]["best_epoch"] for n in runs if mts.get(n)])))
            n70 = f"{KITNAME}_fc_all_70"
            args = list(FC)
            args[args.index("--epochs") + 1] = str(be)
            mt = train(n70, args + ["--seed", "0", "--trainval"])
            if mt:
                dst = HERE / "models" / f"risk_{n70}.pt"
                shutil.copy2(KIT / "runs" / n70 / "best.pt", dst)
                f = (mt.get("test_vs_zenodo_reference") or {}).get("flooded") or {}
                (dst.with_suffix(".json")).write_text(json.dumps({
                    "file": dst.name, "task": "forecast", "channels": meta["tasks"]["forecast"]["channels"], "base": 32,
                    "note": f"kit {KITNAME}, all layers, train+val, {be} epochs", "run": str(KIT / "runs" / n70),
                    "test_vs_published_map_flooded": {k: round(v, 3) for k, v in f.items() if isinstance(v, float)}}, indent=1),
                    encoding="utf-8")
                md.append(f"Saved ops/models/{dst.name}. Test (published map, flooded class): "
                          + ", ".join(f"{k} {v:.3f}" for k, v in f.items() if isinstance(v, float)))

    md += ["", f"Run time {(time.time() - t0) / 3600:.1f} h. The models in use were not changed.",
           "Caveat: the training labels come from the published Mondal map (Zenodo 4390084); the independent map is the honest check."]
    rep = ROOT / f"REPORT_{KITNAME}.md"
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    tlog(f"report written: {rep.name}")
    subprocess.run([PY, str(HERE / "bridge.py"), "send", f"Laptop finished testing kit {KITNAME}. Report attached.",
                    "--subject", f"kit {KITNAME} result", "--attach", str(rep)], cwd=str(HERE), timeout=1800)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
