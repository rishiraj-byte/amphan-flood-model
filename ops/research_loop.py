"""research_loop.py - unattended, BOUNDED search for a better detection recipe (kit v4), with a plain-English log.

Waits for train_queue.py (v5) to finish, then greedy one-change-at-a-time search:
  best = best recipe so far (old recipe, or the v5 winner)
  for each idea: train best+change on TRAIN blocks, score VALIDATION blocks (TTA 8, best cutoff, mean IoU of
  published + independent maps). Keep the change only if it beats best by more than the seed spread (luck bar).
At the end, if the best recipe is new: retrain it on train+val with 2 seeds, add to the ops ensemble and score
TEST blocks ONCE (pre-declared: TTA 8, cutoff 0.30). Never switches the console's models on by itself.
Log: D:/sar_flood/TRAINING_LOG.md (every decision + why). State: ops/improve/research_state.json.
Stops at --hours (default 60) or when ops/RESEARCH_STOP exists.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import train_queue as tq

ie = tq.ie
ROOT = Path(r"D:\sar_flood")
LOG = ROOT / "TRAINING_LOG.md"
STATE = tq.HERE / "improve" / "research_state.json"
STOP = tq.HERE / "RESEARCH_STOP"
OLD_RECIPE = ["--patch", "256", "--batch", "12", "--steps", "300", "--epochs", "60", "--patience", "12"]
IDEAS = [  # (name, why, change applied on top of the best recipe)
    ("ff80", "more flood-centred patches, to attack low recall", {"--flood-frac": "0.8"}),
    ("base64", "wider network (more capacity)", {"--base": "64", "--batch": "8"}),
    ("lr1e3", "higher learning rate", {"--lr": "0.001"}),
    ("wd1e3", "stronger weight decay (less overfitting to Mondal labels)", {"--weight-decay": "0.001"}),
    ("ff50", "fewer flood-centred patches (better dry/water context)", {"--flood-frac": "0.5"}),
]


def note(text):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n**{time.strftime('%Y-%m-%d %H:%M')}** {text}\n")
    tq.log("[research] " + text.replace("\n", " ")[:300])


def apply(recipe, change):
    a = list(recipe)
    for k, v in change.items():
        if k in a:
            a[a.index(k) + 1] = v
        else:
            a += [k, v]
    return a


def full_args(recipe):
    """BASE holds --base/--lr/... ; recipe may override them."""
    a = list(tq.BASE)
    i = 0
    while i < len(recipe):
        k, v = recipe[i], recipe[i + 1]
        if k in a:
            a[a.index(k) + 1] = v
        else:
            a += [k, v]
        i += 2
    return a


def val_scores(names, dev):
    X, chans, sel, truth = tq.setup("val")
    out = {}
    for n in names:
        ck = tq.KIT / "runs" / n / "best.pt"
        if not ck.exists():
            continue
        p = ie.prob(n, ck, tq.TTA, X, chans, dev)
        pv = {k: p[sel[k]] for k in sel}
        c, s = max(((c, ie.score(pv, truth, c)) for c in ie.CUTS),
                   key=lambda t: t[1]["published"]["iou"] + t[1]["independent"]["iou"])
        m = json.loads((tq.KIT / "runs" / n / "metrics.json").read_text())
        out[n] = {"cut": c, "pub": s["published"]["iou"], "ind": s["independent"]["iou"],
                  "mean": round((s["published"]["iou"] + s["independent"]["iou"]) / 2, 4), "best_epoch": m.get("best_epoch")}
    return out


def save(st):
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=60)
    a = ap.parse_args()
    end = time.time() + a.hours * 3600
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not LOG.exists():
        LOG.write_text("# Training log (laptop, kit v4)\n\nEvery experiment and decision, newest at the bottom. Rule: a change is kept "
                       "only if its VALIDATION score (mean IoU of published + independent map, TTA 8, best cutoff) beats the best "
                       "so far by more than the seed spread of the old recipe. Test blocks are scored only at the very end, once.\n",
                       encoding="utf-8")
    note("Research loop started. Waiting for training queue v5 to finish.")
    while not tq.REP.exists() and time.time() < end and not STOP.exists():
        time.sleep(300)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not st:
        sc = val_scores(tq.OLD + list(tq.CANDS), dev)
        old = [sc[b]["mean"] for b in tq.OLD if b in sc]
        spread = round(max(old) - min(old), 4)
        best = {"name": "old recipe (v4_B x3 seeds)", "recipe": OLD_RECIPE, "mean": round(float(np.mean(old)), 4)}
        lines = [f"Old recipe, 3 seeds: {', '.join(f'{b} {sc[b]['mean']:.3f}' for b in tq.OLD if b in sc)} -> mean "
                 f"{best['mean']:.3f}, luck spread {spread:.3f}."]
        for n in tq.CANDS:
            if n in sc:
                win = sc[n]["mean"] > best["mean"] + spread
                lines.append(f"{n} ({' '.join(tq.CANDS[n])}): {sc[n]['mean']:.3f} (pub {sc[n]['pub']:.3f}, ind {sc[n]['ind']:.3f}) -> "
                             f"{'KEEP, beats the luck bar' if win else 'drop, within luck'}.")
                if win and sc[n]["mean"] > best.get("mean", 0) and n not in ("",):
                    best = {"name": n, "recipe": tq.CANDS[n], "mean": sc[n]["mean"]}
        st = {"best": best, "spread": spread, "tried": [], "scores": sc}
        save(st)
        note("Queue v5 result.\n- " + "\n- ".join(lines) + f"\n- Best recipe now: **{best['name']}** ({best['mean']:.3f}).")
    for key, why, change in IDEAS:
        if time.time() > end or STOP.exists():
            note("Stopping: time limit or RESEARCH_STOP.")
            break
        if key in st["tried"]:
            continue
        name = f"v6_{key}"
        recipe = apply(st["best"]["recipe"], change)
        note(f"Experiment **{name}**: {why}. Change {change} on top of {st['best']['name']}.")
        tq.train(name, full_args(recipe))
        s = val_scores([name], dev).get(name)
        st["tried"].append(key)
        if not s:
            note(f"{name} failed (no model written). Skipped. See train_kit_v4/runs_{name}.out.")
            save(st)
            continue
        st["scores"][name] = s
        bar = st["best"]["mean"] + st["spread"]
        keep = s["mean"] > bar
        note(f"{name}: validation {s['mean']:.3f} (pub {s['pub']:.3f}, ind {s['ind']:.3f}, cutoff {s['cut']}, best epoch "
             f"{s['best_epoch']}). Bar {bar:.3f} -> **{'KEEP: new best recipe' if keep else 'DROP: within luck'}**.")
        if keep:
            st["best"] = {"name": name, "recipe": recipe, "mean": s["mean"]}
        save(st)
    best = st["best"]
    if best["recipe"] == OLD_RECIPE or st.get("final_done"):
        note("Final: no recipe beat the old one by more than luck. Models in use stay as they are (TTA 8, cutoff 0.30). "
             "Conclusion: the limit is the data (one storm, one label map), not the training settings." if not st.get("final_done") else "Final already done.")
    elif time.time() < end and not STOP.exists():
        be = st["scores"].get(best["name"], {}).get("best_epoch") or 40
        names = []
        for seed in ("0", "1"):
            rec = apply(best["recipe"], {"--epochs": str(be), "--seed": seed})
            n70 = f"{best['name']}_70s{seed}"
            note(f"Final: retraining best recipe {best['name']} on train+val, seed {seed}, {be} epochs -> {n70}.")
            tq.train(n70, full_args(rec) + ["--trainval"])
            if (tq.KIT / "runs" / n70 / "best.pt").exists():
                dst = tq.HERE / "models" / f"detect_{n70}.pt"
                shutil.copy2(tq.KIT / "runs" / n70 / "best.pt", dst)
                ck = torch.load(dst, map_location="cpu", weights_only=False)
                dst.with_suffix(".json").write_text(json.dumps({
                    "file": dst.name, "task": "detect", "channels": ck["channels"], "base": ck["base"],
                    "note": f"storm-agnostic, kit v4, recipe {best['name']}, train+val, seed {seed}", "run": str(tq.KIT / "runs" / n70)},
                    indent=1), encoding="utf-8")
                names.append(dst.name)
        if names:
            X, chans, sel, truth = tq.setup("test")
            P = {m: ie.prob(Path(m).stem, tq.HERE / "models" / m, tq.TTA, X, chans, dev) for m in tq.OPS + names}
            rows = []
            for lab, mem in (("current ensemble", tq.OPS), ("current + new", tq.OPS + names), ("new only", names)):
                p = np.mean([P[x] for x in mem], 0)
                s = ie.score({k: p[sel[k]] for k in sel}, truth, tq.CUT)
                rows.append(f"| {lab} | {s['published']['iou']:.3f} | {s['independent']['iou']:.3f} | "
                            f"{s['independent']['p']:.3f} / {s['independent']['r']:.3f} |")
            note("Final TEST, scored once (TTA 8, cutoff 0.30):\n\n| ensemble | IoU published | IoU independent | P / R independent |\n"
                 "|---|---|---|---|\n" + "\n".join(rows) + f"\n\nNew models: {', '.join(names)}. Not switched on automatically.")
        st["final_done"] = True
        save(st)
    subprocess.run([tq.PY, str(tq.HERE / "bridge.py"), "send", "Laptop research loop finished. Training log attached.",
                    "--subject", "training log", "--attach", str(LOG)], cwd=str(tq.HERE), timeout=900)
    note("Research loop ended. Log sent to the PC.")


if __name__ == "__main__":
    main()
