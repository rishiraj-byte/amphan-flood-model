"""live_run.py - run ONE detection model tile by tile and publish every step, so the console can show the model
working in real time: the map filling in, the network's real layer activity for the current tile, and (if the stack
has reference maps) a running score on the held-out test blocks.

  python live_run.py --stack STACK_DIR --out OUT/live --model models/detect_v4_plus70.pt [--pace 0.35] [--cpu]

--pace only adds a pause between tiles so a person can watch; state.json reports the real compute time per tile.
Writes OUT/live: state.json, radar.png, tiles/t_<k>.png (flood overlay of each tile), act_<layer>.png (current tile),
in.png / out.png (current tile). Create OUT/live/STOP to stop early.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from predict import load_model, pixel_area_rows, preview_step, radar_png, ramp, rgba, tile_starts, write_png

LAYERS = ["enc0", "enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1", "dec0"]


def dump(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    for _ in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.02)


def act_png(a, path, max_px=128):
    """a: (C,h,w) activations -> mean |a| over channels, colour-mapped, as a small PNG."""
    m = np.abs(a).mean(0)
    s = max(1, int(np.ceil(max(m.shape) / max_px)))
    m = m[::s, ::s]
    hi = float(np.percentile(m, 99)) or 1.0
    write_png(path, ramp(m / hi, [(0, (8, 10, 30)), (0.35, (90, 60, 200)), (0.7, (40, 190, 255)), (1, (235, 255, 255))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pace", type=float, default=0.35)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    stack, out = Path(a.stack), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "tiles").mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    meta = json.loads((stack / "meta.json").read_text(encoding="utf-8"))
    g = meta["grid"]
    H, W = g["H"], g["W"]
    X = np.load(stack / "inputs_detect.npy", mmap_mode="r")
    chans = meta["tasks"]["detect"]["channels"]
    model, ck = load_model(a.model, device)
    miss = [c for c in ck["channels"] if c not in chans]
    if miss:
        raise SystemExit(f"model needs layers this stack lacks: {miss}")
    idx = [chans.index(c) for c in ck["channels"]]
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    s = preview_step(H, W)
    radar_png(X, chans, "a_vv", s, out / "radar.png")
    area = np.asarray(meta.get("pixel_area_km2_rows") or pixel_area_rows(g), np.float64)[:, None]
    valid = np.asarray(X[chans.index("valid")]) > 0 if "valid" in chans else np.ones((H, W), bool)

    # optional running score vs the published map on held-out test blocks
    ref = np.load(stack / "ref_labels.npy") if (stack / "ref_labels.npy").exists() else None
    test = None
    if ref is not None and (stack / "split.npy").exists() and "split_code" in meta:
        circ = np.load(stack / "circle.npy").astype(bool) if (stack / "circle.npy").exists() else np.ones((H, W), bool)
        test = (np.load(stack / "split.npy") == meta["split_code"]["test"]) & circ & (ref >= 0)
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.uint8)
    flag = np.zeros((H, W), bool)
    tot = {"flood_km2": 0.0, "covered_km2": 0.0, "tp": 0, "fp": 0, "fn": 0}

    acts = {}
    hooks = [m.register_forward_hook(lambda _m, _i, o, n=n: acts.__setitem__(n, o)) for n, m in
             [(f"enc{i}", model.enc[i]) for i in range(5)] + [(f"dec{i}", model.dec[i]) for i in range(4)]]
    tiles = [(r, c) for r in tile_starts(H, 512, 64) for c in tile_starts(W, 512, 64)]
    st = {"state": "running", "model": Path(a.model).name, "device": device.type, "stack": str(stack),
          "grid": {"H": H, "W": W, "step": s, "w": -(-W // s), "h": -(-H // s), "res": g["res"], "north": g["north"]},
          "total": len(tiles), "done": 0, "tiles": [], "pace": a.pace, "started": time.time(),
          "layers": LAYERS, "channels": {}, "has_score": test is not None, "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    compute = []
    for k, (r0, c0) in enumerate(tiles):
        if (out / "STOP").exists():
            st["state"] = "stopped"
            break
        x = np.asarray(X[idx, r0:r0 + 512, c0:c0 + 512], np.float32)
        th, tw = x.shape[1:]
        v = valid[r0:r0 + th, c0:c0 + tw]
        if v.mean() < 0.01:
            st["tiles"].append([k, r0, c0, th, tw, 0])
            st["done"] = k + 1
            continue
        x = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0, posinf=0.0, neginf=0.0)
        ph, pw = (-th) % 16, (-tw) % 16
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(th, tw) > 16 else "constant")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pr = model(torch.from_numpy(x)[None].to(device)).float().softmax(1)[0, :, :th, :tw].cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute.append(time.time() - t0)
        # merge into the running map (overlaps averaged) and update totals for this tile's area only
        sl = (slice(r0, r0 + th), slice(c0, c0 + tw))
        was = cnt[sl] > 0                      # pixels already scored by an earlier (overlapping) tile
        acc[sl] += pr[1]
        cnt[sl] += 1
        new = (acc[sl] / np.maximum(cnt[sl], 1) >= 0.5) & v
        ar = area[r0:r0 + th]
        tot["flood_km2"] += float(((new.astype(np.int8) - flag[sl]) * ar).sum())
        tot["covered_km2"] += float(((v & ~was) * ar).sum())
        if test is not None:                   # running score over the tiles done so far only
            t_, rf = test[sl], ref[sl] == 1
            for key, f in (("tp", lambda p, c: p & rf & t_ & c), ("fp", lambda p, c: p & ~rf & t_ & c),
                           ("fn", lambda p, c: ~p & rf & t_ & c)):
                tot[key] += int(f(new, np.ones_like(was)).sum()) - int(f(flag[sl], was).sum())
        flag[sl] = new
        # pictures: this tile's flood overlay, its radar input, its output, the network's layer activity
        cls = pr.argmax(0)
        sub = (slice(None, None, s), slice(None, None, s))
        p1, c1, v1 = pr[1][sub], cls[sub], v[sub]
        write_png(out / "tiles" / f"t_{k}.png", rgba(p1.shape, [((c1 == 2) & v1, (24, 44, 96, 210)),
                                                            ((p1 >= 0.5) & v1, (40, 200, 255, 240))]))
        ai = np.asarray(X[chans.index("a_vv"), r0:r0 + th:2, c0:c0 + tw:2], np.float32) if "a_vv" in chans else x[0, ::2, ::2]
        g8 = (np.clip((np.nan_to_num(ai, nan=-30) + 25) / 25, 0, 1) ** 0.8 * 255).astype(np.uint8)
        write_png(out / "in.png", np.repeat(g8[..., None], 3, 2))
        write_png(out / "out.png", ramp(pr[1][::2, ::2], [(0, (12, 14, 40)), (0.5, (60, 200, 255)), (1, (240, 255, 255))]))
        lstats = {}
        for n in LAYERS:
            o = acts[n][0].float().cpu().numpy()
            act_png(o, out / f"act_{n}.png")
            lstats[n] = {"mean": round(float(np.abs(o).mean()), 4), "active": round(float((o > 0).mean()), 3),
                         "channels": int(o.shape[0]), "size": [int(o.shape[1]), int(o.shape[2])]}
        st["tiles"].append([k, r0, c0, th, tw, 1])
        iou = tot["tp"] / max(tot["tp"] + tot["fp"] + tot["fn"], 1) if test is not None else None
        st.update(done=k + 1, current={"k": k, "row": r0, "col": c0, "h": th, "w": tw,
                                       "flood_pct": round(float((pr[1] >= 0.5)[v].mean() * 100), 1),
                                       "mean_prob": round(float(pr[1][v].mean()), 3)},
                  stats=lstats, compute_ms=round(compute[-1] * 1000, 1), avg_compute_ms=round(float(np.mean(compute)) * 1000, 1),
                  flood_km2=round(tot["flood_km2"], 1), covered_km2=round(tot["covered_km2"], 1),
                  test_iou=None if iou is None else round(iou, 3),
                  test_precision=None if test is None else round(tot["tp"] / max(tot["tp"] + tot["fp"], 1), 3),
                  test_recall=None if test is None else round(tot["tp"] / max(tot["tp"] + tot["fn"], 1), 3),
                  updated=time.time())
        dump(out / "state.json", st)
        time.sleep(a.pace)
    for h in hooks:
        h.remove()
    if st["state"] == "running":
        st["state"] = "done"
    st["updated"] = time.time()
    st["elapsed_s"] = round(time.time() - st["started"], 1)
    dump(out / "state.json", st)
    print(f"{st['state']}: {st['done']}/{st['total']} tiles, flood {tot['flood_km2']:.0f} km2, "
          f"avg compute {st.get('avg_compute_ms')} ms/tile", flush=True)


if __name__ == "__main__":
    main()
