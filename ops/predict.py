"""predict.py - run the trained flood models on ANY storm's input stack (no labels needed).

  python predict.py --stack STACK_DIR --out OUT_DIR [--detect a.pt b.pt ...] [--risk r.pt] [--cpu]

STACK_DIR is what the PC kit builder writes (same format as a kit's data/ folder, labels not needed):
  meta.json            grid {west, north, res, W, H, crs} and tasks.detect.channels / tasks.forecast.channels
  inputs_detect.npy    (C, H, W) float, radar layers AFTER the storm   -> flood detection
  inputs_forecast.npy  (C, H, W) float, radar BEFORE landfall + storm track layers -> flood risk (early warning)
Each model picks its own channels by NAME, so the stack may hold extra layers.

OUT_DIR gets: detect_prob.npy, detect_class.npy (0 dry, 1 flood, 2 water, 255 no data), risk_prob.npy,
flood_map.png, risk_map.png, *.tif (if rasterio is installed) and summary.json.
Several --detect models are averaged (ensemble); flood = probability >= --threshold (0.5, fixed, not tuned).
"""
import argparse
import json
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


# ---------------- model (identical to train.py) ----------------
def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, in_ch, n_cls, base=32, dropout=0.1):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([conv_block(in_ch, ch[0])] + [conv_block(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([conv_block(ch[i] * 2, ch[i]) for i in range(4)])
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(ch[0], n_cls, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.drop(skips.pop())
        for i in reversed(range(4)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)


def tile_starts(n, tile, overlap):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, tile - overlap)) + [n - tile]
    return sorted(set(s))


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = UNet(ck["in_ch"], ck["n_cls"], ck["base"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


TTA_VIEWS = [(False, False, False), (True, False, False), (False, True, False), (True, True, False),
             (False, False, True), (True, False, True), (False, True, True), (True, True, True)]  # (flip rows, flip cols, transpose)


def _views(x, tta):
    """x (C,h,w) -> list of (view, (fr, fc, tr)) for test-time augmentation: 1 = none, 4 = flips, 8 = flips + transpose."""
    out = []
    for fr, fc, tr in TTA_VIEWS[:max(1, int(tta))]:
        v = x.transpose(0, 2, 1) if tr else x
        v = v[:, ::-1] if fr else v
        v = v[:, :, ::-1] if fc else v
        out.append((np.ascontiguousarray(v), (fr, fc, tr)))
    return out


def _unview(p, how):
    fr, fc, tr = how
    p = p[:, :, ::-1] if fc else p
    p = p[:, ::-1] if fr else p
    return p.transpose(0, 2, 1) if tr else p


def run_model(path, stack_chans, inputs, device, log, tile=512, overlap=64, tta=1):
    """-> flood probability (H, W) float32, full class probabilities (n_cls, H, W).
    tta: 1 = plain; 4 = average over the 4 flips; 8 = flips + transpose (the same network, 8 views of each tile)."""
    model, ck = load_model(path, device)
    missing = [c for c in ck["channels"] if c not in stack_chans]
    if missing:
        log(f"  SKIP {Path(path).name}: this stack lacks its layers {missing}")
        return None
    idx = [stack_chans.index(c) for c in ck["channels"]]
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    C, H, W = inputs.shape
    acc = np.zeros((ck["n_cls"], H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    t0 = time.time()
    rows = tile_starts(H, tile, overlap)
    for k, r0 in enumerate(rows):
        for c0 in tile_starts(W, tile, overlap):
            x = np.asarray(inputs[idx, r0:r0 + tile, c0:c0 + tile], np.float32)
            x = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0, posinf=0.0, neginf=0.0)
            th, tw = x.shape[1:]
            ph, pw = (-th) % 16, (-tw) % 16
            if ph or pw:
                x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(th, tw) > 16 else "constant")
            pr = 0
            views = _views(x, tta)
            for grp in ([v for v in views if not v[1][2]], [v for v in views if v[1][2]]):   # same-shape views batched
                if not grp:
                    continue
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                     enabled=device.type == "cuda"):
                    o = model(torch.from_numpy(np.stack([v for v, _ in grp])).to(device)).float().softmax(1).cpu().numpy()
                for oi, (_, how) in zip(o, grp):
                    pr = pr + _unview(oi, how)
            pr = (pr / len(views))[:, :th, :tw]
            acc[:, r0:r0 + th, c0:c0 + tw] += pr
            cnt[r0:r0 + th, c0:c0 + tw] += 1
        log(f"  {Path(path).stem}: row {k + 1}/{len(rows)}  ({time.time() - t0:.0f} s)")
    acc /= np.maximum(cnt, 1)
    return acc[1], acc, ck


# ---------------- outputs ----------------
# ---------------- map layers + scoring (for the control panel) ----------------
def write_png(path, img):
    h, w, c = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if c == 4 else 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def rgba(shape, mask_colors):
    """mask_colors: list of (bool mask, (r,g,b,a)) painted in order on a transparent image."""
    img = np.zeros(shape + (4,), np.uint8)
    for m, col in mask_colors:
        img[m] = col
    return img


def ramp(t, stops):
    """t in [0,1] (H,W) -> RGB via linear colour stops [(pos,(r,g,b)),...]."""
    t = np.clip(np.nan_to_num(t), 0, 1)
    out = np.zeros(t.shape + (3,), np.float32)
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        m = (t >= p0) & (t <= p1)
        u = ((t[m] - p0) / max(p1 - p0, 1e-6))[:, None]
        out[m] = np.array(c0) + (np.array(c1) - np.array(c0)) * u
    return out.astype(np.uint8)


def radar_png(X, chans, name, s, path):
    if name not in chans:
        return False
    a = np.asarray(X[chans.index(name), ::s, ::s], np.float32)
    g = np.clip((np.nan_to_num(a, nan=-30) + 25) / 25, 0, 1) ** 0.8 * 255
    img = np.repeat(g.astype(np.uint8)[..., None], 3, 2)
    img[~np.isfinite(a)] = (8, 12, 22)
    write_png(path, img)
    return True


def score(pf, ref, mask, area):
    k = mask & (ref >= 0) & (ref != 255)
    rf = (ref == 1) & k
    tp, fp, fn = int((pf & rf).sum()), int((pf & ~rf & k).sum()), int((~pf & rf).sum())
    d = lambda a, b: round(a / b, 3) if b else None
    return {"iou": d(tp, tp + fp + fn), "precision": d(tp, tp + fp), "recall": d(tp, tp + fn),
            "f1": d(2 * tp, 2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn,
            "ref_flood_km2": round(float((rf * area).sum()), 1), "pred_flood_km2": round(float(((pf & k) * area).sum()), 1),
            "scored_km2": round(float((k * area).sum()), 1)}


def pr_auc(p, y):
    """average precision of scores p for binary truth y."""
    o = np.argsort(-p, kind="stable")
    y = y[o].astype(np.float64)
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return round(float((prec * y).sum() / max(y.sum(), 1)), 3)


def evaluate(stack, meta, out, area, s, det_probs, ens_prob, risk, log, thr=0.5):
    """Scores vs reference maps, IF the stack carries them (training kits do; a new storm usually does not)."""
    refs = {k: stack / f for k, f in (("published", "ref_labels.npy"), ("independent", "ref2_labels.npy"))
            if (stack / f).exists()}
    if not refs:
        return None
    H, W = meta["grid"]["H"], meta["grid"]["W"]
    circle = np.load(stack / "circle.npy").astype(bool) if (stack / "circle.npy").exists() else np.ones((H, W), bool)
    test = None
    if (stack / "split.npy").exists() and "split_code" in meta:
        test = (np.load(stack / "split.npy") == meta["split_code"]["test"]) & circle
    ev = {"threshold": thr, "areas": {"test": "held-out test blocks (never trained on)" if test is not None else None,
                                      "all": "whole area (includes areas the models trained on)"},
          "reference_notes": {"published": "Mondal et al. / Zenodo 4390084 - the SAME map the training labels came from",
                              "independent": (meta.get("ref2_info") or {}).get("source", "second map, never used in training")},
          "ensemble": {}, "per_model": {}}
    R = {k: np.load(p) for k, p in refs.items()}
    pf = np.nan_to_num(ens_prob, nan=0) >= thr
    for k, ref in R.items():
        if test is not None:
            ev["ensemble"][f"{k}_test"] = score(pf, ref, test, area)
        ev["ensemble"][f"{k}_all"] = score(pf, ref, circle, area)
        for name, p in det_probs.items():
            if test is not None:
                ev["per_model"].setdefault(name, {})[f"{k}_test"] = score(np.nan_to_num(p, nan=0) >= thr, ref, test, area)
    if risk is not None and "published" in R and test is not None:
        ref = R["published"]
        k = test & ((ref == 0) | (ref == 1)) & np.isfinite(risk)
        p, y = risk[k], ref[k] == 1
        cut = float(np.quantile(p, 0.9)) if p.size else 1
        ev["risk"] = {"pr_auc_test": pr_auc(p, y), "flood_base_rate_test": round(float(y.mean()), 3),
                      "share_of_real_flood_in_top10pct_risk": round(float(y[p >= cut].sum() / max(y.sum(), 1)), 3),
                      "note": "PR-AUC of random guessing = the base rate"}
    # agreement layer vs the published map + test-block layer
    ref = R.get("published", next(iter(R.values())))[::s, ::s]
    pv = pf[::s, ::s]
    kn = (ref >= 0) & (ref != 255)
    write_png(out / "layers" / "agree.png", rgba(pv.shape, [((pv & (ref == 1)), (40, 220, 255, 235)),
                                                         ((pv & (ref != 1) & kn), (255, 150, 40, 235)),
                                                         ((~pv & (ref == 1)), (255, 60, 170, 235))]))
    for k, r in R.items():
        r = r[::s, ::s]
        write_png(out / "layers" / f"ref_{k}.png", rgba(r.shape, [((r == 2), (24, 44, 96, 200)), ((r == 1), (60, 160, 255, 235))]))
    if test is not None:
        t = test[::s, ::s]
        edge = t ^ (np.roll(t, 1, 0) & np.roll(t, -1, 0) & np.roll(t, 1, 1) & np.roll(t, -1, 1))
        write_png(out / "layers" / "test_blocks.png", rgba(t.shape, [(t, (150, 120, 255, 40)), (edge, (170, 150, 255, 255))]))
    (out / "eval.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    e = ev["ensemble"]
    log("scores: " + "; ".join(f"{k} IoU {v['iou']}" for k, v in e.items()))
    return ev


def preview_step(H, W, max_px=1400):
    return max(1, math.ceil(max(H, W) / max_px))


def flood_png(cls, path):
    s = preview_step(*cls.shape)
    c = cls[::s, ::s]
    pal = np.zeros((256, 3), np.uint8) + 255
    pal[0], pal[1], pal[2] = (226, 229, 222), (30, 110, 235), (20, 40, 90)
    write_png(path, pal[c])


def risk_png(risk, valid, path):
    s = preview_step(*risk.shape)
    r, v = np.nan_to_num(risk[::s, ::s]), valid[::s, ::s]
    t = np.clip(r, 0, 1)[..., None]
    lo, hi = np.array([255, 245, 200], np.float32), np.array([190, 20, 30], np.float32)
    img = (lo + (hi - lo) * t).astype(np.uint8)
    img[~v] = 255
    write_png(path, img)


def write_tifs(out, grid, arrays):
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return []
    tf = from_origin(grid["west"], grid["north"], grid["res"], grid["res"])
    done = []
    for name, (arr, dtype, nodata) in arrays.items():
        prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                    crs=grid.get("crs", "EPSG:4326"), transform=tf, compress="deflate", tiled=True)
        with rasterio.open(out / name, "w", dtype=dtype, nodata=nodata, **prof) as d:
            d.write(arr.astype(dtype), 1)
        done.append(name)
    return done


def pixel_area_rows(grid):
    lat = grid["north"] - (np.arange(grid["H"]) + 0.5) * grid["res"]
    return (grid["res"] * 111.32) ** 2 * np.cos(np.radians(lat))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detect", nargs="*", default=None, help="detection checkpoints (default: models/detect_*.pt)")
    ap.add_argument("--risk", default=None, help="risk checkpoint (default: first models/risk_*.pt)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", type=int, default=1, choices=[1, 4, 8],
                    help="test-time augmentation: 1 none, 4 flips, 8 flips+transpose (about N times slower)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    stack, out = Path(a.stack), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        s = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
        (out / "progress.txt").write_text(s, encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    meta = json.loads((stack / "meta.json").read_text(encoding="utf-8"))
    grid = meta["grid"]
    area = np.asarray(meta.get("pixel_area_km2_rows") or pixel_area_rows(grid), np.float64)[:, None]
    det = a.detect if a.detect is not None else sorted(str(p) for p in (HERE / "models").glob("detect_*.pt"))
    risks = sorted((HERE / "models").glob("risk_*.pt"))   # first = preferred (risk_0_... sorts first)
    risk_ck = a.risk or (str(risks[0]) if risks else None)
    log(f"device {device}; stack {stack}; detect models {[Path(p).name for p in det]}; risk {risk_ck}")
    summary = {"name": a.name or stack.name, "created": time.strftime("%Y-%m-%d %H:%M"), "grid": grid,
               "threshold": a.threshold, "tta": a.tta, "device": device.type}
    step = preview_step(grid["H"], grid["W"])
    (out / "layers").mkdir(exist_ok=True)
    det_probs, ens_prob, risk_arr = {}, None, None

    # ---- detection (after-storm radar) ----
    fi = stack / "inputs_detect.npy"
    if det and fi.exists():
        X = np.load(fi, mmap_mode="r")
        chans = meta["tasks"]["detect"]["channels"]
        valid = np.asarray(X[chans.index("valid")]) > 0 if "valid" in chans else np.isfinite(X[0])
        probs, full = [], []
        used = []
        for p in det:
            res = run_model(p, chans, X, device, log, tta=a.tta)
            if res:
                probs.append(res[0])
                full.append(res[1])
                used.append(Path(p).name)
                det_probs[Path(p).stem] = np.where(valid, res[0], np.nan)
        if not probs:
            raise SystemExit("no detection model fits this stack's layers")
        prob = np.mean(probs, 0)
        cls_prob = np.mean(full, 0)
        cls = np.where(prob >= a.threshold, 1, np.where(cls_prob.argmax(0) == 2, 2, 0)).astype(np.uint8)
        cls[~valid] = 255
        prob = np.where(valid, prob, np.nan).astype(np.float32)
        np.save(out / "detect_prob.npy", prob.astype(np.float16))
        np.save(out / "detect_class.npy", cls)
        flood_png(cls, out / "flood_map.png")
        ens_prob = prob
        for nm, lbl in (("a_vv", "radar_after"), ("b_vv", "radar_before")):
            radar_png(X, chans, nm, step, out / "layers" / f"{lbl}.png")
        c = cls[::step, ::step]
        write_png(out / "layers" / "flood.png", rgba(c.shape, [((c == 2), (24, 44, 96, 210)), ((c == 1), (40, 200, 255, 240))]))
        pv = np.nan_to_num(prob[::step, ::step], nan=0)
        img = np.concatenate([ramp(pv, [(0, (40, 30, 120)), (0.5, (60, 200, 255)), (1, (240, 255, 255))]),
                              (np.clip((pv - 0.15) / 0.5, 0, 1) * 235).astype(np.uint8)[..., None]], 2)
        write_png(out / "layers" / "prob.png", img)
        g8 = (np.clip(pv, 0, 1) * 255 + 0.5).astype(np.uint8)       # raw probability (grey) + valid mask, for the live cutoff slider
        write_png(out / "layers" / "prob_gray.png", np.stack([g8, g8, g8, (valid[::step, ::step] * 255).astype(np.uint8)], 2))
        summary["detect"] = {
            "models": used,
            "flood_km2": round(float(((cls == 1) * area).sum()), 1),
            "permanent_water_km2": round(float(((cls == 2) * area).sum()), 1),
            "radar_covered_km2": round(float((valid * area).sum()), 1),
            "flood_km2_if_cutoff_0.3": round(float(((prob >= 0.3) * area).sum()), 1),
            "flood_km2_if_cutoff_0.7": round(float(((prob >= 0.7) * area).sum()), 1),
        }
        summary["tifs"] = write_tifs(out, grid, {"flood_class.tif": (cls, "uint8", 255),
                                                 "flood_probability.tif": (prob, "float32", np.nan)})
        log(f"flood {summary['detect']['flood_km2']:,.0f} km2 of {summary['detect']['radar_covered_km2']:,.0f} km2 covered")

    # ---- risk / early warning (pre-landfall radar + storm track) ----
    fr = stack / "inputs_forecast.npy"
    if risk_ck and fr.exists():
        X = np.load(fr, mmap_mode="r")
        chans = meta["tasks"]["forecast"]["channels"]
        valid = np.asarray(X[chans.index("valid")]) > 0 if "valid" in chans else np.isfinite(X[0])
        res = run_model(risk_ck, chans, X, device, log)   # TTA not validated for the risk model: plain
        if res is None:
            raise SystemExit("the risk model does not fit this stack's layers")
        r = res[0]
        r = np.where(valid, r, np.nan).astype(np.float32)
        np.save(out / "risk_prob.npy", r.astype(np.float16))
        risk_png(r, valid, out / "risk_map.png")
        risk_arr = r
        rv = np.nan_to_num(r[::step, ::step], nan=0)
        q = float(np.nanquantile(r[valid], 0.98)) if valid.any() else 1.0
        t = np.clip(rv / max(q, 1e-3), 0, 1)
        img = np.concatenate([ramp(t, [(0, (255, 210, 60)), (0.6, (255, 90, 40)), (1, (255, 30, 120))]),
                              (np.clip(t, 0, 1) ** 0.7 * 220).astype(np.uint8)[..., None]], 2)
        img[..., 3][~valid[::step, ::step]] = 0
        write_png(out / "layers" / "risk.png", img)
        if not (out / "layers" / "radar_before.png").exists():
            radar_png(X, chans, "pre_med_vv", step, out / "layers" / "radar_before.png")
        v = r[valid]
        zones = {}
        for q in (5, 10, 20):
            cut = float(np.quantile(v, 1 - q / 100)) if v.size else 1.0
            zones[f"top_{q}pct"] = {"risk_at_least": round(cut, 3),
                                    "km2": round(float((((r >= cut) & valid) * area).sum()), 1)}
        summary["risk"] = {"model": Path(risk_ck).name, "zones": zones,
                           "risk_scale_note": "relative ranking; the map colours scale to the 98th percentile"}
        summary["tifs"] = summary.get("tifs", []) + write_tifs(out, grid, {"flood_risk.tif": (r, "float32", np.nan)})
        log(f"risk map done; top 10 % zone = {zones['top_10pct']['km2']:,.0f} km2")

    if "detect" not in summary and "risk" not in summary:
        raise SystemExit("nothing to do: no inputs_detect.npy / inputs_forecast.npy or no models")
    if ens_prob is not None:
        log("scoring against reference maps (if the stack has any)")
        summary["eval"] = evaluate(stack, meta, out, area, step, det_probs, ens_prob, risk_arr, log, a.threshold) is not None
    summary["preview"] = {"step": step, "w": -(-grid["W"] // step), "h": -(-grid["H"] // step)}
    summary["layers"] = sorted(f.stem for f in (out / "layers").glob("*.png"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
