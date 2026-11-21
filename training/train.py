"""Train + test the Amphan flood models on the GPU laptop.

  python train.py --task detect --smoke      # 2-minute check that everything works
  python train.py --task detect              # flood map from before/after SAR (now-cast)
  python train.py --task forecast            # flood risk from PRE-landfall SAR + terrain (early warning)

Model: U-Net (4 levels). Loss: cross-entropy (class-weighted, unlabelled pixels ignored) + Dice
on the flood class. AdamW + cosine LR, mixed precision on GPU, early stopping on validation
flood IoU. Scores on the 30 % test blocks are written to runs/<name>/metrics.json, plus the
same scores against the independent Zenodo reference map and for the Otsu rule baseline.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kit_utils import (CLASS_NAMES, PatchSampler, augment, average_precision, confusion, flood_km2,
                       get_patch, load_kit, make_target, normalize, rule_baseline, scores,
                       tile_starts)

HERE = Path(__file__).resolve().parent
STATUS = None


class Status:
    """Live progress for dashboard.py: runs/<name>/status.json, history.json, preview.npz."""

    def __init__(self, out, **base):
        self.out = out
        self.d = dict(base, state="starting", started=time.time())
        self.write()

    def _dump(self, name, obj):
        p, tmp = self.out / name, self.out / (name + ".tmp")
        txt = json.dumps(obj, default=float)
        tmp.write_text(txt, encoding="utf-8")
        for _ in range(5):
            try:
                tmp.replace(p)
                return
            except PermissionError:      # Windows: file open in the dashboard right now
                time.sleep(0.05)
        p.write_text(txt, encoding="utf-8")

    def write(self, **kw):
        self.d.update(kw, updated=time.time())
        try:
            self._dump("status.json", self.d)
        except OSError:
            pass

    def history(self, hist):
        try:
            self._dump("history.json", hist)
        except OSError:
            pass

    def activations(self, maps, stats, extra):
        try:
            np.savez(self.out / "activations_tmp.npz", **maps, **extra)
            (self.out / "activations_tmp.npz").replace(self.out / "activations.npz")
            self._dump("activations.json", stats)
        except OSError:
            pass

    def preview(self, prob, circle, step):
        try:
            cls = prob[:, ::step, ::step].argmax(0).astype(np.uint8)
            cls[~circle[::step, ::step]] = 255
            fl = np.clip(prob[1, ::step, ::step] * 255, 0, 255).astype(np.uint8)
            np.savez(self.out / "preview_tmp.npz", cls=cls, flood=fl)
            (self.out / "preview_tmp.npz").replace(self.out / "preview.npz")
        except OSError:
            pass


# ------------------------------------------------------------------ model
def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, in_ch, n_cls, base=32, dropout=0.1):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([conv_block(in_ch, ch[0])] +
                                 [conv_block(ch[i], ch[i + 1]) for i in range(4)])
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


def dice_loss(logits, y, cls=1):
    p = logits.float().softmax(1)[:, cls]
    m = (y >= 0).float()
    t = (y == cls).float()
    inter = (p * t * m).sum()
    return 1 - (2 * inter + 1) / ((p * m).sum() + (t * m).sum() + 1)


def reduce_activation(a):
    """(C,h,w) float -> (h,w) uint8 heat map (mean |activation| over channels) + stats."""
    a = np.asarray(a, np.float32)
    m = np.abs(a).mean(0)
    hi = float(np.percentile(m, 99)) or 1.0
    img = np.clip(m / hi * 255, 0, 255).astype(np.uint8)
    return img, {"mean_abs": float(np.abs(a).mean()), "active_frac": float((a > 0).mean()),
                 "channels": int(a.shape[0]), "size": [int(a.shape[1]), int(a.shape[2])]}


@torch.no_grad()
def capture_activations(model, x, device):
    """Forward one patch with hooks on every encoder/decoder block (the 'brain scan')."""
    acts, hooks = {}, []
    for i, blk in enumerate(model.enc):
        hooks.append(blk.register_forward_hook(lambda m, inp, out, k=f"enc{i}": acts.__setitem__(k, out)))
    for i, blk in enumerate(model.dec):
        hooks.append(blk.register_forward_hook(lambda m, inp, out, k=f"dec{i}": acts.__setitem__(k, out)))
    model.eval()
    try:
        prob = model(torch.from_numpy(x)[None].to(device)).float().softmax(1)[0].cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    maps, stats = {}, {}
    for k, v in acts.items():
        maps[k], stats[k] = reduce_activation(v[0].float().cpu().numpy())
    return maps, stats, prob


def snapshot_center(target, split, SC, region):
    """Fixed spot for activation snapshots: a validation flood pixel (else any labelled pixel)."""
    r_lo, r_hi, c_lo, c_hi = region
    sub_t, sub_s = target[r_lo:r_hi, c_lo:c_hi], split[r_lo:r_hi, c_lo:c_hi]
    for cond in ((sub_s == SC["val"]) & (sub_t == 1), (sub_t == 1), (sub_t >= 0)):
        idx = np.flatnonzero(cond)
        if len(idx):
            k = idx[len(idx) // 2]
            r, c = divmod(int(k), c_hi - c_lo)
            return r + r_lo, c + c_lo
    return (r_lo + r_hi) // 2, (c_lo + c_hi) // 2


# ------------------------------------------------------------------ inference
@torch.no_grad()
def predict_map(model, inputs, mean, std, device, n_cls, region=None, tile=512, overlap=64, amp=True):
    model.eval()
    C, H, W = inputs.shape
    r_lo, r_hi, c_lo, c_hi = region or (0, H, 0, W)
    h, w = r_hi - r_lo, c_hi - c_lo
    acc = np.zeros((n_cls, h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for r0 in tile_starts(h, tile, overlap):
        for c0 in tile_starts(w, tile, overlap):
            R0, C0 = r_lo + r0, c_lo + c0
            x = normalize(inputs[:, R0:R0 + tile, C0:C0 + tile], mean, std)
            th, tw = x.shape[1], x.shape[2]
            ph, pw = (-th) % 16, (-tw) % 16
            if ph or pw:
                x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(th, tw) > 16 else "constant")
            xt = torch.from_numpy(x)[None].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                pr = model(xt).float().softmax(1)[0, :, :th, :tw].cpu().numpy()
            acc[:, r0:r0 + th, c0:c0 + tw] += pr
            cnt[r0:r0 + th, c0:c0 + tw] += 1
    return acc / np.maximum(cnt, 1)


def evaluate(prob, target, mask, names):
    pred = prob.argmax(0).astype(np.uint8)
    cm = confusion(pred, target, len(names), mask)
    s = scores(cm, names)
    flood_prob = prob[1]
    s["flood_pr_auc"] = average_precision(flood_prob[mask & (target >= 0)],
                                          (target[mask & (target >= 0)] == 1).astype(np.int8))
    return s, pred


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detect", "forecast"], default="detect")
    ap.add_argument("--kit", default=str(HERE))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300, help="training batches per epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--base", type=int, default=32, help="U-Net width (32 fits 4 GB VRAM)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8, help="evals without val improvement")
    ap.add_argument("--flood-frac", type=float, default=0.5, help="share of patches centred on flood")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mmap", action="store_true", help="do not load inputs into RAM")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the setup")
    ap.add_argument("--name", default=None)
    ap.add_argument("--drop-channels", default="",
                    help="comma list of input channels to leave out, or 'terrain' "
                         "(= jrc_occ,dem,hand_proxy,slope,dist_water_km) for a SAR-only check")
    ap.add_argument("--trainval", action="store_true",
                    help="final refit: train on train + val blocks (the full 70 %%), run all --epochs, "
                         "no validation selection; choose --epochs from a normal run's best epoch")
    ap.add_argument("--alldata", action="store_true",
                    help="deployment model: train on train + val + test + buffers (100 %%); its test numbers are "
                         "in-sample, NOT a score. Implies --trainval")
    a = ap.parse_args()
    if a.alldata:
        a.trainval = True
    if a.smoke:
        a.epochs, a.steps, a.eval_every, a.batch = 1, 3, 1, min(a.batch, 2)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    name = a.name or f"{a.task}_{time.strftime('%Y%m%d_%H%M')}{'_smoke' if a.smoke else ''}"
    out = Path(a.kit) / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(*m):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    global STATUS
    STATUS = Status(out, name=name, task=a.task, device=device.type,
                    gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                    epochs=a.epochs, steps=a.steps, batch=a.batch, patch=a.patch, smoke=a.smoke)
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                "  <-- no GPU found: this will be slow"))
    meta, inputs, labels, split, circle, ref = load_kit(a.kit, a.task, mmap=a.mmap)
    tinfo = meta["tasks"][a.task]
    chans = tinfo["channels"]
    mean, std = np.array(tinfo["mean"], np.float32), np.array(tinfo["std"], np.float32)
    if a.drop_channels:
        drop = set()
        for d in a.drop_channels.split(","):
            drop |= {"jrc_occ", "dem", "hand_proxy", "slope", "dist_water_km"} if d.strip() == "terrain" else {d.strip()}
        keep = [i for i, n in enumerate(chans) if n not in drop]
        inputs = np.ascontiguousarray(np.asarray(inputs)[keep])
        chans = [chans[i] for i in keep]
        mean, std = mean[keep], std[keep]
        tinfo = dict(tinfo, channels=chans, mean=mean.tolist(), std=std.tolist())
        log(f"dropped channels {sorted(drop)}; using {len(chans)}: {chans}")
    names = CLASS_NAMES[a.task]
    n_cls = len(names)
    target = make_target(labels, a.task)
    SC = meta["split_code"]
    split_orig = split   # the real test blocks, for the test report
    if a.trainval:   # val joins training -> 70 % train / 30 % test; --alldata: everything trains (100 %)
        merge = [SC["val"]] + ([SC["test"], SC["ignore"]] if a.alldata else [])
        split = np.where(np.isin(np.asarray(split), merge), SC["train"], np.asarray(split)).astype(np.uint8)
        if a.alldata:
            log("ALL DATA: test blocks are trained on -> the TEST numbers below are in-sample, NOT a score")
    _, H, W = inputs.shape
    log(f"task {a.task}: {len(chans)} channels, map {W} x {H}, classes {names}")

    region = None
    if a.smoke:   # small crop around the centre of the map
        r, c = H // 2, W // 2
        region = (max(r - 512, 0), min(r + 512, H), max(c - 512, 0), min(c + 512, W))
    in_region = np.zeros((H, W), bool)
    r_lo, r_hi, c_lo, c_hi = region or (0, H, 0, W)
    in_region[r_lo:r_hi, c_lo:c_hi] = True

    sampler = PatchSampler(target, split, a.patch, a.flood_frac, a.seed, SC["train"])
    tr = (split == SC["train"]) & (target >= 0)
    counts = np.bincount(target[tr].astype(np.int64), minlength=n_cls).astype(np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    w = w / w.mean()
    log(f"train pixels per class {dict(zip(names, counts.astype(int).tolist()))}, "
        f"loss weights {np.round(w, 2).tolist()}")
    sar_idx = np.array([i for i, n in enumerate(chans)
                        if n.split("_")[0] in ("b", "a", "pre") and "inc" not in n
                        and "count" not in n and "tex" not in n and "std" not in n])

    model = UNet(len(chans), n_cls, a.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"U-Net parameters: {n_params / 1e6:.1f} M")
    STATUS.write(state="training", params_m=round(n_params / 1e6, 2), channels=chans,
                 classes=names, map=[W, H], train_px=dict(zip(names, counts.astype(int).tolist())))
    pv_step = max(1, max(H, W) // 700)
    sr, sc_ = snapshot_center(target, split, SC, region or (0, H, 0, W))
    ps = a.patch
    sr0 = int(np.clip(sr - ps // 2, 0, max(H - ps, 0)))
    sc0 = int(np.clip(sc_ - ps // 2, 0, max(W - ps, 0)))
    snap_x, _ = get_patch(inputs, mean, std, target, np.zeros_like(split), sr0, sc0, ps, 0)
    snap_lab = target[sr0:sr0 + ps, sc0:sc0 + ps].astype(np.int8)
    ch_ix = {n: i for i, n in enumerate(chans)}

    def snap_band(name):
        if name not in ch_ix:
            return np.zeros((ps, ps), np.uint8)
        v = np.asarray(inputs[ch_ix[name], sr0:sr0 + ps, sc0:sc0 + ps], np.float32)
        lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        return np.nan_to_num(np.clip((v - lo) / max(hi - lo, 1e-6) * 255, 0, 255)).astype(np.uint8)
    snap_inputs = {"in_before": snap_band("b_vh" if a.task == "detect" else "pre_med_vh"),
                   "in_after": snap_band("a_vh" if a.task == "detect" else "pre_min_vh"),
                   "in_change": snap_band("d_vh" if a.task == "detect" else "pre_std_vh")}
    STATUS.write(snapshot={"row": sr0, "col": sc0, "size": ps})
    recent, step_times, eval_sec = [], [], 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * total))) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    cw = torch.tensor(w, dtype=torch.float32, device=device)
    rng = np.random.default_rng(a.seed)

    best, best_ep, bad, history = -1.0, -1, 0, []
    val_mask = (split == SC["val"]) & circle & in_region
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for step in range(a.steps):
            ts = time.time()
            xs, ys = [], []
            for r0, c0 in sampler.sample(a.batch):
                x, y = get_patch(inputs, mean, std, target, split, r0, c0, a.patch, SC["train"])
                x, y = augment(x, y, rng, sar_idx, std=std)
                xs.append(x)
                ys.append(y)
            xb = torch.from_numpy(np.stack(xs)).to(device, non_blocking=True)
            yb = torch.from_numpy(np.stack(ys)).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), yb, weight=cw, ignore_index=-1) \
                    + dice_loss(logits, yb, 1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            lv = float(loss)
            run += lv
            recent.append(lv)
            recent = recent[-300:]
            step_times.append(time.time() - ts)
            step_times = step_times[-50:]
            if step % 10 == 0 or step == a.steps - 1:
                sps = float(np.mean(step_times))
                left_steps = (a.epochs - ep) * a.steps + (a.steps - step - 1)
                left_evals = len([e for e in range(ep, a.epochs + 1) if e % a.eval_every == 0])
                gpu_mem = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0
                STATUS.write(state="training", epoch=ep, step=step + 1, loss=lv,
                             loss_avg=float(np.mean(recent[-50:])), recent_losses=recent[-300:],
                             lr=opt.param_groups[0]["lr"], sec_per_step=sps,
                             eta_sec=left_steps * sps + left_evals * eval_sec,
                             gpu_mem_peak_gb=round(gpu_mem, 2), best_val_iou=best, best_epoch=best_ep)
        rec = {"epoch": ep, "loss": run / a.steps, "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - t0, 1)}
        if (ep % a.eval_every == 0 or ep == a.epochs) and not a.trainval:
            STATUS.write(state="validating", epoch=ep)
            te = time.time()
            prob = predict_map(model, inputs, mean, std, device, n_cls, region)
            full = np.zeros((n_cls, H, W), np.float32)
            full[:, r_lo:r_hi, c_lo:c_hi] = prob
            STATUS.preview(full, circle & in_region, pv_step)
            try:
                maps, astats, sprob = capture_activations(model, snap_x, device)
                astats["_meta"] = {"epoch": ep, "row": sr0, "col": sc0, "size": ps, "task": a.task,
                                   "classes": names}
                STATUS.activations(maps, astats, dict(snap_inputs, label=snap_lab,
                                                       pred=sprob.argmax(0).astype(np.uint8),
                                                       flood_prob=(sprob[1] * 255).astype(np.uint8)))
            except Exception as e:      # never let the monitor break training
                log(f"activation snapshot skipped: {e}")
            s, _ = evaluate(full, target, val_mask, names)
            rec.update(val_flood_iou=s[names[1]]["iou"], val_flood_f1=s[names[1]]["f1"],
                       val_pr_auc=s["flood_pr_auc"])
            score = s[names[1]]["iou"] if np.isfinite(s[names[1]]["iou"]) else 0.0
            if score > best:
                best, best_ep, bad = score, ep, 0
                torch.save({"model": model.state_dict(), "in_ch": len(chans), "n_cls": n_cls,
                            "base": a.base, "task": a.task, "channels": chans,
                            "mean": tinfo["mean"], "std": tinfo["std"]}, out / "best.pt")
            else:
                bad += 1
            eval_sec = time.time() - te
            rec["eval_sec"] = round(eval_sec, 1)
        history.append(rec)
        STATUS.history(history)
        STATUS.write(best_val_iou=best, best_epoch=best_ep, patience_left=a.patience - bad)
        log(json.dumps(rec))
        if bad >= a.patience:
            log(f"early stop: no val improvement for {a.patience} evaluations")
            break
    if a.trainval:   # no val left to choose by: keep the model after the last epoch
        best_ep = history[-1]["epoch"] if history else 0
        torch.save({"model": model.state_dict(), "in_ch": len(chans), "n_cls": n_cls,
                    "base": a.base, "task": a.task, "channels": chans,
                    "mean": tinfo["mean"], "std": tinfo["std"]}, out / "best.pt")
        log(f"train+val refit: kept the final model (epoch {best_ep}); val was used for training")
    log(f"best val flood IoU {best:.4f} at epoch {best_ep}")
    STATUS.write(state="testing", best_val_iou=best, best_epoch=best_ep)

    # ------------------------------------------------------------ test
    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    prob_r = predict_map(model, inputs, mean, std, device, n_cls, region)
    prob = np.zeros((n_cls, H, W), np.float32)
    prob[:, r_lo:r_hi, c_lo:c_hi] = prob_r
    # pixels without SAR input (valid == 0) hold fill values: never report flood there
    has_data = (np.asarray(inputs[chans.index("valid")]) > 0.5) if "valid" in chans else np.ones((H, W), bool)
    test_mask = (split_orig == SC["test"]) & circle & in_region
    res = {"task": a.task, "best_epoch": best_ep, "best_val_flood_iou": best, "args": vars(a),
           "history": history}
    res["test_vs_weak_labels"], pred = evaluate(prob, target, test_mask, names)
    area = meta["pixel_area_km2_rows"]
    res["flood_km2_predicted_whole_circle"] = flood_km2(pred, circle & in_region & has_data, area)
    res["flood_km2_predicted_test_blocks"] = flood_km2(pred, test_mask & has_data, area)
    res["sar_valid_km2_in_circle"] = float(((circle & has_data).sum(1) * np.asarray(area)).sum())
    if ref is not None:
        rt = make_target(ref, a.task)
        res["test_vs_zenodo_reference"], _ = evaluate(prob, rt, test_mask, names)
        res["reference_flood_km2_test_blocks"] = flood_km2(rt, test_mask, area)
    if a.task == "detect":
        raw = inputs
        base = rule_baseline(raw, chans, meta["labels_info"]["thresholds_db"])
        onehot = np.eye(n_cls, dtype=np.float32)[base].transpose(2, 0, 1)
        res["baseline_otsu_vs_weak_labels"], _ = evaluate(onehot, target, test_mask, names)
        if ref is not None:
            res["baseline_otsu_vs_zenodo_reference"], _ = evaluate(onehot, rt, test_mask, names)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    np.save(out / "pred_class.npy", np.where(circle & has_data, pred, 255).astype(np.uint8))
    np.save(out / "pred_flood_prob.npy", np.where(circle & has_data, prob[1], np.nan).astype(np.float16))
    STATUS.preview(prob, circle & in_region & has_data, max(1, max(H, W) // 700))

    fl = res["test_vs_weak_labels"][names[1]]
    log(f"TEST (weak labels)  flood IoU {fl['iou']:.3f}  F1 {fl['f1']:.3f}  "
        f"P {fl['precision']:.3f}  R {fl['recall']:.3f}  PR-AUC {res['test_vs_weak_labels']['flood_pr_auc']:.3f}")
    if ref is not None:
        fr = res["test_vs_zenodo_reference"][names[1]]
        log(f"TEST (Zenodo ref)   flood IoU {fr['iou']:.3f}  F1 {fr['f1']:.3f}  "
            f"P {fr['precision']:.3f}  R {fr['recall']:.3f}")
        if a.task == "detect":
            fb = res["baseline_otsu_vs_zenodo_reference"][names[1]]
            log(f"BASELINE Otsu vs ref flood IoU {fb['iou']:.3f}  F1 {fb['f1']:.3f}  (model must beat this)")
    log(f"predicted flood area {res['flood_km2_predicted_whole_circle']:,.0f} km2 inside the "
        f"250 km circle (WB + Bangladesh + N. Odisha)")
    try:
        from to_geotiff import write_run
        write_run(out, meta)
        log("GeoTIFFs written: flood_class.tif, flood_probability.tif")
    except ImportError:
        log("rasterio not installed -> .npy outputs only (run to_geotiff.py on the PC)")
    except Exception as e:
        log(f"GeoTIFF export failed ({e}); .npy outputs are complete")
    log(f"outputs in {out}")
    STATUS.write(state="done", eta_sec=0)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        if STATUS is not None:
            import traceback
            STATUS.write(state="stopped" if isinstance(e, KeyboardInterrupt) else "error",
                         error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        raise
