"""numpy-only helpers for train.py: data loading, patch sampling, tiling, metrics, baseline."""
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = {"detect": ["dry", "flood", "water"], "forecast": ["not flooded", "flooded"]}


def load_kit(root, task, mmap=False):
    root = Path(root)
    d = root / "data"
    meta = json.loads((d / "meta.json").read_text())
    inputs = np.load(d / f"inputs_{task}.npy", mmap_mode="r" if mmap else None)
    labels = np.load(d / "labels.npy")
    split = np.load(d / "split.npy")
    circle = np.load(d / "circle.npy")
    ref = np.load(d / "ref_labels.npy") if (d / "ref_labels.npy").exists() else None
    return meta, inputs, labels, split, circle, ref


def make_target(labels, task):
    """detect: -1/0/1/2 as is. forecast: flooded (1) vs not (0); permanent water ignored."""
    if task == "detect":
        return labels.astype(np.int8)
    t = labels.astype(np.int8).copy()
    t[t == 2] = -1
    return t


class PatchSampler:
    def __init__(self, target, split, patch, flood_frac=0.5, seed=0, split_code=0):
        self.H, self.W = target.shape
        self.p = patch
        ok = (split == split_code) & (target >= 0)
        self.any_pix = np.flatnonzero(ok)
        self.flood_pix = np.flatnonzero(ok & (target == 1))
        self.flood_frac = flood_frac if len(self.flood_pix) else 0.0
        self.rng = np.random.default_rng(seed)
        if len(self.any_pix) == 0:
            raise ValueError("no labelled training pixels")

    def sample(self, n):
        out = []
        for _ in range(n):
            pool = self.flood_pix if self.rng.random() < self.flood_frac else self.any_pix
            k = pool[self.rng.integers(len(pool))]
            r, c = divmod(int(k), self.W)
            r0 = int(np.clip(r - self.p // 2 + self.rng.integers(-self.p // 4, self.p // 4 + 1),
                             0, max(self.H - self.p, 0)))
            c0 = int(np.clip(c - self.p // 2 + self.rng.integers(-self.p // 4, self.p // 4 + 1),
                             0, max(self.W - self.p, 0)))
            out.append((r0, c0))
        return out


def normalize(x, mean, std):
    x = (x.astype(np.float32) - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def get_patch(inputs, mean, std, target, split, r0, c0, p, split_code=0):
    x = normalize(inputs[:, r0:r0 + p, c0:c0 + p], mean, std)
    y = target[r0:r0 + p, c0:c0 + p].astype(np.int64)
    y = np.where(split[r0:r0 + p, c0:c0 + p] == split_code, y, -1)
    ph, pw = p - x.shape[1], p - x.shape[2]
    if ph or pw:   # only if the whole map is smaller than a patch
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
        y = np.pad(y, ((0, ph), (0, pw)), constant_values=-1)
    return x, y


def augment(x, y, rng, sar_idx, noise=0.05, offset_db=1.0, std=None):
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    if len(sar_idx):
        x[sar_idx] += rng.normal(0, noise, size=(len(sar_idx),) + x.shape[1:]).astype(np.float32)
        if std is not None and offset_db:
            # same random calibration offset for the whole image (mimics track differences)
            off = rng.uniform(-offset_db, offset_db)
            x[sar_idx] += (off / std[sar_idx])[:, None, None].astype(np.float32)
    return x, y


def tile_starts(n, tile, overlap):
    if n <= tile:
        return [0]
    step = tile - overlap
    s = list(range(0, n - tile, step))
    s.append(n - tile)
    return sorted(set(s))


def confusion(pred, target, n_cls, mask):
    m = mask & (target >= 0) & (target < n_cls)
    return np.bincount(target[m].astype(np.int64) * n_cls + pred[m].astype(np.int64),
                       minlength=n_cls * n_cls).reshape(n_cls, n_cls)


def scores(cm, names):
    out = {}
    for i, nm in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        if tp + fp + fn == 0:      # class absent in both truth and prediction: not scoreable
            out[nm] = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                       "iou": float("nan"), "support": 0}
            continue
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        out[nm] = {"precision": float(p), "recall": float(r),
                   "f1": float(2 * p * r / max(p + r, 1e-12)),
                   "iou": float(tp / max(tp + fp + fn, 1)), "support": int(cm[i].sum())}
    out["overall_accuracy"] = float(np.trace(cm) / max(cm.sum(), 1))
    out["confusion_matrix(rows=true,cols=pred)"] = cm.tolist()
    return out


def average_precision(prob, y, max_n=5_000_000, seed=0):
    """PR-AUC (average precision) for binary y in {0,1}."""
    ok = np.isfinite(prob) & (y >= 0)
    prob, y = prob[ok], y[ok]
    if len(y) > max_n:
        i = np.random.default_rng(seed).choice(len(y), max_n, replace=False)
        prob, y = prob[i], y[i]
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-prob, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def rule_baseline(raw, channels, thresholds, drop_db=3.0):
    """Otsu change detection, the method of the published Amphan studies.
    raw = unnormalized detect inputs (C,H,W). Returns 0 dry / 1 flood / 2 water."""
    ix = {n: i for i, n in enumerate(channels)}
    f = lambda n: raw[ix[n]].astype(np.float32)
    tvh, tvv = thresholds["VH"], thresholds["VV"]
    wb = (f("b_vh") < tvh) & (f("b_vv") < tvv)
    wa = (f("a_vh") < tvh) & (f("a_vv") < tvv)
    out = np.zeros(wb.shape, np.uint8)
    out[wa & wb] = 2
    out[wa & ~wb & (f("d_vh") < -drop_db)] = 1
    return out


def flood_km2(class_map, mask, area_rows, flood_class=1):
    return float(((class_map == flood_class) & mask).sum(1) @ np.asarray(area_rows))
