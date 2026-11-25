"""ops_dashboard.py - Flood Ops control panel (web page; Python standard library, numpy optional).

  python ops_dashboard.py                       # http://127.0.0.1:8765  (this computer only)
  python ops_dashboard.py --host 0.0.0.0        # also reachable over Tailscale, e.g. from your phone

Pages: Overview, Live map (layers, zoom, click a pixel), Forecast (early-warning risk), Evaluation (scores vs
reference maps when the stack has them), Storms (enter storm data -> Build stack -> Run models), Models (choose
the ensemble + risk model), Training (runs + learning curves), Bridge (PC <-> laptop messages), System.
"""
import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
STORMS = HERE / "storms"
STORMS.mkdir(exist_ok=True)
MODELS = HERE / "models"
CFG_PATH = HERE / "ops_config.json"
DEFAULT_CFG = {
    "python": sys.executable,
    "build_cmd": ["{python}", "build_stack.py", "--request", "{request}", "--out", "{stack}"],
    "build_cwd": str(HERE),
    "predict_extra": [],
    "detect_models": None,      # None = every models/detect_*.pt (models that do not fit a stack are skipped)
    "risk_model": None,         # None = first models/risk_*.pt
    "threshold": 0.5,
    "tta": 1,                   # test-time augmentation views: 1 none, 4 flips, 8 flips+transpose
    "kit_dirs": [],             # training kits to show on the Training page
    "kit_logs": [],             # run_kit logs (glob patterns) to tail
}
JOBS = {}
LOCK = threading.Lock()
SAFE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
SERVE = {".md": "text/plain; charset=utf-8", ".png": "image/png", ".tif": "image/tiff", ".json": "application/json", ".txt": "text/plain; charset=utf-8"}


def cfg():
    c = dict(DEFAULT_CFG)
    if CFG_PATH.exists():
        c.update(json.loads(CFG_PATH.read_text(encoding="utf-8")))
    else:
        CFG_PATH.write_text(json.dumps(DEFAULT_CFG, indent=1), encoding="utf-8")
    return c


def save_cfg(upd):
    c = json.loads(CFG_PATH.read_text(encoding="utf-8")) if CFG_PATH.exists() else dict(DEFAULT_CFG)
    c.update(upd)
    CFG_PATH.write_text(json.dumps(c, indent=1), encoding="utf-8")


def rjson(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def storm_dir(name):
    if not SAFE.match(name or ""):
        raise ValueError("storm name: letters, digits, _ or - only (max 40)")
    return STORMS / name


def stack_path(d):
    req = rjson(d / "request.json", {})
    return Path(req.get("stack_path") or (d / "stack"))


def storm_info(d):
    req = rjson(d / "request.json", {})
    st, out = stack_path(d), d / "out"
    prog = (out / "progress.txt").read_text(encoding="utf-8") if (out / "progress.txt").exists() else ""
    return {"name": d.name, "request": req, "stack_ready": (st / "meta.json").exists(), "stack": str(st),
            "summary": rjson(out / "summary.json"), "eval": rjson(out / "eval.json"), "job": JOBS.get(d.name),
            "progress": prog, "maps": [f for f in ("flood_map.png", "risk_map.png") if (out / f).exists()],
            "tifs": sorted(f.name for f in out.glob("*.tif")) if out.exists() else [],
            "has_refs": (st / "ref_labels.npy").exists() or (st / "ref2_labels.npy").exists()}


def all_storms():
    ds = sorted((d for d in STORMS.iterdir() if (d / "request.json").exists()),
                key=lambda d: d.stat().st_mtime, reverse=True)
    return [storm_info(d) for d in ds]


def run_job(name, kind, cmd, cwd, log_path):
    with LOCK:
        JOBS[name] = {"kind": kind, "state": "running", "started": time.strftime("%H:%M:%S")}
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {kind} {time.strftime('%Y-%m-%d %H:%M:%S')}\n$ {' '.join(cmd)}\n")
            lf.flush()
            rc = subprocess.call(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
        state = "done" if rc == 0 else f"failed (exit {rc})"
    except Exception as e:  # noqa: BLE001
        state = f"failed ({e})"
    with LOCK:
        JOBS[name] = dict(JOBS[name], state=state, ended=time.strftime("%H:%M:%S"))


def start(name, kind):
    d = storm_dir(name)
    if JOBS.get(name, {}).get("state") == "running":
        raise ValueError("a job is already running for this storm")
    c = cfg()
    st = stack_path(d)
    if kind == "build":
        sub = {"{python}": c["python"], "{request}": str(d / "request.json"), "{stack}": str(st)}
        cmd, cwd = [sub.get(x, x) for x in c["build_cmd"]], c["build_cwd"]
    else:
        if not (st / "meta.json").exists():
            raise ValueError(f"no stack yet at {st} - build it first")
        cmd = [c["python"], str(HERE / "predict.py"), "--stack", str(st), "--out", str(d / "out"), "--name", name,
               "--threshold", str(float(c.get("threshold") or 0.5)), "--tta", str(int(c.get("tta") or 1))]
        if c.get("detect_models"):
            cmd += ["--detect"] + [str(MODELS / m) for m in c["detect_models"] if (MODELS / m).exists()]
        if c.get("risk_model") and (MODELS / c["risk_model"]).exists():
            cmd += ["--risk", str(MODELS / c["risk_model"])]
        cmd += list(c.get("predict_extra", []))
        cwd = str(HERE)
        for f in (d / "out").glob("*") if (d / "out").exists() else []:
            if f.name in ("progress.txt", "log.txt"):
                f.unlink()
    threading.Thread(target=run_job, args=(name, kind, cmd, cwd, d / "jobs.log"), daemon=True).start()


# ---------------- models / training / bridge / system ----------------
def models_info():
    c = cfg()
    out = []
    for p in sorted(MODELS.glob("*.pt")):
        card = rjson(p.with_suffix(".json"), {}) or {}
        task = card.get("task") or ("forecast" if p.name.startswith("risk") else "detect")
        if task == "detect":
            on = (c.get("detect_models") is None) or (p.name in c["detect_models"])
        else:
            risks = sorted(MODELS.glob("risk_*.pt"))
            on = p.name == (c.get("risk_model") or (risks[0].name if risks else None))
        out.append({"file": p.name, "task": task, "size_mb": round(p.stat().st_size / 1e6, 1), "enabled": on, **card})
    return {"models": out, "threshold": c.get("threshold", 0.5), "tta": int(c.get("tta") or 1)}


def parse_history(kit, run):
    h = rjson(kit / "runs" / run / "history.json")
    if h is None:
        h = []
        f = kit / f"runs_{run}.out"
        if f.exists():
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                i = line.find('{"epoch"')
                if i >= 0:
                    try:
                        h.append(json.loads(line[i:]))
                    except ValueError:
                        pass
    return [{k: e.get(k) for k in ("epoch", "loss", "val_flood_iou", "val_pr_auc")} for e in h if isinstance(e, dict)]


def training_info():
    c = cfg()
    kits = []
    for k in c.get("kit_dirs", []):
        kit = Path(k)
        if not (kit / "runs").exists():
            continue
        runs = []
        for r in sorted((kit / "runs").iterdir(), key=lambda r: r.stat().st_mtime):
            if not r.is_dir() or r.name.startswith("_") or "smoke" in r.name:
                continue
            m = rjson(r / "metrics.json")
            outf = kit / f"runs_{r.name}.out"
            live = (not m) and outf.exists() and time.time() - outf.stat().st_mtime < 900
            t = (m or {}).get("test_vs_zenodo_reference") or {}
            f = t.get("flood") or t.get("flooded") or {}
            runs.append({"name": r.name, "status": "done" if m else ("training" if live else "stopped"),
                         "task": (m or {}).get("task"), "best_epoch": (m or {}).get("best_epoch"),
                         "test_iou": f.get("iou"), "test_f1": f.get("f1"), "pr_auc": t.get("flood_pr_auc"),
                         "in_sample": "full100" in r.name or "ensemble_map" in r.name,
                         "history": parse_history(kit, r.name)})
        kits.append({"kit": kit.name, "path": str(kit), "runs": runs})
    logs = []
    for pat in c.get("kit_logs", []):
        pp = Path(pat)
        logs += [f for f in pp.parent.glob(pp.name)]
    tail = ""
    if logs:
        f = max(logs, key=lambda f: f.stat().st_mtime)
        lines = [ln for ln in f.read_text(encoding="utf-8", errors="replace").splitlines() if re.search(r"START|END|DONE|STOP|RESUME|sent", ln)]
        tail = f.name + "\n" + "\n".join(lines[-14:])
    return {"kits": kits, "log": tail}


def bridge_msgs():
    msgs = []
    for box in ("inbox", "outbox"):
        for f in (HERE / "bridge" / box).glob("msg_*.md"):
            t = f.read_text(encoding="utf-8", errors="replace")
            head, _, body = t.partition("\n---")
            h = {}
            for ln in head.splitlines():
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    h[k.strip().lower()] = v.strip()
            msgs.append({"id": f.name, "dir": "in" if box == "inbox" else "out", "from": h.get("from", "?"),
                         "time": h.get("time", f.name[4:19]), "subject": h.get("subject", ""),
                         "attachments": h.get("attachments", ""), "body": body.strip()[:1500]})
    msgs.sort(key=lambda m: m["id"][4:19])
    me = (rjson(HERE / "bridge_config.json", {}) or {})
    return {"me": me.get("me"), "peer": me.get("peer_host"), "messages": msgs[-40:]}


def bridge_send(subject, text):
    if not text.strip():
        raise ValueError("empty message")
    r = subprocess.run([cfg()["python"], str(HERE / "bridge.py"), "send", text, "--subject", subject or "from console"],
                       cwd=str(HERE), capture_output=True, text=True, timeout=180)
    if r.returncode:
        raise ValueError((r.stderr or r.stdout)[-400:])
    return (r.stdout or "sent").strip()[-300:]


def ram():
    try:
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
            m = MS()
            m.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return {"total_gb": round(m.tot / 2**30, 1), "free_gb": round(m.avail / 2**30, 1)}
        mi = dict(ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines())
        kb = lambda k: int(mi[k].split()[0]) / 2**20
        return {"total_gb": round(kb("MemTotal"), 1), "free_gb": round(kb("MemAvailable"), 1)}
    except Exception:  # noqa: BLE001
        return None


_PC_CACHE = {"t": 0, "v": None}


def pc_online():
    """mainpc online over Tailscale (cached 60 s)."""
    if time.time() - _PC_CACHE["t"] < 60:
        return _PC_CACHE["v"]
    v = None
    try:
        b = rjson(HERE / "bridge_config.json", {}) or {}
        r = subprocess.run([b.get("tailscale", "tailscale"), "status", "--json"], capture_output=True, text=True, timeout=8)
        peers = json.loads(r.stdout).get("Peer", {}) if r.returncode == 0 else {}
        for p in peers.values():
            if p.get("HostName", "").lower() == str(b.get("peer_host", "mainpc")).lower():
                v = bool(p.get("Online"))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    _PC_CACHE.update(t=time.time(), v=v)
    return v


def brain_info():
    """Everything the Brain window shows besides the live run: research loop scores, recent log, GPU, PC link."""
    st = rjson(HERE / "improve" / "research_state.json", {}) or {}
    best, spread = st.get("best", {}), st.get("spread", 0)
    log = HERE.parent / "TRAINING_LOG.md"
    entries = []
    if log.exists():
        for blk in log.read_text(encoding="utf-8", errors="replace").split("\n\n"):
            m = re.match(r"\*\*(\d{4}-\d\d-\d\d \d\d:\d\d)\*\*\s*(.*)", blk.strip(), re.S)
            if m:
                entries.append({"time": m[1], "text": re.sub(r"\*\*", "", m[2]).strip()[:400]})
    gpu = system_info().get("gpu")
    inbox = sorted((HERE / "bridge" / "inbox").glob("msg_*_pc.md"))
    last_pc = None
    if inbox:
        h = dict(ln.split(":", 1) for ln in inbox[-1].read_text(encoding="utf-8", errors="replace").split("\n---")[0].splitlines() if ":" in ln)
        last_pc = {"time": h.get("time", "").strip(), "subject": h.get("subject", "").strip()}
    yaas = None
    for d in (HERE / "bridge" / "files", Path.home() / "Downloads", HERE / "storms" / "Yaas_2021"):
        f = d / "yaas_stack_v1.tar"
        if f.exists():
            yaas = {"where": str(f), "gb": round(f.stat().st_size / 2**30, 2)}
            break
    done = (HERE / "storms" / "Yaas_2021" / "out" / "summary.json")
    return {"scores": st.get("scores", {}), "best": {"name": best.get("name"), "mean": best.get("mean")},
            "spread": spread, "bar": round((best.get("mean") or 0) + spread, 4), "tried": st.get("tried", []),
            "log": entries[-10:], "gpu": gpu, "time": time.strftime("%H:%M:%S"),
            "pc": {"online": pc_online(), "last_msg": last_pc, "yaas_tar": yaas, "yaas_mapped": done.exists()}}


def autopilot_info():
    st = rjson(HERE / "autopilot_state.json")
    if not st:
        return None
    st["alive"] = st.get("state") == "running" and time.time() - st.get("updated", 0) < 600
    log = HERE / "autopilot.log"
    st["log"] = chr(10).join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]) if log.exists() else ""
    return st


def system_info():
    gpu = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        if r.returncode == 0 and r.stdout.strip():
            n, u, mu, mt, t = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            gpu = {"name": n, "util": int(u), "mem_used_mb": int(mu), "mem_total_mb": int(mt), "temp_c": int(t)}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    du = shutil.disk_usage(HERE)
    return {"host": socket.gethostname(), "os": platform.platform(terse=True), "python": sys.version.split()[0],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": gpu, "ram": ram(),
            "disk": {"free_gb": round(du.free / 2**30, 1), "total_gb": round(du.total / 2**30, 1)},
            "jobs": {k: v for k, v in JOBS.items() if v.get("state") == "running"},
            "autopilot": autopilot_info(),
            "config": {k: cfg().get(k) for k in ("python", "build_cmd", "build_cwd", "predict_extra", "threshold")}}


def pixel(name, x, y):
    import numpy as np
    d = storm_dir(name)
    out, st = d / "out", stack_path(d)
    s = rjson(out / "summary.json") or {}
    g, step = s["grid"], s.get("preview", {}).get("step", 1)
    r, c = int(y) * step + step // 2, int(x) * step + step // 2
    if not (0 <= r < g["H"] and 0 <= c < g["W"]):
        raise ValueError("outside the map")
    v = {"row": r, "col": c, "lat": round(g["north"] - (r + .5) * g["res"], 4), "lon": round(g["west"] + (c + .5) * g["res"], 4)}

    def at(f):
        return np.load(f, mmap_mode="r")[r, c].item() if f.exists() else None
    p = at(out / "detect_prob.npy")
    v["flood_prob"] = None if p is None or p != p else round(p, 3)
    cl = at(out / "detect_class.npy")
    v["class"] = {0: "dry land", 1: "FLOOD", 2: "permanent water", 255: "no radar"}.get(cl)
    rk = at(out / "risk_prob.npy")
    v["risk"] = None if rk is None or rk != rk else round(rk, 3)
    ref = at(st / "ref_labels.npy")
    v["published_map"] = {-1: "unlabelled", 0: "dry", 1: "flood", 2: "water"}.get(ref) if ref is not None else None
    ref2 = at(st / "ref2_labels.npy")
    v["independent_map"] = {0: "dry", 1: "flood", 255: "not mapped"}.get(ref2) if ref2 is not None else None
    sp = at(st / "split.npy")
    v["block"] = {0: "training", 1: "validation", 2: "TEST (held out)", 255: "unused"}.get(sp) if sp is not None else None
    return v


# ---------------- live model test (live_run.py) ----------------
LIVE = {}


def live_start(name, model, pace):
    d = storm_dir(name)
    st = stack_path(d)
    if not (st / "inputs_detect.npy").exists():
        raise ValueError("this storm has no stack yet")
    if any(p.poll() is None for p in LIVE.values()):
        raise ValueError("a live test is already running")
    if not (MODELS / model).exists() or not model.startswith("detect"):
        raise ValueError("pick a detection model")
    pace = min(max(float(pace), 0.0), 2.0)
    c = cfg()
    cmd = [c["python"], str(HERE / "live_run.py"), "--stack", str(st), "--out", str(d / "out" / "live"),
           "--model", str(MODELS / model), "--pace", str(pace)] + (["--cpu"] if "--cpu" in c.get("predict_extra", []) else [])
    (d / "out").mkdir(exist_ok=True)
    LIVE[name] = subprocess.Popen(cmd, cwd=str(HERE), stdout=open(d / "live.log", "w", encoding="utf-8"),
                                  stderr=subprocess.STDOUT)


def live_stop(name):
    f = storm_dir(name) / "out" / "live"
    if f.exists():
        (f / "STOP").write_text("stop", encoding="utf-8")


def live_state(name):
    d = storm_dir(name)
    s = rjson(d / "out" / "live" / "state.json") or {"state": "none"}
    p = LIVE.get(name)
    s["process"] = None if p is None else ("running" if p.poll() is None else f"exit {p.returncode}")
    if p is not None and p.poll() not in (None, 0) and (d / "live.log").exists():
        s["error"] = (d / "live.log").read_text(encoding="utf-8", errors="replace")[-600:]
    return s


# ---------------- live training (reads train.py's status.json / activations.npz / preview.npz) ----------------
PRESETS = {
    "quick_demo": ("Quick demo: 4 short epochs, validation every epoch (about 10-15 min). Results NOT used.",
                   "--task detect --epochs 4 --steps 100 --batch 12 --base 48 --lr 0.0005 --flood-frac 0.65 "
                   "--eval-every 1 --patience 99"),
    "seed_check": ("Seed check: the v4_B recipe with another random seed (about 3.5-4 h). Shows how much scores move by chance.",
                   "--task detect --epochs 60 --steps 300 --batch 12 --base 48 --lr 0.0005 --flood-frac 0.65 "
                   "--patience 12 --seed {seed}"),
}


def kit_path(kit):
    for k in cfg().get("kit_dirs", []):
        if Path(k).name == kit:
            return Path(k)
    raise ValueError("unknown kit")


def run_live(kit, run):
    s = rjson(kit / "runs" / run / "status.json") or {}
    return s.get("state") in ("starting", "training", "validating") and time.time() - s.get("updated", 0) < 300


def train_live(kit_name=None, run=None):
    kits = [Path(k) for k in cfg().get("kit_dirs", []) if (Path(k) / "runs").exists()]
    if kit_name and run:
        kit = kit_path(kit_name)
        if not SAFE.match(run):
            raise ValueError("bad run name")
    else:
        cands = [(kit, r.name, (rjson(r / "status.json") or {}).get("updated", 0)) for kit in kits
                 for r in (kit / "runs").iterdir() if (r / "status.json").exists() and not r.name.startswith("_")]
        live = [c for c in cands if run_live(c[0], c[1])]
        pick = max(live or cands, key=lambda c: c[2]) if cands else None
        if not pick:
            return {"run": None}
        kit, run = pick[0], pick[1]
    rd = kit / "runs" / run
    st = rjson(rd / "status.json") or {}
    pidf = kit / f"runs_{run}.pid"
    return {"kit": kit.name, "run": run, "live": run_live(kit, run), "status": st,
            "history": parse_history(kit, run), "act": rjson(rd / "activations.json"),
            "has_preview": (rd / "preview.npz").exists(), "stoppable": pidf.exists() and run_live(kit, run),
            "img_t": int(max([(rd / f).stat().st_mtime for f in ("activations.npz", "preview.npz") if (rd / f).exists()] or [0])),
            "runs": [{"kit": k.name, "run": r.name, "live": run_live(k, r.name)} for k in kits
                     for r in sorted((k / "runs").iterdir(), key=lambda r: r.stat().st_mtime, reverse=True)
                     if (r / "status.json").exists() and not r.name.startswith("_")][:30],
            "presets": {k: v[0] for k, v in PRESETS.items()}, "kits": [k.name for k in kits]}


def png_bytes(img):
    import zlib
    import struct
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, {1: 0, 3: 2, 4: 6}[c], 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def train_img(kit_name, run, key):
    import numpy as np
    kit = kit_path(kit_name)
    if not SAFE.match(run) or not re.match(r"^[a-z_0-9]+$", key):
        raise ValueError("bad request")
    rd = kit / "runs" / run
    pal = np.zeros((256, 3), np.uint8) + 40
    pal[0], pal[1], pal[2], pal[255] = (70, 76, 92), (40, 200, 255), (24, 44, 110), (12, 16, 28)
    for _ in range(5):
        try:
            if key == "preview":
                z = np.load(rd / "preview.npz")
                return png_bytes(pal[z["cls"]])
            z = np.load(rd / "activations.npz")
            a = z[key]
            break
        except (OSError, ValueError, KeyError, EOFError):
            time.sleep(0.1)
    else:
        raise ValueError("image not available yet")
    if key in ("label", "pred"):
        return png_bytes(pal[(a.astype(np.int16) % 256).astype(np.uint8)])
    t = a.astype(np.float32) / 255.0
    if key.startswith("in_"):
        return png_bytes(a.astype(np.uint8))
    stops = [(0, (8, 10, 30)), (0.35, (90, 60, 200)), (0.7, (40, 190, 255)), (1, (235, 255, 255))]
    out = np.zeros(t.shape + (3,), np.float32)
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        m = (t >= p0) & (t <= p1)
        u = ((t[m] - p0) / (p1 - p0))[:, None]
        out[m] = np.array(c0) + (np.array(c1) - np.array(c0)) * u
    return png_bytes(out.astype(np.uint8))


def spawn(cmd, cwd):
    """cmd: argument list. Detached so it survives a console restart; on Windows keep_awake_run.py stops sleep."""
    if os.name == "nt":
        # start through WMI: the new process is not a child of this console (or of any app that runs it),
        # so closing/restarting the console or that app cannot take the training down with it
        line = subprocess.list2cmdline(cmd).replace("'", "''")
        ps = (f"$r=Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{{CommandLine='{line}';"
              f"CurrentDirectory='{cwd}'}}; if($r.ReturnValue -ne 0){{exit 1}}; $r.ProcessId")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not r.stdout.strip().isdigit():
            raise ValueError("could not start the run: " + (r.stderr or r.stdout)[-300:])
        return int(r.stdout.strip())
    else:
        p = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
    return p.pid


def train_start(kit_name, preset, seed):
    kit = kit_path(kit_name)
    if preset not in PRESETS:
        raise ValueError("unknown preset")
    for k in cfg().get("kit_dirs", []):
        for r in (Path(k) / "runs").iterdir() if (Path(k) / "runs").exists() else []:
            if run_live(Path(k), r.name):
                raise ValueError(f"{r.name} is already training: one run at a time (6 GB GPU)")
    pre = kit.name.replace("train_kit_", "")
    seed = int(seed or 1)
    name = f"{pre}_B_seed{seed}" if preset == "seed_check" else f"{pre}_demo_{time.strftime('%m%d_%H%M')}"
    if (kit / "runs" / name).exists():
        raise ValueError(f"{name} already exists")
    args = PRESETS[preset][1].format(seed=seed).split()
    py = cfg()["python"]
    cmd = [py, str(HERE / "keep_awake_run.py"), "--out", str(kit / f"runs_{name}.out"), "--",
           py, "-u", "train.py", *args, "--mmap", "--name", name]
    pid = spawn(cmd, str(kit))
    (kit / f"runs_{name}.pid").write_text(str(pid), encoding="utf-8")
    return {"run": name, "pid": pid, "kit": kit.name}


def train_stop(kit_name, run):
    kit = kit_path(kit_name)
    f = kit / f"runs_{run}.pid"
    if not SAFE.match(run) or not f.exists():
        raise ValueError("only runs started from this console can be stopped here")
    pid = int(f.read_text().strip())
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        os.killpg(pid, signal.SIGTERM)
    st = kit / "runs" / run / "status.json"
    s = rjson(st)
    if s:
        s.update(state="stopped", updated=time.time())
        st.write_text(json.dumps(s), encoding="utf-8")
    f.unlink()
    return {"stopped": run}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        try:
            if p == "/":
                page = HERE / "ops_console.html"
                return self.send(200, page.read_bytes() if page.exists() else b"ops_console.html missing", "text/html; charset=utf-8")
            if p == "/brain":
                page = HERE / "brain.html"
                return self.send(200, page.read_bytes() if page.exists() else b"brain.html missing", "text/html; charset=utf-8")
            if p == "/api/brain":
                return self.send(200, brain_info())
            if p == "/api/storms":
                return self.send(200, all_storms())
            if p == "/api/models":
                return self.send(200, models_info())
            if p == "/api/training":
                return self.send(200, training_info())
            if p == "/api/bridge":
                return self.send(200, bridge_msgs())
            if p == "/api/system":
                return self.send(200, system_info())
            if p == "/api/improve":
                d = HERE / "improve"
                rd = lambda f: (d / f).read_text(encoding="utf-8") if (d / f).exists() else None
                return self.send(200, {"recipe": rjson(d / "recipe.json"), "test": rjson(d / "test_scores.json"),
                                       "val_md": rd("val_report.md"), "test_md": rd("test_report.md"), "progress": rd("progress.txt")})
            if p == "/api/todo":
                f = HERE.parent / "TODO_Rj.md"
                return self.send(200, f.read_text(encoding="utf-8") if f.exists() else "", "text/plain; charset=utf-8")
            m = re.match(r"^/api/storm/([^/]+)/log$", p)
            if m:
                f = storm_dir(m[1]) / "jobs.log"
                t = f.read_text(encoding="utf-8", errors="replace")[-8000:] if f.exists() else ""
                return self.send(200, t, "text/plain; charset=utf-8")
            m = re.match(r"^/api/storm/([^/]+)/pixel$", p)
            if m:
                return self.send(200, pixel(m[1], q["x"][0], q["y"][0]))
            if p == "/api/train/live":
                return self.send(200, train_live(q.get("kit", [None])[0], q.get("run", [None])[0]))
            if p == "/api/train/img":
                return self.send(200, train_img(q["kit"][0], q["run"][0], q["k"][0]), "image/png")
            m = re.match(r"^/api/live/([^/]+)$", p)
            if m:
                return self.send(200, live_state(m[1]))
            m = re.match(r"^/files/([^/]+)/((?:layers/|live/|live/tiles/)?[A-Za-z0-9_.-]+)$", p)
            if m:
                f = storm_dir(m[1]) / "out" / m[2]
                if f.suffix in SERVE and f.exists():
                    return self.send(200, f.read_bytes(), SERVE[f.suffix])
            self.send(404, {"error": "not found"})
        except (ValueError, KeyError, IndexError, ImportError) as e:
            self.send(400, {"error": str(e)})

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if p == "/api/storm":
                d = storm_dir(body.get("name", ""))
                d.mkdir(exist_ok=True)
                keep = ("name", "landfall", "pre_from", "pre_to", "post_from", "post_to", "bbox", "track_csv",
                        "notes", "stack_path")
                req = {k: str(body.get(k, "")).strip() for k in keep if str(body.get(k, "")).strip()}
                req["saved"] = time.strftime("%Y-%m-%d %H:%M")
                (d / "request.json").write_text(json.dumps(req, indent=1), encoding="utf-8")
                return self.send(200, storm_info(d))
            m = re.match(r"^/api/storm/([^/]+)/(build|predict)$", p)
            if m:
                start(m[1], m[2])
                return self.send(200, {"ok": True})
            if p == "/api/config":
                names = {f.name for f in MODELS.glob("*.pt")}
                upd = {}
                if "detect_models" in body:
                    dm = [x for x in body["detect_models"] if x in names and x.startswith("detect")]
                    if not dm:
                        raise ValueError("pick at least one detection model")
                    upd["detect_models"] = dm
                if "risk_model" in body:
                    if body["risk_model"] not in names:
                        raise ValueError("unknown risk model")
                    upd["risk_model"] = body["risk_model"]
                if "threshold" in body:
                    t = float(body["threshold"])
                    if not 0.05 <= t <= 0.95:
                        raise ValueError("threshold must be 0.05-0.95")
                    upd["threshold"] = round(t, 2)
                if "tta" in body:
                    if int(body["tta"]) not in (1, 4, 8):
                        raise ValueError("tta must be 1, 4 or 8")
                    upd["tta"] = int(body["tta"])
                save_cfg(upd)
                return self.send(200, models_info())
            m = re.match(r"^/api/live/([^/]+)/(start|stop)$", p)
            if m:
                if m[2] == "start":
                    live_start(m[1], str(body.get("model", "")), body.get("pace", 0.35))
                else:
                    live_stop(m[1])
                return self.send(200, {"ok": True})
            if p == "/api/train/start":
                return self.send(200, train_start(str(body.get("kit", "")), str(body.get("preset", "")), body.get("seed", 1)))
            if p == "/api/train/stop":
                return self.send(200, train_stop(str(body.get("kit", "")), str(body.get("run", ""))))
            if p == "/api/autopilot/stop":
                (HERE / "AUTOPILOT_STOP").write_text("stop", encoding="utf-8")
                return self.send(200, {"ok": True})
            if p == "/api/bridge/send":
                return self.send(200, {"result": bridge_send(str(body.get("subject", "")), str(body.get("text", "")))})
            self.send(404, {"error": "not found"})
        except (ValueError, json.JSONDecodeError, subprocess.SubprocessError, OSError) as e:
            self.send(400, {"error": str(e)})




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    cfg()
    print(f"Flood Ops Console: http://{a.host}:{a.port}   (Ctrl+C to stop)")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
