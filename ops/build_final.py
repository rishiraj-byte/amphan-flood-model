"""build_final.py - pack the flood model in use into one self-contained deliverable folder (+ zip + checksums).

  python build_final.py --version v1 [--results ops/storms/Amphan_2020_kitv4/out]

Takes the models and settings from ops_config.json (detect_models, risk_model, threshold, tta), a copy of predict.py
with those settings as its defaults, the verified Amphan results (GeoTIFFs, scores, previews) and the handbook PDF.
Writes D:/sar_flood/FINAL/FloodModel_<version>/ and FloodModel_<version>.zip. Nothing outside FINAL/ is changed.
"""
import argparse
import hashlib
import json
import re
import shutil
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_png(path):
    """Decoder for the PNGs predict.py writes (8-bit RGB/RGBA, filter 0 on every row)."""
    b = Path(path).read_bytes()
    i, idat, w, h, c = 8, b"", 0, 0, 3
    while i < len(b):
        n = struct.unpack(">I", b[i:i + 4])[0]
        t, d = b[i + 4:i + 8], b[i + 8:i + 8 + n]
        if t == b"IHDR":
            w, h, _, ct = struct.unpack(">IIBB", d[:10])
            c = 4 if ct == 6 else 3
        elif t == b"IDAT":
            idat += d
        i += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * c + 1)
    if raw[:, 0].any():
        raise ValueError(f"{path}: unsupported PNG filter")
    return raw[:, 1:].reshape(h, w, c)


def overlay(bg_png, fg_png, out_png, write_png):
    bg = read_png(bg_png)[..., :3].astype(np.float32)
    fg = read_png(fg_png).astype(np.float32)
    a = fg[..., 3:4] / 255.0
    write_png(out_png, (bg * (1 - a) + fg[..., :3] * a).clip(0, 255).astype(np.uint8))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--results", default=str(HERE / "storms" / "Amphan_2020_kitv4" / "out"))
    ap.add_argument("--readme", default=str(HERE / "FINAL_README.md"))
    ap.add_argument("--pdf", default=str(ROOT / "Amphan_Flood_Model_Handbook.pdf"))
    ap.add_argument("--extra", action="append", default=[],
                    help="more storm results: FOLDER_NAME=PATH, e.g. results_yaas_2021=ops/storms/Yaas_2021/out")
    ap.add_argument("--log", default=str(ROOT / "TRAINING_LOG.md"))
    a = ap.parse_args()
    cfg = json.loads((HERE / "ops_config.json").read_text(encoding="utf-8"))
    res = Path(a.results)
    summ = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    det, risk = summ["detect"]["models"], summ["risk"]["model"]
    thr, tta = summ["threshold"], summ.get("tta", 1)
    if (sorted(det), risk, thr, tta) != (sorted(cfg["detect_models"]), cfg["risk_model"], cfg["threshold"], cfg.get("tta", 1)):
        raise SystemExit("the results were not made with the settings in ops_config.json - re-run the models first")
    pkg = ROOT / "FINAL" / f"FloodModel_{a.version}"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "models").mkdir(parents=True)
    for m in det + [risk]:
        shutil.copy2(HERE / "models" / m, pkg / "models" / m)
        card = (HERE / "models" / m).with_suffix(".json")
        if card.exists():
            shutil.copy2(card, pkg / "models" / card.name)
    # predict.py with the final settings as defaults
    src = (HERE / "predict.py").read_text(encoding="utf-8")
    src = src.replace('ap.add_argument("--threshold", type=float, default=0.5)', f'ap.add_argument("--threshold", type=float, default={thr})')
    src = re.sub(r'(ap\.add_argument\("--tta", type=int, default=)1(,)', rf"\g<1>{tta}\2", src)
    src = src.replace("flood = probability >= --threshold (0.5, fixed, not tuned).",
                      f"flood = probability >= --threshold (default {thr}: chosen on validation blocks, 20 Nov 2026).")
    assert f"default={thr})" in src and f"default={tta}," in src
    (pkg / "predict.py").write_text(src, encoding="utf-8")
    # runners
    (pkg / "run_flood_model.bat").write_text(
        "@echo off\r\nREM Flood model: run_flood_model.bat STACK_FOLDER OUTPUT_FOLDER [--cpu]\r\n"
        "set PY=python\r\nif exist \"D:\\sar_flood\\train_kit\\.venv\\Scripts\\python.exe\" set PY=D:\\sar_flood\\train_kit\\.venv\\Scripts\\python.exe\r\n"
        "if \"%~2\"==\"\" (echo Usage: run_flood_model.bat STACK_FOLDER OUTPUT_FOLDER [--cpu] & exit /b 1)\r\n"
        "\"%PY%\" \"%~dp0predict.py\" --stack \"%~1\" --out \"%~2\" %3\r\n", encoding="utf-8")
    (pkg / "run_amphan.bat").write_text(
        "@echo off\r\nREM Re-make the Amphan 2020 map from kit v4 (about 2 min on the laptop GPU)\r\n"
        "call \"%~dp0run_flood_model.bat\" D:\\sar_flood\\train_kit_v4\\data \"%~dp0rerun_amphan_2020\" %1\r\n", encoding="utf-8")
    (pkg / "run_flood_model.sh").write_text(
        "#!/bin/sh\n# Flood model: ./run_flood_model.sh STACK_FOLDER OUTPUT_FOLDER [--cpu]\n"
        "[ -z \"$2\" ] && { echo 'usage: run_flood_model.sh STACK_FOLDER OUTPUT_FOLDER [--cpu]'; exit 1; }\n"
        "exec python3 \"$(dirname \"$0\")/predict.py\" --stack \"$1\" --out \"$2\" $3\n", encoding="utf-8", newline="\n")
    (pkg / "requirements.txt").write_text(
        "# tested with these versions (laptop, Windows 11, RTX 3050 6 GB, CUDA 12.6)\n"
        "torch==2.14.1\nnumpy==2.5.4\nrasterio==1.5.2   # optional: without it no GeoTIFFs, the .npy/.png outputs still work\n",
        encoding="utf-8")
    # verified Amphan results
    r = pkg / "results_amphan_2020"
    (r / "layers").mkdir(parents=True)
    for f in ("flood_class.tif", "flood_probability.tif", "flood_risk.tif", "flood_map.png", "risk_map.png",
              "summary.json", "eval.json", "log.txt"):
        if (res / f).exists():
            shutil.copy2(res / f, r / f)
    for f in (res / "layers").glob("*.png"):
        if f.name != "prob_gray.png":
            shutil.copy2(f, r / "layers" / f.name)
    import sys
    sys.path.insert(0, str(HERE))
    from predict import write_png
    overlay(res / "layers" / "radar_after.png", res / "layers" / "flood.png", r / "preview_flood_over_radar.png", write_png)
    overlay(res / "layers" / "radar_before.png", res / "layers" / "risk.png", r / "preview_risk_over_radar.png", write_png)
    # more storms (same models + settings only)
    for spec in a.extra:
        name, _, path = spec.partition("=")
        xr = Path(path) if Path(path).is_absolute() else ROOT / path
        xs = json.loads((xr / "summary.json").read_text(encoding="utf-8"))
        if (sorted(xs["detect"]["models"]), xs["risk"]["model"], xs["threshold"], xs.get("tta", 1)) != \
                (sorted(cfg["detect_models"]), cfg["risk_model"], cfg["threshold"], cfg.get("tta", 1)):
            raise SystemExit(f"{xr}: not made with the settings in ops_config.json")
        xd = pkg / name
        (xd / "layers").mkdir(parents=True)
        for f in ("flood_class.tif", "flood_probability.tif", "flood_risk.tif", "flood_map.png", "risk_map.png",
                  "summary.json", "eval.json", "log.txt"):
            if (xr / f).exists():
                shutil.copy2(xr / f, xd / f)
        for f in (xr / "layers").glob("*.png"):
            if f.name != "prob_gray.png":
                shutil.copy2(f, xd / "layers" / f.name)
        if (xr / "layers" / "radar_after.png").exists():
            overlay(xr / "layers" / "radar_after.png", xr / "layers" / "flood.png", xd / "preview_flood_over_radar.png", write_png)
            overlay(xr / "layers" / "radar_before.png", xr / "layers" / "risk.png", xd / "preview_risk_over_radar.png", write_png)
    shutil.copy2(a.readme, pkg / "README.md")
    if Path(a.log).exists():
        shutil.copy2(a.log, pkg / "TRAINING_LOG.md")
    from build_dashboard import build as build_dashboard
    build_dashboard(pkg)
    if Path(a.pdf).exists():
        shutil.copy2(a.pdf, pkg / Path(a.pdf).name)
    files = sorted(p for p in pkg.rglob("*") if p.is_file())
    (pkg / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(pkg).as_posix()}\n" for p in files), encoding="utf-8", newline="\n")
    z = pkg.parent / f"FloodModel_{a.version}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(pkg.rglob("*")):
            if p.is_file():
                zf.write(p, Path(pkg.name) / p.relative_to(pkg))
    (pkg.parent / f"FloodModel_{a.version}.zip.sha256").write_text(f"{sha(z)}  {z.name}\n", encoding="utf-8", newline="\n")
    print(f"{pkg}: {len(files) + 1} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB; zip {z.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
