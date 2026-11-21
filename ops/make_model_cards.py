"""make_model_cards.py - write models/<name>.json 'model cards' (task, layers, test scores) for the Models page.
Scores are copied from each training run's metrics.json, never typed by hand."""
import json, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
SRC = {  # checkpoint -> (training run folder, plain-language note)
    "detect_v3_final70_amphan_only.pt": (r"D:\sar_flood\train_kit_v3\runs\v3_final70", "Amphan only: needs date-named p_2020 layers"),
    "detect_v3_ops70.pt": (r"D:\sar_flood\train_kit_v3\runs\v3_ops70", "storm-agnostic, 16 radar layers"),
    "detect_v4_plus70.pt": (r"D:\sar_flood\train_kit_v4\runs\v4_final70", "storm-agnostic, kit v4 'plus' (no rain yet)"),
    "risk_0_v3_ops.pt": (r"D:\sar_flood\train_kit_v3\runs\v3_risk_ops", "early warning: pre-landfall radar + storm track"),
    "risk_v3_forecast_amphan_only.pt": (r"D:\sar_flood\train_kit_v3\runs\v3_forecast", "Amphan only"),
    "risk_v4_plus.pt": (r"D:\sar_flood\train_kit_v4\runs\v4_forecast", "early warning, kit v4 (no rain yet)"),
}
for name, (run, note) in SRC.items():
    p = HERE / "models" / name
    if not p.exists():
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    card = {"file": name, "task": ck.get("task"), "channels": ck.get("channels"), "base": ck.get("base"),
            "params_M": round(sum(v.numel() for v in ck["model"].values()) / 1e6, 1), "note": note, "run": run}
    mp = Path(run) / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        t = m.get("test_vs_zenodo_reference") or {}
        f = t.get("flood") or t.get("flooded") or {}
        card["test_vs_published_map"] = {k: round(float(f[k]), 3) for k in ("iou", "precision", "recall", "f1") if k in f}
        if "flood_pr_auc" in t:
            card["test_vs_published_map"]["pr_auc"] = round(float(t["flood_pr_auc"]), 3)
        card["best_epoch"] = m.get("best_epoch")
    (HERE / "models" / (p.stem + ".json")).write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(name, card.get("task"), len(card["channels"] or []), card.get("test_vs_published_map"))
