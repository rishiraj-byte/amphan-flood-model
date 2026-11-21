"""Write a run's predictions as GeoTIFFs (EPSG:4326). Needs rasterio.

  python to_geotiff.py runs/detect_20261101_1200
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def write_run(run_dir, meta=None):
    run_dir = Path(run_dir)
    if meta is None:
        meta = json.loads((run_dir.parents[1] / "data" / "meta.json").read_text())
    g = meta["grid"]
    tf = from_origin(g["west"], g["north"], g["res"], g["res"])
    prof = dict(driver="GTiff", height=g["H"], width=g["W"], count=1, crs="EPSG:4326",
                transform=tf, compress="deflate", tiled=True)
    cls = np.load(run_dir / "pred_class.npy")
    with rasterio.open(run_dir / "flood_class.tif", "w", dtype="uint8", nodata=255, **prof) as d:
        d.write(cls, 1)
        d.update_tags(classes="detect: 0 dry, 1 flood, 2 water | forecast: 0 not flooded, 1 flooded")
    prob = np.load(run_dir / "pred_flood_prob.npy").astype(np.float32)
    with rasterio.open(run_dir / "flood_probability.tif", "w", dtype="float32", nodata=np.nan, **prof) as d:
        d.write(prob, 1)


if __name__ == "__main__":
    write_run(sys.argv[1])
    print("written:", Path(sys.argv[1]) / "flood_class.tif")
