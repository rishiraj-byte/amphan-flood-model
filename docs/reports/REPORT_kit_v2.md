# Kit v2 results (27 input layers) vs v1 (16)

| model | VAL IoU | TEST IoU | F1 | P | R | note |
|---|---|---|---|---|---|---|
| v1 B | 0.493 | 0.427 | 0.599 | 0.561 | 0.643 |  |
| v1 final 70 % | - | 0.433 | 0.604 | 0.632 | 0.578 | v1 score to beat |
| v2 B | 0.461 | 0.441 | 0.612 | 0.664 | 0.567 |  |
| v2 final 70 % | - | 0.457 | 0.627 | 0.655 | 0.602 | v2 SCORE |
| v2 SAR only (no terrain) | 0.476 | 0.422 | 0.594 | 0.698 | 0.517 | compare with v2 B |
| v2 full 100 % | - | 0.577 | 0.732 | 0.659 | 0.823 | map model; TEST is in-sample, NOT a score |
| v2 forecast | 0.245 | 0.230 | 0.374 | 0.368 | 0.381 |  |

Otsu baseline IoU: 0.287
Flood area (full model): 1,655 km2
Map: train_kit_v2/runs/v2_full100/flood_class.tif, flood_probability.tif
CAVEAT: labels and test reference are the same published map.
