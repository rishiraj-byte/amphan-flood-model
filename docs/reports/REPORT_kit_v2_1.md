# Kit v2_1 results (31 detect / 22 forecast layers) vs v2

| model | VAL IoU | TEST IoU | F1 | P | R | note |
|---|---|---|---|---|---|---|
| v2 B | 0.461 | 0.441 | 0.612 | 0.664 | 0.567 |  |
| v2 final 70 % | - | 0.457 | 0.627 | 0.655 | 0.602 | score to beat (0.457) |
| v2_1 B | 0.466 | 0.369 | 0.539 | 0.563 | 0.517 |  |
| v2_1 final 70 % | - | 0.354 | 0.523 | 0.487 | 0.565 | v2_1 SCORE |
| v2_1 SAR only | 0.469 | 0.465 | 0.635 | 0.657 | 0.615 | no terrain; compare with v2_1 B |
| v2_1 full 100 % | - | 0.482 | 0.650 | 0.549 | 0.798 | map model; TEST in-sample, NOT a score |
| v2 forecast | 0.245 | 0.230 | 0.374 | 0.368 | 0.381 |  |
| v2_1 forecast | 0.247 | 0.195 | 0.327 | 0.295 | 0.366 |  |

Otsu baseline IoU: 0.287
Labels: {'-1': 16819944, '0': 3964362, '1': 107481, '2': 966658}  pairs: 1
Flood area (full model): 1,613 km2
Map: train_kit_v2_1/runs/v2_1_full100/flood_class.tif, flood_probability.tif
CAVEAT: labels and test reference are the same published map.
NOTE v2_1: PC announced 32 detect layers, kit has 31 (b_ratio, a_ratio, d_ratio, dist_water_km added vs v2). Trained as delivered.

## BEST MODEL (9 Nov): radar-only (no terrain layers)
| model | TEST IoU | F1 | P | R |
|---|---|---|---|---|
| v2_1 SAR-only final 70 % | **0.482** | 0.651 | 0.633 | 0.669 |
| v2 final 70 % (previous best) | 0.457 | 0.627 | 0.655 | 0.602 |
Chosen because SAR-only had the best VAL IoU (0.469). Map: train_kit_v2_1/runs/v2_1_sarOnly_full100 (1,564 km2; its test 0.534 is in-sample, not a score).
