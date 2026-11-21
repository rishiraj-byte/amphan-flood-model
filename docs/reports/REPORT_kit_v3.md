# Kit v3 results (26 detect / 19 forecast layers) vs v2_1 radar-only

| model | VAL IoU | TEST IoU | F1 | P | R | note |
|---|---|---|---|---|---|---|
| v2_1 radar-only final 70 % | - | 0.482 | 0.651 | 0.633 | 0.669 | BEST SO FAR (0.482) |
| v3 B | 0.445 | 0.512 | 0.677 | 0.691 | 0.664 |  |
| v3 final 70 % | - | 0.540 | 0.701 | 0.693 | 0.709 | v3 SCORE |
| v3 SAR only | 0.454 | 0.523 | 0.687 | 0.698 | 0.676 | drop-terrain run |
| v3 full 100 % | - | 0.613 | 0.760 | 0.691 | 0.845 | map model; TEST in-sample, NOT a score |
| v2_1 forecast | 0.247 | 0.195 | 0.327 | 0.295 | 0.366 |  |
| v3 forecast | 0.244 | 0.245 | 0.394 | 0.399 | 0.388 |  |

Otsu baseline IoU: 0.293
Pairs: 7  [{'pre': '20200516_S1B_T048D_o21599', 'post': '20200522_S1A_T048D_o32670'}, {'pre': '20200516_S1A_T150D_o32597', 'post': '20200522_S1B_T150D_o21701'}, {'pre': '20200512_S1A_T085A_o32532', 'post': '20200524_S1A_T085A_o32707'}, {'pre': '20200519_S1A_T012A_o32634', 'post': '20200525_S1B_T012A_o21738'}, {'pre': '20200514_S1A_T114A_o32561', 'post': '20200526_S1A_T114A_o32736'}, {'pre': '20200516_S1B_T048D_o21599', 'post': '20200528_S1B_T048D_o21774'}, {'pre': '20200516_S1A_T150D_o32597', 'post': '20200528_S1A_T150D_o32772'}]
Labels: {'-1': 11702345, '0': 8761576, '1': 231752, '2': 1162772}  (v2_1: 107,481 flood px)
Labelled km2: 116,116  (v2_1: 57,536)
Flood area (full model): 4,260 km2
Map: train_kit_v3/runs/v3_full100/flood_class.tif, flood_probability.tif
CAVEAT: labels and test reference are the same published map. If the test area changed (new blocks),
the v3 vs v2_1 test numbers are on different areas: compare with care.

## Pair checks

### Calibration: mean after - before VH on dry land (expect ~0 dB)

| pair layers | dry px | mean dB | std dB |
|---|---|---|---|
| a_vh - b_vh | 8,756,986 | +0.49 | 1.59 |

| pair (pre -> post) | dry px | mean dB | std dB |
|---|---|---|---|
| 20200516_S1B_T048D_o21599 -> 20200522_S1A_T048D_o32670 | 5,718,153 | +0.56 | 1.54 |
| 20200516_S1A_T150D_o32597 -> 20200522_S1B_T150D_o21701 | 2,789,913 | +0.34 | 1.72 |
| 20200512_S1A_T085A_o32532 -> 20200524_S1A_T085A_o32707 | 248,920 | +0.41 | 1.14 |
| 20200519_S1A_T012A_o32634 -> 20200525_S1B_T012A_o21738 | 0 | n/a | n/a |
| 20200514_S1A_T114A_o32561 -> 20200526_S1A_T114A_o32736 | 0 | n/a | n/a |
| 20200516_S1B_T048D_o21599 -> 20200528_S1B_T048D_o21774 | 0 | n/a | n/a |
| 20200516_S1A_T150D_o32597 -> 20200528_S1A_T150D_o32772 | 0 | n/a | n/a |

### Per-pair test IoU

run v3_final70

| pair | test px | flood IoU |
|---|---|---|
| 20200516_S1B_T048D_o21599 -> 20200522_S1A_T048D_o32670 | 1,474,578 | 0.543 |
| 20200516_S1A_T150D_o32597 -> 20200522_S1B_T150D_o21701 | 1,088,625 | 0.535 |
| 20200512_S1A_T085A_o32532 -> 20200524_S1A_T085A_o32707 | 45,930 | 0.006 |
| 20200519_S1A_T012A_o32634 -> 20200525_S1B_T012A_o21738 | 0 | no flood |
| 20200514_S1A_T114A_o32561 -> 20200526_S1A_T114A_o32736 | 0 | no flood |
| 20200516_S1B_T048D_o21599 -> 20200528_S1B_T048D_o21774 | 0 | no flood |
| 20200516_S1A_T150D_o32597 -> 20200528_S1A_T150D_o32772 | 0 | no flood |

