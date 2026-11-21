# Seed check, kit v4 (2026-11-17 03:06)

Same recipe (v4_B), only the random seed differs: seed 0 = v4_B, seed 1 = v4_B_seed1.

| run | map | area | IoU | P | R | F1 |
|---|---|---|---|---|---|---|
| v4_B | Mondal (published) | test blocks | 0.412 | 0.665 | 0.520 | 0.583 |
| v4_B | ref2 (independent) | test blocks | 0.306 | 0.674 | 0.359 | 0.468 |
| v4_B | ref2 (independent) | whole circle | 0.318 | 0.715 | 0.364 | 0.483 |
| v4_B_seed1 | Mondal (published) | test blocks | 0.448 | 0.721 | 0.543 | 0.619 |
| v4_B_seed1 | ref2 (independent) | test blocks | 0.291 | 0.684 | 0.336 | 0.451 |
| v4_B_seed1 | ref2 (independent) | whole circle | 0.305 | 0.737 | 0.342 | 0.467 |

- Test IoU vs published map (from metrics.json): v4_B 0.4155489319423746 (best epoch 38), v4_B_seed1 0.4524183727635126 (best epoch 42); difference 0.037.
- Difference vs the independent map (test blocks): 0.015.
- Meaning: two models whose scores differ by less than about this much cannot be called better or worse from one training each. (v3_ops70 0.465 vs v4_plus70 0.468 differ by 0.003.)
- Caveat: the published map is the source of the training labels; the independent map is the honest one.
