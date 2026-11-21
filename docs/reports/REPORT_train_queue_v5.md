# Training queue v5 (2026-11-21 21:18, 17.9 h)

Validation blocks, single model, TTA 8, best cutoff. Old recipe = v4_B / v4_B_seed1 / v4_sarOnly (3 seeds).

| run | recipe change | best epoch | cutoff | IoU published | IoU independent | mean |
|---|---|---|---|---|---|---|
| v4_B | old recipe | 38 | 0.30 | 0.444 | 0.323 | 0.383 |
| v4_B_seed1 | old recipe | 42 | 0.25 | 0.443 | 0.325 | 0.384 |
| v4_sarOnly | old recipe | 34 | 0.30 | 0.451 | 0.347 | 0.399 |
| v5_wide384 | --patch 384 --batch 6 --steps 300 --epochs 60 --patience 12 | 50 | 0.25 | 0.444 | 0.340 | 0.392 |
| v5_long | --patch 256 --batch 12 --steps 450 --epochs 80 --patience 16 | 20 | 0.30 | 0.449 | 0.330 | 0.390 |

Bar to beat = old mean 0.389 + old seed spread 0.016 = **0.405**.
Best candidate: v5_wide384 (0.392).
**Verdict: no recipe beat the old one by more than luck; nothing changed.**

Caveat: the published map is the source of the training labels; the independent (ARIA) map is the honest one.
