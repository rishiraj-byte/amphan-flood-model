# Training log (laptop, kit v4)

Every experiment and decision, newest at the bottom. Rule: a change is kept only if its VALIDATION score (mean IoU of published + independent map, TTA 8, best cutoff) beats the best so far by more than the seed spread of the old recipe. Test blocks are scored only at the very end, once.

**2026-11-21 03:41** Research loop started. Waiting for training queue v5 to finish.

**2026-11-21 21:23** Queue v5 result.
- Old recipe, 3 seeds: v4_B 0.383, v4_B_seed1 0.384, v4_sarOnly 0.399 -> mean 0.389, luck spread 0.016.
- v5_wide384 (--patch 384 --batch 6 --steps 300 --epochs 60 --patience 12): 0.392 (pub 0.444, ind 0.340) -> drop, within luck.
- v5_long (--patch 256 --batch 12 --steps 450 --epochs 80 --patience 16): 0.390 (pub 0.449, ind 0.330) -> drop, within luck.
- Best recipe now: **old recipe (v4_B x3 seeds)** (0.389).

**2026-11-21 21:23** Experiment **v6_ff80**: more flood-centred patches, to attack low recall. Change {'--flood-frac': '0.8'} on top of old recipe (v4_B x3 seeds).

**2026-11-22 00:05** v6_ff80: validation 0.396 (pub 0.455, ind 0.338, cutoff 0.3, best epoch 28). Bar 0.405 -> **DROP: within luck**.

**2026-11-22 00:05** Experiment **v6_base64**: wider network (more capacity). Change {'--base': '64', '--batch': '8'} on top of old recipe (v4_B x3 seeds).

**2026-11-22 02:39** v6_base64: validation 0.395 (pub 0.449, ind 0.341, cutoff 0.25, best epoch 44). Bar 0.405 -> **DROP: within luck**.

**2026-11-22 02:39** Experiment **v6_lr1e3**: higher learning rate. Change {'--lr': '0.001'} on top of old recipe (v4_B x3 seeds).
