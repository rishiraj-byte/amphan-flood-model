# Flood maps compared on Amphan test blocks (2026-11-25 13:24)

Model = ops ensemble, TTA 8, cutoff 0.30. Each row judges map A against map B as the truth, on test-block cells where both have data. Reference-vs-reference rows = how much two expert maps agree.

| A (judged) | B (truth) | flood IoU | precision | recall | cells compared (km²) | flood in B (cells) |
|---|---|---|---|---|---|---|
| model | published (Mondal) | **0.511** | 0.639 | 0.719 | 29,809 | 77,988 |
| model | independent (ARIA) | **0.344** | 0.635 | 0.429 | 11,336 | 41,743 |
| published (Mondal) | independent (ARIA) | **0.341** | 0.646 | 0.419 | 9,916 | 40,302 |

## Same, only where all 2 reference maps have data (9,916 km²)

| A (judged) | B (truth) | flood IoU | precision | recall |
|---|---|---|---|---|
| model | published (Mondal) | **0.559** | 0.692 | 0.744 |
| model | independent (ARIA) | **0.353** | 0.635 | 0.442 |
| published (Mondal) | independent (ARIA) | **0.341** | 0.646 | 0.419 |

Caveat: the model's training labels come from the published Mondal map (Zenodo 4390084).
