# Gate 3 input: self-correspondence 2D error (OpenD4RT 48CLIP, design 4, clip [0, 48))

`scripts/l3d/run_gate3_selfcorr.py --dump output/l3d/dumps/flow3r_T48 --out results/l3d_flow3r/gate3_selfcorr` (18 s). Error of the model's own `u2 = uv_hat(u1, t1, t2, t2)` against GT, in
pixels of the 256x256 model input, rows with `t1 != t2`. Gate 3 uses self-correspondences if the median is
<= 2 px, otherwise CoTracker3. GT-visible: visible at `t2` by GT. Model-visible: `sigmoid(vis) > 0.5`
at `t2`, the label-free filter training would use. Design 4 queries only frames where the track is GT-visible,
so every row is GT-visible and "kept by model vis." is the share the model's filter would keep.

| Subset | Rows (GT-visible) | Median px | Mean px | p90 px | <= 2 px | Gate 3 | Median px (model-visible) | Gate 3 (model-visible) | Kept by model vis. | Median native px |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| `adt_mini` | 473089 | 1.54 | 6.46 | 15.08 | 0.61 | **pass** | 1.54 | **pass** | 1.00 | 3.08 |
| `po_mini` | 473021 | 1.75 | 4.64 | 11.23 | 0.55 | **pass** | 1.75 | **pass** | 1.00 | 5.25 |
| `pstudio_mini` | 402068 | 9.07 | 13.69 | 30.29 | 0.09 | fail (use CoTracker3) | 9.07 | fail (use CoTracker3) | 1.00 | 18.29 |
| `ds_mini` | 496247 | 1.05 | 2.78 | 6.03 | 0.74 | **pass** | 1.05 | **pass** | 1.00 | 4.45 |

## Median px (GT-visible) by frame gap `|t2 - t1|`

| Subset | [1, 2) | [2, 4) | [4, 8) | [8, 16) | [16, 32) | [32, 48) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 1.21 | 1.47 | 1.53 | 1.52 | 1.57 | 1.73 |
| `po_mini` | 1.17 | 1.33 | 1.56 | 1.80 | 2.05 | 2.26 |
| `pstudio_mini` | 3.98 | 4.71 | 6.31 | 8.92 | 13.16 | 16.49 |
| `ds_mini` | 0.78 | 0.87 | 0.97 | 1.04 | 1.14 | 1.24 |

`selfcorr.png`: share of GT-visible self-correspondences within each error (the table above is its
text view).
