# A4: pose baselines from the decoded grid (OpenD4RT 48CLIP, design 2, clip [0, 48))

`scripts/l3d/run_a4_pose.py --dump output/l3d/dumps/flow3r_T48 --out results/l3d_flow3r/a4_pose` (197 s). Per grid pair `(t, c)`: a 32x32 pixel grid of frame t decoded in
cameras t and c (`c` in {0, t-4, t+4}), outputs scaled by the clip scale, `T_{t->c}` fitted by each method and
compared with the GT relative pose. Translation errors are relative to the clip's median GT depth. ATE / RPE:
trajectory `T_{t->0}` vs GT after a Sim(3) alignment, median over clips (not computed for static cameras).

Label-free K on the grid: fx(fit) / fx(GT), median over clips: `adt_mini` 0.979, `po_mini` 1.033, `pstudio_mini` 0.859, `ds_mini` 1.029.

## Relative pose per pair (median over pairs)

| Subset | Pairs | GT rot (deg) | GT trans / depth | Method | rot err (deg) | trans err / depth | beats Umeyama (trans) | ATE / depth | RPE trans / depth | RPE rot (deg) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 6850 | 3.62 | 0.0488 | Umeyama | 0.199 | 0.0064 | n/a | 0.0049 | 0.0035 | 0.090 |
| | | | | Umeyama, conf-weighted | 0.199 | 0.0064 | 0.51 | 0.0049 | 0.0035 | 0.090 |
| | | | | ray center (K fit) | 0.199 | 0.0154 | 0.14 | 0.0055 | 0.0039 | 0.090 |
| | | | | ray center (GT K) | 0.199 | 0.0460 | 0.07 | 0.0080 | 0.0056 | 0.090 |
| `po_mini` | 6850 | 1.11 | 0.0078 | Umeyama | 0.076 | 0.0028 | n/a | 0.0041 | 0.0033 | 0.069 |
| | | | | Umeyama, conf-weighted | 0.076 | 0.0028 | 0.50 | 0.0041 | 0.0033 | 0.069 |
| | | | | ray center (K fit) | 0.076 | 0.0037 | 0.39 | 0.0032 | 0.0028 | 0.069 |
| | | | | ray center (GT K) | 0.076 | 0.0455 | 0.08 | 0.0060 | 0.0049 | 0.069 |
| `pstudio_mini` | 6850 | 0.00 | 0.0000 | Umeyama | 0.135 | 0.0054 | n/a | n/a | n/a | n/a |
| | | | | Umeyama, conf-weighted | 0.135 | 0.0054 | 0.48 | n/a | n/a | n/a |
| | | | | ray center (K fit) | 0.135 | 0.0063 | 0.40 | n/a | n/a | n/a |
| | | | | ray center (GT K) | 0.135 | 0.2166 | 0.00 | n/a | n/a | n/a |
| `ds_mini` | 6850 | 0.22 | 0.0034 | Umeyama | 0.041 | 0.0008 | n/a | 0.0007 | 0.0006 | 0.026 |
| | | | | Umeyama, conf-weighted | 0.041 | 0.0008 | 0.49 | 0.0007 | 0.0006 | 0.026 |
| | | | | ray center (K fit) | 0.041 | 0.0020 | 0.12 | 0.0007 | 0.0006 | 0.026 |
| | | | | ray center (GT K) | 0.041 | 0.0245 | 0.01 | 0.0032 | 0.0024 | 0.026 |

## Label-free camera residuals vs pose error (Gate 1 (iv): median residual >= 20% of the median pose error)

| Subset | median camera-swap residual (SE(3)) | median ray-center spread | median Umeyama trans err / depth | camera swap / pose err | ray center / pose err |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.0000 | 0.0019 | 0.0064 | 0.00 | 0.30 |
| `po_mini` | 0.0000 | 0.0007 | 0.0028 | 0.00 | 0.25 |
| `pstudio_mini` | 0.0000 | 0.0009 | 0.0054 | 0.00 | 0.17 |
| `ds_mini` | 0.0000 | 0.0008 | 0.0008 | 0.00 | 1.03 |
