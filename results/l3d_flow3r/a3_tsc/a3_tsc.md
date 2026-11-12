# A3: T/S/C split of the real error (OpenD4RT 48CLIP, design 1, clip [0, 48))

`scripts/l3d/run_a3_tsc.py --dump output/l3d/dumps/flow3r_T48 --out results/l3d_flow3r/a3_tsc` (133 s). Per `(track, t_t)` cell, design 1 decodes 6 source frames x 6
camera frames (`t_t` and 0 always among the cameras); the balanced two-way split gives the part shared by
every query of the cell (T), the source and camera main effects (S, C) and their interaction (SC). Shares
are sums over complete cells of all clips. Disagreement = 1 - T = the part that differs between queries of
the same point.

## 3D error (camera t_t, / depth)

| Subset | Cells | T | S | C | SC | Disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 101104 | 0.998 | 0.002 | 0.000 | 0.000 | 0.002 |
| `po_mini` | 102400 | 0.996 | 0.004 | 0.000 | 0.000 | 0.004 |
| `pstudio_mini` | 85536 | 0.648 | 0.352 | 0.000 | 0.000 | 0.352 |
| `ds_mini` | 102400 | 0.771 | 0.229 | 0.000 | 0.000 | 0.229 |

## depth (log ratio along ray)

| Subset | Cells | T | S | C | SC | Disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 100255 | 0.777 | 0.160 | 0.056 | 0.007 | 0.223 |
| `po_mini` | 102395 | 0.821 | 0.176 | 0.003 | 0.000 | 0.179 |
| `pstudio_mini` | 85536 | 0.630 | 0.370 | 0.000 | 0.000 | 0.370 |
| `ds_mini` | 102400 | 0.841 | 0.158 | 0.000 | 0.000 | 0.159 |

## image-plane (perpendicular to ray)

| Subset | Cells | T | S | C | SC | Disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 101104 | 0.491 | 0.192 | 0.268 | 0.048 | 0.509 |
| `po_mini` | 102400 | 0.860 | 0.056 | 0.066 | 0.018 | 0.140 |
| `pstudio_mini` | 85536 | 0.731 | 0.268 | 0.001 | 0.000 | 0.269 |
| `ds_mini` | 102400 | 0.727 | 0.265 | 0.006 | 0.002 | 0.273 |

## S share vs `|t_s - t_t|` (depth (log ratio along ray))

| Subset | [0, 1) | [1, 2) | [2, 4) | [4, 8) | [8, 16) | [16, 32) | [32, 64) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.489 | 0.249 | 0.256 | 0.247 | 0.213 | 0.152 | 0.073 |
| `po_mini` | 1.161 | 0.422 | 0.335 | 0.256 | 0.185 | 0.115 | 0.127 |
| `pstudio_mini` | 5.837 | 0.604 | 0.510 | 0.434 | 0.335 | 0.260 | 0.224 |
| `ds_mini` | 0.679 | 0.232 | 0.209 | 0.194 | 0.155 | 0.115 | 0.099 |

## S share vs `|t_s - t_t|` (3D error (camera t_t, / depth))

| Subset | [0, 1) | [1, 2) | [2, 4) | [4, 8) | [8, 16) | [16, 32) | [32, 64) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 1.748 | 0.609 | 0.429 | 0.316 | 0.215 | 0.002 | 0.002 |
| `po_mini` | 2.432 | 0.459 | 0.340 | 0.227 | 0.005 | 0.002 | 0.007 |
| `pstudio_mini` | 10.657 | 0.652 | 0.520 | 0.433 | 0.324 | 0.247 | 0.215 |
| `ds_mini` | 1.152 | 0.301 | 0.298 | 0.286 | 0.230 | 0.168 | 0.148 |
