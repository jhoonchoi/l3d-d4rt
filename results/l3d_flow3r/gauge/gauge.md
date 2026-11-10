# GAUGE baseline (GT anchors; a reference, not label-free), OpenD4RT 48CLIP, clip [0, 48)

`scripts/l3d/run_gauge.py --dump output/l3d/dumps/flow3r_T48 --out results/l3d_flow3r/gauge` (279 s). World-frame outputs (design 0, `t_c = 0`) scaled by the clip's global
median scale; GAUGE (`l3d/gauge.py`: per-group per-frame ray scale about the model's camera centers + median
translation) and a per-group Sim(3) from the same anchors. Groups = GT motion groups (the paper's oracle
variant); anchors = whole GT tracks. Metrics on non-anchor points, upstream APD thresholds, no re-alignment,
mean over clips. Dynamic = total GT motion > 0.01 (upstream rule).

| Subset | Anchors | Points | APD base | APD GAUGE | APD Sim(3)/group | EPE base | EPE GAUGE | EPE Sim(3)/group | EPE gain GAUGE | Paper EPE gain (DQS / Full) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `adt_mini` | 1% (4/clip) | all | 0.780 | 0.800 | 0.684 | 0.2898 | 0.2901 | 0.7619 | -0.1% | +24.7% / +29.5% |
| `adt_mini` | 1% (4/clip) | dyn | 0.618 | 0.676 | 0.653 | 0.4441 | 0.4050 | 0.3968 | 8.8% | +24.7% / +29.5% |
| `adt_mini` | 5% (22/clip) | all | 0.781 | 0.868 | 0.854 | 0.2890 | 0.1902 | 0.2144 | 34.2% | +24.7% / +29.5% |
| `adt_mini` | 5% (22/clip) | dyn | 0.619 | 0.747 | 0.752 | 0.4438 | 0.2978 | 0.2600 | 32.9% | +24.7% / +29.5% |
| `po_mini` | 1% (10/clip) | all | 0.413 | 0.565 | 0.443 | 0.9735 | 0.7605 | 0.8959 | 21.9% | +43.4% / +18.2% |
| `po_mini` | 1% (10/clip) | dyn | 0.361 | 0.446 | 0.370 | 1.0615 | 0.9424 | 0.8470 | 11.2% | +43.4% / +18.2% |
| `po_mini` | 5% (53/clip) | all | 0.413 | 0.637 | 0.573 | 0.9729 | 0.6345 | 0.5575 | 34.8% | +43.4% / +18.2% |
| `po_mini` | 5% (53/clip) | dyn | 0.362 | 0.508 | 0.418 | 1.0599 | 0.8503 | 0.6852 | 19.8% | +43.4% / +18.2% |
| `pstudio_mini` | 1% (1/clip) | all | 0.425 | 0.453 | 0.484 | 0.9062 | 0.7948 | 0.6595 | 12.3% | +51.8% / +25.6% |
| `pstudio_mini` | 1% (1/clip) | dyn | 0.425 | 0.453 | 0.484 | 0.9062 | 0.7948 | 0.6595 | 12.3% | +51.8% / +25.6% |
| `pstudio_mini` | 5% (8/clip) | all | 0.426 | 0.546 | 0.622 | 0.9035 | 0.6164 | 0.3430 | 31.8% | +51.8% / +25.6% |
| `pstudio_mini` | 5% (8/clip) | dyn | 0.426 | 0.546 | 0.622 | 0.9035 | 0.6164 | 0.3430 | 31.8% | +51.8% / +25.6% |
| `ds_mini` | 1% (10/clip) | all | 0.643 | 0.690 | 0.630 | 0.4075 | 0.3581 | 0.4607 | 12.1% | +31.9% / -8.3% |
| `ds_mini` | 1% (10/clip) | dyn | 0.544 | 0.544 | 0.502 | 0.6050 | 0.5327 | 0.5961 | 12.0% | +31.9% / -8.3% |
| `ds_mini` | 5% (52/clip) | all | 0.643 | 0.709 | 0.711 | 0.4067 | 0.3610 | 0.3023 | 11.2% | +31.9% / -8.3% |
| `ds_mini` | 5% (52/clip) | dyn | 0.544 | 0.608 | 0.602 | 0.6051 | 0.4601 | 0.3747 | 24.0% | +31.9% / -8.3% |

Paper column: GAUGE Table A1 for D4RT (their unsupervised groups, 5% anchors, full sequences); for
orientation only.
