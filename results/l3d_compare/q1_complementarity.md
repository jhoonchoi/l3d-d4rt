# Q1: do the models fail on the same points?

Stage 1 point table (`docs/l3d/STAGE1_PLAN.md`, section 4.1). Unit = (clip, frame-0 track, target frame), 48-frame clips, camera-0 coordinates, upstream global median scale per model and clip. APD = mean over clips of the clip's mean unit score (the upstream protocol). Static/dynamic rows keep the clip-level scale (upstream `dyn` refits it). 95% CIs: paired bootstrap over clips.

## OpenD4RT vs flow3r

### Accuracy and error correlation

| Subset | Points | Units | Clips | APD OpenD4RT | APD flow3r | Δ (OpenD4RT − flow3r) | 95% CI | Real? | ρ per clip (median) | ρ pooled | Both in worst 20% (indep. = 0.04) | Jaccard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `adt_mini` | all | 1064976 | 49 | 0.734 | 0.806 | -0.071 | [-0.124, -0.019] | real | 0.49 | 0.64 | 0.093 | 0.30 |
| `adt_mini` | static | 832992 | 47 | 0.712 | 0.779 | -0.067 | [-0.136, 0.001] | no | 0.48 | 0.63 | 0.086 | 0.27 |
| `adt_mini` | dynamic | 231984 | 21 | 0.645 | 0.654 | -0.009 | [-0.115, 0.098] | no | 0.47 | 0.52 | 0.073 | 0.22 |
| `po_mini` | all | 2566464 | 50 | 0.690 | 0.399 | 0.291 | [0.218, 0.365] | real | 0.25 | 0.18 | 0.054 | 0.15 |
| `po_mini` | static | 1338288 | 50 | 0.595 | 0.374 | 0.221 | [0.145, 0.301] | real | 0.31 | 0.26 | 0.063 | 0.19 |
| `po_mini` | dynamic | 1228176 | 46 | 0.753 | 0.334 | 0.420 | [0.361, 0.478] | real | 0.15 | 0.19 | 0.050 | 0.14 |
| `pstudio_mini` | all | 418560 | 50 | 0.811 | 0.412 | 0.399 | [0.353, 0.445] | real | 0.18 | 0.23 | 0.032 | 0.09 |
| `pstudio_mini` | static | 19536 | 11 | 0.381 | 0.491 | -0.109 | [-0.234, 0.017] | no | 0.07 | 0.27 | 0.083 | 0.26 |
| `pstudio_mini` | dynamic | 399024 | 50 | 0.819 | 0.403 | 0.416 | [0.370, 0.462] | real | 0.23 | 0.25 | 0.036 | 0.10 |
| `ds_mini` | all | 2518176 | 50 | 0.726 | 0.637 | 0.089 | [0.046, 0.135] | real | 0.39 | 0.34 | 0.085 | 0.27 |
| `ds_mini` | static | 1882560 | 50 | 0.723 | 0.667 | 0.056 | [0.010, 0.105] | real | 0.44 | 0.34 | 0.082 | 0.26 |
| `ds_mini` | dynamic | 635616 | 48 | 0.744 | 0.522 | 0.222 | [0.176, 0.269] | real | 0.32 | 0.42 | 0.101 | 0.34 |

### Oracle headroom over the better single model

Gain = oracle APD − APD of the better model in that row. **Track (CV)** chooses the model per track on even frames and scores it on odd frames (and the reverse), so it does not profit from noise; it is the H1 measure. **Unit** is an upper bound only.

| Subset | Points | Better model | Clip | Track | Track (CV) | 95% CI (CV) | Unit |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `adt_mini` | all | flow3r | 0.041 | 0.063 | **0.063** | [0.038, 0.092] | 0.073 |
| `adt_mini` | static | flow3r | 0.059 | 0.077 | **0.077** | [0.041, 0.120] | 0.084 |
| `adt_mini` | dynamic | flow3r | 0.096 | 0.103 | **0.103** | [0.048, 0.170] | 0.122 |
| `po_mini` | all | OpenD4RT | 0.020 | 0.047 | **0.047** | [0.028, 0.067] | 0.054 |
| `po_mini` | static | OpenD4RT | 0.041 | 0.062 | **0.062** | [0.039, 0.087] | 0.068 |
| `po_mini` | dynamic | OpenD4RT | 0.000 | 0.011 | **0.011** | [0.006, 0.017] | 0.019 |
| `pstudio_mini` | all | OpenD4RT | 0.000 | 0.007 | **0.007** | [0.003, 0.012] | 0.011 |
| `pstudio_mini` | static | flow3r | 0.040 | 0.070 | **0.070** | [0.030, 0.118] | 0.077 |
| `pstudio_mini` | dynamic | OpenD4RT | 0.000 | 0.001 | **0.001** | [0.001, 0.002] | 0.006 |
| `ds_mini` | all | OpenD4RT | 0.025 | 0.073 | **0.073** | [0.056, 0.092] | 0.078 |
| `ds_mini` | static | OpenD4RT | 0.039 | 0.085 | **0.085** | [0.067, 0.104] | 0.086 |
| `ds_mini` | dynamic | OpenD4RT | 0.008 | 0.028 | **0.028** | [0.017, 0.042] | 0.039 |

### Who gets a unit right (error ≤ 0.1, score 1)

| Subset | Points | Only OpenD4RT | Only flow3r | Both | Neither |
| --- | --- | ---: | ---: | ---: | ---: |
| `adt_mini` | all | 0.059 | 0.205 | 0.422 | 0.314 |
| `adt_mini` | static | 0.052 | 0.201 | 0.494 | 0.253 |
| `adt_mini` | dynamic | 0.086 | 0.219 | 0.163 | 0.532 |
| `po_mini` | all | 0.156 | 0.068 | 0.027 | 0.749 |
| `po_mini` | static | 0.093 | 0.109 | 0.034 | 0.764 |
| `po_mini` | dynamic | 0.225 | 0.022 | 0.020 | 0.732 |
| `pstudio_mini` | all | 0.402 | 0.018 | 0.057 | 0.524 |
| `pstudio_mini` | static | 0.102 | 0.002 | 0.006 | 0.890 |
| `pstudio_mini` | dynamic | 0.416 | 0.019 | 0.059 | 0.506 |
| `ds_mini` | all | 0.206 | 0.128 | 0.130 | 0.536 |
| `ds_mini` | static | 0.181 | 0.149 | 0.135 | 0.534 |
| `ds_mini` | dynamic | 0.281 | 0.063 | 0.114 | 0.542 |

## H1 check (stated in advance)

H1: on dynamic points, Spearman ρ < 0.5 and the cross-validated per-track oracle gains ≥ 0.03 APD on `po_mini` and `pstudio_mini`.

| Subset | ρ per clip (median) | ρ pooled | CV track gain | 95% CI | ρ < 0.5 | gain ≥ 0.03 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `po_mini` | 0.15 | 0.19 | 0.011 | [0.006, 0.017] | yes | no |
| `pstudio_mini` | 0.23 | 0.25 | 0.001 | [0.001, 0.002] | yes | no |
