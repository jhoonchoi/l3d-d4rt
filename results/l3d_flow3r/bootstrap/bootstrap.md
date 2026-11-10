# Bootstrap 95% CIs of the gate numbers (OpenD4RT 48CLIP, clip [0, 48))

`scripts/l3d/run_bootstrap.py --dump output/l3d/dumps/flow3r_T48 --results results/l3d_flow3r` (547 s, 1000 resamples of each subset's clips, one per sequence). Point = full-data value of the
gate tables (bootstrap row statistics use a per-clip row subsample). Verdict: the whole CI on one side of the
threshold (pass = `>=` threshold, except Gate 3: `<=` 2 px). RQ1 rows: >= 20% disagreement, >= 50% offset.

| Subset | Statistic | Threshold | Point | 95% CI | Verdict |
| --- | --- | ---: | ---: | --- | --- |
| `adt_mini` | source_swap (ii) rho | >= 0.3 | 0.655 | [0.592, 0.704] | robust pass |
| `adt_mini` | camera_swap (ii) rho | >= 0.3 | 0.691 | [0.637, 0.754] | robust pass |
| `adt_mini` | ray_center (ii) rho | >= 0.3 | -0.006 | [-0.087, 0.147] | robust fail |
| `adt_mini` | source_swap (iii) S+SC share of depth | >= 0.2 | 0.167 | [0.086, 0.264] | **uncertain** |
| `adt_mini` | camera_swap (iii) non-rigid share | >= 0.2 | 0.000 | [0.000, 0.000] | robust fail |
| `adt_mini` | ray_center (iii) image-plane share | >= 0.2 | 0.263 | [0.194, 0.373] | **uncertain** |
| `adt_mini` | source_swap (iv) median ratio | >= 0.2 | 0.405 | [0.296, 0.513] | robust pass |
| `adt_mini` | camera_swap (iv) median ratio | >= 0.2 | 0.000 | [0.000, 0.000] | robust fail |
| `adt_mini` | ray_center (iv) median ratio | >= 0.2 | 0.296 | [0.184, 0.427] | **uncertain** |
| `adt_mini` | Gate 3 median px | <= 2 | 1.538 | [1.363, 1.875] | robust pass |
| `adt_mini` | RQ1 disagreement (1 - T), 3D | >= 0.2 | 0.002 | [0.002, 0.049] | robust fail |
| `adt_mini` | RQ1 depth offset share | >= 0.5 | 0.348 | [0.228, 0.459] | robust fail |
| `adt_mini` | GAUGE 5% EPE gain %, all |  | 34.189 | [20.322, 47.666] |  |
| `adt_mini` | GAUGE 5% EPE gain %, dyn |  | 32.892 | [16.821, 51.181] |  |
| `po_mini` | source_swap (ii) rho | >= 0.3 | 0.582 | [0.548, 0.620] | robust pass |
| `po_mini` | camera_swap (ii) rho | >= 0.3 | 0.624 | [0.577, 0.649] | robust pass |
| `po_mini` | ray_center (ii) rho | >= 0.3 | 0.083 | [0.015, 0.110] | robust fail |
| `po_mini` | source_swap (iii) S+SC share of depth | >= 0.2 | 0.176 | [0.106, 0.269] | **uncertain** |
| `po_mini` | camera_swap (iii) non-rigid share | >= 0.2 | 0.000 | [0.000, 0.000] | robust fail |
| `po_mini` | ray_center (iii) image-plane share | >= 0.2 | 0.066 | [0.023, 0.144] | robust fail |
| `po_mini` | source_swap (iv) median ratio | >= 0.2 | 0.099 | [0.081, 0.147] | robust fail |
| `po_mini` | camera_swap (iv) median ratio | >= 0.2 | 0.000 | [0.000, 0.000] | robust fail |
| `po_mini` | ray_center (iv) median ratio | >= 0.2 | 0.248 | [0.181, 0.348] | **uncertain** |
| `po_mini` | Gate 3 median px | <= 2 | 1.747 | [1.520, 2.115] | **uncertain** |
| `po_mini` | RQ1 disagreement (1 - T), 3D | >= 0.2 | 0.004 | [0.002, 0.310] | **uncertain** |
| `po_mini` | RQ1 depth offset share | >= 0.5 | 0.277 | [0.232, 0.327] | robust fail |
| `po_mini` | GAUGE 5% EPE gain %, all |  | 34.784 | [28.304, 40.063] |  |
| `po_mini` | GAUGE 5% EPE gain %, dyn |  | 19.782 | [15.082, 24.563] |  |
| `pstudio_mini` | source_swap (ii) rho | >= 0.3 | 0.351 | [0.307, 0.402] | robust pass |
| `pstudio_mini` | camera_swap (ii) rho | >= 0.3 | 0.573 | [0.505, 0.600] | robust pass |
| `pstudio_mini` | ray_center (ii) rho | >= 0.3 | 0.258 | [0.122, 0.328] | **uncertain** |
| `pstudio_mini` | source_swap (iii) S+SC share of depth | >= 0.2 | 0.370 | [0.333, 0.404] | robust pass |
| `pstudio_mini` | camera_swap (iii) non-rigid share | >= 0.2 | 0.000 | [0.000, 0.000] | robust fail |
| `pstudio_mini` | ray_center (iii) image-plane share | >= 0.2 | 0.114 | [0.095, 0.142] | robust fail |
| `pstudio_mini` | source_swap (iv) median ratio | >= 0.2 | 0.340 | [0.279, 0.384] | robust pass |
| `pstudio_mini` | camera_swap (iv) median ratio | >= 0.2 | 0.000 | [0.000, 0.000] | robust fail |
| `pstudio_mini` | ray_center (iv) median ratio | >= 0.2 | 0.170 | [0.149, 0.190] | robust fail |
| `pstudio_mini` | Gate 3 median px | <= 2 | 9.073 | [8.054, 10.375] | robust fail |
| `pstudio_mini` | RQ1 disagreement (1 - T), 3D | >= 0.2 | 0.352 | [0.315, 0.388] | robust pass |
| `pstudio_mini` | RQ1 depth offset share | >= 0.5 | 0.268 | [0.221, 0.312] | robust fail |
| `pstudio_mini` | GAUGE 5% EPE gain %, all |  | 31.774 | [26.353, 36.684] |  |
| `pstudio_mini` | GAUGE 5% EPE gain %, dyn |  | 31.774 | [26.353, 36.684] |  |
| `ds_mini` | source_swap (ii) rho | >= 0.3 | 0.598 | [0.561, 0.626] | robust pass |
| `ds_mini` | camera_swap (ii) rho | >= 0.3 | 0.695 | [0.660, 0.723] | robust pass |
| `ds_mini` | ray_center (ii) rho | >= 0.3 | 0.031 | [-0.025, 0.109] | robust fail |
| `ds_mini` | source_swap (iii) S+SC share of depth | >= 0.2 | 0.158 | [0.134, 0.187] | robust fail |
| `ds_mini` | camera_swap (iii) non-rigid share | >= 0.2 | 0.000 | [0.000, 0.000] | robust fail |
| `ds_mini` | ray_center (iii) image-plane share | >= 0.2 | 0.146 | [0.083, 0.233] | **uncertain** |
| `ds_mini` | source_swap (iv) median ratio | >= 0.2 | 0.278 | [0.235, 0.327] | robust pass |
| `ds_mini` | camera_swap (iv) median ratio | >= 0.2 | 0.001 | [0.000, 0.001] | robust fail |
| `ds_mini` | ray_center (iv) median ratio | >= 0.2 | 1.027 | [0.886, 1.212] | robust pass |
| `ds_mini` | Gate 3 median px | <= 2 | 1.053 | [0.981, 1.136] | robust pass |
| `ds_mini` | RQ1 disagreement (1 - T), 3D | >= 0.2 | 0.229 | [0.201, 0.261] | robust pass |
| `ds_mini` | RQ1 depth offset share | >= 0.5 | 0.222 | [0.163, 0.283] | robust fail |
| `ds_mini` | GAUGE 5% EPE gain %, all |  | 11.231 | [-11.821, 27.076] |  |
| `ds_mini` | GAUGE 5% EPE gain %, dyn |  | 23.961 | [17.076, 30.365] |  |
