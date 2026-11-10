# A3: detection by each check vs native confidence (OpenD4RT 48CLIP, clip [0, 48))

`scripts/l3d/run_a3_detection.py --dump output/l3d/dumps/flow3r_T48 --out results/l3d_flow3r/a3_detection` (303 s). Rows of all clips of a subset are pooled. Error = `|s P - X| / |X|`
of the scored query (mean over the pair for source swap); target = the component Table 1 says the check
sees. Confidence is scored as `-sigmoid(conf)` on the same rows. AUROC: positives = top 20% of errors.
E-AURC: excess area under the risk-coverage curve over the oracle (lower is better). Gate 1 (ii) asks for
rho(residual, target) >= 0.3.

Label-free K: fx(fit) / fx(GT), median over clips: `adt_mini` 0.981, `po_mini` 1.034, `pstudio_mini` 0.854, `ds_mini` 1.030.

## source swap, model u2 (design 5 vs 1)

Target: S depth: |e_d(t1) - e_d(t2)|.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 3014160 | 0.655 | 0.291 | 0.529 | 0.309 | 0.797 | 0.627 | 0.0592 | 0.1318 | 0.307 | 0.405 |
| `po_mini` | 3068544 | 0.582 | 0.460 | 0.349 | 0.356 | 0.744 | 0.613 | 0.0908 | 0.0930 | 0.374 | 0.099 |
| `pstudio_mini` | 2550672 | 0.351 | 0.012 | 0.357 | -0.120 | 0.760 | 0.353 | 0.0954 | 0.1857 | 0.005 | 0.340 |
| `ds_mini` | 3072000 | 0.598 | 0.337 | 0.346 | 0.142 | 0.689 | 0.609 | 0.0441 | 0.0470 | 0.348 | 0.278 |

## source swap, GT u2 control (design 1)

Target: S depth: |e_d(t1) - e_d(t2)|.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 3014160 | 0.833 | 0.282 | 0.552 | 0.311 | 0.832 | 0.623 | 0.0526 | 0.1350 | 0.320 | 0.303 |
| `po_mini` | 3068544 | 0.916 | 0.469 | 0.494 | 0.364 | 0.799 | 0.629 | 0.0721 | 0.0927 | 0.473 | 0.084 |
| `pstudio_mini` | 2550672 | 0.923 | -0.058 | 0.607 | -0.140 | 0.847 | 0.380 | 0.0727 | 0.2095 | -0.035 | 0.496 |
| `ds_mini` | 3072000 | 0.916 | 0.329 | 0.412 | 0.116 | 0.741 | 0.603 | 0.0384 | 0.0488 | 0.326 | 0.196 |

## camera swap, SE(3) fit (design 1)

Target: non-rigid camera-frame error.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 968853 | 0.691 | 0.021 | 0.131 | 0.270 | 0.530 | 0.604 | 0.1688 | 0.1562 | 0.091 | 0.000 |
| `po_mini` | 867575 | 0.624 | -0.067 | -0.024 | 0.275 | 0.477 | 0.548 | 0.1550 | 0.1103 | -0.005 | 0.000 |
| `pstudio_mini` | 771176 | 0.573 | 0.047 | 0.256 | -0.152 | 0.584 | 0.325 | 0.1237 | 0.2529 | 0.139 | 0.000 |
| `ds_mini` | 947537 | 0.695 | -0.009 | 0.005 | 0.040 | 0.488 | 0.546 | 0.0680 | 0.0568 | 0.050 | 0.000 |

## camera swap, Sim(3) fit (design 1)

Target: non-rigid camera-frame error.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 968853 | 0.928 | 0.021 | 0.069 | 0.270 | 0.478 | 0.604 | 0.2031 | 0.1562 | 0.096 | 0.000 |
| `po_mini` | 867575 | 0.935 | -0.067 | -0.082 | 0.275 | 0.439 | 0.548 | 0.1662 | 0.1103 | -0.046 | 0.000 |
| `pstudio_mini` | 771176 | 0.942 | 0.047 | 0.193 | -0.152 | 0.536 | 0.325 | 0.1484 | 0.2529 | 0.123 | 0.000 |
| `ds_mini` | 947537 | 0.971 | -0.009 | -0.003 | 0.040 | 0.491 | 0.546 | 0.0675 | 0.0568 | 0.019 | 0.000 |

## ray center, K fit + R* (design 1, t_s = t_t)

Target: image-plane error of P(u, t, t, t).

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 400850 | -0.006 | 0.090 | 0.127 | 0.360 | 0.572 | 0.742 | 0.0543 | 0.0408 | 0.119 | 0.063 |
| `po_mini` | 295708 | 0.083 | 0.141 | 0.025 | 0.395 | 0.484 | 0.637 | 0.0664 | 0.0360 | 0.043 | 0.011 |
| `pstudio_mini` | 308609 | 0.258 | 0.063 | 0.229 | 0.023 | 0.623 | 0.456 | 0.0335 | 0.0493 | 0.232 | 0.009 |
| `ds_mini` | 450367 | 0.031 | -0.268 | 0.020 | 0.060 | 0.533 | 0.558 | 0.0382 | 0.0320 | 0.146 | 0.021 |

## ray center, GT K and pose (not label-free)

Target: image-plane error of P(u, t, t, t).

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 400850 | 0.395 | 0.090 | 0.411 | 0.360 | 0.710 | 0.742 | 0.0364 | 0.0408 | 0.225 | 0.087 |
| `po_mini` | 295708 | 0.587 | 0.141 | 0.343 | 0.395 | 0.661 | 0.637 | 0.0432 | 0.0360 | 0.161 | 0.030 |
| `pstudio_mini` | 308609 | 0.673 | 0.063 | 0.605 | 0.023 | 0.802 | 0.456 | 0.0173 | 0.0493 | 0.077 | 0.026 |
| `ds_mini` | 450367 | 0.744 | -0.268 | 0.380 | 0.060 | 0.664 | 0.558 | 0.0292 | 0.0320 | -0.126 | 0.055 |

## reprojection with K fit, px (analysis only)

Target: image-plane error.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 80906 | 0.237 | 0.093 | 0.250 | 0.341 | 0.595 | 0.711 | 0.0527 | 0.0411 | 0.013 | 42.420 |
| `po_mini` | 61622 | -0.038 | 0.130 | -0.159 | 0.394 | 0.398 | 0.634 | 0.0832 | 0.0362 | -0.141 | 14.689 |
| `pstudio_mini` | 64371 | 0.166 | 0.039 | 0.153 | 0.033 | 0.585 | 0.454 | 0.0383 | 0.0495 | 0.176 | 6.978 |
| `ds_mini` | 90120 | 0.035 | -0.268 | 0.020 | 0.059 | 0.527 | 0.558 | 0.0400 | 0.0320 | -0.061 | 34.555 |

## static rigidity, GT static mask (analysis only)

Target: depth error |e_d|.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 1084512 | 0.419 | 0.186 | 0.455 | 0.177 | 0.859 | 0.561 | 0.0358 | 0.1052 | 0.156 | 0.256 |
| `po_mini` | 1193184 | 0.209 | 0.354 | 0.283 | 0.301 | 0.743 | 0.604 | 0.0717 | 0.0887 | 0.177 | 0.049 |
| `pstudio_mini` | 40608 | 0.538 | 0.517 | 0.502 | 0.335 | 0.878 | 0.849 | 0.0255 | 0.0327 | 0.488 | 0.143 |
| `ds_mini` | 1227648 | 0.259 | 0.097 | 0.256 | 0.031 | 0.680 | 0.543 | 0.0295 | 0.0410 | 0.389 | 0.069 |
