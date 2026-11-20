# RQ2: check residuals combined with the native confidence (OpenD4RT 48CLIP, clip [0, 48))

`scripts/l3d/run_a3_combined.py --workers 3` (3688 s). Up to 5000 rows per clip and check (seeded sample of the A3 rows). Positives: the top 20% of errors `|s P - X| / |X|` of the subset.

- **Confidence (logit)** is the native confidence without the sigmoid. A3 scored `sigmoid(conf)` stored as float32; for this model that is within 1e-5 of 1, so many rows tie (column *A3 ties*).
- **Rank average**: mean rank of residual and confidence (no fit). **Logistic (CV)**: fitted on the other half of the clips. Delta AUROC is against the logit confidence; 95% CI over clips.
- **Partial rho**: Spearman rho of residual and error after regressing out the confidence ranks (0: the residual adds nothing the confidence has not).

## AUROC (top 20% errors)

| Subset | Check | Rows | conf (logit) | conf (A3) | A3 ties | residual | rank avg | logistic (CV) | delta logistic - conf, 95% CI | delta A3 conf - logit conf, 95% CI | partial rho (error) | partial rho (target) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `adt_mini` | source swap, model u2 | 250000 | 0.507 | 0.523 | 0.98 | 0.820 | 0.706 | 0.829 | +0.162 to +0.426 | +0.011 to +0.022 | 0.440 | 0.588 |
| `adt_mini` | source swap, GT u2 (control) | 250000 | 0.509 | 0.519 | 0.98 | 0.782 | 0.685 | 0.795 | +0.129 to +0.394 | +0.006 to +0.015 | 0.402 | 0.764 |
| `adt_mini` | camera swap, SE(3) | 250000 | 0.500 | 0.500 | 0.98 | 0.814 | 0.699 | 0.829 | +0.173 to +0.444 | -0.000 to +0.000 | 0.460 | 0.899 |
| `adt_mini` | ray center, K fit + R* | 243415 | 0.631 | 0.631 | 0.97 | 0.713 | 0.706 | 0.694 | -0.037 to +0.156 | -0.000 to +0.000 | 0.332 | 0.196 |
| `adt_mini` | ray center, GT K and pose (control) | 243415 | 0.630 | 0.630 | 0.97 | 0.664 | 0.711 | 0.713 | -0.015 to +0.187 | -0.000 to +0.000 | 0.492 | 0.504 |
| `adt_mini` | reprojection, K fit | 80906 | 0.552 | 0.552 | 0.94 | 0.724 | 0.659 | 0.704 | +0.062 to +0.229 | -0.000 to +0.000 | 0.430 | 0.394 |
| `adt_mini` | static rigidity (GT static mask) | 232784 | 0.421 | 0.421 | 0.97 | 0.611 | 0.519 | 0.605 | +0.018 to +0.290 | -0.000 to +0.000 | 0.231 | 0.235 |
| `po_mini` | source swap, model u2 | 250000 | 0.564 | 0.581 | 0.99 | 0.756 | 0.715 | 0.770 | +0.159 to +0.254 | +0.014 to +0.021 | 0.349 | 0.497 |
| `po_mini` | source swap, GT u2 (control) | 250000 | 0.585 | 0.595 | 0.99 | 0.679 | 0.682 | 0.704 | +0.068 to +0.164 | +0.006 to +0.014 | 0.296 | 0.807 |
| `po_mini` | camera swap, SE(3) | 250000 | 0.561 | 0.561 | 0.99 | 0.612 | 0.615 | 0.620 | +0.016 to +0.106 | -0.000 to +0.000 | 0.184 | 0.913 |
| `po_mini` | ray center, K fit + R* | 230502 | 0.670 | 0.670 | 0.99 | 0.578 | 0.657 | 0.675 | -0.002 to +0.014 | -0.000 to +0.000 | 0.125 | 0.032 |
| `po_mini` | ray center, GT K and pose (control) | 230502 | 0.671 | 0.671 | 0.99 | 0.685 | 0.729 | 0.739 | +0.037 to +0.099 | -0.000 to +0.000 | 0.377 | 0.449 |
| `po_mini` | reprojection, K fit | 61622 | 0.643 | 0.643 | 0.98 | 0.559 | 0.630 | 0.649 | -0.005 to +0.016 | -0.000 to +0.000 | 0.117 | 0.162 |
| `po_mini` | static rigidity (GT static mask) | 250000 | 0.533 | 0.533 | 0.99 | 0.645 | 0.624 | 0.654 | +0.038 to +0.207 | -0.000 to +0.000 | 0.191 | 0.165 |
| `pstudio_mini` | source swap, model u2 | 250000 | 0.574 | 0.588 | 0.99 | 0.713 | 0.684 | 0.710 | +0.082 to +0.216 | +0.010 to +0.019 | 0.336 | 0.496 |
| `pstudio_mini` | source swap, GT u2 (control) | 250000 | 0.553 | 0.564 | 1.00 | 0.628 | 0.621 | 0.628 | +0.019 to +0.148 | +0.005 to +0.015 | 0.230 | 0.788 |
| `pstudio_mini` | camera swap, SE(3) | 243078 | 0.526 | 0.526 | 0.99 | 0.656 | 0.625 | 0.650 | +0.074 to +0.193 | -0.000 to +0.000 | 0.231 | 0.792 |
| `pstudio_mini` | ray center, K fit + R* | 225151 | 0.701 | 0.701 | 0.99 | 0.596 | 0.684 | 0.700 | -0.012 to +0.013 | +0.000 to +0.000 | 0.098 | 0.051 |
| `pstudio_mini` | ray center, GT K and pose (control) | 225151 | 0.702 | 0.702 | 0.99 | 0.676 | 0.730 | 0.732 | +0.000 to +0.070 | -0.000 to +0.000 | 0.215 | 0.180 |
| `pstudio_mini` | reprojection, K fit | 64371 | 0.708 | 0.708 | 0.98 | 0.585 | 0.681 | 0.705 | -0.019 to +0.015 | -0.000 to +0.000 | 0.068 | 0.024 |
| `pstudio_mini` | static rigidity (GT static mask) | 33152 | 0.570 | 0.570 | 0.97 | 0.528 | 0.574 | 0.525 | -0.269 to +0.248 | -0.000 to +0.000 | 0.000 | -0.015 |
| `ds_mini` | source swap, model u2 | 250000 | 0.645 | 0.651 | 0.99 | 0.717 | 0.723 | 0.743 | +0.076 to +0.122 | +0.005 to +0.008 | 0.300 | 0.501 |
| `ds_mini` | source swap, GT u2 (control) | 250000 | 0.646 | 0.649 | 0.99 | 0.644 | 0.684 | 0.690 | +0.024 to +0.066 | +0.002 to +0.004 | 0.231 | 0.851 |
| `ds_mini` | camera swap, SE(3) | 250000 | 0.641 | 0.641 | 0.99 | 0.635 | 0.671 | 0.670 | +0.013 to +0.043 | -0.000 to +0.000 | 0.167 | 0.904 |
| `ds_mini` | ray center, K fit + R* | 250000 | 0.668 | 0.668 | 0.99 | 0.506 | 0.615 | 0.666 | -0.004 to +0.001 | -0.000 to +0.000 | 0.006 | 0.330 |
| `ds_mini` | ray center, GT K and pose (control) | 250000 | 0.669 | 0.669 | 0.99 | 0.510 | 0.618 | 0.666 | -0.006 to -0.000 | -0.000 to +0.000 | 0.024 | 0.403 |
| `ds_mini` | reprojection, K fit | 90120 | 0.669 | 0.669 | 0.97 | 0.544 | 0.633 | 0.668 | -0.003 to +0.002 | -0.000 to +0.000 | 0.027 | 0.607 |
| `ds_mini` | static rigidity (GT static mask) | 250000 | 0.676 | 0.676 | 0.99 | 0.686 | 0.724 | 0.739 | +0.043 to +0.081 | -0.000 to -0.000 | 0.195 | 0.189 |

## E-AURC (excess area under risk-coverage; lower is better)

| Subset | Check | confidence (logit) | confidence (A3: float32 sigmoid) | residual | rank average | logistic (CV) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | source swap, model u2 | 0.1630 | 0.1529 | 0.0523 | 0.0821 | 0.0519 |
| `adt_mini` | source swap, GT u2 (control) | 0.1343 | 0.1329 | 0.0558 | 0.0787 | 0.0544 |
| `adt_mini` | camera swap, SE(3) | 0.1435 | 0.1435 | 0.0493 | 0.0798 | 0.0460 |
| `adt_mini` | ray center, K fit + R* | 0.0765 | 0.0765 | 0.0499 | 0.0530 | 0.0551 |
| `adt_mini` | ray center, GT K and pose (control) | 0.0765 | 0.0765 | 0.0579 | 0.0450 | 0.0518 |
| `adt_mini` | reprojection, K fit | 0.0618 | 0.0618 | 0.0363 | 0.0423 | 0.0430 |
| `adt_mini` | static rigidity (GT static mask) | 0.1249 | 0.1249 | 0.0673 | 0.0885 | 0.0675 |
| `po_mini` | source swap, model u2 | 0.0522 | 0.0497 | 0.0260 | 0.0298 | 0.0248 |
| `po_mini` | source swap, GT u2 (control) | 0.0344 | 0.0336 | 0.0246 | 0.0240 | 0.0224 |
| `po_mini` | camera swap, SE(3) | 0.0348 | 0.0348 | 0.0297 | 0.0290 | 0.0287 |
| `po_mini` | ray center, K fit + R* | 0.0214 | 0.0213 | 0.0301 | 0.0217 | 0.0205 |
| `po_mini` | ray center, GT K and pose (control) | 0.0212 | 0.0212 | 0.0251 | 0.0178 | 0.0170 |
| `po_mini` | reprojection, K fit | 0.0216 | 0.0215 | 0.0324 | 0.0227 | 0.0210 |
| `po_mini` | static rigidity (GT static mask) | 0.0463 | 0.0463 | 0.0315 | 0.0329 | 0.0303 |
| `pstudio_mini` | source swap, model u2 | 0.0308 | 0.0295 | 0.0190 | 0.0198 | 0.0187 |
| `pstudio_mini` | source swap, GT u2 (control) | 0.0238 | 0.0235 | 0.0204 | 0.0193 | 0.0200 |
| `pstudio_mini` | camera swap, SE(3) | 0.0248 | 0.0249 | 0.0186 | 0.0194 | 0.0194 |
| `pstudio_mini` | ray center, K fit + R* | 0.0135 | 0.0135 | 0.0177 | 0.0136 | 0.0133 |
| `pstudio_mini` | ray center, GT K and pose (control) | 0.0135 | 0.0135 | 0.0146 | 0.0116 | 0.0117 |
| `pstudio_mini` | reprojection, K fit | 0.0123 | 0.0123 | 0.0180 | 0.0128 | 0.0122 |
| `pstudio_mini` | static rigidity (GT static mask) | 0.0978 | 0.0977 | 0.0885 | 0.0866 | 0.0803 |
| `ds_mini` | source swap, model u2 | 0.0343 | 0.0333 | 0.0281 | 0.0238 | 0.0238 |
| `ds_mini` | source swap, GT u2 (control) | 0.0287 | 0.0283 | 0.0291 | 0.0229 | 0.0236 |
| `ds_mini` | camera swap, SE(3) | 0.0282 | 0.0282 | 0.0296 | 0.0238 | 0.0246 |
| `ds_mini` | ray center, K fit + R* | 0.0215 | 0.0215 | 0.0391 | 0.0261 | 0.0217 |
| `ds_mini` | ray center, GT K and pose (control) | 0.0214 | 0.0214 | 0.0385 | 0.0258 | 0.0218 |
| `ds_mini` | reprojection, K fit | 0.0215 | 0.0215 | 0.0375 | 0.0249 | 0.0216 |
| `ds_mini` | static rigidity (GT static mask) | 0.0321 | 0.0321 | 0.0293 | 0.0234 | 0.0223 |

## AUROC against the check's target component

| Subset | Check | confidence (logit) | confidence (A3: float32 sigmoid) | residual | rank average | logistic (CV) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | source swap, model u2 | 0.545 | 0.566 | 0.890 | 0.776 | 0.874 |
| `adt_mini` | source swap, GT u2 (control) | 0.555 | 0.571 | 0.960 | 0.830 | 0.946 |
| `adt_mini` | camera swap, SE(3) | 0.577 | 0.577 | 0.960 | 0.843 | 0.944 |
| `adt_mini` | ray center, K fit + R* | 0.382 | 0.382 | 0.556 | 0.467 | 0.535 |
| `adt_mini` | ray center, GT K and pose (control) | 0.381 | 0.381 | 0.768 | 0.598 | 0.656 |
| `adt_mini` | reprojection, K fit | 0.373 | 0.373 | 0.669 | 0.525 | 0.679 |
| `adt_mini` | static rigidity (GT static mask) | 0.537 | 0.537 | 0.640 | 0.611 | 0.598 |
| `po_mini` | source swap, model u2 | 0.483 | 0.508 | 0.809 | 0.699 | 0.787 |
| `po_mini` | source swap, GT u2 (control) | 0.487 | 0.511 | 0.971 | 0.823 | 0.929 |
| `po_mini` | camera swap, SE(3) | 0.508 | 0.508 | 0.971 | 0.830 | 0.921 |
| `po_mini` | ray center, K fit + R* | 0.544 | 0.544 | 0.510 | 0.534 | 0.546 |
| `po_mini` | ray center, GT K and pose (control) | 0.544 | 0.544 | 0.733 | 0.679 | 0.687 |
| `po_mini` | reprojection, K fit | 0.542 | 0.542 | 0.572 | 0.573 | 0.556 |
| `po_mini` | static rigidity (GT static mask) | 0.532 | 0.532 | 0.635 | 0.615 | 0.645 |
| `pstudio_mini` | source swap, model u2 | 0.506 | 0.527 | 0.800 | 0.706 | 0.793 |
| `pstudio_mini` | source swap, GT u2 (control) | 0.502 | 0.524 | 0.958 | 0.813 | 0.945 |
| `pstudio_mini` | camera swap, SE(3) | 0.499 | 0.499 | 0.924 | 0.796 | 0.917 |
| `pstudio_mini` | ray center, K fit + R* | 0.649 | 0.649 | 0.507 | 0.597 | 0.642 |
| `pstudio_mini` | ray center, GT K and pose (control) | 0.649 | 0.649 | 0.613 | 0.660 | 0.663 |
| `pstudio_mini` | reprojection, K fit | 0.651 | 0.651 | 0.530 | 0.613 | 0.651 |
| `pstudio_mini` | static rigidity (GT static mask) | 0.568 | 0.568 | 0.530 | 0.574 | 0.522 |
| `ds_mini` | source swap, model u2 | 0.568 | 0.579 | 0.832 | 0.753 | 0.809 |
| `ds_mini` | source swap, GT u2 (control) | 0.576 | 0.586 | 0.983 | 0.862 | 0.913 |
| `ds_mini` | camera swap, SE(3) | 0.572 | 0.572 | 0.973 | 0.842 | 0.861 |
| `ds_mini` | ray center, K fit + R* | 0.605 | 0.605 | 0.715 | 0.714 | 0.590 |
| `ds_mini` | ray center, GT K and pose (control) | 0.605 | 0.605 | 0.752 | 0.731 | 0.590 |
| `ds_mini` | reprojection, K fit | 0.603 | 0.603 | 0.851 | 0.790 | 0.624 |
| `ds_mini` | static rigidity (GT static mask) | 0.677 | 0.677 | 0.685 | 0.723 | 0.738 |
