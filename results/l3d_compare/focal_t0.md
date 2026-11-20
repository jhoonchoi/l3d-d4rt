# Field-of-view error at the query frame (exploratory)

At ``t = 0`` the target camera is camera 0, so a pure depth error projects onto the GT pixel. Each model's camera-0 point is projected with the GT intrinsics. Per clip and axis, the median ratio ``(u_pred - c) / (u_gt - c)`` over units at least 20 px from the principal point estimates the ratio ``f_gt / f_model``: above 1 = the model's focal length is too short (field of view too wide). The 2D offset is the median distance between the two projections, in 256-px units. ``--dump`` gives ``K_gt`` per clip. 

Per subset: median over clips, with the 10th and 90th percentiles over clips.

| Subset | Model | Clips | Ratio x (median) | Ratio x [p10, p90] | Ratio y (median) | Ratio y [p10, p90] | 2D offset at t = 0 (px) |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| `adt_mini` | OpenD4RT | 49 | 1.038 | [0.802, 1.109] | 0.868 | [0.672, 1.001] | 6.26 |
| `adt_mini` | flow3r | 49 | 1.017 | [1.004, 1.144] | 1.011 | [1.002, 1.147] | 1.23 |
| `po_mini` | OpenD4RT | 50 | 0.976 | [0.853, 1.031] | 0.969 | [0.842, 1.048] | 2.76 |
| `po_mini` | flow3r | 50 | 0.966 | [0.919, 1.007] | 0.966 | [0.918, 1.006] | 2.33 |
| `pstudio_mini` | OpenD4RT | 50 | 0.962 | [0.866, 1.250] | 1.003 | [0.918, 1.177] | 5.59 |
| `pstudio_mini` | flow3r | 50 | 1.157 | [0.930, 1.480] | 1.180 | [1.034, 1.478] | 10.38 |
| `ds_mini` | OpenD4RT | 50 | 0.999 | [0.989, 1.005] | 1.007 | [0.997, 1.012] | 0.63 |
| `ds_mini` | flow3r | 50 | 0.969 | [0.894, 0.998] | 0.969 | [0.893, 0.994] | 1.71 |
