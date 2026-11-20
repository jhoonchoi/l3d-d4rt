# Robust T* fits vs plain Umeyama (OpenD4RT 48CLIP, clip [0, 48))

`scripts/l3d/run_a4_robust.py --workers 2` (2026 s; code `l3d/pose_robust.py`). Per camera pair `(t, c)`, `T*: t -> c` (SE(3)) from the same outputs, scaled by the clip scale; errors against the GT relative pose, translation over the clip's median GT depth. RQ3 predicts pose error falls only as far as fewer outliers improve the fit; the oracle (drop the 20% of points with the largest GT error) bounds that.

- Ratios are per pair, method / plain Umeyama (< 1: better). *Geo-mean ratio* is the mean over clips of each clip's mean log ratio, with a 95% CI from resampling clips.

| Subset | Points | Pairs | Method | rot err (deg) | trans err / depth | median trans ratio | geo-mean trans ratio (95% CI) | geo-mean rot ratio (95% CI) | beats Umeyama (trans) |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: |
| `adt_mini` | grid (design 2) | 6850 | Umeyama (A4) | 0.504 | 0.0156 | 1.000 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | n/a |
| | | | trim 20% by residual | 0.501 | 0.0167 | 1.043 | 1.056 (1.037 to 1.074) | 1.003 (0.990 to 1.018) | 0.30 |
| | | | IRLS Huber | 0.502 | 0.0165 | 1.029 | 1.050 (1.031 to 1.068) | 1.002 (0.988 to 1.017) | 0.31 |
| | | | IRLS Cauchy | 0.502 | 0.0171 | 1.051 | 1.088 (1.060 to 1.116) | 1.005 (0.988 to 1.024) | 0.29 |
| | | | exp(conf) weights | 0.499 | 0.0155 | 1.007 | 1.002 (0.988 to 1.014) | 0.998 (0.988 to 1.008) | 0.43 |
| `adt_mini` | tracks (design 1) | 18799 | Umeyama (A4) | 0.814 | 0.0263 | 1.000 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | n/a |
| | | | trim 20% by residual | 0.806 | 0.0261 | 1.005 | 1.003 (0.983 to 1.021) | 0.994 (0.981 to 1.007) | 0.47 |
| | | | IRLS Huber | 0.800 | 0.0262 | 1.002 | 1.003 (0.986 to 1.022) | 0.985 (0.973 to 0.995) | 0.48 |
| | | | IRLS Cauchy | 0.799 | 0.0262 | 1.005 | 1.015 (0.989 to 1.043) | 0.982 (0.965 to 0.997) | 0.47 |
| | | | exp(conf) weights | 0.820 | 0.0268 | 1.005 | 1.016 (1.005 to 1.027) | 1.006 (1.000 to 1.012) | 0.45 |
| | | | oracle: trim 20% by GT error | 0.827 | 0.0259 | 1.000 | 0.999 (0.981 to 1.015) | 1.045 (1.013 to 1.082) | 0.50 |
| `po_mini` | grid (design 2) | 6850 | Umeyama (A4) | 0.299 | 0.0061 | 1.000 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | n/a |
| | | | trim 20% by residual | 0.296 | 0.0063 | 1.008 | 1.007 (0.991 to 1.023) | 0.990 (0.982 to 0.998) | 0.44 |
| | | | IRLS Huber | 0.297 | 0.0062 | 1.002 | 0.997 (0.981 to 1.012) | 0.987 (0.977 to 0.995) | 0.46 |
| | | | IRLS Cauchy | 0.296 | 0.0063 | 1.007 | 1.010 (0.989 to 1.031) | 0.986 (0.975 to 0.996) | 0.45 |
| | | | exp(conf) weights | 0.299 | 0.0062 | 1.002 | 0.996 (0.983 to 1.010) | 0.995 (0.987 to 1.001) | 0.47 |
| `po_mini` | tracks (design 1) | 18800 | Umeyama (A4) | 0.442 | 0.0101 | 1.000 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | n/a |
| | | | trim 20% by residual | 0.431 | 0.0097 | 1.000 | 0.955 (0.934 to 0.976) | 0.967 (0.954 to 0.980) | 0.54 |
| | | | IRLS Huber | 0.427 | 0.0096 | 0.998 | 0.945 (0.925 to 0.966) | 0.959 (0.946 to 0.971) | 0.55 |
| | | | IRLS Cauchy | 0.429 | 0.0097 | 1.000 | 0.950 (0.928 to 0.973) | 0.961 (0.948 to 0.975) | 0.54 |
| | | | exp(conf) weights | 0.447 | 0.0102 | 1.003 | 1.009 (0.993 to 1.027) | 1.013 (1.002 to 1.026) | 0.46 |
| | | | oracle: trim 20% by GT error | 0.438 | 0.0098 | 1.000 | 0.967 (0.947 to 0.985) | 0.980 (0.969 to 0.991) | 0.52 |
| `ds_mini` | grid (design 2) | 6850 | Umeyama (A4) | 0.119 | 0.0017 | 1.000 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | n/a |
| | | | trim 20% by residual | 0.116 | 0.0017 | 0.999 | 0.993 (0.977 to 1.007) | 0.984 (0.974 to 0.993) | 0.50 |
| | | | IRLS Huber | 0.116 | 0.0016 | 0.997 | 0.985 (0.971 to 0.999) | 0.983 (0.974 to 0.992) | 0.52 |
| | | | IRLS Cauchy | 0.117 | 0.0017 | 1.000 | 0.995 (0.977 to 1.012) | 0.984 (0.972 to 0.995) | 0.50 |
| | | | exp(conf) weights | 0.117 | 0.0017 | 1.004 | 1.002 (0.988 to 1.015) | 0.988 (0.980 to 0.995) | 0.48 |
| `ds_mini` | tracks (design 1) | 18800 | Umeyama (A4) | 0.196 | 0.0031 | 1.000 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | n/a |
| | | | trim 20% by residual | 0.188 | 0.0028 | 0.970 | 0.911 (0.884 to 0.932) | 0.952 (0.933 to 0.966) | 0.58 |
| | | | IRLS Huber | 0.186 | 0.0028 | 0.970 | 0.897 (0.869 to 0.920) | 0.944 (0.926 to 0.959) | 0.61 |
| | | | IRLS Cauchy | 0.188 | 0.0028 | 0.968 | 0.903 (0.874 to 0.926) | 0.948 (0.928 to 0.964) | 0.58 |
| | | | exp(conf) weights | 0.195 | 0.0030 | 0.999 | 0.969 (0.940 to 0.996) | 0.995 (0.977 to 1.011) | 0.50 |
| | | | oracle: trim 20% by GT error | 0.191 | 0.0029 | 0.988 | 0.937 (0.912 to 0.959) | 0.969 (0.952 to 0.981) | 0.54 |

GT motion per pair (median): `adt_mini` grid (design 2): 3.62 deg, 0.0488 depth; `adt_mini` tracks (design 1): 7.44 deg, 0.1044 depth; `po_mini` grid (design 2): 1.11 deg, 0.0078 depth; `po_mini` tracks (design 1): 1.70 deg, 0.0134 depth; `ds_mini` grid (design 2): 0.22 deg, 0.0034 depth; `ds_mini` tracks (design 1): 0.52 deg, 0.0089 depth.
