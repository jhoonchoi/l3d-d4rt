"""Robust T* fits: does dropping outliers lower pose error? (RQ3's pose prediction, at test time.)

The proposal predicts that fine-tuning with the checks leaves the rigid gauge alone, "so
pose error falls only as far as fewer outliers improve the fit". Here the same question
is asked without training: per camera pair ``(t, c)``, ``T*: t -> c`` is refitted from the
same outputs with outlier rejection, and compared with the plain SE(3) Umeyama fit of A4.

Label-free methods (weights from the camera-swap residual ``|T* P_t - P_c| / zbar`` of the
previous fit, or from the confidence):

- ``umeyama``: plain fit (A4's baseline);
- ``trim20``: refit without the 20% largest residuals;
- ``huber`` / ``cauchy``: IRLS with the pair's median residual as scale;
- ``expconf``: weights ``exp(logit_t + logit_c)`` (A4's ``sigmoid`` weights saturate at 1).

Not label-free (design 1 only, where rows have GT): ``oracle_trim20`` drops the 20% of points
with the largest GT error (the larger of the two outputs' relative errors), the most that
removing outliers could do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from l3d import geometry as geo
from l3d.check_rows import MIN_POINTS, cross_pairs, row_errors
from l3d.checks import pose_errors
from l3d.decompose import table_scale
from l3d.t0 import irls_weights

METHODS = ("umeyama", "trim20", "huber", "cauchy", "expconf", "oracle_trim20")
DESIGNS = ("grid", "cross")
IRLS_ITERS = 10
_EPS = 1e-12


def _residual(src: np.ndarray, dst: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    zbar = np.maximum(np.abs(0.5 * (src[:, 2] + dst[:, 2])), _EPS)
    return np.linalg.norm(src @ R.T + t - dst, axis=-1) / zbar


def fit(src: np.ndarray, dst: np.ndarray, method: str, logit: np.ndarray | None = None,
        gt_err: np.ndarray | None = None, frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` of ``T*: src -> dst`` (SE(3)) by one of :data:`METHODS`."""
    _, R, t = geo.umeyama(src, dst, with_scale=False)
    if method == "umeyama" or not np.isfinite(R).all():
        return R, t
    if method == "expconf":
        lg = np.asarray(logit, dtype=np.float64)
        w = np.exp(np.clip(lg - lg.max(), -50.0, 0.0))
        _, R, t = geo.umeyama(src, dst, weights=w, with_scale=False)
        return R, t
    if method in ("trim20", "oracle_trim20"):
        score = _residual(src, dst, R, t) if method == "trim20" else np.asarray(gt_err, dtype=np.float64)
        keep = score <= np.nanquantile(score, 1.0 - frac)
        _, R, t = geo.umeyama(src[keep], dst[keep], with_scale=False)
        return R, t
    for _ in range(IRLS_ITERS):
        r = _residual(src, dst, R, t)
        w = irls_weights(r, method, float(np.median(r)))
        _, R_new, t_new = geo.umeyama(src, dst, weights=w, with_scale=False)
        if not np.isfinite(R_new).all():
            break
        R, t = R_new, t_new
    return R, t


def clip_pose(d: dict[str, Any], min_points: int = MIN_POINTS) -> dict[str, Any]:
    """Per design and camera pair: GT motion and each method's rotation / translation error
    (translation over the clip's median GT depth). Empty without GT poses."""
    out: dict[str, Any] = {}
    if not bool(d["has_pose_gt"]):
        return out
    err = row_errors(d, table_scale(d))
    P, rel = err["P"], err["rel"]
    logit = d["p_conf_logit"].astype(np.float64)
    T_w2c = np.asarray(d["T_w2c_gt"], dtype=np.float64)
    z = np.asarray(d["gt_xyz_cam"], dtype=np.float64)[..., 2]
    ref_depth = float(np.median(z[np.isfinite(z)]))
    for design in DESIGNS:
        j, i = cross_pairs(d, design)
        ok = np.isfinite(P[j]).all(-1) & np.isfinite(P[i]).all(-1)
        j, i = j[ok], i[ok]
        tt, tc = d["q_tt"][i].astype(np.int64), d["q_tc"][i].astype(np.int64)
        key = tt * int(d["T"]) + tc
        order = np.argsort(key, kind="stable")
        rec: dict[str, list[float]] = {k: [] for k in ("t", "c", "gt_rot_deg", "gt_trans_rel")}
        per = {m: {"rot_deg": [], "trans_rel": []} for m in METHODS if design == "cross" or m != "oracle_trim20"}
        for sel in np.split(order, np.flatnonzero(np.diff(key[order])) + 1):
            if sel.size < min_points:
                continue
            t, c = int(tt[sel[0]]), int(tc[sel[0]])
            src, dst = P[j[sel]], P[i[sel]]
            lg = logit[j[sel]] + logit[i[sel]]
            gt_err = np.fmax(rel[j[sel]], rel[i[sel]]) if design == "cross" else None
            T_gt = geo.relative_pose(T_w2c, t, c)
            fits = {m: fit(src, dst, m, lg, gt_err) for m in per}
            if not all(np.isfinite(R).all() for R, _ in fits.values()):
                continue
            rec["t"].append(t)
            rec["c"].append(c)
            rec["gt_rot_deg"].append(float(geo.rotation_angle_deg(T_gt[:3, :3])))
            rec["gt_trans_rel"].append(float(np.linalg.norm(T_gt[:3, 3]) / ref_depth))
            for m, (R, tr) in fits.items():
                pe = pose_errors(geo.sim3_matrix(1.0, R, tr), T_gt)
                per[m]["rot_deg"].append(float(pe["rot_deg"]))
                per[m]["trans_rel"].append(float(pe["trans"]) / ref_depth)
        out[design] = {"pairs": {k: np.asarray(v) for k, v in rec.items()},
                       "methods": {m: {k: np.asarray(v) for k, v in vals.items()} for m, vals in per.items()}}
    return out
