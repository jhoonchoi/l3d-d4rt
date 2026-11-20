"""Stage 1 point table: one row per unit (clip, frame-0-visible track ``n``, target frame ``t``).

See ``docs/l3d/STAGE1_PLAN.md`` (section 3). Every model gives a camera-0 point per unit (Tier A);
models with decoded-query tables also give the target-camera point, the 2D position and the raw logits
(Tier B). All model quantities use the upstream WorldTrack alignment: one global median scale per model
and clip (``median |g| / median |p|`` over the clip's finite camera-0 units), so the mean of the per-unit
score ``s`` over a clip is exactly the upstream APD of that clip.

Per-unit columns (``<m>`` = model name):

- keys: ``clip`` (index into the clip list), ``track``, ``t``;
- GT labels: ``static``, ``vis``, ``parallax`` (deg, cameras 0 and ``t``), ``speed`` (GT 3D step from
  ``t - 1`` to ``t``, / clip median depth; ``t = 0`` uses the step to 1), ``depth_rel`` (GT depth at ``t`` /
  clip median depth), ``cam_rot`` (deg) and ``cam_trans`` (/ clip median depth) between cameras 0 and
  ``t``, ``border`` (GT pixel's distance to the image border / min(H, W); negative = off-image);
- Tier A: ``d_<m>`` (camera-0 error after scale), ``s_<m>`` (APD score), ``e_<m>`` (``[M, 3]`` error
  vector ``scale * p - g``); shared ``g0`` (``[M, 3]`` GT camera-0 point) and ``group`` (A1's rigid GT motion
  group of the track: 0 = static, 1.. = dynamic groups, -1 = none). The nested Tier A split (Sim(3) gauge,
  group offset, jitter, residual) runs at analysis time on the selected units (:func:`l3d.compare.structure_parts`),
  so its parts add up exactly on every selection;
- Tier B: ``point_sq_<m>`` (``|R_gt (p_t - g_t)|^2``, the target-camera point error),
  ``pose_sq_<m>`` (``|e_cam0 - e_point|^2``), ``cross_<m>`` (``2 e_point . e_pose``), ``radial_sq_<m>`` and
  ``perp_sq_<m>`` (depth and image-plane parts of ``point_sq``), ``px_<m>`` (2D error of ``p_uv`` in
  256-px units, as in Gate 3), ``conf_<m>`` and ``vislogit_<m>``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from l3d import geometry as geo
from l3d.decompose import group_tracks, self_frame_errors, world_trajectories
from l3d.gauge import THRESHOLDS
from l3d.splits import parallax_deg

TIER_B = ("point_sq", "pose_sq", "cross", "radial_sq", "perp_sq", "px", "conf", "vislogit")
TIER_A = ("d", "s", "e")
LABELS = ("static", "vis", "parallax", "speed", "depth_rel", "cam_rot", "cam_trans", "border")


def apd_score(d: np.ndarray) -> np.ndarray:
    """Per-unit APD score: mean over the upstream thresholds of ``1[d <= thr]`` (NaN where ``d`` is NaN)."""
    d = np.asarray(d, dtype=np.float64)
    s = np.mean([d <= th for th in THRESHOLDS], axis=0)
    return np.where(np.isfinite(d), s, np.nan)


def median_scale(gt: np.ndarray, pred: np.ndarray) -> float:
    """Upstream ``_compute_scale_factor_global``: ``median |gt| / median |pred|`` over finite pairs."""
    g = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(g).all(-1) & np.isfinite(p).all(-1)
    if not ok.any():
        return 1.0
    gn = np.maximum(np.linalg.norm(g[ok], axis=-1), 1e-12)
    pn = np.maximum(np.linalg.norm(p[ok], axis=-1), 1e-12)
    return float(np.median(gn) / max(float(np.median(pn)), 1e-12))


def design0_arrays(d: dict[str, Any]) -> dict[str, Any]:
    """Design-0 rows of a decoded-query table as ``[T, N0]`` arrays over the frame-0-visible tracks.

    Returns ``tracks [N0]``, GT ``g0`` / ``gown`` ``[T, N0, 3]`` (camera 0 / own camera), ``guv [T, N0, 2]``,
    and the model's ``p0``, ``pown``, ``puv``, ``conf``, ``vislogit`` (NaN where a row is missing).
    """
    m = np.asarray(d["design"]) == 0
    tr = np.asarray(d["q_track"])[m].astype(np.int64)
    tt = np.asarray(d["q_tt"])[m].astype(np.int64)
    tc = np.asarray(d["q_tc"])[m].astype(np.int64)
    if np.any(np.asarray(d["q_ts"])[m] != 0):
        raise ValueError("design-0 rows must have t_s = 0")
    T = int(d["T"])
    tracks = np.unique(tr)
    col = np.searchsorted(tracks, tr)
    out: dict[str, Any] = {"tracks": tracks}
    shape3, shape2, shape1 = (T, tracks.size, 3), (T, tracks.size, 2), (T, tracks.size)
    for key, shape in (("g0", shape3), ("gown", shape3), ("p0", shape3), ("pown", shape3),
                       ("guv", shape2), ("puv", shape2), ("conf", shape1), ("vislogit", shape1)):
        out[key] = np.full(shape, np.nan, dtype=np.float64)
    p, g = np.asarray(d["p_xyz"])[m], np.asarray(d["g_xyz"])[m]
    for sel, pk, gk in ((tc == 0, "p0", "g0"), (tc == tt, "pown", "gown")):
        out[pk][tt[sel], col[sel]] = p[sel]
        out[gk][tt[sel], col[sel]] = g[sel]
    own = tc == tt
    out["puv"][tt[own], col[own]] = np.asarray(d["p_uv"])[m][own]
    out["guv"][tt[own], col[own]] = np.asarray(d["g_uv"])[m][own]
    out["conf"][tt[own], col[own]] = np.asarray(d["p_conf_logit"])[m][own]
    out["vislogit"][tt[own], col[own]] = np.asarray(d["p_vis_logit"])[m][own]
    return out


def gt_labels(d: dict[str, Any], tracks: np.ndarray, guv: np.ndarray) -> dict[str, np.ndarray]:
    """GT-only condition labels ``[T, N0]`` for the frame-0 tracks (see the module docstring)."""
    T = int(d["T"])
    n = tracks.size
    gt_cam = np.asarray(d["gt_xyz_cam"], dtype=np.float64)[:, tracks]
    T_w2c = np.asarray(d["T_w2c_gt"], dtype=np.float64)
    world = world_trajectories(gt_cam, T_w2c)
    z = gt_cam[..., 2]
    med_depth = float(np.nanmedian(np.abs(z))) if np.isfinite(z).any() else 1.0
    step = np.linalg.norm(np.diff(world, axis=0), axis=-1)  # [T-1, N]
    speed = np.empty((T, n))
    speed[1:] = step
    speed[0] = step[0] if T > 1 else np.nan
    rel = geo.relative_pose(T_w2c, np.arange(T), 0)  # camera t -> camera 0
    c2w = geo.invert_se3(geo.rigidify(T_w2c))
    hw = np.asarray(d["hw_native"], dtype=np.float64)
    x, y = guv[..., 0] * (hw[1] - 1), guv[..., 1] * (hw[0] - 1)
    border = np.minimum.reduce([x, hw[1] - 1 - x, y, hw[0] - 1 - y]) / hw.min()
    tt = np.repeat(np.arange(T), n)
    par = parallax_deg(d, np.tile(tracks, T), np.zeros(T * n, dtype=np.int64), tt).reshape(T, n)
    return {
        "static": np.broadcast_to(np.asarray(d["is_static"], dtype=bool)[tracks], (T, n)),
        "vis": np.asarray(d["gt_vis"], dtype=bool)[:, tracks],
        "parallax": par,
        "speed": speed / med_depth,
        "depth_rel": z / med_depth,
        "cam_rot": np.broadcast_to(geo.rotation_angle_deg(rel[:, :3, :3])[:, None], (T, n)),
        "cam_trans": np.broadcast_to((np.linalg.norm(c2w[:, :3, 3] - c2w[0, :3, 3], axis=-1) / med_depth)[:, None], (T, n)),
        "border": border,
        "_world": world,
        "_rel": rel,
    }


def build_clip(d_ref: dict[str, Any], models: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Point-table rows of one clip.

    ``d_ref`` is a decoded-query table (for GT and the track set). ``models`` maps a model name to a
    dict with ``p0 [T, N0, 3]`` (camera-0 prediction, any scale) and, for Tier B, ``pown``, ``puv``,
    ``conf``, ``vislogit`` from :func:`design0_arrays`. Units are the ``(t, n)`` with a finite GT camera-0
    point; a model's columns are NaN where its prediction is missing.
    """
    ref = design0_arrays(d_ref)
    tracks, g0, gown, guv = ref["tracks"], ref["g0"], ref["gown"], ref["guv"]
    lab = gt_labels(d_ref, tracks, guv)
    unit = np.isfinite(g0).all(-1)
    tt, nn = np.nonzero(unit)
    rows: dict[str, np.ndarray] = {"track": tracks[nn].astype(np.int32), "t": tt.astype(np.int16)}
    for k in LABELS:
        rows[k] = np.asarray(lab[k])[unit]
    groups = group_tracks(lab["_world"], np.asarray(d_ref["is_static"], dtype=bool)[tracks], mode="rigid")
    rows["group"] = np.broadcast_to(groups[None], unit.shape)[unit].astype(np.int16)
    rows["g0"] = g0[unit]
    rel = lab["_rel"]
    meta: dict[str, Any] = {"subset": str(d_ref["subset"]), "sequence": str(d_ref["sequence"]),
                            "T": int(d_ref["T"]), "tracks": int(tracks.size), "units": int(unit.sum()), "models": {}}
    for name, mdl in models.items():
        p0 = np.asarray(mdl["p0"], dtype=np.float64)
        if p0.shape != g0.shape:
            raise ValueError(f"{name}: p0 {p0.shape} vs GT {g0.shape}")
        scale = median_scale(g0[unit], p0[unit])
        e = scale * p0 - g0
        dist = np.linalg.norm(e, axis=-1)
        rows[f"d_{name}"] = dist[unit]
        rows[f"s_{name}"] = apd_score(dist)[unit]
        rows[f"e_{name}"] = e[unit]
        fin = np.isfinite(dist[unit])
        meta["models"][name] = {"scale": scale, "finite_units": int(fin.sum()),
                                "apd": float(np.nanmean(rows[f"s_{name}"])), "epe": float(np.nanmean(rows[f"d_{name}"]))}
        if "pown" in mdl:
            pown = scale * np.asarray(mdl["pown"], dtype=np.float64)
            e_point = np.einsum("tij,tnj->tni", rel[:, :3, :3], pown - gown)
            e_pose = e - e_point
            sf = self_frame_errors(pown, gown)
            puv = np.asarray(mdl["puv"], dtype=np.float64)
            vals = {"point_sq": (e_point**2).sum(-1), "pose_sq": (e_pose**2).sum(-1),
                    "cross": 2.0 * (e_point * e_pose).sum(-1), "radial_sq": sf["radial"] ** 2,
                    "perp_sq": (sf["perp"] ** 2).sum(-1), "px": np.linalg.norm((puv - guv) * 255.0, axis=-1),
                    "conf": np.asarray(mdl["conf"]), "vislogit": np.asarray(mdl["vislogit"])}
            for k, v in vals.items():
                rows[f"{k}_{name}"] = np.asarray(v, dtype=np.float64)[unit]
    for k, v in rows.items():
        if v.dtype == np.float64:
            rows[k] = v.astype(np.float32)
    return rows, meta
