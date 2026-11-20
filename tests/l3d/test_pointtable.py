"""CPU tests for l3d.pointtable (Stage 1 point table) on a synthetic design-0 table."""

from __future__ import annotations

import numpy as np

from eval_track3d_in_worldtrack import _compute_average_pts_within_thresh
from l3d import geometry as geo
from l3d.compare import structure_parts
from l3d.pointtable import apd_score, build_clip, design0_arrays

T, N, H, W = 6, 40, 48, 64


def _scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    K = np.array([[50.0, 0, (W - 1) / 2], [0, 50.0, (H - 1) / 2], [0, 0, 1]])
    T_w2c = np.stack([np.eye(4) for _ in range(T)])
    for t in range(T):
        a = np.deg2rad(2.0 * t)
        T_w2c[t, :3, :3] = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        T_w2c[t, :3, 3] = [0.05 * t, 0.0, 0.0]
    X0 = np.stack([rng.uniform(-1, 1, N), rng.uniform(-0.7, 0.7, N), rng.uniform(3, 6, N)], -1)
    is_static = np.arange(N) < N // 2
    vel = np.where(is_static[:, None], 0.0, rng.normal(0, 0.05, (N, 3)))
    world = X0[None] + np.arange(T)[:, None, None] * vel[None]  # world = camera 0
    cam = geo.apply_transform(T_w2c[:, None], world)
    uv = geo.uv_px_to_norm(geo.project(K, cam), (H, W))
    return {"K": K, "T_w2c": T_w2c, "world": world, "cam": cam, "uv": uv, "is_static": is_static}


def _table(sc, pred_cam0, pred_own, pred_uv):
    """Design-0 rows (t_s = 0, every t_t, t_c in {t_t, 0}) for every track."""
    rows = []
    for t in range(T):
        for tc in sorted({t, 0}):
            for n in range(N):
                rows.append((n, t, tc))
    r = np.array(rows)
    n, tt, tc = r[:, 0], r[:, 1], r[:, 2]
    rel = geo.relative_pose(sc["T_w2c"], tt, tc)
    g = geo.apply_transform(rel, sc["cam"][tt, n])
    p = np.where((tc == 0)[:, None], pred_cam0[tt, n], pred_own[tt, n])
    M = r.shape[0]
    return {
        "subset": np.array("po_mini"), "sequence": np.array("synth"), "T": np.array(T), "clip_start": np.array(0),
        "hw_native": np.array([H, W]), "K_gt": sc["K"], "T_w2c_gt": sc["T_w2c"], "has_pose_gt": np.array(True),
        "gt_xyz_cam": sc["cam"].astype(np.float32), "gt_uv": sc["uv"].astype(np.float32),
        "gt_vis": np.ones((T, N), bool), "is_static": sc["is_static"],
        "design": np.zeros(M, np.int8), "q_track": n.astype(np.int32), "q_parent": np.full(M, -1, np.int32),
        "q_u": sc["uv"][0, n, 0].astype(np.float32), "q_v": sc["uv"][0, n, 1].astype(np.float32),
        "q_ts": np.zeros(M, np.int16), "q_tt": tt.astype(np.int16), "q_tc": tc.astype(np.int16),
        "p_xyz": p.astype(np.float32), "p_uv": pred_uv[tt, n].astype(np.float32),
        "p_vis_logit": np.zeros(M, np.float32), "p_conf_logit": np.zeros(M, np.float32),
        "g_xyz": g.astype(np.float32), "g_uv": sc["uv"][tt, n].astype(np.float32), "g_vis": np.ones(M, bool),
    }


def _build(sc, cam0, own, uv):
    d = _table(sc, cam0, own, uv)
    return build_clip(d, {"m": design0_arrays(d)})


def test_gt_prediction_scores_one():
    sc = _scene()
    rows, meta = _build(sc, sc["world"], sc["cam"], sc["uv"])
    assert meta["units"] == T * N
    assert meta["models"]["m"]["apd"] == 1.0
    assert np.nanmax(rows["d_m"]) < 1e-5
    for k in ("point_sq", "pose_sq", "px"):
        assert np.nanmax(np.abs(rows[f"{k}_m"])) < 1e-4, k
    rows["clip"] = np.zeros(rows["t"].shape, np.int16)
    parts = structure_parts(rows, "m", np.ones(rows["t"].shape, bool))
    assert max(abs(v) for v in parts.values()) < 1e-8


def test_apd_equals_upstream_metric_and_parts_add_up():
    sc = _scene(1)
    rng = np.random.default_rng(1)
    own = 1.7 * (sc["cam"] + rng.normal(0, 0.3, sc["cam"].shape))  # wrong scale + noise
    cam0 = geo.apply_transform(geo.relative_pose(sc["T_w2c"], np.arange(T), 0)[:, None], own) + [0.2, 0, 0]
    rows, meta = _build(sc, cam0, own, sc["uv"] + 0.01)
    upstream = _compute_average_pts_within_thresh(sc["world"], cam0, scaling="global")[0]
    assert abs(meta["models"]["m"]["apd"] - upstream) < 1e-6
    assert abs(np.mean(apd_score(rows["d_m"])) - upstream) < 1e-6
    d2 = rows["d_m"].astype(np.float64) ** 2
    rows["clip"] = np.zeros(rows["t"].shape, np.int16)
    # parts add up exactly and are non-negative on any selection (all, static only, dynamic only, d <= 1)
    for sel in (np.ones(d2.shape, bool), rows["static"], ~rows["static"], rows["d_m"] <= 1.0):
        parts = structure_parts(rows, "m", sel)
        assert min(parts.values()) >= -1e-9
        np.testing.assert_allclose(sum(parts.values()), d2[sel].mean(), rtol=1e-4)
    tier_b = rows["point_sq_m"].astype(np.float64) + rows["pose_sq_m"] + rows["cross_m"]
    np.testing.assert_allclose(tier_b, d2, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(rows["radial_sq_m"] + rows["perp_sq_m"], rows["point_sq_m"], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(rows["px_m"], np.hypot(0.01 * 255, 0.01 * 255), rtol=1e-3)


def test_sim3_error_is_all_gauge_and_pose_error_is_all_pose():
    sc = _scene(2)
    R = geo.rotvec_to_matrix(np.array([0.0, 0.05, 0.0]))
    # A rotation and scale of the GT about camera 0's origin: the median scale is exactly 2, so the
    # own-camera points (exact up to the same scale) have no error and all camera-0 error is pose error.
    cam0 = 0.5 * np.einsum("ij,tnj->tni", R, sc["world"])
    rows, _ = _build(sc, cam0, 0.5 * sc["cam"], sc["uv"])
    d2 = (rows["d_m"].astype(np.float64) ** 2).sum()
    assert d2 > 0.1
    rows["clip"] = np.zeros(rows["t"].shape, np.int16)
    parts = structure_parts(rows, "m", np.ones(rows["t"].shape, bool))
    assert parts["gauge"] / (d2 / rows["t"].size) > 0.999
    # At t = 0 camera 0 is the target camera (one row, no pose), so only t > 0 is pose error.
    later = rows["t"] > 0
    d2_later = (rows["d_m"][later].astype(np.float64) ** 2).sum()
    assert rows["point_sq_m"][later].astype(np.float64).sum() / d2_later < 1e-6
    np.testing.assert_allclose(rows["pose_sq_m"][later].astype(np.float64).sum(), d2_later, rtol=1e-4)
    assert np.nanmax(rows["pose_sq_m"][~later]) < 1e-9
