"""Robust T* fits (l3d.pose_robust)."""

from __future__ import annotations

import numpy as np
import pytest

from l3d import geometry as geo
from l3d import pose_robust as pr
from l3d import sensitivity as sens


def _pair(n: int = 400, n_out: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.normal(size=(n, 3)) + np.array([0.0, 0.0, 4.0])
    R = geo.rotvec_to_matrix(np.array([0.05, -0.1, 0.02]))
    t = np.array([0.2, -0.1, 0.3])
    dst = src @ R.T + t + 1e-4 * rng.normal(size=(n, 3))
    out = np.zeros(n, dtype=bool)
    out[rng.choice(n, n_out, replace=False)] = True
    dst[out] += rng.normal(scale=0.5, size=(n_out, 3)) + np.array([1.0, 0.0, 0.0])  # gross, biased outliers
    return src, dst, R, t, out


@pytest.mark.parametrize("method", ["trim20", "huber", "cauchy", "oracle_trim20"])
def test_robust_fits_reject_outliers(method: str) -> None:
    src, dst, R, t, out = _pair()
    R0, t0 = pr.fit(src, dst, "umeyama")
    Rm, tm = pr.fit(src, dst, method, gt_err=out.astype(float))
    assert np.linalg.norm(tm - t) < 0.2 * np.linalg.norm(t0 - t)
    assert geo.rotation_angle_deg(Rm.T @ R) < geo.rotation_angle_deg(R0.T @ R)


def test_expconf_ignores_low_confidence_points() -> None:
    src, dst, R, t, out = _pair()
    logit = np.where(out, -20.0, 10.0)
    Rm, tm = pr.fit(src, dst, "expconf", logit=logit)
    assert np.linalg.norm(tm - t) < 1e-3


def test_clip_pose_is_exact_for_a_clean_decoder() -> None:
    scene = sens.make_scene(0, T=12, n_static=120, n_objects=2, n_per_object=40, occlusion_prob=0.2)
    d = sens.synthetic_table(scene, sens.VirtualDecoder(scene), max_tracks=32, model_scale=0.5)
    res = pr.clip_pose(d)
    for design in pr.DESIGNS:
        assert res[design]["pairs"]["t"].size > 0, design
        for m, v in res[design]["methods"].items():
            assert np.max(v["trans_rel"]) < 1e-4 and np.max(v["rot_deg"]) < 1e-3, (design, m)
