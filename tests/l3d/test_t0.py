"""T0 and B0 ablations (l3d.t0)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from l3d import ensemble as ens
from l3d import sensitivity as sens
from l3d import t0

SMALL = dict(T=12, n_static=120, n_objects=2, n_per_object=40, occlusion_prob=0.2)
ESTS = ("mean", "geomedian", "conf_select", "oracle", "t0_l2_1", "t0_cauchy_0.25")


@pytest.fixture(scope="module")
def scene() -> sens.Scene:
    return sens.make_scene(0, **SMALL)


def _sums(scene: sens.Scene, inj: sens.Injection | None = None) -> dict:
    dec = sens.VirtualDecoder(scene) if inj is None else sens.VirtualDecoder(scene, inj, seed=7)
    return t0.clip_sums(sens.synthetic_table(scene, dec, max_tracks=32, model_scale=0.5), ks=(2, 4), estimators=ESTS)


def _objective(x: np.ndarray, a0: np.ndarray, B0: np.ndarray, lam: float, psi: str, c: float) -> float:
    a, B = x[:3], x[3:].reshape(-1, 3)
    z = abs(a0[2])
    r = np.linalg.norm(a - B, axis=-1) / z
    rho = 0.5 * r**2 if psi == "l2" else 0.5 * c**2 * np.log1p((r / c) ** 2)
    return rho.sum() + 0.5 * lam * (np.sum((a - a0) ** 2) + np.sum((B - B0) ** 2)) / z**2


@pytest.mark.parametrize("psi", ["l2", "cauchy"])
def test_t0_solve_is_the_minimizer(psi: str) -> None:
    rng = np.random.default_rng(0)
    a0 = np.array([0.1, -0.2, 2.0])
    B0 = a0 + 0.05 * rng.normal(size=(4, 3))
    B0[3] += np.array([0.0, 0.0, 0.8])  # one far partner
    lam, c = 0.5, 0.03
    a = t0.t0_solve(a0, B0, lam, psi, c, iters=200, tol=0.0)
    x0 = np.concatenate([a0, B0.reshape(-1)])
    best = minimize(_objective, x0, args=(a0, B0, lam, psi, c), method="BFGS", options={"gtol": 1e-12})
    assert a == pytest.approx(best.x[:3], abs=1e-5)


def test_t0_l2_interpolates_single_and_mean() -> None:
    a0 = np.zeros(3) + np.array([0, 0, 1.0])
    B0 = np.array([[1.0, 0, 1], [3.0, 0, 1]])
    assert t0.t0_solve(a0, B0, 1e-9) == pytest.approx(np.r_[(0 + 1 + 3) / 3, 0, 1])
    assert t0.t0_solve(a0, B0, 1e9) == pytest.approx(a0, abs=1e-6)
    assert t0.t0_solve(a0, B0, 1.0) == pytest.approx(np.r_[0.5 * 4 / 2, 0, 1])  # beta = 1/2: (0 + 2) / 2


def test_t0_cauchy_drops_an_outlier() -> None:
    a0 = np.array([0.0, 0.0, 1.0])
    B0 = np.array([[0.01, 0, 1], [-0.01, 0, 1], [0.0, 0.01, 1], [2.0, 0, 1]])
    l2 = t0.t0_solve(a0, B0, 0.1, "l2")
    robust = t0.t0_solve(a0, B0, 0.1, "cauchy", c=0.01)
    assert abs(robust[0]) < 0.01 < abs(l2[0])


def test_line_solve_removes_part_of_the_offset() -> None:
    P0 = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    perp = np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = t0.line_solve(P0, perp, lam=1.0)
    assert out[0] == pytest.approx([-0.1, 0.0, 2.0]) and out[1] == pytest.approx(P0[1])


def test_estimators_pick_and_median() -> None:
    anchor = np.array([[0.0, 0, 1]])
    O = np.array([[[[1.0, 0, 1], [10.0, 0, 1]]]])  # one cell, one subset of two partners
    wa, W, X = np.array([1.0]), np.array([[[5.0, 2.0]]]), np.array([[9.0, 0, 1]])
    assert t0.make_estimator("conf_select")(anchor, O, wa, W, X)[0, 0] == pytest.approx([1.0, 0, 1])
    assert t0.make_estimator("oracle")(anchor, O, wa, W, X)[0, 0] == pytest.approx([10.0, 0, 1])
    assert t0.make_estimator("geomedian")(anchor, O, wa, W, X)[0, 0] == pytest.approx([1.0, 0, 1], abs=1e-4)


def test_clean_decoder_gives_zero_error(scene: sens.Scene) -> None:
    s = _sums(scene)
    for m in ("b0_model_u2", "b0_gt_u2", "b0c_cam0_tstar", "b0_model_u2_uv2", "b0c_cam0_tstar_sim3"):
        est = "t0_l2_1" if m in ens.METHODS else "mean"
        c = ens.summarize_cell(s[m][est]["all"][4])
        assert c["n_cells"] > 0 and c["rel"] < 1e-4 and c["rel_1"] < 1e-4, m
    rc = ens.summarize_cell(s["t0_rc_gt"]["t0_l2_1"]["all"][2])
    assert rc["n_cells"] > 0 and rc["rel"] < 1e-4


def test_t0_l2_sits_between_single_and_b0(scene: sens.Scene) -> None:
    s = _sums(scene, sens.Injection("depth_jitter", "S", 0.05))
    mean = ens.summarize_cell(s["b0_gt_u2"]["mean"]["all"][4])
    damped = ens.summarize_cell(s["b0_gt_u2"]["t0_l2_1"]["all"][4])
    oracle = ens.summarize_cell(s["b0_gt_u2"]["oracle"]["all"][4])
    assert 0 < damped["gain"] < mean["gain"] and 0 < oracle["gain"] < mean["gain"]  # picking one output < averaging
    assert damped["gain_dep"] > 0.3 and damped["dep_1"] > damped["ang_1"]  # the injected error is mostly depth


def test_ray_center_t0_removes_off_ray_error(scene: sens.Scene) -> None:
    s = _sums(scene, sens.Injection("nonrigid", "C", 0.05))
    rc = ens.summarize_cell(s["t0_rc_gt"]["t0_l2_1"]["all"][2])
    assert rc["n_cells"] > 0 and rc["gain"] > 0.05 and rc["gain_ang"] > rc["gain_dep"]
