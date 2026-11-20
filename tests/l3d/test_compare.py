"""CPU tests for l3d.compare (Stage 1 analyses) on synthetic point tables."""

from __future__ import annotations

import numpy as np

from l3d.compare import condition_bins, q1_pair, q2_pair, q2_regression
from l3d.pointtable import apd_score


def _points(err_a, err_b, clips=20, tracks=30, T=12, seed=0):
    """Synthetic point table: per-unit errors from ``err_a(rng, shape)`` and ``err_b``."""
    rng = np.random.default_rng(seed)
    c, n, t = np.meshgrid(np.arange(clips), np.arange(tracks), np.arange(T), indexing="ij")
    shape = c.shape
    P = {"clip": c.ravel().astype(np.int16), "track": n.ravel().astype(np.int32), "t": t.ravel().astype(np.int16),
         "static": (n.ravel() % 2 == 0), "vis": np.ones(c.size, bool),
         "parallax": rng.uniform(0, 10, c.size), "speed": rng.uniform(0, 0.02, c.size),
         "depth_rel": rng.uniform(0.3, 3, c.size), "cam_rot": t.ravel() * 0.5, "cam_trans": t.ravel() * 0.01,
         "border": rng.uniform(-0.1, 0.4, c.size)}
    for m, f in (("a", err_a), ("b", err_b)):
        d = np.abs(f(rng, shape)).ravel()
        P[f"d_{m}"] = d.astype(np.float32)
        P[f"s_{m}"] = apd_score(d).astype(np.float32)
    return P


def test_oracles_never_worse_than_either_model():
    P = _points(lambda r, s: r.gamma(2.0, 0.2, s), lambda r, s: r.gamma(2.0, 0.25, s))
    r = q1_pair(P, "a", "b", "all")
    best = max(r["apd"].values())
    for name in ("unit", "track", "clip"):
        assert r["oracle"][name]["apd"] >= best - 1e-12
    assert r["oracle"]["unit"]["apd"] >= r["oracle"]["track"]["apd"] >= r["oracle"]["clip"]["apd"] - 1e-12


def test_cv_oracle_has_no_gain_for_equal_iid_models():
    """Same error distribution, independent per unit: per-unit/track oracles gain, the CV oracle does not."""
    P = _points(lambda r, s: r.gamma(2.0, 0.2, s), lambda r, s: r.gamma(2.0, 0.2, s), seed=3)
    r = q1_pair(P, "a", "b", "all")
    assert r["oracle"]["unit"]["gain"] > 0.05
    assert r["oracle"]["track"]["gain"] > 0.0
    assert abs(r["oracle"]["track_cv"]["gain"]) < 0.01
    assert abs(r["rho_pooled"]) < 0.05


def test_cv_oracle_finds_persistent_track_differences():
    """Each model is bad on half of the tracks (a fixed per-track offset): the CV oracle gains."""
    def err(flip):
        def f(r, s):
            bad = (np.arange(s[1]) % 2 == flip)[None, :, None]
            return r.gamma(2.0, 0.1, s) + np.where(bad, 0.8, 0.0)
        return f
    P = _points(err(0), err(1), seed=5)
    r = q1_pair(P, "a", "b", "all")
    assert r["oracle"]["track_cv"]["gain"] > 0.2
    assert r["rho_pooled"] < 0


def test_q2_bins_and_regression_recover_a_condition_effect():
    """Model a is worse only on occluded units: the regression finds the effect, other terms stay near 0."""
    P = _points(lambda r, s: r.gamma(2.0, 0.15, s), lambda r, s: r.gamma(2.0, 0.15, s), seed=7)
    rng = np.random.default_rng(1)
    P["vis"] = rng.random(P["t"].size) > 0.3
    occ = ~P["vis"]
    P["d_a"] = np.where(occ, P["d_a"] + 1.0, P["d_a"]).astype(np.float32)
    P["s_a"] = apd_score(P["d_a"]).astype(np.float32)
    bins = condition_bins(P)
    assert set(bins) >= {"motion", "visibility", "frame gap", "speed", "image position"}
    rows = {r["bin"]: r for r in q2_pair(P, "a", "b")["visibility"]}
    assert rows["occluded"]["delta"] < -0.3 and abs(rows["visible"]["delta"]) < 0.03
    coef = {r["term"]: r for r in q2_regression(P, "a", "b")}
    assert coef["visibility: occluded"]["coef"] < -0.3 and coef["visibility: occluded"]["verdict"] == "real"
    others = [r for k, r in coef.items() if k not in ("intercept", "visibility: occluded")]
    assert all(abs(r["coef"]) < 0.05 for r in others)
