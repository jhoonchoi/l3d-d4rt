"""B0 / B0-C test-time ensembling (l3d.ensemble)."""

from __future__ import annotations

import numpy as np
import pytest

from l3d import ensemble as ens
from l3d import sensitivity as sens
from l3d import tsc

SMALL = dict(T=12, n_static=120, n_objects=2, n_per_object=40, occlusion_prob=0.2)


@pytest.fixture(scope="module")
def scene() -> sens.Scene:
    return sens.make_scene(0, **SMALL)


def _sums(scene: sens.Scene, inj: sens.Injection | None = None) -> dict:
    dec = sens.VirtualDecoder(scene) if inj is None else sens.VirtualDecoder(scene, inj, seed=7)
    return ens.clip_sums(sens.synthetic_table(scene, dec, max_tracks=32, model_scale=0.5), ks=(2, 4))


def test_pad_groups_scatters_in_order() -> None:
    mask, (v,) = ens.pad_groups(np.array([2, 0, 2, 2]), 3, np.array([10.0, 11.0, 12.0, 13.0]))
    assert mask.tolist() == [[True, False, False], [False, False, False], [True, True, True]]
    assert v[0, 0] == 11.0 and v[2].tolist() == [10.0, 12.0, 13.0] and np.isnan(v[1]).all()


def test_ensemble_errors_average_over_subsets() -> None:
    X = np.zeros((1, 3))
    anchor = np.array([[1.0, 0, 0]])
    others = np.array([[[-1.0, 0, 0], [3.0, 0, 0], [np.nan, 0, 0]]])  # third slot invalid
    res = ens.ensemble_errors(anchor, others, np.ones((1, 3), bool), X, ks=(2, 3, 4))
    assert res[2]["e"][0] == pytest.approx(1.0)  # subsets {-1}: 0, {3}: 2
    assert res[2]["e_sq"][0] == pytest.approx(2.0)
    assert res[3]["e"][0] == pytest.approx(1.0) and res[2]["e1"][0] == pytest.approx(1.0)
    assert res[4]["cells"].size == 0  # only two valid others
    w = ens.ensemble_errors(anchor, others, np.ones((1, 3), bool), X, ks=(3,), w_anchor=np.array([1.0]),
                            w_others=np.array([[1.0, 3.0, 1.0]]))
    assert w[3]["e"][0] == pytest.approx(1.8)  # (1 - 1 + 3 * 3) / 5; unweighted it is 1


def test_full_source_ensemble_mse_is_t_plus_c() -> None:
    """Averaging over every source leaves mu + b_c, so its MSE over cameras is T + C."""
    rng = np.random.default_rng(0)
    E = rng.normal(size=(5, 3, 4, 3, 3)) + rng.normal(size=(5, 3, 1, 1, 3))  # [n, t, S, C, d]
    split = tsc.two_way_split(E)
    n, t, S, C, _ = E.shape
    anchor = E[:, :, 0].reshape(-1, 3)  # [(n, t, c), 3]
    others = E[:, :, 1:].transpose(0, 1, 3, 2, 4).reshape(-1, S - 1, 3)
    res = ens.ensemble_errors(anchor, others, np.ones(others.shape[:2], bool), np.zeros_like(anchor), ks=(S,))
    assert res[S]["e_sq"].mean() == pytest.approx(np.nanmean(split["T"] + split["C"]))
    assert (res[S]["e1"] ** 2).mean() == pytest.approx(np.mean((E[:, :, 0] ** 2).sum(-1)))


def test_clean_decoder_gives_zero_error(scene: sens.Scene) -> None:
    s = _sums(scene)
    for m in ens.METHODS:
        for k in (2, 4):
            c = ens.summarize_cell(s[m]["mean"]["all"][k])
            assert c["n_cells"] > 0, m
            assert c["rel"] < 1e-4 and c["rel_1"] < 1e-4, m


def test_source_noise_helps_b0_not_b0c(scene: sens.Scene) -> None:
    s = _sums(scene, sens.Injection("image_plane", "S", 0.05))
    b0 = ens.summarize_cell(s["b0_gt_u2"]["mean"]["all"][4])
    b0c = ens.summarize_cell(s["b0c_gt_pose"]["mean"]["all"][4])
    assert b0["gain"] > 0.2 and b0["gain_mse"] > b0["gain"]
    assert abs(b0c["gain"]) < 1e-4  # each camera carries the same source error, mapped by the GT pose


def test_tstar_absorbs_a_rigid_gauge(scene: sens.Scene) -> None:
    s = _sums(scene, sens.Injection("gauge", "C", 0.05))
    star = ens.summarize_cell(s["b0c_tstar"]["mean"]["all"][4])
    gt = ens.summarize_cell(s["b0c_gt_pose"]["mean"]["all"][4])
    assert star["rel"] < 1e-4 and gt["rel"] > 1e-3  # the self-frame anchor is clean; T* undoes each camera's gauge
    b0 = ens.summarize_cell(s["b0_gt_u2"]["mean"]["all"][4])
    assert abs(b0["gain"]) < 1e-4  # a camera-only error is the same for every source
    cam0 = ens.summarize_cell(s["b0c_cam0_tstar"]["mean"]["all"][4])
    assert abs(cam0["gain"]) < 1e-4 and cam0["rel_1"] > 1e-3  # T* maps every camera onto camera 0's own gauge


def test_camera_ensemble_averages_out_nonrigid_error(scene: sens.Scene) -> None:
    s = _sums(scene, sens.Injection("nonrigid", "C", 0.05))
    cam0 = ens.summarize_cell(s["b0c_cam0_tstar"]["mean"]["all"][4])
    assert cam0["n_cells"] > 0 and cam0["gain"] > 0.2


def test_pool_adds_and_bootstrap_brackets() -> None:
    a = ens.empty_sums((2,))
    a["b0_gt_u2"]["mean"]["all"][2].update(n=2.0, rel=1.0, rel1=2.0, rel_sq=1.0, rel1_sq=4.0)
    p = ens.pool([a, a], ks=(2,))
    c = ens.summarize_cell(p["b0_gt_u2"]["mean"]["all"][2])
    assert c["n_cells"] == 4 and c["gain"] == pytest.approx(0.5) and c["gain_mse"] == pytest.approx(0.75)
    lo, hi = ens.bootstrap_gain(np.array([1.0, 2.0, 1.5]), np.array([2.0, 2.0, 2.0]), n_boot=200)
    assert lo <= 1 - 4.5 / 6.0 <= hi
