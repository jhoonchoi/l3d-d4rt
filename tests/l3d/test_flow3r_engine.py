"""CPU tests for l3d.flow3r_engine: query composition on an analytic scene, plus a random-weight run of the real
flow3r architecture (skipped without third_party/flow3r; run scripts/l3d/setup_flow3r.sh with SKIP_CKPT=1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from l3d import checks
from l3d.flow3r_engine import Flow3rEngine, bilinear, flow3r_size, ndc_to_unit

REPO = Path(__file__).resolve().parents[2]
FLOW3R_ROOT = REPO / "third_party" / "flow3r"

T, H, W = 4, 29, 43
FX = FY = 30.0
CX, CY = (W - 1) / 2, (H - 1) / 2


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _pose(t: int) -> np.ndarray:
    P = np.eye(4)
    P[:3, :3] = _rot_y(5.0 * t)
    P[:3, 3] = [0.1 * t, -0.05 * t, 0.02 * t]
    return P


def _shift(i: int, j: int) -> tuple[float, float]:
    return 2.0 * (j - i), -1.0 * (j - i)


def _local(t: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.full_like(x, 5.0 + t)
    return np.stack([(x - CX) / FX * z, (y - CY) / FY * z, z], -1)


class FakeEngine(Flow3rEngine):
    """Analytic scene: planar local maps, known poses, flow = constant pixel shift per pair."""

    def __init__(self, pair_chunk: int = 8) -> None:
        super().__init__(model=None, device="cpu", pair_chunk=pair_chunk)
        ys, xs = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
        local = np.stack([_local(t, xs, ys) for t in range(T)])
        conf = np.stack([xs / (W - 1) + t for t in range(T)])
        self._cache = {
            "local": torch.tensor(local, dtype=torch.float32),
            "conf": torch.tensor(conf, dtype=torch.float32),
            "poses": torch.tensor(np.stack([_pose(t) for t in range(T)]), dtype=torch.float64),
            "hw": (H, W),
        }
        self.flow_calls: list[int] = []

    def _flow_maps(self, pairs: torch.Tensor) -> torch.Tensor:
        self.flow_calls.append(int(pairs.shape[0]))
        ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
        out = []
        for i, j in pairs.tolist():
            dx, dy = _shift(i, j)
            out.append(torch.stack([1 - 2 * (xs + dx) / (W - 1), 1 - 2 * (ys + dy) / (H - 1)], -1))
        return torch.stack(out)


def _expected(u, v, ts, tt, tc):
    dx, dy = _shift(ts, tt)
    x, y = u * (W - 1) + dx, v * (H - 1) + dy
    X = _local(tt, np.array([x]), np.array([y]))[0]
    rel = np.linalg.inv(_pose(tc)) @ _pose(tt)
    P = rel[:3, :3] @ X + rel[:3, 3]
    return P, (x / (W - 1), y / (H - 1)), x / (W - 1) + tt


def test_flow3r_size_rule():
    for hw in [(256, 256), (540, 960), (480, 640), (1080, 1920)]:
        h, w = flow3r_size(*hw)
        assert h % 14 == 0 and w % 14 == 0 and h * w <= 255_000
        assert abs(w / h - hw[1] / hw[0]) < 0.06
    assert flow3r_size(256, 256) == (504, 504)


def test_ndc_and_bilinear_corners():
    assert torch.allclose(ndc_to_unit(torch.tensor([1.0, -1.0, 0.0])), torch.tensor([0.0, 1.0, 0.5]))
    maps = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4, 1)
    k = torch.tensor([1, 1, 0])
    out = bilinear(maps, k, torch.tensor([0.0, 1.0, 0.5]), torch.tensor([0.0, 1.0, 0.5]))[:, 0]
    assert torch.allclose(out, torch.tensor([12.0, 23.0, 5.5]))


@pytest.mark.parametrize("pair_chunk", [1, 3, 8])
def test_decode_matches_analytic_scene(pair_chunk):
    eng = FakeEngine(pair_chunk=pair_chunk)
    rng = np.random.default_rng(0)
    M = 200
    ts, tt, tc = (rng.integers(0, T, M) for _ in range(3))
    u = rng.uniform(0.2, 0.8, M)
    v = rng.uniform(0.2, 0.8, M)
    out = eng.decode(u, v, ts, tt, tc)
    for m in range(M):
        P, uv, conf = _expected(u[m], v[m], int(ts[m]), int(tt[m]), int(tc[m]))
        np.testing.assert_allclose(out["xyz_3d"][m], P, atol=1e-4)
        np.testing.assert_allclose(out["uv_2d"][m], uv, atol=1e-5)
        np.testing.assert_allclose(out["confidence"][m], conf, atol=1e-4)
    assert (out["visibility"] > 0).all()
    n_pairs = len({(a, b) for a, b in zip(ts, tt) if a != b})
    assert sum(eng.flow_calls) == n_pairs
    assert max(eng.flow_calls) <= pair_chunk


def test_self_frame_needs_no_flow_and_out_of_frame_is_invisible():
    eng = FakeEngine()
    out = eng.decode(np.array([0.5]), np.array([0.5]), np.array([2]), np.array([2]), np.array([2]))
    assert eng.flow_calls == []
    np.testing.assert_allclose(out["xyz_3d"][0], _local(2, np.array([CX]), np.array([CY]))[0], atol=1e-5)
    # shift (j - i) * 2 px to the right from the right edge leaves the image
    out = eng.decode(np.array([1.0]), np.array([0.5]), np.array([0]), np.array([3]), np.array([3]))
    assert out["visibility"][0] < 0


def test_camera_swap_residual_is_zero_by_construction():
    eng = FakeEngine()
    g = np.linspace(0.1, 0.9, 8)
    u, v = (a.ravel() for a in np.meshgrid(g, g))
    n = u.size
    t = np.full(n, 1)
    Pa = eng.decode(u, v, t, t, np.full(n, 0))["xyz_3d"]
    Pb = eng.decode(u, v, t, t, np.full(n, 3))["xyz_3d"]
    assert checks.camera_swap(Pa, Pb)["residual"].max() < 1e-5


@pytest.mark.skipif(not (FLOW3R_ROOT / "flow3r").exists(), reason="no third_party/flow3r clone")
def test_real_architecture_random_weights():
    from l3d.flow3r_engine import import_flow3r

    torch.manual_seed(0)
    model = import_flow3r(FLOW3R_ROOT)()  # only the released "large" config builds
    eng = Flow3rEngine(model, device="cpu", pixel_limit=28 * 42)
    video = np.random.default_rng(0).integers(0, 255, (3, 60, 90, 3), dtype=np.uint8)
    cache = eng.encode(video)
    assert eng.image_hw == (28, 42)
    assert cache["local"].shape == (3, 28, 42, 3) and cache["poses"].shape == (3, 4, 4)
    out = eng.decode(np.array([0.5, 0.3, 0.7]), np.array([0.5, 0.2, 0.9]), np.array([0, 1, 2]),
                     np.array([2, 1, 0]), np.array([1, 0, 2]))
    assert out["xyz_3d"].shape == (3, 3) and out["uv_2d"].shape == (3, 2)
    assert np.isfinite(out["xyz_3d"]).all() and np.isfinite(out["confidence"]).all()
    # the self-frame query keeps its pixel
    np.testing.assert_allclose(out["uv_2d"][1], [0.3, 0.2], atol=1e-6)


@pytest.mark.skipif(not (FLOW3R_ROOT / "flow3r").exists(), reason="no third_party/flow3r clone")
def test_dump_queries_flow3r_end_to_end(tmp_path, monkeypatch):
    """`dump_queries.py --model flow3r` writes valid tables (random weights, all six designs)."""
    import json

    from l3d import table as l3d_table
    from l3d.flow3r_engine import import_flow3r
    from tests.l3d.test_engine import _load_dump_module, make_synthetic_worldtrack_npz

    data_root = tmp_path / "worldtrack_release"
    make_synthetic_worldtrack_npz(data_root / "po_mini" / "synth_pose.npz", with_extrinsics=True, seed=1)

    def fake_from_checkpoint(ckpt_path, **kwargs):
        torch.manual_seed(0)
        return Flow3rEngine(import_flow3r(FLOW3R_ROOT)(), device="cpu", pixel_limit=28 * 42,
                            pair_chunk=kwargs.get("pair_chunk", 8), model_config="flow3r@test")

    monkeypatch.setattr(Flow3rEngine, "from_checkpoint", staticmethod(fake_from_checkpoint))
    dump = _load_dump_module()
    out_root = tmp_path / "dumps"
    rc = dump.main(["--model", "flow3r", "--ckpt-path", "unused.bin", "--data-root", str(data_root),
                    "--subsets", "po_mini", "--clip-len", "8", "--designs", "0,1,2,3,4,5",
                    "--out-root", str(out_root), "--device", "cpu", "--pair-chunk", "4"])
    assert rc == 0
    run_dir = out_root / "flow3r_T8"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [c["status"] for c in manifest["clips"]] == ["ok"]
    path = run_dir / "po_mini" / "synth_pose__c0.npz"
    assert l3d_table.validate_table(l3d_table.load_table(path)) == []
    t = l3d_table.load_table(path)
    assert set(np.unique(t["design"]).tolist()) == {0, 1, 2, 3, 4, 5}
    assert np.isfinite(t["p_xyz"]).all()
