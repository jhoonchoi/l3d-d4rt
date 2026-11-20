"""T0 (training-free optimization of the outputs with a check plus a proximity term) and B0 ablations.

The proposal's Part B: "T0: training-free optimization of the outputs with each of the three
loss checks plus a proximity term; it counts only if it beats B0 (source swap) or B0-C
(camera checks) at matched k." Everything runs on the decoded table of :mod:`l3d.ensemble`,
on the same cells and partner sets as B0 / B0-C, so the comparison is at matched ``k``.

**Source swap and camera swap.** A cell is a query's output ``a0`` and ``k - 1`` partner
outputs ``b0_j`` of the same point at the same ``(t3, c)``: decoded from other source frames
(source swap; the model's ``u2`` or the GT-``u2`` control) or in other cameras and mapped by
``T*`` (camera swap; GT pose as a control). With ``u2`` and ``T*`` held fixed (stop-gradient,
as in the proposal's losses), T0 solves

    min_{a, b}  sum_j rho(|a - b_j| / z) + lam / 2 * (|a - a0|^2 + sum_j |b_j - b0_j|^2) / z^2,

``z = |a0_z|``, and scores the optimized ``a``. ``T*`` is an isometry, so the camera partners'
proximity term is the same in their own camera. Each ``b_j`` appears in one pair only, so
the minimum has a closed form for pair weights ``w_j``: ``b_j = (w_j a + lam b0_j) / (w_j + lam)``
and ``a = (a0 + sum_j beta_j b0_j) / (1 + sum_j beta_j)``, ``beta_j = w_j / (w_j + lam)``;
robust ``rho`` is solved by IRLS (``w = rho'(r) / r``) from ``a = a0``. With the squared
penalty (``l2``, ``w = 1``) T0 is a damped B0: the B0 mean as ``lam -> 0``, the single output
as ``lam -> inf``. It can only beat B0 through the damping or a robust ``rho`` that drops
bad partners (``huber``, ``cauchy``; scale = the clip's median residual ``|a0 - b0_j| / z``
over the method's pairs, label-free).

**Ray center.** Design-1 rows with ``t_s = t_t = t`` decoded in camera 0 (the B0-C-to-cam-0
anchors whose query is at the target). Per ``(t, 0)``: ``R*`` from the camera-swap fit
between the self-frame and camera-0 outputs, lines through each output with direction
``R* K^-1 [u; 1]`` (``K`` fitted label-free, or GT ``K`` and GT rotation as a control) and
their least-squares center ``o`` (held fixed). T0 moves each output toward its line through
``o``: it removes ``beta`` of its perpendicular offset, the same closed form. It uses two
decodes per query (cameras ``t`` and 0), so it is stored at ``k = 2``.

**B0 ablations** (estimators other than the mean on the same cells): ``geomedian`` (the
geometric median of the ``k`` outputs, Weiszfeld), ``conf_select`` (the most confident
output), ``oracle`` (the output closest to GT: a selection upper bound, not label-free).
Partner-set variants: ``b0_model_u2_novis`` (no visibility filter on ``u2``),
``b0_model_u2_uv{2,5}`` (label-free 2D agreement: keep a partner only if its 2D output at
``t3`` is within 2 / 5 px, at 256x256, of the query's), ``b0c_cam0_tstar_sim3`` (``T*`` fitted
as Sim(3)).

Every result is an additive sum per ``[method][estimator][group][k]`` (:func:`l3d.ensemble.add_sums`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from l3d import checks
from l3d import ensemble as ens
from l3d import geometry as geo
from l3d.check_rows import MIN_POINTS, cross_pairs, fit_K, row_errors, uv_px
from l3d.decompose import table_scale

_EPS = 1e-12
KS = ens.KS
GROUPS = ens.GROUPS
PSIS = ("l2", "huber", "cauchy")
LAMS = (0.25, 1.0, 4.0)
T0_ESTIMATORS = tuple(f"t0_{p}_{lam:g}" for p in PSIS for lam in LAMS)
B0_ESTIMATORS = ("mean", "conf", "geomedian", "conf_select", "oracle")
CORE_METHODS = ens.METHODS
VARIANT_METHODS = ("b0_model_u2_novis", "b0_model_u2_uv2", "b0_model_u2_uv5", "b0c_cam0_tstar_sim3")
RC_METHODS = ("t0_rc_fit", "t0_rc_gt")
UV_GATES = {"b0_model_u2_uv2": 2.0, "b0_model_u2_uv5": 5.0}
IRLS_ITERS = 30
TOL = 1e-4  # IRLS / Weiszfeld stop: largest update / depth


def parse_t0(name: str) -> tuple[str, float]:
    """``"t0_cauchy_1"`` -> ``("cauchy", 1.0)``."""
    _, psi, lam = name.split("_")
    return psi, float(lam)


def irls_weights(r: np.ndarray, psi: str, c: float) -> np.ndarray:
    """``rho'(r) / r`` for ``l2`` (``r^2 / 2``), ``huber`` and ``cauchy`` with scale ``c``."""
    if psi == "l2":
        return np.ones_like(r)
    rc = r / max(float(c), _EPS)
    if psi == "huber":
        return np.where(rc <= 1.0, 1.0, 1.0 / np.maximum(rc, _EPS))
    if psi == "cauchy":
        return 1.0 / (1.0 + rc**2)
    raise ValueError(f"unknown psi '{psi}'")


def t0_solve(a0: np.ndarray, B0: np.ndarray, lam: float, psi: str = "l2", c: float = 1.0,
             valid: np.ndarray | None = None, iters: int = IRLS_ITERS, tol: float = TOL) -> np.ndarray:
    """The T0 star problem (module docstring): ``a0 [..., 3]``, partners ``B0 [..., m, 3]`` -> optimized ``a``.

    Works on the partners' offsets ``D_j = b0_j - a0`` (partner-major, one array per axis):
    ``a - a0 = sum_j beta_j D_j / (1 + sum_j beta_j)`` and
    ``|a - b0_j|^2 = |a - a0|^2 - 2 (a - a0) . D_j + |D_j|^2``.
    """
    a0 = np.asarray(a0, dtype=np.float64)
    B0 = np.asarray(B0, dtype=np.float64)
    lead, m = a0.shape[:-1], B0.shape[-2]
    A = a0.reshape(-1, 3)
    Bt = np.moveaxis(B0.reshape(-1, m, 3), 1, 0)  # [m, L, 3]
    v = np.isfinite(Bt).all(-1)
    if valid is not None:
        v &= np.moveaxis(np.asarray(valid, dtype=bool).reshape(-1, m), 1, 0)
    D = [np.ascontiguousarray(np.where(v, Bt[..., i] - A[None, :, i], 0.0)) for i in range(3)]
    DD = D[0] * D[0] + D[1] * D[1] + D[2] * D[2]
    z = np.maximum(np.abs(A[:, 2]), _EPS)
    r = np.sqrt(DD) / z
    delta = [np.zeros(A.shape[0]) for _ in range(3)]
    for _ in range(1 if psi == "l2" else iters):
        w = irls_weights(r, psi, c) * v
        beta = w / (w + lam)
        sb = 1.0 + beta.sum(0)
        new = [(beta * Di).sum(0) / sb for Di in D]
        d2 = (new[0] ** 2 + new[1] ** 2 + new[2] ** 2) - 2.0 * (new[0] * D[0] + new[1] * D[1] + new[2] * D[2]) + DD
        r = lam / (w + lam) * np.sqrt(np.maximum(d2, 0.0)) / z
        change = np.maximum(np.maximum(np.abs(new[0] - delta[0]), np.abs(new[1] - delta[1])), np.abs(new[2] - delta[2]))
        delta = new
        if change.size == 0 or float((change / z).max()) < tol:
            break
    return (A + np.stack(delta, -1)).reshape(lead + (3,))


def line_solve(P0: np.ndarray, perp0: np.ndarray, lam: float, psi: str = "l2", c: float = 1.0,
               iters: int = IRLS_ITERS) -> np.ndarray:
    """T0 for the ray center: ``min rho(|e| / z) + lam / 2 |P - P0|^2 / z^2`` with ``e`` the offset of
    ``P`` from its (fixed) line, ``perp0`` that offset at ``P0``; removes ``w / (w + lam)`` of it."""
    P0 = np.asarray(P0, dtype=np.float64)
    perp0 = np.asarray(perp0, dtype=np.float64)
    z = np.maximum(np.abs(P0[:, 2]), _EPS)
    r0 = np.linalg.norm(perp0, axis=-1) / z
    r = r0
    w = np.ones_like(r0)
    for _ in range(1 if psi == "l2" else iters):
        w = irls_weights(r, psi, c)
        r = lam / (w + lam) * r0
    return P0 - (w / (w + lam))[:, None] * perp0


def _broadcast_anchor(anchor: np.ndarray, O: np.ndarray) -> np.ndarray:
    return np.broadcast_to(anchor[:, None, :], O.shape[:2] + (3,))


def geomedian(points: np.ndarray, iters: int = IRLS_ITERS) -> np.ndarray:
    """Geometric median over axis -2 of ``points [..., k, 3]`` (Weiszfeld from the mean)."""
    y = points.mean(-2)
    for _ in range(iters):
        dist = np.linalg.norm(points - y[..., None, :], axis=-1)
        w = 1.0 / np.maximum(dist, 1e-9)
        y_new = (w[..., None] * points).sum(-2) / w.sum(-1)[..., None]
        scale = np.maximum(np.abs(y[..., 2]), _EPS)
        done = y.size == 0 or float(np.max(np.abs(y_new - y).max(-1) / scale)) < TOL
        y = y_new
        if done:
            break
    return y


def make_estimator(name: str, c: float = 1.0) -> Callable[..., np.ndarray] | None:
    """Estimator for :func:`l3d.ensemble.ensemble_errors` (``None``: the plain mean)."""
    if name in ("mean", "conf"):
        return None

    def stack(anchor: np.ndarray, O: np.ndarray) -> np.ndarray:
        return np.concatenate([_broadcast_anchor(anchor, O)[:, :, None], O], axis=2)  # [s, n, k, 3]

    if name == "geomedian":
        return lambda anchor, O, wa, W, X: geomedian(stack(anchor, O))
    if name in ("conf_select", "oracle"):
        def pick(anchor: np.ndarray, O: np.ndarray, wa: np.ndarray, W: np.ndarray, X: np.ndarray) -> np.ndarray:
            pts = stack(anchor, O)
            if name == "oracle":
                score = -np.linalg.norm(pts - X[:, None, None], axis=-1)
            else:
                score = np.concatenate([np.broadcast_to(wa[:, None, None], W.shape[:2] + (1,)), W], axis=2)
            idx = np.argmax(score, axis=2)
            return np.take_along_axis(pts, idx[..., None, None], 2)[:, :, 0]
        return pick
    if name.startswith("t0_"):
        psi, lam = parse_t0(name)
        return lambda anchor, O, wa, W, X: t0_solve(_broadcast_anchor(anchor, O), O, lam, psi, c)
    raise ValueError(f"unknown estimator '{name}'")


def median_residual(anchor: np.ndarray, others: np.ndarray, valid: np.ndarray) -> float:
    """Label-free robust scale: median of ``|a0 - b0_j| / |a0_z|`` over a builder's valid pairs."""
    z = np.maximum(np.abs(anchor[:, 2]), _EPS)[:, None]
    r = np.linalg.norm(others - anchor[:, None], axis=-1) / z
    r = r[np.asarray(valid, dtype=bool) & np.isfinite(r)]
    return float(np.median(r)) if r.size else 1.0


# --------------------------------------------------------------------------------------
# Ray center
# --------------------------------------------------------------------------------------


def rc_rows(d: dict[str, Any], scale: float, gt: bool = False, min_points: int = MIN_POINTS) -> dict[str, np.ndarray]:
    """Camera-0 outputs of design-1 queries with ``t_s = t_t > 0`` and their offsets from the
    ray-center lines (module docstring): ``rows``, ``P``, ``X``, ``perp``."""
    j, i = cross_pairs(d, "cross")
    keep = (d["q_ts"][i] == d["q_tt"][i]) & (d["q_tc"][i] == 0)
    j, i = j[keep], i[keep]
    err = row_errors(d, scale)
    P = err["P"]
    X = d["g_xyz"].astype(np.float64)
    ok = np.isfinite(P[j]).all(-1) & np.isfinite(P[i]).all(-1)  # the fit uses every output, GT or not
    j, i = j[ok], i[ok]
    perp = np.full((i.size, 3), np.nan)
    if i.size == 0:
        return {"rows": i, "P": P[i], "X": X[i], "perp": perp}
    K = np.asarray(d["K_gt"], dtype=np.float64) if gt else fit_K(d, err, "cross")
    uv = uv_px(d, i)
    tt = d["q_tt"][i].astype(np.int64)
    order = np.argsort(tt, kind="stable")
    for sel in np.split(order, np.flatnonzero(np.diff(tt[order])) + 1):
        if sel.size < min_points:
            continue
        a, b = P[j[sel]], P[i[sel]]
        if gt:
            R = geo.relative_pose(np.asarray(d["T_w2c_gt"], dtype=np.float64), int(tt[sel[0]]), 0)[:3, :3]
        else:
            R = checks.camera_swap(a, b)["R"]
        if not np.isfinite(R).all():
            continue
        dirs = geo.ray_dirs(K, uv[sel]) @ R.T
        o, _ = geo.ls_ray_center(b, dirs)
        diff = b - o
        perp[sel] = diff - (diff * dirs).sum(-1, keepdims=True) * dirs
    keep = np.isfinite(perp).all(-1) & np.isfinite(X[i]).all(-1)
    return {"rows": i[keep], "P": P[i][keep], "X": X[i][keep], "perp": perp[keep]}


def point_errors(est: np.ndarray, anchor: np.ndarray, X: np.ndarray) -> dict[str, np.ndarray]:
    """An :func:`l3d.ensemble.ensemble_errors`-style entry for one estimate per cell."""
    e = np.linalg.norm(est - X, axis=-1)
    dep, ang = ens.components(est, X)
    dep1, ang1 = ens.components(anchor, X)
    return {"cells": np.arange(est.shape[0]), "e": e, "e_sq": e**2, "apd": ens._apd(e), "dep": dep, "ang": ang,
            "e1": np.linalg.norm(anchor - X, axis=-1), "dep1": dep1, "ang1": ang1}


# --------------------------------------------------------------------------------------
# Per-clip sums
# --------------------------------------------------------------------------------------


def _slot(acc: dict[str, Any], method: str, est: str, k: int) -> dict[str, dict[str, float]]:
    node = acc.setdefault(method, {}).setdefault(est, {})
    return {g: node.setdefault(g, {}).setdefault(k, dict.fromkeys(ens.SUM_KEYS, 0.0)) for g in GROUPS}


def clip_sums(d: dict[str, Any], scale: float | None = None, ks: tuple[int, ...] = KS,
              estimators: tuple[str, ...] = B0_ESTIMATORS + T0_ESTIMATORS) -> dict[str, Any]:
    """Additive sums ``[method][estimator][group][k]`` for one decoded table, plus each method's
    robust scale under ``"_scale"``."""
    if scale is None:
        scale = table_scale(d)
    static = np.asarray(d["is_static"], dtype=bool)
    acc: dict[str, Any] = {"_scale": {}}
    vis = ens.b0_cells(d, scale, vis_filter=True)
    novis = ens.b0_cells(d, scale, vis_filter=False)
    plan: list[tuple[str, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]] = []
    for cells in (vis, ens.b0c_cells(d, scale, "target"), ens.b0c_cells(d, scale, "cam0")):
        for method, (O, W) in cells["others"].items():
            plan.append((method, cells, O, W, cells["valid"], estimators))
    O, W = novis["others"]["b0_model_u2"]
    plan.append(("b0_model_u2_novis", novis, O, W, novis["valid"], ("mean",)))
    O, W = vis["others"]["b0_model_u2"]
    for method, gate in UV_GATES.items():
        plan.append((method, vis, O, W, vis["valid"] & (vis["uv_dist"] <= gate), ("mean",)))
    sim3 = ens.b0c_cells(d, scale, "cam0", with_scale=True)
    O, W = sim3["others"]["b0c_cam0_tstar"]
    plan.append(("b0c_cam0_tstar_sim3", sim3, O, W, sim3["valid"], ("mean",)))

    for method, cells, O, W, valid, ests in plan:
        dist = np.linalg.norm(cells["X"], axis=-1)
        labels = ens.group_labels(cells, static)
        c = median_residual(cells["anchor"], O, valid)
        acc["_scale"][method] = c
        for est in ests:
            wa, wo = (cells["w_anchor"], W) if est in ("conf", "conf_select") else (None, None)
            res = ens.ensemble_errors(cells["anchor"], O, valid, cells["X"], ks, wa, wo, make_estimator(est, c))
            for k, r in res.items():
                slots = _slot(acc, method, est, k)
                for g, lab in labels.items():
                    ens.add_sums(slots[g], r, dist[r["cells"]], lab[r["cells"]])

    for method, gt in (("t0_rc_fit", False), ("t0_rc_gt", True)):
        if gt and not bool(d["has_pose_gt"]):
            continue
        rc = rc_rows(d, scale, gt=gt)
        if rc["rows"].size == 0:
            continue
        z = np.maximum(np.abs(rc["P"][:, 2]), _EPS)
        c = float(np.median(np.linalg.norm(rc["perp"], axis=-1) / z))
        acc["_scale"][method] = c
        dist = np.linalg.norm(rc["X"], axis=-1)
        cells = {"track": d["q_track"][rc["rows"]].astype(np.int64), "at_target": np.ones(rc["rows"].size, dtype=bool)}
        labels = ens.group_labels(cells, static)
        for est in estimators:
            if not est.startswith("t0_"):
                continue
            psi, lam = parse_t0(est)
            r = point_errors(line_solve(rc["P"], rc["perp"], lam, psi, c), rc["P"], rc["X"])
            slots = _slot(acc, method, est, 2)
            for g, lab in labels.items():
                ens.add_sums(slots[g], r, dist, lab)
    return acc


def add_tree(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """``a += b`` for nested sum dicts (missing keys are copied); ``"_scale"`` entries are lists."""
    for key, v in b.items():
        if key == "_scale":
            dst = a.setdefault("_scale", {})
            for m, c in v.items():
                dst.setdefault(m, []).extend(c if isinstance(c, list) else [c])
        elif isinstance(v, dict):
            add_tree(a.setdefault(key, {}), v)
        else:
            a[key] = a.get(key, 0.0) + v
    return a


def pool(sums: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in sums:
        add_tree(out, s)
    return out
