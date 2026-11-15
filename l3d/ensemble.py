"""B0 and B0-C: test-time ensembling over source frames and over camera frames.

The proposal's Part B baselines: "B0: test-time ensembling over k source frames; B0-C:
the same over camera frames, mapped by T*". T0 counts only if it beats them at matched k.
Both are computed from a decoded table without new decodes, so k is capped by the dump:
design 1 decodes each (track, target frame) from up to 6 source frames and in up to 6
camera frames, so k <= 6 here (k = 8 needs a new dump).

**B0** (source ensemble), one cell per (track ``n``, target ``t3``, camera ``c``). The query
is the design-4/5 anchor ``(u1, t1)`` (``t1 = t3`` if it is a source, else the nearest
source) and its single output is the design-1 row ``P(u1, t1, t3, c)``. The ensemble adds
``k - 1`` of the cell's other source frames ``t2``, decoded from

- ``b0_model_u2``: the model's own correspondence ``u2 = uv_hat(u1, t1, t2, t2)`` (design
  5), kept only where the model calls the point visible at ``t2`` (the source-swap check's
  ``u2``); this is the label-free B0;
- ``b0_gt_u2``: the GT pixel of track ``n`` at ``t2`` (design 1), a control with perfect
  correspondences, on the same ``t2``.

**B0-C** (camera ensemble), one cell per design-1 query ``(n, t_s, t_t)``. The single output
is the query decoded in a reference camera; the ensemble adds ``k - 1`` of the query's
other camera frames ``c``, each mapped to the reference by

- ``*_tstar``: ``T*``, the SE(3) Umeyama fit per ``(t_t, c)`` from every design-1 output of
  the clip decoded in both cameras (label-free; the camera-swap check's ``T*``);
- ``*_gt_pose``: the GT relative pose, a control, on the same cameras.

Reference ``b0c_*``: camera ``t_t`` (the self-frame output, which has no camera-frame
error); ``b0c_cam0_*``: camera 0, the clip's world frame (a cross-frame output for
``t_t > 0``, where the camera checks apply; ``t_t = 0`` is left out).

For each ``k`` the error is averaged over every size-``(k - 1)`` subset of the cell's other
frames (the expected error of a random pick) and compared with ``k = 1`` on the same
cells. Aggregators: ``mean`` and ``conf``, weighted by ``exp(confidence logit)`` (a softmax
over the ensemble; ``sigmoid(confidence)`` would not do: the released model's logits are
7.5-12, so it is 0.9998-1 everywhere and gives the plain mean). Errors are
taken in the reference camera after the clip's global median scale: EPE (m), relative
error ``|P - X| / |X|`` and APD over the WorldTrack thresholds. :func:`clip_sums` returns
additive sums, so clips and subsets pool by addition.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from l3d import geometry as geo
from l3d.check_rows import MIN_POINTS, cross_pairs
from l3d.decompose import table_scale
from l3d.gauge import THRESHOLDS
from l3d.table import DESIGNS, source_swap_rows

KS = (2, 4, 6)
METHODS = ("b0_model_u2", "b0_gt_u2", "b0c_tstar", "b0c_gt_pose", "b0c_cam0_tstar", "b0c_cam0_gt_pose")
AGGREGATORS = ("mean", "conf")
GROUPS = ("all", "static", "dynamic", "query_at_target", "query_elsewhere")
SUM_KEYS = ("n", "e", "rel", "rel_sq", "apd", "e1", "rel1", "rel1_sq", "apd1", "better")


def conf_weight(logit: np.ndarray) -> np.ndarray:
    """Ensemble weight ``exp(logit)`` (softmax once normalized; clipped to avoid overflow)."""
    return np.exp(np.clip(np.asarray(logit, dtype=np.float64), -50.0, 50.0))


def _apd(e: np.ndarray) -> np.ndarray:
    """Mean over the WorldTrack thresholds of ``e <= th`` (last axis added)."""
    return (np.asarray(e)[..., None] <= np.asarray(THRESHOLDS)).mean(-1)


def pad_groups(cell: np.ndarray, n_cells: int, *values: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Scatter member rows into ``[n_cells, m_max, ...]`` slots by ``cell``; returns the slot mask and arrays.

    Float arrays are NaN on empty slots, others 0. Members keep their input order within a cell.
    """
    cell = np.asarray(cell, dtype=np.int64)
    order = np.argsort(cell, kind="stable")
    cs = cell[order]
    counts = np.bincount(cs, minlength=n_cells) if n_cells else np.zeros(0, dtype=np.int64)
    m_max = int(counts.max()) if counts.size else 0
    start = np.r_[0, np.cumsum(counts)[:-1]] if counts.size else np.zeros(0, dtype=np.int64)
    rank = np.arange(cs.size) - start[cs] if cs.size else np.zeros(0, dtype=np.int64)
    mask = np.zeros((n_cells, m_max), dtype=bool)
    mask[cs, rank] = True
    out = []
    for v in values:
        v = np.asarray(v)
        fill = np.nan if v.dtype.kind == "f" else 0
        arr = np.full((n_cells, m_max) + v.shape[1:], fill, dtype=v.dtype)
        arr[cs, rank] = v[order]
        out.append(arr)
    return mask, out


def ensemble_errors(
    anchor: np.ndarray,
    others: np.ndarray,
    valid: np.ndarray,
    X: np.ndarray,
    ks: tuple[int, ...] = KS,
    w_anchor: np.ndarray | None = None,
    w_others: np.ndarray | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Error of the ``k``-ensemble (anchor + ``k - 1`` valid others), averaged over all subsets.

    ``anchor [C, 3]``, ``others [C, m, 3]``, ``valid [C, m]``, ``X [C, 3]`` (GT in the same
    camera); weights default to 1. Per ``k``: ``cells`` (indices with at least ``k - 1`` valid
    others), ``e`` and ``e_sq`` (means over subsets of ``|ensemble - X|`` and its square),
    ``apd`` (mean over subsets) and ``e1`` (the anchor's own error) on those cells.
    """
    anchor = np.asarray(anchor, dtype=np.float64)
    others = np.asarray(others, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    C, m = others.shape[:2]
    wa = np.ones(C) if w_anchor is None else np.asarray(w_anchor, dtype=np.float64)
    wo = np.ones((C, m)) if w_others is None else np.asarray(w_others, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(others).all(-1) & np.isfinite(wo)
    cnt = valid.sum(1)
    order = np.argsort(~valid, axis=1, kind="stable")  # valid others first
    others = np.take_along_axis(others, order[..., None], 1)
    wo = np.take_along_axis(wo, order, 1)
    e1_all = np.linalg.norm(anchor - X, axis=-1)
    out: dict[int, dict[str, np.ndarray]] = {}
    for k in ks:
        parts: list[tuple[np.ndarray, ...]] = []
        for mc in range(k - 1, m + 1):
            sel = np.flatnonzero(cnt == mc)
            if sel.size == 0:
                continue
            combos = np.asarray(list(combinations(range(mc), k - 1)), dtype=np.int64).reshape(-1, k - 1)
            O = others[sel][:, combos]  # [s, n_comb, k - 1, 3]
            W = wo[sel][:, combos]  # [s, n_comb, k - 1]
            num = (wa[sel, None, None] * anchor[sel, None]) + (W[..., None] * O).sum(2)
            ens = num / (wa[sel, None] + W.sum(2))[..., None]
            e = np.linalg.norm(ens - X[sel, None], axis=-1)  # [s, n_comb]
            parts.append((sel, e.mean(1), (e**2).mean(1), _apd(e).mean(1)))
        if parts:
            cells, e_k, esq_k, apd_k = (np.concatenate([p[i] for p in parts]) for i in range(4))
        else:
            cells, e_k, esq_k, apd_k = np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0)
        out[k] = {"cells": cells, "e": e_k, "e_sq": esq_k, "apd": apd_k, "e1": e1_all[cells]}
    return out


def b0_cells(d: dict[str, Any], scale: float, vis_filter: bool = True) -> dict[str, Any]:
    """B0 cells of a full table: anchor rows and, per cell, the other sources' outputs (both variants)."""
    rows = source_swap_rows(d)
    a, b, g, parent = rows["a"], rows["b"], rows["g"], rows["parent"]
    P = d["p_xyz"].astype(np.float64) * scale
    X = d["g_xyz"].astype(np.float64)
    conf = conf_weight(d["p_conf_logit"])
    keep = (np.isfinite(P[a]).all(-1) & np.isfinite(X[a]).all(-1) & np.isfinite(P[b]).all(-1)
            & np.isfinite(P[g]).all(-1))
    if vis_filter:
        keep &= d["p_vis_logit"][parent] > 0  # the model sees the point at t2
    a, b, g = a[keep], b[keep], g[keep]
    cells, inv = np.unique(a, return_inverse=True)
    mask, (Pb, Pg, cb, cg) = pad_groups(inv.reshape(-1), cells.size, P[b], P[g], conf[b], conf[g])
    return {"anchor": P[cells], "X": X[cells], "w_anchor": conf[cells], "track": d["q_track"][cells].astype(np.int64),
            "at_target": d["q_ts"][cells] == d["q_tt"][cells],
            "valid": mask, "others": {"b0_model_u2": (Pb, cb), "b0_gt_u2": (Pg, cg)}}


def ref_pairs(d: dict[str, Any], ref: str = "target") -> tuple[np.ndarray, np.ndarray]:
    """Design-1 row pairs ``(j, i)``: ``j`` a query decoded in the reference camera, ``i`` the same
    query (track, ``t_s``, ``t_t``) decoded in another camera.

    ``ref="target"``: the reference is camera ``t_t`` (the self-frame output). ``ref="cam0"``:
    camera 0, the clip's world frame; queries with ``t_t = 0`` are left out (camera 0 is their own).
    """
    if ref == "target":
        return cross_pairs(d, "cross")
    if ref != "cam0":
        raise ValueError(f"unknown reference '{ref}'")
    rows = np.flatnonzero((d["design"] == DESIGNS["cross"]) & (d["q_tt"] != 0))
    T = int(d["T"])
    n, s, t, c = (d[k][rows].astype(np.int64) for k in ("q_track", "q_ts", "q_tt", "q_tc"))
    key = ((n + 1) * T + s) * T + t
    is_ref = c == 0
    order = np.argsort(key[is_ref], kind="stable")
    ref_key, ref_rows = key[is_ref][order], rows[is_ref][order]
    want, other = key[~is_ref], rows[~is_ref]
    if ref_key.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos = np.clip(np.searchsorted(ref_key, want), 0, ref_key.size - 1)
    hit = ref_key[pos] == want
    return ref_rows[pos[hit]], other[hit]


def b0c_cells(d: dict[str, Any], scale: float, ref: str = "target", min_points: int = MIN_POINTS) -> dict[str, Any]:
    """B0-C cells of a full table: reference-camera rows (see :func:`ref_pairs`) and the same
    queries in other cameras, mapped to the reference by ``T*`` and by the GT pose."""
    j, i = ref_pairs(d, ref)
    P = d["p_xyz"].astype(np.float64) * scale
    X = d["g_xyz"].astype(np.float64)
    conf = conf_weight(d["p_conf_logit"])
    ok = np.isfinite(P[j]).all(-1) & np.isfinite(P[i]).all(-1) & np.isfinite(X[j]).all(-1)
    j, i = j[ok], i[ok]
    tt, tc, rc = (d[k][r].astype(np.int64) for k, r in (("q_tt", i), ("q_tc", i), ("q_tc", j)))
    mapped_star = np.full((i.size, 3), np.nan)
    if i.size:
        _, pair = np.unique(np.stack([tt, tc], -1), axis=0, return_inverse=True)
        pair = pair.reshape(-1)
        order = np.argsort(pair, kind="stable")
        for sel in np.split(order, np.flatnonzero(np.diff(pair[order])) + 1):
            if sel.size < min_points:
                continue
            s, R, t = geo.umeyama(P[i[sel]], P[j[sel]], with_scale=False)  # camera c -> reference camera
            if np.isfinite(s):
                mapped_star[sel] = geo.apply_sim3(P[i[sel]], s, R, t)
    T_rel = geo.relative_pose(np.asarray(d["T_w2c_gt"], dtype=np.float64), tc, rc)  # camera c -> reference camera
    mapped_gt = geo.apply_transform(T_rel, P[i])
    cells, inv = np.unique(j, return_inverse=True)
    mask, (Ms, Mg, ci) = pad_groups(inv.reshape(-1), cells.size, mapped_star, mapped_gt, conf[i])
    mask &= np.isfinite(Ms).all(-1)  # both variants use the cameras that have a T*
    prefix = "b0c" if ref == "target" else "b0c_cam0"
    return {"anchor": P[cells], "X": X[cells], "w_anchor": conf[cells], "track": d["q_track"][cells].astype(np.int64),
            "at_target": d["q_ts"][cells] == d["q_tt"][cells],
            "valid": mask, "others": {f"{prefix}_tstar": (Ms, ci), f"{prefix}_gt_pose": (Mg, ci)}}


def empty_sums(ks: tuple[int, ...] = KS) -> dict[str, Any]:
    return {m: {a: {g: {k: dict.fromkeys(SUM_KEYS, 0.0) for k in ks} for g in GROUPS} for a in AGGREGATORS}
            for m in METHODS}


def clip_sums(d: dict[str, Any], scale: float | None = None, ks: tuple[int, ...] = KS,
              vis_filter: bool = True) -> dict[str, Any]:
    """Additive sums ``[method][aggregator][group][k]`` for one decoded table (see the module docstring)."""
    if scale is None:
        scale = table_scale(d)
    static = np.asarray(d["is_static"], dtype=bool)
    acc = empty_sums(ks)
    for cells in (b0_cells(d, scale, vis_filter), b0c_cells(d, scale, "target"), b0c_cells(d, scale, "cam0")):
        dist = np.linalg.norm(cells["X"], axis=-1)
        st, at = static[cells["track"]], np.asarray(cells["at_target"], dtype=bool)
        labels = {"all": np.ones(st.shape, dtype=bool), "static": st, "dynamic": ~st,
                  "query_at_target": at, "query_elsewhere": ~at}
        for method, (O, W) in cells["others"].items():
            for agg in AGGREGATORS:
                wa, wo = (None, None) if agg == "mean" else (cells["w_anchor"], W)
                res = ensemble_errors(cells["anchor"], O, cells["valid"], cells["X"], ks, wa, wo)
                for k, r in res.items():
                    dd = dist[r["cells"]]
                    for group, lab in labels.items():
                        m = lab[r["cells"]]
                        e, e1, dm = r["e"][m], r["e1"][m], dd[m]
                        vals = {"n": m.sum(), "e": e.sum(), "rel": (e / dm).sum(), "rel_sq": (r["e_sq"][m] / dm**2).sum(),
                                "apd": r["apd"][m].sum(), "e1": e1.sum(), "rel1": (e1 / dm).sum(),
                                "rel1_sq": ((e1 / dm) ** 2).sum(), "apd1": _apd(e1).sum(), "better": (e < e1).sum()}
                        s = acc[method][agg][group][k]
                        for key, v in vals.items():
                            s[key] += float(v)
    return acc


def pool(sums: list[dict[str, Any]], ks: tuple[int, ...] = KS) -> dict[str, Any]:
    """Add per-clip sums."""
    out = empty_sums(ks)
    for s in sums:
        for m in METHODS:
            for a in AGGREGATORS:
                for g in GROUPS:
                    for k in ks:
                        for key in SUM_KEYS:
                            out[m][a][g][k][key] += s[m][a][g][k][key]
    return out


def summarize_cell(s: dict[str, float]) -> dict[str, float]:
    """Metrics of one pooled ``[method][aggregator][group][k]`` entry (NaN when empty)."""
    n = s["n"]

    def r(x: float) -> float:
        return x / n if n > 0 else float("nan")

    rel, rel1 = r(s["rel"]), r(s["rel1"])
    ok = n > 0 and rel1 > 0
    return {"n_cells": int(n), "epe": r(s["e"]), "epe_1": r(s["e1"]), "rel": rel, "rel_1": rel1,
            "apd": r(s["apd"]), "apd_1": r(s["apd1"]),
            "gain": 1.0 - rel / rel1 if ok else float("nan"),
            "gain_mse": 1.0 - s["rel_sq"] / s["rel1_sq"] if ok and s["rel1_sq"] > 0 else float("nan"),
            "frac_better": r(s["better"])}


def bootstrap_gain(rel: np.ndarray, rel1: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of ``1 - sum(rel) / sum(rel1)`` over clips resampled with replacement."""
    rel, rel1 = np.asarray(rel, dtype=np.float64), np.asarray(rel1, dtype=np.float64)
    if rel.size == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, rel.size, size=(n_boot, rel.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = 1.0 - rel[idx].sum(1) / rel1[idx].sum(1)
    lo, hi = np.nanpercentile(g, [2.5, 97.5])
    return float(lo), float(hi)
