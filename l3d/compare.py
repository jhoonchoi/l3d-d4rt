"""Stage 1 analyses on the point table (``docs/l3d/STAGE1_PLAN.md``, sections 4 and 5).

Q1 (complementarity), Q2 (conditions) and Q3 (error type) for every pair of models in the point table.
Units are paired: a unit counts only where every model has a finite error. Clip-level numbers follow the
upstream protocol (mean over clips of the clip's mean score); bin-level numbers pool units inside a
subset, as in ``results/l3d/splits``. CIs come from a paired bootstrap over clips within a subset.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy.stats import spearmanr

N_BOOT = 2000
FAIL_SHARE = 0.2
MIN_CLIPS, MIN_UNITS = 5, 1000
MOTION = {"all": None, "static": True, "dynamic": False}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def paired_mask(P: dict[str, np.ndarray], models: list[str]) -> np.ndarray:
    return np.logical_and.reduce([np.isfinite(P[f"d_{m}"]) for m in models])


def motion_mask(P: dict[str, np.ndarray], motion: str) -> np.ndarray:
    want = MOTION[motion]
    return np.ones(P["t"].shape, bool) if want is None else (P["static"] == want)


def clip_means(clip: np.ndarray, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(clip ids, mean of values per clip)`` over masked units."""
    c = clip[mask]
    ids, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=values[mask].astype(np.float64))
    return ids, sums / np.bincount(inv)


def boot_ci(per_clip: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of the mean over clips (clip resampling)."""
    x = np.asarray(per_clip, dtype=np.float64)
    if x.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (N_BOOT, x.size))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio_ci(num: np.ndarray, den: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% CI of ``sum(num) / sum(den)`` with clip resampling (``num``/``den`` per clip)."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    if num.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, (N_BOOT, num.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num[idx].sum(1) / den[idx].sum(1)
    return float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def verdict(diff: float, ci: tuple[float, float], min_effect: float = 0.02) -> str:
    """'real' if the CI excludes 0 and |diff| >= min_effect (plan section 5)."""
    if not np.all(np.isfinite(ci)):
        return "n/a"
    if (ci[0] > 0 or ci[1] < 0) and abs(diff) >= min_effect:
        return "real"
    return "no"


# --------------------------------------------------------------------------------------
# Q1: complementarity
# --------------------------------------------------------------------------------------


def _track_oracles(P: dict[str, np.ndarray], mask: np.ndarray, a: str, b: str) -> dict[str, np.ndarray]:
    """Per-unit chosen score of the per-track oracle and its even/odd cross-validated version."""
    key = P["clip"][mask].astype(np.int64) * 1_000_000 + P["track"][mask].astype(np.int64)
    _, inv = np.unique(key, return_inverse=True)
    sa, sb = P[f"s_{a}"][mask].astype(np.float64), P[f"s_{b}"][mask].astype(np.float64)
    da, db = P[f"d_{a}"][mask].astype(np.float64), P[f"d_{b}"][mask].astype(np.float64)
    odd = (P["t"][mask] % 2).astype(bool)
    k = inv.max() + 1

    def pick(sel: np.ndarray) -> np.ndarray:
        """True where model a is chosen for the track, from units in ``sel`` (mean s, then mean d)."""
        n = np.bincount(inv[sel], minlength=k)
        ms = (np.bincount(inv[sel], weights=sa[sel] - sb[sel], minlength=k)) / np.maximum(n, 1)
        md = (np.bincount(inv[sel], weights=da[sel] - db[sel], minlength=k)) / np.maximum(n, 1)
        return np.where(ms != 0, ms > 0, md <= 0)

    full = pick(np.ones_like(odd))
    from_even, from_odd = pick(~odd), pick(odd)
    cv_choose_a = np.where(odd, from_even[inv], from_odd[inv])
    return {"track": np.where(full[inv], sa, sb), "track_cv": np.where(cv_choose_a, sa, sb)}


def q1_pair(P: dict[str, np.ndarray], a: str, b: str, motion: str) -> dict[str, Any] | None:
    mask = paired_mask(P, [a, b]) & motion_mask(P, motion)
    if mask.sum() < MIN_UNITS:
        return None
    clip = P["clip"]
    ids, apd_a = clip_means(clip, P[f"s_{a}"], mask)
    _, apd_b = clip_means(clip, P[f"s_{b}"], mask)
    if ids.size < MIN_CLIPS:
        return None
    sa, sb = P[f"s_{a}"][mask].astype(np.float64), P[f"s_{b}"][mask].astype(np.float64)
    da, db = P[f"d_{a}"][mask].astype(np.float64), P[f"d_{b}"][mask].astype(np.float64)
    c = clip[mask]
    # correlation
    rho_clip = []
    for i in ids:
        sel = c == i
        if sel.sum() >= 20:
            rho_clip.append(spearmanr(da[sel], db[sel]).statistic)
    rho_pooled = float(spearmanr(da, db).statistic)
    # failure overlap (worst 20% of the subset/motion cell, per model)
    fa = da >= np.quantile(da, 1 - FAIL_SHARE)
    fb = db >= np.quantile(db, 1 - FAIL_SHARE)
    both = float((fa & fb).mean())
    jacc = float((fa & fb).sum() / max((fa | fb).sum(), 1))
    # oracles
    tr = _track_oracles(P, mask, a, b)
    sub = np.zeros(clip.shape, np.float64)
    per = {}
    for name, vals in (("unit", np.maximum(sa, sb)), ("track", tr["track"]), ("track_cv", tr["track_cv"])):
        sub[mask] = vals
        per[name] = clip_means(clip, sub, mask)[1]
    per["clip"] = np.maximum(apd_a, apd_b)
    best_single = apd_a if apd_a.mean() >= apd_b.mean() else apd_b
    best_name = a if apd_a.mean() >= apd_b.mean() else b
    out: dict[str, Any] = {
        "units": int(mask.sum()), "clips": int(ids.size),
        "apd": {a: float(apd_a.mean()), b: float(apd_b.mean())},
        "delta_apd": float((apd_a - apd_b).mean()), "delta_ci": boot_ci(apd_a - apd_b),
        "rho_clip_median": float(np.nanmedian(rho_clip)) if rho_clip else float("nan"), "rho_pooled": rho_pooled,
        "both_fail": both, "both_fail_if_independent": FAIL_SHARE**2, "jaccard_fail": jacc,
        "best_single": best_name, "oracle": {}, "all_model": {},
    }
    for name, v in per.items():
        gain = v - best_single
        out["oracle"][name] = {"apd": float(v.mean()), "gain": float(gain.mean()), "gain_ci": boot_ci(gain)}
    ok_a, ok_b = sa == 1.0, sb == 1.0
    out["all_model"] = {"only_" + a: float((ok_a & ~ok_b).mean()), "only_" + b: float((ok_b & ~ok_a).mean()),
                        "both": float((ok_a & ok_b).mean()), "neither": float((~ok_a & ~ok_b).mean())}
    out["delta_verdict"] = verdict(out["delta_apd"], out["delta_ci"])
    return out


def q1(points: dict[str, dict[str, np.ndarray]], models: list[str]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for a, b in itertools.combinations(models, 2):
        res[f"{a}|{b}"] = {s: {mo: q1_pair(P, a, b, mo) for mo in MOTION} for s, P in points.items()}
    return res


# --------------------------------------------------------------------------------------
# Q2: conditions
# --------------------------------------------------------------------------------------

GAP_EDGES = (0, 4, 8, 16, 32, 48)
PARALLAX_EDGES = (0.0, 1.0, 5.0)
DEPTH_EDGES = (0.0, 0.5, 1.0, 2.0)
BORDER_EDGES = (0.0, 0.05)


def _edges_bins(x: np.ndarray, edges: tuple[float, ...], fmt: str = "{:g}") -> tuple[np.ndarray, list[str]]:
    idx = np.digitize(x, edges[1:])
    names = [f"[{fmt.format(lo)}, {fmt.format(hi)})" for lo, hi in zip(edges[:-1], edges[1:])] + [f">= {fmt.format(edges[-1])}"]
    idx = np.where(np.isfinite(x), idx, -1)
    used = sorted(set(idx[idx >= 0].tolist()))
    return idx, names[: max(used) + 1] if used else names


def condition_bins(P: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Bin index per unit (``-1`` = no label) and bin names for every Q2 label of one subset."""
    out: dict[str, tuple[np.ndarray, list[str]]] = {}
    out["motion"] = (np.where(P["static"], 0, 1), ["static", "dynamic"])
    out["visibility"] = (np.where(P["vis"], 0, 1), ["visible", "occluded"])
    gap = np.digitize(P["t"], GAP_EDGES[1:-1])
    out["frame gap"] = (gap, [f"[{lo}, {hi})" for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:])])
    out["parallax (deg)"] = _edges_bins(P["parallax"].astype(np.float64), PARALLAX_EDGES)
    # speed: static units are their own bin, dynamic units in quartiles of this subset
    dyn = ~P["static"] & np.isfinite(P["speed"])
    q = np.quantile(P["speed"][dyn], [0.25, 0.5, 0.75]) if dyn.any() else np.array([np.inf] * 3)
    sp = np.where(P["static"], 0, 1 + np.digitize(P["speed"], q))
    out["speed"] = (np.where(np.isfinite(P["speed"]), sp, -1), ["static", "dyn Q1", "dyn Q2", "dyn Q3", "dyn Q4"])
    out["depth / median"] = _edges_bins(P["depth_rel"].astype(np.float64), DEPTH_EDGES)
    for key, name in (("cam_rot", "camera rotation"), ("cam_trans", "camera translation")):
        x = P[key].astype(np.float64)
        e = np.unique(np.quantile(x[np.isfinite(x)], [0.25, 0.5, 0.75]))
        if x[np.isfinite(x)].max() - x[np.isfinite(x)].min() < 1e-9:
            out[name] = (np.zeros(x.shape, int), ["fixed camera"])
            continue
        # right=True puts ties at an edge (e.g. many exact zeros) in the lower bin, so no bin is empty
        idx = np.digitize(x, e, right=True)
        labels = [f"Q{i + 1} (≤ {e[i]:.3g})" for i in range(e.size)] + [f"Q{e.size + 1} (> {e[-1]:.3g})"]
        out[name] = (np.where(np.isfinite(x), idx, -1), labels)
    b = P["border"].astype(np.float64)
    out["image position"] = (np.where(b < 0, 0, np.where(b < 0.05, 1, 2)), ["off-image", "border (< 5%)", "inside"])
    return out


def q2_pair(P: dict[str, np.ndarray], a: str, b: str) -> dict[str, list[dict[str, Any]]]:
    mask = paired_mask(P, [a, b])
    sa, sb = P[f"s_{a}"].astype(np.float64), P[f"s_{b}"].astype(np.float64)
    clip = P["clip"].astype(np.int64)
    ids = np.unique(clip[mask])
    cidx = np.searchsorted(ids, clip)
    nc = ids.size
    res: dict[str, list[dict[str, Any]]] = {}
    for label, (idx, names) in condition_bins(P).items():
        rows = []
        for k, name in enumerate(names):
            sel = mask & (idx == k)
            n = int(sel.sum())
            cl = np.bincount(cidx[sel], minlength=nc)
            n_clips = int((cl > 0).sum())
            row: dict[str, Any] = {"bin": name, "units": n, "clips": n_clips}
            if n < MIN_UNITS or n_clips < MIN_CLIPS:
                row["na"] = True
                rows.append(row)
                continue
            sum_a = np.bincount(cidx[sel], weights=sa[sel], minlength=nc)
            sum_b = np.bincount(cidx[sel], weights=sb[sel], minlength=nc)
            win = np.bincount(cidx[sel], weights=(sa[sel] > sb[sel]).astype(float), minlength=nc)
            lose = np.bincount(cidx[sel], weights=(sa[sel] < sb[sel]).astype(float), minlength=nc)
            row.update({
                f"apd_{a}": float(sum_a.sum() / n), f"apd_{b}": float(sum_b.sum() / n),
                "delta": float((sum_a.sum() - sum_b.sum()) / n), "delta_ci": boot_ratio_ci(sum_a - sum_b, cl),
                "win_rate": float(win.sum() / max(win.sum() + lose.sum(), 1)),
                "win_ci": boot_ratio_ci(win, win + lose), "ties": float(1 - (win.sum() + lose.sum()) / n),
            })
            row["verdict"] = verdict(row["delta"], row["delta_ci"])
            rows.append(row)
        res[label] = rows
    return res


REG_LABELS = ("visibility", "frame gap", "speed", "depth / median", "camera rotation", "camera translation", "image position")


def q2_regression(P: dict[str, np.ndarray], a: str, b: str) -> list[dict[str, Any]]:
    """OLS of ``s_a - s_b`` on bin indicators of the Q2 labels (first bin of each label = reference).

    Parallax is left out: inside a clip it is a function of the camera motion and the point. Motion is
    the "static" level of speed. CIs come from resampling clips (sums of ``X'X`` and ``X'y`` per clip).
    """
    mask = paired_mask(P, [a, b])
    bins = condition_bins(P)
    cols, names = [np.ones(mask.sum())], ["intercept"]
    keep = mask.copy()
    for lab in REG_LABELS:
        keep &= bins[lab][0] >= 0
    for lab in REG_LABELS:
        idx, bn = bins[lab]
        for k in range(1, len(bn)):
            col = (idx[mask] == k).astype(np.float64)
            if col.sum() >= MIN_UNITS:
                cols.append(col)
                names.append(f"{lab}: {bn[k]}")
    X = np.stack(cols, 1)[keep[mask]]
    y = (P[f"s_{a}"] - P[f"s_{b}"]).astype(np.float64)[keep]
    clip = P["clip"][keep]
    ids, inv = np.unique(clip, return_inverse=True)
    p = X.shape[1]
    XtX = np.zeros((ids.size, p, p))
    Xty = np.zeros((ids.size, p))
    for i in range(ids.size):
        sel = inv == i
        XtX[i] = X[sel].T @ X[sel]
        Xty[i] = X[sel].T @ y[sel]
    beta = np.linalg.lstsq(XtX.sum(0), Xty.sum(0), rcond=None)[0]
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(500):
        pick = rng.integers(0, ids.size, ids.size)
        boots.append(np.linalg.lstsq(XtX[pick].sum(0), Xty[pick].sum(0), rcond=None)[0])
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    return [{"term": n, "coef": float(c), "ci": (float(l), float(h)), "verdict": verdict(float(c), (float(l), float(h)))}
            for n, c, l, h in zip(names, beta, lo, hi)]


def q2_fixed_camera_test(points: dict[str, dict[str, np.ndarray]], a: str, b: str) -> list[dict[str, Any]]:
    """ΔAPD (a - b) by total camera motion of the clip, over the moving-camera subsets (plan 4.2, item 3)."""
    rows = []
    for s, P in points.items():
        mask = paired_mask(P, [a, b])
        if np.nanmax(P["cam_trans"]) < 1e-9:
            continue
        ids, apd_a = clip_means(P["clip"], P[f"s_{a}"], mask)
        _, apd_b = clip_means(P["clip"], P[f"s_{b}"], mask)
        _, rot = clip_means(P["clip"], P["cam_rot"], mask)
        _, tra = clip_means(P["clip"], P["cam_trans"], mask)
        for i, c in enumerate(ids):
            sel = P["clip"] == c
            still = bool(P["cam_trans"][sel].max() < 1e-6 and P["cam_rot"][sel].max() < 1e-3)
            rows.append({"subset": s, "clip": int(c), "delta": float(apd_a[i] - apd_b[i]), "still_camera": still,
                         "cam_rot_mean": float(rot[i]), "cam_trans_mean": float(tra[i])})
    return rows


# --------------------------------------------------------------------------------------
# Q3: error type
# --------------------------------------------------------------------------------------

TIER_A_PARTS = ("gauge", "offset", "jitter", "resid")
TIER_B_PARTS = ("point_sq", "pose_sq", "cross")


def structure_parts(P: dict[str, np.ndarray], model: str, mask: np.ndarray) -> dict[str, float]:
    """Nested Tier A split of the mean of ``|e|^2`` over the units in ``mask``.

    Per clip, one least-squares Sim(3) of the (scaled) prediction ``p = g + e`` onto ``g`` is fit on the
    selected units; **gauge** is what it removes. Of the remaining error ``r``, per clip and GT motion
    group: **offset** = mean of ``r`` over the group's units, **jitter** = per-frame group mean minus the
    offset, **resid** = ``r`` minus the per-frame group mean. Fit on exactly the selected units, every
    part is non-negative and the four parts add up to the mean of ``|e|^2``.
    """
    from l3d import geometry as geo

    e = P[f"e_{model}"][mask].astype(np.float64)
    g = P["g0"][mask].astype(np.float64)
    clip = P["clip"][mask].astype(np.int64)
    n = e.shape[0]
    r = np.empty_like(e)
    for c in np.unique(clip):
        sel = clip == c
        s, R, t = geo.umeyama(g[sel] + e[sel], g[sel], with_scale=True)
        r[sel] = e[sel] if not np.isfinite(s) else s * (g[sel] + e[sel]) @ R.T + t - g[sel]
    grp = clip * 100_000 + P["group"][mask].astype(np.int64)
    _, gi = np.unique(grp, return_inverse=True)
    _, gti = np.unique(grp * 1000 + P["t"][mask].astype(np.int64), return_inverse=True)

    def means(idx: np.ndarray) -> np.ndarray:
        cnt = np.bincount(idx)
        return np.stack([np.bincount(idx, weights=r[:, k]) / cnt for k in range(3)], 1)

    offset = means(gi)[gi]
    m_t = means(gti)[gti]
    parts = {"gauge": float(((e**2).sum() - (r**2).sum()) / n), "offset": float((offset**2).sum() / n),
             "jitter": float(((m_t - offset) ** 2).sum() / n), "resid": float(((r - m_t) ** 2).sum() / n)}
    return parts


def q3_cell(P: dict[str, np.ndarray], models: list[str], motion: str, d_max: float | None = None,
            px_max: float | None = None) -> dict[str, Any] | None:
    """Mean squared error parts per model in one (subset, motion) cell, on paired units.

    ``d_max`` keeps only units where every model's error is at most ``d_max`` (the APD range is 1.0).
    ``px_max`` keeps only ``t > 0`` units where every Tier B model's predicted pixel is within ``px_max``
    (256-px units) of the GT pixel, so depth error there is not caused by a wrong correspondence.
    """
    mask = paired_mask(P, models) & motion_mask(P, motion)
    if d_max is not None:
        for m in models:
            mask &= P[f"d_{m}"] <= d_max
    if px_max is not None:
        mask &= P["t"] > 0
        for m in models:
            if f"px_{m}" in P:
                mask &= P[f"px_{m}"] <= px_max
    if mask.sum() < MIN_UNITS:
        return None
    out: dict[str, Any] = {"units": int(mask.sum())}
    for m in models:
        d2 = float(np.mean(P[f"d_{m}"][mask].astype(np.float64) ** 2))
        r = {"mse": d2, "median_d": float(np.median(P[f"d_{m}"][mask]))}
        r["tierA"] = structure_parts(P, m, mask)
        if f"point_sq_{m}" in P:
            r["tierB"] = {k: float(np.mean(P[f"{k}_{m}"][mask].astype(np.float64)))
                          for k in TIER_B_PARTS + ("radial_sq", "perp_sq")}
            later = mask & (P["t"] > 0) & np.isfinite(P[f"px_{m}"])
            r["px_median"] = float(np.median(P[f"px_{m}"][later])) if later.any() else float("nan")
        out[m] = r
    return out


def q3(points: dict[str, dict[str, np.ndarray]], models: list[str]) -> dict[str, Any]:
    return {s: {mo: {"all_units": q3_cell(P, models, mo), "d_le_1": q3_cell(P, models, mo, d_max=1.0),
                     "px_le_2": q3_cell(P, models, mo, d_max=1.0, px_max=2.0)}
                for mo in MOTION} for s, P in points.items()}
