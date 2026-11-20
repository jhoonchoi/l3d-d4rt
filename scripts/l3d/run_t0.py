#!/usr/bin/env python
"""T0 vs B0 / B0-C at matched k, and B0 ablations; writes results/l3d/t0/t0.{json,md}.

Per clip, :func:`l3d.t0.clip_sums` runs every estimator (B0 mean and its ablations, the T0
sweep over penalty x proximity weight) on the B0 / B0-C cells of the existing dump, plus T0
with the ray-center check. Clips run in parallel; their sums are cached in
``output/l3d/t0/per_clip.pkl`` (``--from-cache`` re-renders the report from it). The T0
setting is chosen by 2-fold cross-validation over clips (even / odd clip of the subset's
sorted list): pick the setting with the best pooled gain on one fold, score it on the other.
CPU only.

Example:
    .venv/bin/python scripts/l3d/run_t0.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import ensemble as ens  # noqa: E402
from l3d import t0  # noqa: E402
from l3d.report import SUBSETS, fmt, jsonable, model_label  # noqa: E402
from l3d.table import load_table  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
DEFAULT_CACHE = REPO / "output/l3d/t0/per_clip.pkl"
K_MAIN = 4
NAMES = {
    "b0_model_u2": "source swap, model u2",
    "b0_gt_u2": "source swap, GT u2 (control)",
    "b0c_tstar": "camera swap to t_t, T*",
    "b0c_gt_pose": "camera swap to t_t, GT pose (control)",
    "b0c_cam0_tstar": "camera swap to cam 0, T*",
    "b0c_cam0_gt_pose": "camera swap to cam 0, GT pose (control)",
    "b0_model_u2_novis": "source swap, model u2, no visibility filter",
    "b0_model_u2_uv2": "source swap, model u2, 2D agreement <= 2 px",
    "b0_model_u2_uv5": "source swap, model u2, 2D agreement <= 5 px",
    "b0c_cam0_tstar_sim3": "camera swap to cam 0, Sim(3) T*",
    "t0_rc_fit": "ray center, K fit + R*",
    "t0_rc_gt": "ray center, GT K + GT rotation (control)",
}
EST_NAMES = {"mean": "mean (B0)", "conf": "exp(conf)-weighted", "geomedian": "geometric median",
             "conf_select": "most confident", "oracle": "closest to GT (oracle)"}
GATE2 = (  # (T0 method, B0 comparator method, comparator group): the proposal's "matched k" pairs
    ("b0_model_u2", "b0_model_u2", "all"),
    ("b0_gt_u2", "b0_gt_u2", "all"),
    ("b0c_cam0_tstar", "b0c_cam0_tstar", "all"),
    ("b0c_tstar", "b0c_tstar", "all"),
    ("b0c_cam0_gt_pose", "b0c_cam0_gt_pose", "all"),
    ("t0_rc_fit", "b0c_cam0_tstar", "query_at_target"),
    ("t0_rc_gt", "b0c_cam0_tstar", "query_at_target"),
)
DETAIL_ESTS = ("mean", "t0_l2_1", "t0_huber_1", "t0_cauchy_1", "oracle")  # kept per group in the JSON


def pct(x: float | None) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"


def est_label(e: str) -> str:
    if e in EST_NAMES:
        return EST_NAMES[e]
    psi, lam = t0.parse_t0(e)
    return f"T0 {psi}, lam={lam:g}"


def work(task: tuple[str, str, str]) -> tuple[str, str, dict[str, Any], float]:
    path, subset, name = task
    start = time.time()
    return subset, name, t0.clip_sums(load_table(path)), time.time() - start


def compute(dump: Path, subsets: list[str], limit: int, workers: int) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    manifest = json.loads((dump / "manifest.json").read_text())
    tasks, seen = [], {s: 0 for s in subsets}
    for entry in manifest["clips"]:
        s = entry["subset"]
        if s not in seen or entry.get("status") != "ok" or (limit and seen[s] >= limit):
            continue
        seen[s] += 1
        tasks.append((str(dump / entry["file"]), s, entry["file"]))
    out: dict[str, list[tuple[str, dict[str, Any]]]] = {s: [] for s in subsets}
    start = time.time()
    with Pool(workers) as pool:
        for i, (s, name, sums, dt) in enumerate(pool.imap_unordered(work, tasks), 1):
            out[s].append((name, sums))
            print(f"[{i}/{len(tasks)}] {name} {dt:.0f}s (elapsed {time.time() - start:.0f}s)", flush=True)
    for s in out:
        out[s].sort(key=lambda x: x[0])
    return out


def _leaf(sums: dict[str, Any], m: str, e: str, g: str, k: int) -> dict[str, float] | None:
    try:
        return sums[m][e][g][k]
    except KeyError:
        return None


def _rel(clips: list[dict[str, Any]], m: str, e: str, g: str, k: int, key: str = "rel") -> np.ndarray:
    vals = []
    for c in clips:
        leaf = _leaf(c, m, e, g, k)
        vals.append(leaf[key] if leaf else 0.0)
    return np.array(vals)


def _gain(clips: list[dict[str, Any]], m: str, e: str, k: int) -> float:
    rel, rel1 = _rel(clips, m, e, "all", k), _rel(clips, m, e, "all", k, "rel1")
    return 1.0 - rel.sum() / rel1.sum() if rel1.sum() > 0 else float("nan")


def cross_validate(clips: list[dict[str, Any]], m: str, k: int, cands: list[str]) -> dict[str, Any]:
    """2-fold CV over clips of the T0 setting: per-clip sums of the held-out choice, and the choices."""
    folds = np.arange(len(clips)) % 2
    chosen, per_clip = [], [None] * len(clips)
    for f in (0, 1):
        train = [c for c, ff in zip(clips, folds) if ff != f]
        best = max(cands, key=lambda e: np.nan_to_num(_gain(train, m, e, k), nan=-np.inf))
        chosen.append(best)
        for i in np.flatnonzero(folds == f):
            per_clip[i] = best
    return {"chosen": chosen, "per_clip": per_clip}


def summarize(clips_named: list[tuple[str, dict[str, Any]]], n_boot: int) -> dict[str, Any]:
    clips = [c for _, c in clips_named]
    pooled = t0.pool(clips)
    scales = {m: float(np.median(v)) for m, v in pooled.pop("_scale", {}).items()}
    summ: dict[str, Any] = {}
    for m, by_est in pooled.items():
        for e, by_group in by_est.items():
            for g, by_k in by_group.items():
                if g != "all" and e not in DETAIL_ESTS:
                    continue
                for k, leaf in by_k.items():
                    cell = ens.summarize_cell(leaf)
                    if g == "all":
                        rel, rel1 = _rel(clips, m, e, g, k), _rel(clips, m, e, g, k, "rel1")
                        cell["gain_ci"] = list(ens.bootstrap_gain(rel, rel1, n_boot, seed=k))
                        if e != "mean" and "mean" in by_est:
                            relb = _rel(clips, m, "mean", g, k)
                            cell["vs_b0"] = 1.0 - rel.sum() / relb.sum() if relb.sum() > 0 else float("nan")
                            cell["vs_b0_ci"] = list(ens.bootstrap_gain(rel, relb, n_boot, seed=k))
                    summ.setdefault(m, {}).setdefault(e, {}).setdefault(g, {})[str(k)] = cell
    cv: dict[str, Any] = {}
    for m in list(t0.CORE_METHODS) + list(t0.RC_METHODS):
        if m not in pooled:
            continue
        cands = [e for e in t0.T0_ESTIMATORS if e in pooled[m]]
        for k in sorted(next(iter(pooled[m].values()))["all"]):
            res = cross_validate(clips, m, k, cands)
            leaves = [_leaf(c, m, e, "all", k) for c, e in zip(clips, res["per_clip"])]
            tot = dict.fromkeys(ens.SUM_KEYS, 0.0)
            for leaf in leaves:
                if leaf:
                    for key in ens.SUM_KEYS:
                        tot[key] += leaf[key]
            cell = ens.summarize_cell(tot)
            rel = np.array([leaf["rel"] if leaf else 0.0 for leaf in leaves])
            rel1 = np.array([leaf["rel1"] if leaf else 0.0 for leaf in leaves])
            cell["gain_ci"] = list(ens.bootstrap_gain(rel, rel1, n_boot, seed=k))
            if "mean" in pooled[m]:
                relb = _rel(clips, m, "mean", "all", k)
                cell["vs_b0"] = 1.0 - rel.sum() / relb.sum() if relb.sum() > 0 else float("nan")
                cell["vs_b0_ci"] = list(ens.bootstrap_gain(rel, relb, n_boot, seed=k))
            gains = {e: _gain(clips, m, e, k) for e in cands}
            cell["chosen"] = res["chosen"]
            cell["hindsight"] = max(gains, key=lambda e: np.nan_to_num(gains[e], nan=-np.inf))
            cell["hindsight_gain"] = gains[cell["hindsight"]]
            cv.setdefault(m, {})[str(k)] = cell
    return {"n_clips": len(clips), "summary": summ, "cv": cv, "robust_scale": scales}


def _row(S: dict[str, Any], s: str, m: str, e: str, g: str = "all", k: int = K_MAIN) -> dict[str, Any] | None:
    try:
        return S[s]["summary"][m][e][g][str(k)]
    except KeyError:
        return None


def write_md(res: dict[str, Any], path: Path, label: str, argv: list[str], runtime_s: float | None) -> None:
    S = res["subsets"]
    subs = list(S)
    k = K_MAIN
    when = f"{runtime_s:.0f} s" if runtime_s else "from cache"
    lines = [
        f"# T0 vs B0 / B0-C, and B0 ablations ({label}, clip [0, 48))",
        "",
        f"`{' '.join(['scripts/l3d/run_t0.py', *argv])}` ({when}; code `l3d/t0.py`, `l3d/ensemble.py`). Same cells and "
        "partner outputs as `results/l3d/b0/` (k <= 6 from the existing dump).",
        "",
        "- **T0** optimizes the outputs of one cell (the query's output and its `k - 1` partners: other source frames for "
        "source swap, other cameras mapped by `T*` for camera swap) to minimize `sum_j rho(|a - b_j| / z) + lam/2 "
        "(|a - a0|^2 + sum_j |b_j - b0_j|^2) / z^2`, with `u2` and `T*` held fixed, and scores the optimized query "
        "output. `rho`: `l2` (then T0 is a damped B0: the B0 mean as `lam -> 0`), `huber` or `cauchy` with the "
        "clip's median residual as scale (label-free). **Ray center**: the camera-0 outputs of queries with "
        "`t_s = t_t` move toward their lines through the frame's least-squares ray center (2 decodes per query, so "
        "compared with B0-C to cam 0 at k = 2 on the same kind of query).",
        "- **Gain** = 1 - mean relative error `|P - X| / |X|` / that of the single output on the same cells (positive: "
        "helps). **vs B0** = 1 - error / B0-mean error on the same cells (positive: T0 beats B0). 95% CIs: 1000 "
        "resamples of each subset's clips.",
        "- **T0 (CV)**: the penalty and `lam` are picked on one half of the clips (even / odd) and scored on the other, "
        "so the choice is not tuned on the clips it is scored on. **Hindsight**: the best setting on all clips "
        "(optimistic).",
        "",
        f"## Gate 2: T0 vs B0 / B0-C at matched k (k = {k}; ray center at k = 2)",
        "",
        "Gate 2: \"If T0 does not beat B0/B0-C, we report it as ensembling, not consistency.\" A check's T0 counts if "
        "its CV gain beats B0 at matched k with a CI on *vs B0* above 0; it also has to beat the single output "
        "(Gain CI above 0) to be a correction.",
        "",
        "| Subset | Check | Cells | B0 gain | T0 (CV) gain | T0 gain, 95% CI | T0 vs B0 | vs B0, 95% CI | Hindsight | CV picks | T0 > B0? | T0 > single? |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for s in subs:
        for m, bm, bg in GATE2:
            kk = 2 if m.startswith("t0_rc") else k
            cv = S[s]["cv"].get(m, {}).get(str(kk))
            b0 = _row(S, s, bm, "mean", bg, kk)
            if cv is None or b0 is None or not cv["n_cells"]:
                continue
            if m.startswith("t0_rc"):
                vs, vs_ci = float("nan"), (float("nan"), float("nan"))
                beats_b0 = "yes" if cv["gain_ci"][0] > b0["gain"] else ("no" if cv["gain_ci"][1] < b0["gain"] else "tie")
                vs_txt, vs_ci_txt = "n/a (other cells)", "n/a"
            else:
                vs, vs_ci = cv["vs_b0"], cv["vs_b0_ci"]
                beats_b0 = "yes" if vs_ci[0] > 0 else ("no" if vs_ci[1] < 0 else "tie")
                vs_txt, vs_ci_txt = pct(vs), f"{pct(vs_ci[0])} to {pct(vs_ci[1])}"
            beats_1 = "yes" if cv["gain_ci"][0] > 0 else ("no" if cv["gain_ci"][1] < 0 else "tie")
            picks = ", ".join(sorted(set(cv["chosen"])))
            lines.append(
                f"| `{s}` | {NAMES[m]} | {cv['n_cells']} | {pct(b0['gain'])} | {pct(cv['gain'])} | "
                f"{pct(cv['gain_ci'][0])} to {pct(cv['gain_ci'][1])} | {vs_txt} | {vs_ci_txt} | "
                f"{pct(cv['hindsight_gain'])} ({cv['hindsight'].removeprefix('t0_')}) | {picks} | {beats_b0} | {beats_1} |")

    lines += ["", "## T0 vs B0 by k (source and camera swap, all cells)", "",
              "| Subset | Check | B0 k=2 | T0 (CV) k=2 | B0 k=4 | T0 (CV) k=4 | B0 k=6 | T0 (CV) k=6 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in subs:
        for m in t0.CORE_METHODS:
            cells = []
            for kk in ens.KS:
                b0 = _row(S, s, m, "mean", "all", kk)
                cv = S[s]["cv"].get(m, {}).get(str(kk))
                cells += [pct(b0["gain"]) if b0 else "n/a", pct(cv["gain"]) if cv else "n/a"]
            lines.append(f"| `{s}` | {NAMES[m]} | " + " | ".join(cells) + " |")

    lines += ["", f"## T0 sweep (gain, k = {k}; ray center k = 2)", "",
              "Rows: check and penalty; columns: subset and proximity weight `lam` (larger = closer to the single output).", "",
              "| Check | rho | " + " | ".join(f"`{s}` lam={lam:g}" for s in subs for lam in t0.LAMS) + " |",
              "| --- | --- |" + " ---: |" * (len(subs) * len(t0.LAMS))]
    for m in list(t0.CORE_METHODS) + list(t0.RC_METHODS):
        kk = 2 if m.startswith("t0_rc") else k
        for psi in t0.PSIS:
            cells = []
            for s in subs:
                for lam in t0.LAMS:
                    r = _row(S, s, m, f"t0_{psi}_{lam:g}", "all", kk)
                    cells.append(pct(r["gain"]) if r else "n/a")
            lines.append(f"| {NAMES[m]} | {psi} | " + " | ".join(cells) + " |")

    lines += ["", f"## B0 estimator ablation (gain, k = {k})", "",
              "Same cells and partners as B0; only the way the k outputs are combined changes. The oracle picks the "
              "output closest to GT (not label-free); it can lose to the mean, which averages noise down.", "",
              "| Subset | Check | " + " | ".join(EST_NAMES[e] for e in t0.B0_ESTIMATORS) + " |",
              "| --- | --- |" + " ---: |" * len(t0.B0_ESTIMATORS)]
    for s in subs:
        for m in t0.CORE_METHODS:
            cells = [pct(r["gain"]) if (r := _row(S, s, m, e)) else "n/a" for e in t0.B0_ESTIMATORS]
            lines.append(f"| `{s}` | {NAMES[m]} | " + " | ".join(cells) + " |")

    lines += ["", f"## Partner-set ablation (B0 mean, k = {k})", "",
              "Label-free filters on the model's `u2` and a Sim(3) `T*`. Cells: queries with at least `k - 1` partners "
              "left after the filter; the gain is against the single output on those cells.", "",
              "| Subset | Partners | Cells | Single rel. err | Gain | 95% CI | Better |",
              "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for s in subs:
        for m in ("b0_model_u2", "b0_model_u2_novis", "b0_model_u2_uv5", "b0_model_u2_uv2", "b0c_cam0_tstar",
                  "b0c_cam0_tstar_sim3"):
            r = _row(S, s, m, "mean")
            if r is None:
                continue
            lines.append(f"| `{s}` | {NAMES[m]} | {r['n_cells']} | {fmt(r['rel_1'])} | {pct(r['gain'])} | "
                         f"{pct(r['gain_ci'][0])} to {pct(r['gain_ci'][1])} | {fmt(r['frac_better'], 2)} |")

    lines += ["", f"## Which error component moves (RQ3; k = {k}, ray center k = 2)", "",
              "Gain on the depth part (`|log depth ratio|` along the GT ray) and the image-plane part (perpendicular "
              "offset / `|X|`) of the scored output. The proposal predicts source swap acts on S depth error, camera "
              "swap on non-rigid camera-frame error, ray center on image-plane error.", "",
              "| Subset | Check | depth / image-plane share of single error | B0 depth | B0 image-plane | T0 (CV) depth | T0 (CV) image-plane |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for s in subs:
        for m in list(t0.CORE_METHODS) + list(t0.RC_METHODS):
            kk = 2 if m.startswith("t0_rc") else k
            cv = S[s]["cv"].get(m, {}).get(str(kk))
            b0 = _row(S, s, m, "mean", "all", kk)
            if cv is None or not cv["n_cells"]:
                continue
            share = f"{fmt(cv['dep_1'], 4)} / {fmt(cv['ang_1'], 4)}"
            lines.append(f"| `{s}` | {NAMES[m]} | {share} | {pct(b0['gain_dep']) if b0 else 'n/a'} | "
                         f"{pct(b0['gain_ang']) if b0 else 'n/a'} | {pct(cv['gain_dep'])} | {pct(cv['gain_ang'])} |")

    lines += ["", f"## Static vs dynamic points (gain, k = {k})", "",
              "| Subset | Check | B0 static | B0 dynamic | T0 l2 lam=1 static | T0 l2 lam=1 dynamic | T0 cauchy lam=1 static | T0 cauchy lam=1 dynamic |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in subs:
        for m in t0.CORE_METHODS:
            cells = []
            for e in ("mean", "t0_l2_1", "t0_cauchy_1"):
                for g in ("static", "dynamic"):
                    r = _row(S, s, m, e, g)
                    cells.append(f"{pct(r['gain'])} ({r['n_cells']})" if r and r["n_cells"] else "n/a")
            lines.append(f"| `{s}` | {NAMES[m]} | " + " | ".join(cells) + " |")

    lines += ["", "Robust scale (median residual / depth, median over clips): " + "; ".join(
        f"`{s}` " + ", ".join(f"{m} {S[s]['robust_scale'].get(m, float('nan')):.4f}" for m in
                              ("b0_model_u2", "b0_gt_u2", "b0c_cam0_tstar", "t0_rc_fit")) for s in subs) + "."]
    path.write_text("\n".join(lines) + "\n")


def _round(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, float):
        return float(f"{x:.5g}") if np.isfinite(x) else None
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/l3d/t0")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--from-cache", action="store_true", help="re-render from the per-clip cache")
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--model-label", default=None, help="model name in the report (default: from the dump manifest)")
    args = ap.parse_args()

    start = time.time()
    subsets = args.subsets.split(",")
    if args.from_cache:
        per_clip = pickle.loads(args.cache.read_bytes())
        per_clip = {s: per_clip[s] for s in subsets if s in per_clip}
        runtime = None
    else:
        per_clip = compute(args.dump, subsets, args.limit, args.workers)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_bytes(pickle.dumps(per_clip))
        runtime = time.time() - start
    res: dict[str, Any] = {
        "dump": str(args.dump.relative_to(REPO) if args.dump.is_relative_to(REPO) else args.dump),
        "ks": list(ens.KS), "lams": list(t0.LAMS), "psis": list(t0.PSIS), "subsets": {}}
    for s, clips in per_clip.items():
        res["subsets"][s] = summarize(clips, args.n_boot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "t0.json").write_text(json.dumps(_round(jsonable(res))) + "\n")
    write_md(res, args.out_dir / "t0.md", model_label(args.dump, args.model_label), sys.argv[1:], runtime)
    print((args.out_dir / "t0.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
