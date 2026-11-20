#!/usr/bin/env python
"""RQ2: does each check add information beyond the native confidence? Writes results/l3d/a3_combined/.

A3 (``run_a3_detection.py``) scores each residual and the confidence separately. Here, on
the same rows (up to ``--max-rows`` per clip and check, seeded), each check's residual is
combined with the confidence and scored against the top 20% of errors:

- ``confidence``: the raw logit (A3 scores ``sigmoid`` in float32, which is within 1e-5 of
  1 for this model and ties many rows; both are reported);
- ``rank average``: mean of the residual's and the confidence's ranks (label-free, no fit);
- ``logistic (CV)``: logistic regression on ``log residual``, the logit and their product,
  fitted on one half of the clips and scored on the other (even / odd clip), so it uses
  labels only from other clips;
- partial Spearman ``rho(residual, error | confidence)``: rank correlation after regressing
  the confidence ranks out of both.

Delta AUROC 95% CIs resample clips (``--n-boot``). CPU only.

Example:
    .venv/bin/python scripts/l3d/run_a3_combined.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import check_rows as cr  # noqa: E402
from l3d import detection as det  # noqa: E402
from l3d.decompose import table_scale  # noqa: E402
from l3d.report import SUBSETS, fmt, jsonable, model_label  # noqa: E402
from l3d.table import load_table  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
CHECKS = {
    "source_swap": "source swap, model u2",
    "source_swap_gt_u2": "source swap, GT u2 (control)",
    "camera_swap_se3": "camera swap, SE(3)",
    "ray_center": "ray center, K fit + R*",
    "ray_center_gt": "ray center, GT K and pose (control)",
    "reprojection": "reprojection, K fit",
    "static_rigidity": "static rigidity (GT static mask)",
}
SCORES = ("confidence", "confidence_a3", "residual", "rank_average", "logistic_cv")
SCORE_NAMES = {"confidence": "confidence (logit)", "confidence_a3": "confidence (A3: float32 sigmoid)",
               "residual": "residual", "rank_average": "rank average", "logistic_cv": "logistic (CV)"}
TOP = 0.2


def clip_rows(task: tuple[str, str, int, int]) -> tuple[str, str, dict[str, dict[str, np.ndarray]]]:
    path, subset, max_rows, seed = task
    d = load_table(path)
    err = cr.row_errors(d, table_scale(d))
    logit = d["p_conf_logit"].astype(np.float64)
    err["conf"] = np.stack([logit, 1.0 / (1.0 + np.exp(-logit))], -1)  # every check indexes it row-wise
    K = cr.fit_K(d, err)
    ss = cr.source_swap_detection(d, err)
    cs = cr.camera_swap_detection(d, err)
    rc = cr.ray_center_detection(d, err, K)
    base = lambda r: {k: r[k] for k in ("error", "target", "conf")}  # noqa: E731
    rows = {
        "source_swap": ss["model_u2"],
        "source_swap_gt_u2": ss["gt_u2"],
        "camera_swap_se3": {"residual": cs["residual_se3"], **base(cs)},
        "ray_center": {"residual": rc["residual"], **base(rc)},
        "ray_center_gt": {"residual": rc["residual_gt"], **base(rc)},
        "reprojection": cr.reprojection_detection(d, err, K),
        "static_rigidity": cr.static_rigidity_detection(d, err),
    }
    rng = np.random.default_rng(seed)
    out = {}
    for name, r in rows.items():
        conf = np.asarray(r["conf"], dtype=np.float64).reshape(-1, 2)
        res, e, tgt = (np.asarray(r[k], dtype=np.float64).reshape(-1) for k in ("residual", "error", "target"))
        ok = np.flatnonzero(np.isfinite(res) & np.isfinite(e) & np.isfinite(conf).all(-1))
        if ok.size > max_rows:
            ok = np.sort(rng.choice(ok, max_rows, replace=False))
        out[name] = {"residual": res[ok], "error": e[ok], "target": tgt[ok], "logit": conf[ok, 0],
                     "sig32": conf[ok, 1].astype(np.float32)}
    return subset, path, out


def _std(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (x - ref.mean()) / max(float(ref.std()), 1e-12)


def _features(res: np.ndarray, logit: np.ndarray, ref_res: np.ndarray, ref_logit: np.ndarray) -> np.ndarray:
    lr, lref = np.log(res + 1e-9), np.log(ref_res + 1e-9)
    a, b = _std(lr, lref), _std(logit, ref_logit)
    return np.stack([np.ones_like(a), a, b, a * b], -1)


def logistic_fit(F: np.ndarray, y: np.ndarray, ridge: float = 1e-4, iters: int = 30) -> np.ndarray:
    """Newton's method for L2-regularized logistic regression (features include the intercept)."""
    w = np.zeros(F.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(F @ w, -30, 30)))
        g = F.T @ (p - y) / y.size + ridge * w
        H = (F * (p * (1 - p))[:, None]).T @ F / y.size + ridge * np.eye(F.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman rho of ``x`` and ``y`` after regressing the ranks of ``z`` out of both."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    A = np.stack([np.ones_like(rz), rz], -1)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def scores_for(rows: dict[str, np.ndarray], clip: np.ndarray) -> dict[str, np.ndarray]:
    res, logit, e = rows["residual"], rows["logit"], rows["error"]
    out = {"confidence": -logit, "confidence_a3": -rows["sig32"].astype(np.float64), "residual": res,
           "rank_average": 0.5 * (rankdata(res) + rankdata(-logit)) / res.size}
    fold = clip % 2
    lc = np.full(res.size, np.nan)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if tr.sum() < 50 or te.sum() == 0:
            continue
        y = (e[tr] >= np.quantile(e[tr], 1.0 - TOP)).astype(np.float64)
        w = logistic_fit(_features(res[tr], logit[tr], res[tr], logit[tr]), y)
        lc[te] = _features(res[te], logit[te], res[tr], logit[tr]) @ w
    out["logistic_cv"] = lc
    return out


def score_check(rows: dict[str, np.ndarray], clip: np.ndarray, n_boot: int) -> dict[str, Any]:
    e, tgt = rows["error"], rows["target"]
    if e.size < 100:
        return {"n": int(e.size)}
    sc = scores_for(rows, clip)
    okt = np.isfinite(tgt)
    out: dict[str, Any] = {"n": int(e.size), "n_clips": int(np.unique(clip).size), "vs_error": {}, "vs_target": {}}
    for name, s in sc.items():
        out["vs_error"][name] = det.detection_metrics(s, e, TOP)
        out["vs_target"][name] = {"spearman": det.spearman(s[okt], tgt[okt]), "auroc": det.auroc(s[okt], tgt[okt], TOP)}
    out["partial_spearman_error"] = partial_spearman(rows["residual"], e, rows["logit"])
    out["partial_spearman_target"] = (partial_spearman(rows["residual"][okt], tgt[okt], rows["logit"][okt])
                                      if okt.sum() > 10 else float("nan"))
    out["ties_a3"] = 1.0 - np.unique(rows["sig32"]).size / e.size
    # clip bootstrap of delta AUROC (vs error) against the logit confidence
    uniq = np.unique(clip)
    members = [np.flatnonzero(clip == c) for c in uniq]
    rng = np.random.default_rng(0)
    deltas: dict[str, list[float]] = {k: [] for k in ("residual", "rank_average", "logistic_cv", "confidence_a3")}
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, uniq.size, uniq.size)])
        base = det.auroc(sc["confidence"][idx], e[idx], TOP)
        for k in deltas:
            deltas[k].append(det.auroc(sc[k][idx], e[idx], TOP) - base)
    out["delta_auroc_ci"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                             for k, v in deltas.items()}
    return out


def write_md(per_subset: dict[str, dict[str, Any]], path: Path, label: str, argv: list[str], runtime_s: float,
             max_rows: int) -> None:
    lines = [
        f"# RQ2: check residuals combined with the native confidence ({label}, clip [0, 48))",
        "",
        f"`{' '.join(['scripts/l3d/run_a3_combined.py', *argv])}` ({runtime_s:.0f} s). Up to {max_rows} rows per clip and "
        "check (seeded sample of the A3 rows). Positives: the top 20% of errors `|s P - X| / |X|` of the subset.",
        "",
        "- **Confidence (logit)** is the native confidence without the sigmoid. A3 scored `sigmoid(conf)` stored as "
        "float32; for this model that is within 1e-5 of 1, so many rows tie (column *A3 ties*).",
        "- **Rank average**: mean rank of residual and confidence (no fit). **Logistic (CV)**: fitted on the other half "
        "of the clips. Delta AUROC is against the logit confidence; 95% CI over clips.",
        "- **Partial rho**: Spearman rho of residual and error after regressing out the confidence ranks (0: the "
        "residual adds nothing the confidence has not).",
        "",
        "## AUROC (top 20% errors)",
        "",
        "| Subset | Check | Rows | conf (logit) | conf (A3) | A3 ties | residual | rank avg | logistic (CV) | "
        "delta logistic - conf, 95% CI | delta A3 conf - logit conf, 95% CI | partial rho (error) | partial rho (target) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for s, checks_ in per_subset.items():
        for c, r in checks_.items():
            if "vs_error" not in r:
                continue
            a = {k: r["vs_error"][k]["auroc"] for k in SCORES}
            ci, ci3 = r["delta_auroc_ci"]["logistic_cv"], r["delta_auroc_ci"]["confidence_a3"]
            lines.append(
                f"| `{s}` | {CHECKS[c]} | {r['n']} | {fmt(a['confidence'])} | {fmt(a['confidence_a3'])} | "
                f"{fmt(r['ties_a3'], 2)} | {fmt(a['residual'])} | {fmt(a['rank_average'])} | {fmt(a['logistic_cv'])} | "
                f"{ci[0]:+.3f} to {ci[1]:+.3f} | {ci3[0]:+.3f} to {ci3[1]:+.3f} | {fmt(r['partial_spearman_error'])} | "
                f"{fmt(r['partial_spearman_target'])} |")
    lines += ["", "## E-AURC (excess area under risk-coverage; lower is better)", "",
              "| Subset | Check | " + " | ".join(SCORE_NAMES[k] for k in SCORES) + " |",
              "| --- | --- |" + " ---: |" * len(SCORES)]
    for s, checks_ in per_subset.items():
        for c, r in checks_.items():
            if "vs_error" not in r:
                continue
            lines.append(f"| `{s}` | {CHECKS[c]} | " + " | ".join(fmt(r["vs_error"][k]["eaurc"], 4) for k in SCORES) + " |")
    lines += ["", "## AUROC against the check's target component", "",
              "| Subset | Check | " + " | ".join(SCORE_NAMES[k] for k in SCORES) + " |",
              "| --- | --- |" + " ---: |" * len(SCORES)]
    for s, checks_ in per_subset.items():
        for c, r in checks_.items():
            if "vs_target" not in r:
                continue
            lines.append(f"| `{s}` | {CHECKS[c]} | " + " | ".join(fmt(r["vs_target"][k]["auroc"]) for k in SCORES) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/l3d/a3_combined")
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--max-rows", type=int, default=5000, help="rows per clip and check")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--model-label", default=None)
    args = ap.parse_args()

    start = time.time()
    subsets = args.subsets.split(",")
    manifest = json.loads((args.dump / "manifest.json").read_text())
    tasks, seen = [], {s: 0 for s in subsets}
    for entry in manifest["clips"]:
        s = entry["subset"]
        if s not in seen or entry.get("status") != "ok" or (args.limit and seen[s] >= args.limit):
            continue
        seen[s] += 1
        tasks.append((str(args.dump / entry["file"]), s, args.max_rows, len(tasks)))
    per: dict[str, list[tuple[str, dict[str, dict[str, np.ndarray]]]]] = {s: [] for s in subsets}
    with Pool(args.workers) as pool:
        for i, (s, p, rows) in enumerate(pool.imap_unordered(clip_rows, tasks), 1):
            per[s].append((p, rows))
            print(f"[{i}/{len(tasks)}] {Path(p).name} (elapsed {time.time() - start:.0f}s)", flush=True)
    per_subset: dict[str, dict[str, Any]] = {}
    for s, clips in per.items():
        clips.sort(key=lambda x: x[0])
        per_subset[s] = {}
        for c in CHECKS:
            parts = [(i, r[c]) for i, (_, r) in enumerate(clips) if r[c]["error"].size]
            if not parts:
                continue
            rows = {k: np.concatenate([p[k] for _, p in parts]) for k in parts[0][1]}
            clip = np.concatenate([np.full(p["error"].size, i) for i, p in parts])
            per_subset[s][c] = score_check(rows, clip, args.n_boot)
    runtime = time.time() - start
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "combined.json").write_text(json.dumps(jsonable(per_subset), indent=1) + "\n")
    write_md(per_subset, args.out_dir / "combined.md", model_label(args.dump, args.model_label), sys.argv[1:], runtime,
             args.max_rows)
    print((args.out_dir / "combined.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
