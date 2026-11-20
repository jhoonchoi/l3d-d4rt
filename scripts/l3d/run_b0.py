#!/usr/bin/env python
"""B0 / B0-C: test-time ensembling over source and camera frames; writes results/l3d/b0/b0.{json,md}.

Per clip, :func:`l3d.ensemble.clip_sums` ensembles the decoded outputs already in the dump
(no new decodes, so k <= 6) and sums the errors of the k-ensemble and of the single output
on the same cells. Sums are pooled over each subset's clips; the gain's 95% CI resamples
clips with replacement. CPU only.

Example:
    .venv/bin/python scripts/l3d/run_b0.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import ensemble as ens  # noqa: E402
from l3d.report import SUBSETS, fmt, iter_clips, jsonable, model_label  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
NAMES = {
    "b0_model_u2": "B0, model u2",
    "b0_gt_u2": "B0, GT u2 (control)",
    "b0c_tstar": "B0-C to t_t, T*",
    "b0c_gt_pose": "B0-C to t_t, GT pose (control)",
    "b0c_cam0_tstar": "B0-C to cam 0, T*",
    "b0c_cam0_gt_pose": "B0-C to cam 0, GT pose (control)",
}
GROUP_NAMES = {"static": "Static", "dynamic": "Dynamic", "query_at_target": "Query at t_t", "query_elsewhere": "Query elsewhere"}
K_MAIN = 4


def pct(x: float | None) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"


def summarize(per_clip: list[dict[str, Any]], n_boot: int) -> dict[str, Any]:
    pooled = ens.pool(per_clip)
    out: dict[str, Any] = {}
    for m in ens.METHODS:
        out[m] = {}
        for a in ens.AGGREGATORS:
            out[m][a] = {}
            for g in ens.GROUPS:
                out[m][a][g] = {}
                for k in ens.KS:
                    cell = ens.summarize_cell(pooled[m][a][g][k])
                    rel = np.array([c[m][a][g][k]["rel"] for c in per_clip])
                    rel1 = np.array([c[m][a][g][k]["rel1"] for c in per_clip])
                    cell["gain_ci"] = list(ens.bootstrap_gain(rel, rel1, n_boot, seed=k))
                    out[m][a][g][str(k)] = cell
    return {"summary": out, "sums": pooled}


def write_md(res: dict[str, Any], path: Path, label: str, argv: list[str], runtime_s: float) -> None:
    S = {s: res["subsets"][s]["summary"] for s in res["subsets"]}
    k = str(K_MAIN)
    lines = [
        f"# B0 / B0-C: test-time ensembling ({label}, clip [0, 48))",
        "",
        f"`{' '.join(['scripts/l3d/run_b0.py', *argv])}` ({runtime_s:.0f} s; code `l3d/ensemble.py`). The ensembles average "
        "outputs already in the dump, so k <= 6 (design 1 has up to 6 source and 6 camera frames per point); k = 8 needs a new dump.",
        "",
        "- **B0, model u2**: the query's output `P(u1, t1, t3, c)` averaged with `k - 1` outputs decoded from other source "
        "frames `t2` at the model's own `u2 = uv_hat(u1, t1, t2, t2)`, where the model calls the point visible (label-free). "
        "**GT u2** decodes the same `t2` from the GT pixel (perfect correspondences).",
        "- **B0-C, T\\***: the query's output in a reference camera averaged with `k - 1` outputs of the same query decoded in "
        "other camera frames, mapped to the reference by the SE(3) Umeyama fit `T*` per camera pair (label-free). **GT pose** "
        "maps them by the GT pose. Reference **t_t**: the self-frame output, which has no camera-frame error; **cam 0**: the "
        "clip's world frame (`t_t > 0`), a cross-frame output, where the camera checks apply.",
        "- Every k averages over all size-`(k - 1)` subsets of a cell's other frames. **Gain** = 1 - mean relative error "
        "`|P - X| / |X|` of the ensemble / that of the single output, on the same cells (positive: ensembling helps); "
        "95% CI from 1000 resamples of each subset's clips. **MSE gain** is the same on squared error, which ties to A3: "
        "with GT u2 and every source, what is left is T + C.",
        "- Errors are in the reference camera (`c` for B0, `t_t` for B0-C) after the clip's global median scale.",
        "",
        "## Gain by k (plain mean, all cells)",
        "",
        "| Subset | Method | Cells (k=4) | Rel. err (k=1) | k=2 | k=4 | k=6 | k=4, 95% CI | MSE gain (k=4) | Better (k=4) | APD k=1 -> 4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for s, summ in S.items():
        for m in ens.METHODS:
            c = summ[m]["mean"]["all"]
            ck = c[k]
            ci = ck["gain_ci"]
            lines.append(
                f"| `{s}` | {NAMES[m]} | {ck['n_cells']} | {fmt(c['2']['rel_1'])} | {pct(c['2']['gain'])} | {pct(ck['gain'])} | "
                f"{pct(c['6']['gain'])} | {pct(ci[0])} to {pct(ci[1])} | {pct(ck['gain_mse'])} | {fmt(ck['frac_better'], 2)} | "
                f"{fmt(ck['apd_1'])} -> {fmt(ck['apd'])} |")
    lines += ["", f"## Confidence-weighted vs plain mean (gain at k={K_MAIN})", "",
              "Weights `exp(confidence logit)` (a softmax over the ensemble). `sigmoid(confidence)` is 0.9998-1 for every "
              "output of this model (logits 7.5-12), so weighting by it is the plain mean.", "",
              "| Method | " + " | ".join(f"`{s}` mean / conf" for s in S) + " |",
              "| --- |" + " --- |" * len(S)]
    for m in ens.METHODS:
        cells = [f"{pct(S[s][m]['mean']['all'][k]['gain'])} / {pct(S[s][m]['conf']['all'][k]['gain'])}" for s in S]
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    lines += ["", f"## Gain by motion and query frame (plain mean, k={K_MAIN})", "",
              "Query at `t_t`: the query's source frame is the target frame (B0: the point is GT-visible at `t3`). "
              "Query elsewhere: it is not (B0: the point is occluded at `t3`, so the query is its nearest visible frame).", "",
              "| Subset | Method | " + " | ".join(GROUP_NAMES.values()) + " |",
              "| --- | --- |" + " ---: |" * len(GROUP_NAMES)]
    for s, summ in S.items():
        for m in ens.METHODS:
            cells = []
            for g in GROUP_NAMES:
                c = summ[m]["mean"][g][k]
                cells.append(f"{pct(c['gain'])} ({c['n_cells']})" if c["n_cells"] else "0")
            lines.append(f"| `{s}` | {NAMES[m]} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/l3d/b0")
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-vis-filter", action="store_true", help="B0: keep model u2 the model calls occluded")
    ap.add_argument("--model-label", default=None, help="model name in the report (default: from the dump manifest)")
    args = ap.parse_args()

    start = time.time()
    subsets = args.subsets.split(",")
    per_clip: dict[str, list] = {s: [] for s in subsets}
    for entry, d in iter_clips(args.dump, subsets, args.limit):
        per_clip[entry["subset"]].append(ens.clip_sums(d, vis_filter=not args.no_vis_filter))
    res: dict[str, Any] = {"dump": str(args.dump.relative_to(REPO) if args.dump.is_relative_to(REPO) else args.dump),
                           "ks": list(ens.KS), "vis_filter": not args.no_vis_filter, "subsets": {}}
    for s in subsets:
        res["subsets"][s] = {"n_clips": len(per_clip[s]), **summarize(per_clip[s], args.n_boot)}
    runtime = time.time() - start
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b0.json").write_text(json.dumps(jsonable(res), indent=1) + "\n")
    write_md(res, args.out_dir / "b0.md", model_label(args.dump, args.model_label), sys.argv[1:], runtime)
    print((args.out_dir / "b0.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
