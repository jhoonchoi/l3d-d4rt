#!/usr/bin/env python
"""Robust T* fits vs plain Umeyama (RQ3's pose prediction at test time); writes results/l3d/a4_robust/.

Per clip with GT poses, :func:`l3d.pose_robust.clip_pose` refits ``T*`` for every camera
pair of the grid (design 2, as in A4) and of the tracks (design 1, where rows have GT, so
an oracle that drops the worst 20% of points by GT error bounds what outlier removal can
do). Per subset: median rotation / translation error over pairs, the median per-pair
ratio of translation error to plain Umeyama's, the share of pairs it improves, and a 95%
CI (clips resampled) of the mean log ratio. CPU only.

Example:
    .venv/bin/python scripts/l3d/run_a4_robust.py
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import pose_robust as pr  # noqa: E402
from l3d.report import fmt, jsonable, model_label  # noqa: E402
from l3d.table import load_table  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
POSE_SUBSETS = ("adt_mini", "po_mini", "ds_mini")  # pstudio_mini is a fixed camera
LABEL = {"umeyama": "Umeyama (A4)", "trim20": "trim 20% by residual", "huber": "IRLS Huber", "cauchy": "IRLS Cauchy",
         "expconf": "exp(conf) weights", "oracle_trim20": "oracle: trim 20% by GT error"}
DESIGN_LABEL = {"grid": "grid (design 2)", "cross": "tracks (design 1)"}


def work(task: tuple[str, str]) -> tuple[str, str, dict[str, Any]]:
    path, subset = task
    return subset, path, pr.clip_pose(load_table(path))


def summarize(clips: list[dict[str, Any]], n_boot: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for design in pr.DESIGNS:
        cl = [c[design] for c in clips if design in c and c[design]["pairs"]["t"].size]
        if not cl:
            continue
        cat = lambda f: np.concatenate([f(c) for c in cl])  # noqa: E731
        base_t = cat(lambda c: c["methods"]["umeyama"]["trans_rel"])
        base_r = cat(lambda c: c["methods"]["umeyama"]["rot_deg"])
        clip_id = np.concatenate([np.full(c["pairs"]["t"].size, i) for i, c in enumerate(cl)])
        res: dict[str, Any] = {"n_clips": len(cl), "n_pairs": int(base_t.size),
                               "gt_rot_deg_median": float(np.median(cat(lambda c: c["pairs"]["gt_rot_deg"]))),
                               "gt_trans_rel_median": float(np.median(cat(lambda c: c["pairs"]["gt_trans_rel"]))),
                               "methods": {}}
        rng = np.random.default_rng(0)
        boot = rng.integers(0, len(cl), size=(n_boot, len(cl)))
        for m in cl[0]["methods"]:
            tr = cat(lambda c, m=m: c["methods"][m]["trans_rel"])
            rot = cat(lambda c, m=m: c["methods"][m]["rot_deg"])
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(np.maximum(tr, 1e-12) / np.maximum(base_t, 1e-12))
                lrr = np.log(np.maximum(rot, 1e-9) / np.maximum(base_r, 1e-9))
            per_clip = np.array([lr[clip_id == i].mean() for i in range(len(cl))])
            per_clip_r = np.array([lrr[clip_id == i].mean() for i in range(len(cl))])
            ci = np.percentile(per_clip[boot].mean(1), [2.5, 97.5])
            ci_r = np.percentile(per_clip_r[boot].mean(1), [2.5, 97.5])
            res["methods"][m] = {
                "rot_deg_median": float(np.median(rot)), "trans_rel_median": float(np.median(tr)),
                "trans_rel_mean": float(np.mean(tr)),
                "trans_ratio_median": float(np.exp(np.median(lr))), "rot_ratio_median": float(np.exp(np.median(lrr))),
                "trans_ratio_geomean": float(np.exp(per_clip.mean())), "trans_ratio_ci": [float(np.exp(x)) for x in ci],
                "rot_ratio_geomean": float(np.exp(per_clip_r.mean())), "rot_ratio_ci": [float(np.exp(x)) for x in ci_r],
                "beats_umeyama_trans": float(np.mean(tr < base_t)) if m != "umeyama" else float("nan"),
            }
        out[design] = res
    return out


def write_md(per_subset: dict[str, Any], path: Path, label: str, argv: list[str], runtime_s: float) -> None:
    lines = [
        f"# Robust T* fits vs plain Umeyama ({label}, clip [0, 48))",
        "",
        f"`{' '.join(['scripts/l3d/run_a4_robust.py', *argv])}` ({runtime_s:.0f} s; code `l3d/pose_robust.py`). Per camera "
        "pair `(t, c)`, `T*: t -> c` (SE(3)) from the same outputs, scaled by the clip scale; errors against the GT "
        "relative pose, translation over the clip's median GT depth. RQ3 predicts pose error falls only as far as "
        "fewer outliers improve the fit; the oracle (drop the 20% of points with the largest GT error) bounds that.",
        "",
        "- Ratios are per pair, method / plain Umeyama (< 1: better). *Geo-mean ratio* is the mean over clips of each "
        "clip's mean log ratio, with a 95% CI from resampling clips.",
        "",
        "| Subset | Points | Pairs | Method | rot err (deg) | trans err / depth | median trans ratio | geo-mean trans "
        "ratio (95% CI) | geo-mean rot ratio (95% CI) | beats Umeyama (trans) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for s, by_design in per_subset.items():
        for design, r in by_design.items():
            first = True
            for m, v in r["methods"].items():
                head = f"| `{s}` | {DESIGN_LABEL[design]} | {r['n_pairs']} |" if first else "| | | |"
                first = False
                ci, cir = v["trans_ratio_ci"], v["rot_ratio_ci"]
                lines.append(
                    f"{head} {LABEL[m]} | {fmt(v['rot_deg_median'])} | {fmt(v['trans_rel_median'], 4)} | "
                    f"{fmt(v['trans_ratio_median'])} | {fmt(v['trans_ratio_geomean'])} ({fmt(ci[0])} to {fmt(ci[1])}) | "
                    f"{fmt(v['rot_ratio_geomean'])} ({fmt(cir[0])} to {fmt(cir[1])}) | {fmt(v['beats_umeyama_trans'], 2)} |")
    lines += ["", "GT motion per pair (median): " + "; ".join(
        f"`{s}` {DESIGN_LABEL[dz]}: {fmt(r['gt_rot_deg_median'], 2)} deg, {fmt(r['gt_trans_rel_median'], 4)} depth"
        for s, bd in per_subset.items() for dz, r in bd.items()) + "."]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/l3d/a4_robust")
    ap.add_argument("--subsets", default=",".join(POSE_SUBSETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--n-boot", type=int, default=1000)
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
        tasks.append((str(args.dump / entry["file"]), s))
    per: dict[str, list[tuple[str, dict[str, Any]]]] = {s: [] for s in subsets}
    with Pool(args.workers) as pool:
        for i, (s, p, res) in enumerate(pool.imap_unordered(work, tasks), 1):
            per[s].append((p, res))
            print(f"[{i}/{len(tasks)}] {Path(p).name} (elapsed {time.time() - start:.0f}s)", flush=True)
    per_subset = {}
    for s, clips in per.items():
        clips.sort(key=lambda x: x[0])
        per_subset[s] = summarize([c for _, c in clips], args.n_boot)
    runtime = time.time() - start
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "a4_robust.json").write_text(json.dumps(jsonable(per_subset), indent=1) + "\n")
    write_md(per_subset, args.out_dir / "a4_robust.md", model_label(args.dump, args.model_label), sys.argv[1:], runtime)
    print((args.out_dir / "a4_robust.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
