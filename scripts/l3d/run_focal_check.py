#!/usr/bin/env python3
"""Stage 1 (exploratory): field-of-view error at the query frame, from the point table.

    python scripts/l3d/run_focal_check.py --points output/l3d/compare --dump output/l3d/dumps/flow3r_T48 \
        --out results/l3d_compare/focal_t0.md

At ``t = 0`` the target camera is camera 0, so a pure depth error projects onto the GT pixel. Each model's
camera-0 point is projected with the GT intrinsics. Per clip and axis, the median ratio
``(u_pred - c) / (u_gt - c)`` over units at least 20 px from the principal point estimates the ratio
``f_gt / f_model``: above 1 = the model's focal length is too short (field of view too wide). The 2D offset
is the median distance between the two projections, in 256-px units. ``--dump`` gives ``K_gt`` per clip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from l3d import geometry as geo  # noqa: E402
from l3d.report import SUBSETS, jsonable  # noqa: E402
from l3d.table import load_table  # noqa: E402

LABEL = {"opend4rt": "OpenD4RT", "flow3r": "flow3r", "arrow": "ARROW", "trackeverything": "TrackEverything (Pi3)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=Path, default=Path("output/l3d/compare"))
    ap.add_argument("--dump", type=Path, default=Path("output/l3d/dumps/flow3r_T48"))
    ap.add_argument("--out", type=Path, default=Path("results/l3d_compare/focal_t0.md"))
    args = ap.parse_args()
    meta = json.loads((args.points / "clips.json").read_text())
    models = meta["checks"]["models"]
    res: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
    for s in SUBSETS:
        with np.load(args.points / f"points_{s}.npz") as z:
            P = {k: z[k] for k in ("clip", "t", "g0", *(f"e_{m}" for m in models))}
        for c in (c for c in meta["clips"] if c["subset"] == s):
            K = np.asarray(load_table(args.dump / c["file"])["K_gt"], dtype=np.float64)
            sel = (P["clip"] == c["clip"]) & (P["t"] == 0)
            g = P["g0"][sel].astype(np.float64)
            ug = geo.project(K, g)
            unit = 255.0 / (2.0 * max(K[0, 2], K[1, 2]))  # native px -> 256-px units (image size ~ 2c)
            for m in models:
                up = geo.project(K, g + P[f"e_{m}"][sel].astype(np.float64))
                fin = np.isfinite(up).all(1)
                r = []
                for ax in (0, 1):
                    dg = ug[:, ax] - K[ax, 2]
                    ok = fin & (np.abs(dg) > 20)
                    r.append(float(np.median((up[ok, ax] - K[ax, 2]) / dg[ok])) if ok.sum() > 10 else np.nan)
                off = float(np.median(np.linalg.norm(up[fin] - ug[fin], axis=1)) * unit) if fin.any() else np.nan
                res.setdefault(s, {}).setdefault(m, []).append((r[0], r[1], off))
    L = ["# Field-of-view error at the query frame (exploratory)", "", __doc__.split("\n\n")[2].replace("\n", " "), "",
         "Per subset: median over clips, with the 10th and 90th percentiles over clips.", "",
         "| Subset | Model | Clips | Ratio x (median) | Ratio x [p10, p90] | Ratio y (median) | Ratio y [p10, p90] | 2D offset at t = 0 (px) |",
         "| --- | --- | ---: | ---: | --- | ---: | --- | ---: |"]
    summary = {}
    for s, by_m in res.items():
        for m, v in by_m.items():
            v = np.array(v)
            q = {k: np.nanpercentile(v[:, i], [10, 50, 90]).tolist() for i, k in enumerate(("x", "y", "px"))}
            summary.setdefault(s, {})[m] = q
            L.append(f"| `{s}` | {LABEL.get(m, m)} | {len(v)} | {q['x'][1]:.3f} | [{q['x'][0]:.3f}, {q['x'][2]:.3f}] | "
                     f"{q['y'][1]:.3f} | [{q['y'][0]:.3f}, {q['y'][2]:.3f}] | {q['px'][1]:.2f} |")
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(jsonable(summary), indent=1))
    print("\n".join(L[-(len(L) - 6):]))


if __name__ == "__main__":
    main()
