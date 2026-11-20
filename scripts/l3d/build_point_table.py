#!/usr/bin/env python3
"""Stage 1, Phase 0: build the point table from decoded-query dumps (Tier B models).

    python scripts/l3d/build_point_table.py --dump opend4rt=output/l3d/dumps/flow3r_T48_opend4rt_d0 \
        --dump flow3r=output/l3d/dumps/flow3r_T48 --out output/l3d/compare

Writes ``<out>/points_<subset>.npz`` (flat per-unit columns, see ``l3d/pointtable.py``; ``clip`` indexes
``clips.json``) and ``<out>/clips.json`` with per-clip metadata and the Phase 0 checks:

- the design-0 queries (``q_track``, ``q_ts``, ``q_tt``, ``q_tc``, ``q_u``, ``q_v``) are identical in every dump;
- each model's per-clip APD from the point table equals the manifest's ``d0_apd_global`` (to 1e-6).

The first ``--dump`` gives the GT and the track set. Only clips present in every dump are used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from l3d.pointtable import build_clip, design0_arrays  # noqa: E402
from l3d.report import SUBSETS, jsonable  # noqa: E402
from l3d.table import load_table  # noqa: E402

QUERY_KEYS = ("q_track", "q_ts", "q_tt", "q_tc", "q_u", "q_v")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", action="append", required=True, help="name=path to a dump run directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    args = ap.parse_args()
    dumps = dict(item.split("=", 1) for item in args.dump)
    manifests = {m: {c["file"]: c for c in json.loads((Path(p) / "manifest.json").read_text())["clips"]
                     if c.get("status") == "ok"} for m, p in dumps.items()}
    names = list(dumps)
    files = sorted(set.intersection(*(set(v) for v in manifests.values())))
    args.out.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    max_apd_diff = {m: 0.0 for m in names}
    t0 = time.time()
    for subset in SUBSETS:
        sub_files = [f for f in files if f.startswith(subset + "/")]
        if args.limit:
            sub_files = sub_files[: args.limit]
        cols: dict[str, list[np.ndarray]] = {}
        for f in sub_files:
            tables = {m: load_table(Path(dumps[m]) / f) for m in names}
            ref = tables[names[0]]
            d0 = {m: np.asarray(t["design"]) == 0 for m, t in tables.items()}
            for m in names[1:]:
                for k in QUERY_KEYS:
                    if not np.array_equal(np.asarray(ref[k])[d0[names[0]]], np.asarray(tables[m][k])[d0[m]]):
                        raise ValueError(f"{f}: design-0 {k} differs between {names[0]} and {m}")
            rows, meta = build_clip(ref, {m: design0_arrays(t) for m, t in tables.items()})
            if meta["units"] == 0:
                continue
            meta["clip"] = len(clips)
            meta["file"] = f
            for m in names:
                ref_apd = manifests[m][f].get("d0_apd_global", float("nan"))
                meta["models"][m]["manifest_apd"] = ref_apd
                if np.isfinite(ref_apd):
                    max_apd_diff[m] = max(max_apd_diff[m], abs(meta["models"][m]["apd"] - ref_apd))
            rows["clip"] = np.full(rows["t"].shape, meta["clip"], dtype=np.int16)
            for k, v in rows.items():
                cols.setdefault(k, []).append(v)
            clips.append(meta)
        if cols:
            np.savez(args.out / f"points_{subset}.npz", **{k: np.concatenate(v) for k, v in cols.items()})
        n_units = sum(c["units"] for c in clips if c["subset"] == subset)
        print(f"{subset}: {len([c for c in clips if c['subset'] == subset])} clips, {n_units} units ({time.time() - t0:.0f} s)")
    checks = {"models": names, "dumps": dumps, "query_sets_identical": True,
              "max_abs_apd_diff_vs_manifest": max_apd_diff,
              "apd_check_pass": all(v <= 1e-6 for v in max_apd_diff.values())}
    (args.out / "clips.json").write_text(json.dumps(jsonable({"checks": checks, "clips": clips}), indent=1))
    print(json.dumps(jsonable(checks), indent=1))


if __name__ == "__main__":
    main()
