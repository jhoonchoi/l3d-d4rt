#!/usr/bin/env python3
"""Design-0 tracking accuracy per subset from dump manifests, for models dumped on the same clips.

Each clip's ``d0_apd_global`` / ``d0_epe_global`` is the upstream WorldTrack metric
(``_compute_average_pts_within_thresh``, global median scale) on design-0 rows of that clip
(frame-0-visible tracks, every target frame, in the target frame's camera). This is the 48-frame-clip
counterpart of Gate 0, not the README's 64-frame protocol.

    python scripts/l3d/run_track_apd.py --dump flow3r=output/l3d/dumps/flow3r_T48 \
        --dump opend4rt=output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48_d0 --out results/l3d_flow3r/track_apd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from l3d.report import SUBSETS, jsonable  # noqa: E402


def per_subset(manifest: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for s in SUBSETS:
        rows = [c for c in manifest["clips"] if c.get("subset") == s and c.get("status") == "ok"
                and np.isfinite(c.get("d0_apd_global", np.nan))]
        if not rows:
            continue
        out[s] = {
            "clips": len(rows),
            "apd": float(np.mean([c["d0_apd_global"] for c in rows])),
            "epe": float(np.mean([c["d0_epe_global"] for c in rows])),
            "sequences": sorted(c["sequence"] for c in rows),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", action="append", required=True, help="name=path to a dump run directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    models = {}
    for item in args.dump:
        name, path = item.split("=", 1)
        models[name] = per_subset(json.loads((Path(path) / "manifest.json").read_text()))
    names = list(models)
    lines = ["# Design-0 tracking accuracy on the same 48-frame clips (clip [0, 48))", "",
             "Mean over clips of the upstream WorldTrack APD / EPE (global median scale) on design-0 rows. "
             "Only clips present for every model are compared.", "",
             "| Subset | Clips | " + " | ".join(f"APD {n}" for n in names) + " | " + " | ".join(f"EPE {n}" for n in names) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in names * 2) + " |"]
    for s in SUBSETS:
        if not all(s in models[n] for n in names):
            continue
        common = set.intersection(*(set(models[n][s]["sequences"]) for n in names))
        flag = "" if all(len(common) == models[n][s]["clips"] for n in names) else " (mismatch)"
        lines.append(f"| `{s}` | {len(common)}{flag} | " + " | ".join(f"{models[n][s]['apd']:.4f}" for n in names)
                     + " | " + " | ".join(f"{models[n][s]['epe']:.4f}" for n in names) + " |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "track_apd.json").write_text(json.dumps(jsonable(models), indent=1))
    (args.out / "track_apd.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
