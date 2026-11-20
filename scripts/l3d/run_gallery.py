#!/usr/bin/env python3
"""Stage 1 Q4: failure-gallery selection and per-clip figures from the point table (no RGB needed).

    python scripts/l3d/run_gallery.py --points output/l3d/compare --out results/l3d_compare/gallery

Selection rule (fixed in docs/l3d/STAGE1_PLAN.md, section 4.4): for each model pair, the 3 clips with the
largest clip-APD difference in each direction; the 2 clips where the best model is worst; 2 random clips
(seed 0) as a reference. Per clip: per-frame APD curves of every model, and a top-down (x-z, camera 0)
plot of up to 8 dynamic GT tracks (or static, if the clip has none) with each model's tracks.
Writes ``index.md`` with the reasons and an empty failure-tag table for two people to fill in.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from l3d.report import SUBSETS  # noqa: E402

LABEL = {"opend4rt": "OpenD4RT", "flow3r": "flow3r", "arrow": "ARROW", "trackeverything": "TrackEverything (Pi3)"}
COLORS = {"gt": "black", "opend4rt": "#d1495b", "flow3r": "#00798c", "arrow": "#edae49", "trackeverything": "#66a182"}
TAGS = ["lost after occlusion", "jumps to another object", "stuck to background", "depth drift", "scale jump", "other"]


def load(points_dir: Path):
    meta = json.loads((points_dir / "clips.json").read_text())
    pts = {}
    for s in SUBSETS:
        p = points_dir / f"points_{s}.npz"
        if p.exists():
            with np.load(p) as z:
                pts[s] = {k: z[k] for k in z.files}
    return pts, meta["checks"]["models"], meta["clips"]


def select(clips: list[dict], models: list[str]) -> list[dict]:
    apd = {c["clip"]: {m: c["models"][m]["apd"] for m in models} for c in clips}
    chosen: dict[int, list[str]] = {}

    def add(cid: int, why: str) -> None:
        chosen.setdefault(cid, []).append(why)

    for a, b in itertools.combinations(models, 2):
        diff = sorted(apd, key=lambda c: apd[c][a] - apd[c][b])
        for c in diff[-3:]:
            add(c, f"{LABEL[a]} ≫ {LABEL[b]} (Δ {apd[c][a] - apd[c][b]:+.2f})")
        for c in diff[:3]:
            add(c, f"{LABEL[b]} ≫ {LABEL[a]} (Δ {apd[c][b] - apd[c][a]:+.2f})")
    for c in sorted(apd, key=lambda c: max(apd[c].values()))[:2]:
        add(c, f"all models fail (best APD {max(apd[c].values()):.2f})")
    rng = np.random.default_rng(0)
    for c in rng.choice(sorted(apd), size=2, replace=False):
        add(int(c), "random reference")
    by_id = {c["clip"]: c for c in clips}
    return [{"clip": c, "subset": by_id[c]["subset"], "sequence": by_id[c]["sequence"], "why": w,
             "apd": apd[c]} for c, w in sorted(chosen.items(), key=lambda kv: (SUBSETS.index(by_id[kv[0]]["subset"]), kv[0]))]


def clip_figure(P: dict, cid: int, models: list[str], title: str, path: Path) -> None:
    sel = P["clip"] == cid
    t = P["t"][sel].astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in models:
        s = P[f"s_{m}"][sel].astype(float)
        T = t.max() + 1
        curve = np.bincount(t, weights=np.nan_to_num(s), minlength=T) / np.maximum(np.bincount(t, minlength=T), 1)
        ax1.plot(np.arange(T), curve, color=COLORS.get(m), label=LABEL.get(m, m))
    ax1.set_xlabel("target frame t")
    ax1.set_ylabel("APD over the clip's tracks")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("per-frame accuracy", fontsize=9)
    track = P["track"][sel]
    dyn = ~P["static"][sel]
    pool = np.unique(track[dyn]) if dyn.any() else np.unique(track)
    rng = np.random.default_rng(cid)
    pick = rng.choice(pool, size=min(8, pool.size), replace=False)
    g = P["g0"][sel].astype(float)
    for k, n in enumerate(pick):
        on = track == n
        order = np.argsort(t[on])
        gx, gz = g[on][order, 0], g[on][order, 2]
        ax2.plot(gx, gz, color="black", lw=1.6, label="GT" if k == 0 else None)
        ax2.plot(gx[:1], gz[:1], "o", color="black", ms=3)
        for m in models:
            p = g[on][order] + P[f"e_{m}"][sel][on][order].astype(float)
            ax2.plot(p[:, 0], p[:, 2], color=COLORS.get(m), lw=0.9, alpha=0.8, label=LABEL.get(m, m) if k == 0 else None)
    ax2.set_xlabel("x (camera 0, m)")
    ax2.set_ylabel("z = depth (camera 0, m)")
    ax2.set_title(f"top-down view of {pick.size} {'dynamic' if dyn.any() else 'static'} tracks", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_aspect("equal", adjustable="datalim")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=Path, default=Path("output/l3d/compare"))
    ap.add_argument("--out", type=Path, default=Path("results/l3d_compare/gallery"))
    args = ap.parse_args()
    pts, models, clips = load(args.points)
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = select(clips, models)
    L = ["# Q4: failure gallery", "",
         "Clips chosen by the rule in `docs/l3d/STAGE1_PLAN.md` (section 4.4), from the clip APDs of the point table. "
         "Each figure: per-frame APD of every model (left) and a top-down view (x-z in camera 0, after the median scale) "
         "of up to 8 random dynamic tracks (right).", "",
         "RGB overlays (`rgb_*.jpg`, frames 0/16/32/47) and MP4s come from `scripts/l3d/render_gallery.py`, which needs the "
         "clips' WorldTrack npz files; the MP4s are not committed.", "",
         "| # | Subset | Sequence | Why chosen | " + " | ".join(f"APD {LABEL.get(m, m)}" for m in models) + " | Figure | RGB overlay |",
         "| ---: | --- | --- | --- | " + " | ".join("---:" for _ in models) + " | --- | --- |"]
    for i, c in enumerate(chosen, 1):
        name = f"{i:02d}_{c['subset']}_{c['sequence']}.png"
        title = f"{c['subset']}/{c['sequence']} — " + "; ".join(c["why"])
        clip_figure(pts[c["subset"]], c["clip"], models, title, args.out / name)
        L.append(f"| {i} | `{c['subset']}` | `{c['sequence']}` | {'; '.join(c['why'])} | "
                 + " | ".join(f"{c['apd'][m]:.3f}" for m in models) + f" | [{name}]({name}) | [rgb_{name[:-4]}.jpg](rgb_{name[:-4]}.jpg) |")
    L += ["", "## Failure tags (fill in independently, then compare)", "",
          "Tags: " + ", ".join(f"`{t}`" for t in TAGS) + ". Fix the list after the first 5 clips.", "",
          "| # | Sequence | Model | Tags (Jaehoon) | Tags (Alicia) | Notes |", "| ---: | --- | --- | --- | --- | --- |"]
    for i, c in enumerate(chosen, 1):
        for m in models:
            L.append(f"| {i} | `{c['sequence']}` | {LABEL.get(m, m)} |  |  |  |")
    (args.out / "index.md").write_text("\n".join(L) + "\n")
    (args.out / "selection.json").write_text(json.dumps(chosen, indent=1))
    print(f"{len(chosen)} clips -> {args.out}")


if __name__ == "__main__":
    main()
