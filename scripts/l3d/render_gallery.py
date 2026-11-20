#!/usr/bin/env python3
"""Stage 1 Q4: RGB overlays for the gallery clips (needs their WorldTrack npz files in --data-root).

    python scripts/l3d/render_gallery.py --points output/l3d/compare --gallery results/l3d_compare/gallery \
        --data-root data/worldtrack_release --video-dir output/l3d/compare/gallery_videos

For each clip in ``<gallery>/selection.json``: one panel per model over the RGB frame. Dots = GT pixel of a
track at frame ``t`` (filled: GT-visible, ring: GT-occluded); the line goes to the model's camera-0 point
projected into frame ``t`` with the GT pose and intrinsics (the same rule for every model), colored by the
3D error d (green <= 0.1, orange <= 0.5, red > 0.5; the APD thresholds are 0.1 to 1.0). Up to 120 random
tracks (dynamic first). Writes ``<gallery>/rgb_<name>.jpg`` (frames 0, 16, 32, 47) and an MP4 per clip.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from l3d import geometry as geo  # noqa: E402
from l3d.worldtrack_clip import load_clip  # noqa: E402

LABEL = {"opend4rt": "OpenD4RT", "flow3r": "flow3r", "arrow": "ARROW", "trackeverything": "TrackEverything (Pi3)"}
STILL_FRAMES = (0, 16, 32, 47)
MAX_TRACKS = 120


def err_color(d: np.ndarray) -> np.ndarray:
    c = np.empty((d.size, 3))
    c[:] = (0.85, 0.15, 0.15)
    c[d <= 0.5] = (0.98, 0.6, 0.1)
    c[d <= 0.1] = (0.2, 0.8, 0.3)
    return c


def clip_arrays(P: dict, cid: int, models: list[str], clip, rng) -> dict:
    """Per (frame, picked track): GT pixel, GT visibility, and per model the projected pixel and error."""
    sel = P["clip"] == cid
    track, t = P["track"][sel].astype(int), P["t"][sel].astype(int)
    dyn_tracks = np.unique(track[~P["static"][sel]])
    sta_tracks = np.unique(track[P["static"][sel]])
    pool = np.concatenate([rng.permutation(dyn_tracks), rng.permutation(sta_tracks)])[:MAX_TRACKS]
    keep = np.isin(track, pool)
    T = clip.T
    H, W = clip.hw_native
    K = clip.K_gt
    rel = geo.relative_pose(clip.T_w2c_gt, 0, np.arange(T))  # camera 0 -> camera t
    out = {"pool": pool, "t": t[keep], "track": track[keep], "models": {}}
    g0 = P["g0"][sel][keep].astype(float)
    tt = t[keep]
    for m in models:
        p0 = g0 + P[f"e_{m}"][sel][keep].astype(float)
        pc = np.einsum("nij,nj->ni", rel[tt, :3, :3], p0) + rel[tt, :3, 3]
        uv = geo.project(K, pc)
        uv[pc[:, 2] <= 1e-6] = np.nan
        out["models"][m] = {"uv": uv, "d": P[f"d_{m}"][sel][keep].astype(float)}
    out["gt_uv"] = clip.gt_uv[tt, out["track"]] * np.array([W - 1, H - 1])
    out["gt_vis"] = clip.gt_vis[tt, out["track"]]
    return out


def draw(ax, frame: np.ndarray, A: dict, m: str, t: int, title: str) -> None:
    H, W = frame.shape[:2]
    ax.imshow(frame)
    on = A["t"] == t
    g, vis = A["gt_uv"][on], A["gt_vis"][on]
    uv, d = A["models"][m]["uv"][on], A["models"][m]["d"][on]
    col = err_color(d)
    for i in range(g.shape[0]):
        if np.isfinite(uv[i]).all():
            ax.plot([g[i, 0], uv[i, 0]], [g[i, 1], uv[i, 1]], color=col[i], lw=0.7, alpha=0.9)
    ax.scatter(g[vis, 0], g[vis, 1], s=6, c="white", edgecolors="black", linewidths=0.3, zorder=3)
    ax.scatter(g[~vis, 0], g[~vis, 1], s=9, facecolors="none", edgecolors="white", linewidths=0.6, zorder=3)
    ax.scatter(uv[:, 0], uv[:, 1], s=4, c=col, zorder=4)
    apd = np.nanmean([np.mean(d <= th) for th in (0.1, 0.3, 0.5, 1.0)]) if d.size else np.nan
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_axis_off()
    ax.set_title(f"{title}  t={t}  APD(t)={apd:.2f}", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=Path, default=Path("output/l3d/compare"))
    ap.add_argument("--gallery", type=Path, default=Path("results/l3d_compare/gallery"))
    ap.add_argument("--data-root", type=Path, default=Path("data/worldtrack_release"))
    ap.add_argument("--video-dir", type=Path, default=Path("output/l3d/compare/gallery_videos"))
    args = ap.parse_args()
    meta = json.loads((args.points / "clips.json").read_text())
    models = meta["checks"]["models"]
    chosen = json.loads((args.gallery / "selection.json").read_text())
    args.video_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    for i, c in enumerate(chosen, 1):
        s = c["subset"]
        if s not in cache:
            with np.load(args.points / f"points_{s}.npz") as z:
                cache = {s: {k: z[k] for k in z.files}}
        P = cache[s]
        clip = load_clip(args.data_root / s / f"{c['sequence']}.npz", 0, 48, subset=s)
        A = clip_arrays(P, c["clip"], models, clip, np.random.default_rng(c["clip"]))
        name = f"{i:02d}_{s}_{c['sequence']}"
        head = f"{s}/{c['sequence']} — " + "; ".join(c["why"])
        # still
        fig, axes = plt.subplots(len(STILL_FRAMES), len(models), figsize=(4.2 * len(models), 3.4 * len(STILL_FRAMES)))
        for r, t in enumerate(STILL_FRAMES):
            for k, m in enumerate(models):
                draw(axes[r, k], clip.video[t], A, m, t, LABEL.get(m, m))
        fig.suptitle(head + "\nwhite dot = GT (ring = occluded); line to model point; green d ≤ 0.1, orange ≤ 0.5, red > 0.5",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(args.gallery / f"rgb_{name}.jpg", dpi=80, pil_kwargs={"quality": 82})
        plt.close(fig)
        # video
        with tempfile.TemporaryDirectory() as tmp:
            for t in range(clip.T):
                fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6))
                for k, m in enumerate(models):
                    draw(np.atleast_1d(axes)[k], clip.video[t], A, m, t, LABEL.get(m, m))
                fig.suptitle(head, fontsize=8)
                fig.tight_layout()
                fig.savefig(Path(tmp) / f"{t:03d}.png", dpi=100)
                plt.close(fig)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6", "-i", str(Path(tmp) / "%03d.png"),
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(args.video_dir / f"{name}.mp4")],
                           check=True)
        print(f"{name}: {A['pool'].size} tracks")


if __name__ == "__main__":
    main()
