#!/usr/bin/env python3
"""Stage 1 analyses Q1-Q3 on the point table -> results/l3d_compare/ (see docs/l3d/STAGE1_PLAN.md).

    python scripts/l3d/run_compare.py --points output/l3d/compare --out results/l3d_compare --questions q1,q2,q3
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from l3d import compare as C  # noqa: E402
from l3d.report import SUBSETS, jsonable  # noqa: E402

LABEL = {"opend4rt": "OpenD4RT", "flow3r": "flow3r", "arrow": "ARROW", "trackeverything": "TrackEverything (Pi3)"}


def f(x, p=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def ci(c, p=3):
    return "n/a" if c is None or not np.all(np.isfinite(c)) else f"[{c[0]:.{p}f}, {c[1]:.{p}f}]"


def load(points_dir: Path) -> tuple[dict, list[str], dict]:
    meta = json.loads((points_dir / "clips.json").read_text())
    pts = {}
    for s in SUBSETS:
        p = points_dir / f"points_{s}.npz"
        if p.exists():
            with np.load(p) as z:
                pts[s] = {k: z[k] for k in z.files}
    return pts, meta["checks"]["models"], meta


# --------------------------------------------------------------------------------------
# Q1
# --------------------------------------------------------------------------------------


def report_q1(res: dict, models: list[str], out: Path) -> None:
    L = ["# Q1: do the models fail on the same points?", "",
         "Stage 1 point table (`docs/l3d/STAGE1_PLAN.md`, section 4.1). Unit = (clip, frame-0 track, target frame), "
         "48-frame clips, camera-0 coordinates, upstream global median scale per model and clip. APD = mean over clips "
         "of the clip's mean unit score (the upstream protocol). Static/dynamic rows keep the clip-level scale (upstream "
         "`dyn` refits it). 95% CIs: paired bootstrap over clips.", ""]
    for pair, by_sub in res.items():
        a, b = pair.split("|")
        A, B = LABEL.get(a, a), LABEL.get(b, b)
        L += [f"## {A} vs {B}", "",
              "### Accuracy and error correlation", "",
              f"| Subset | Points | Units | Clips | APD {A} | APD {B} | Δ ({A} − {B}) | 95% CI | Real? | ρ per clip (median) | ρ pooled | Both in worst 20% (indep. = 0.04) | Jaccard |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
        for s, by_mo in by_sub.items():
            for mo, r in by_mo.items():
                if r is None:
                    continue
                L.append(f"| `{s}` | {mo} | {r['units']} | {r['clips']} | {f(r['apd'][a])} | {f(r['apd'][b])} | "
                         f"{f(r['delta_apd'])} | {ci(r['delta_ci'])} | {r['delta_verdict']} | {f(r['rho_clip_median'], 2)} | "
                         f"{f(r['rho_pooled'], 2)} | {f(r['both_fail'])} | {f(r['jaccard_fail'], 2)} |")
        L += ["", "### Oracle headroom over the better single model", "",
              "Gain = oracle APD − APD of the better model in that row. **Track (CV)** chooses the model per track on even "
              "frames and scores it on odd frames (and the reverse), so it does not profit from noise; it is the H1 measure. "
              "**Unit** is an upper bound only.", "",
              "| Subset | Points | Better model | Clip | Track | Track (CV) | 95% CI (CV) | Unit |",
              "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |"]
        for s, by_mo in by_sub.items():
            for mo, r in by_mo.items():
                if r is None:
                    continue
                o = r["oracle"]
                L.append(f"| `{s}` | {mo} | {LABEL.get(r['best_single'], r['best_single'])} | {f(o['clip']['gain'])} | "
                         f"{f(o['track']['gain'])} | **{f(o['track_cv']['gain'])}** | {ci(o['track_cv']['gain_ci'])} | {f(o['unit']['gain'])} |")
        L += ["", "### Who gets a unit right (error ≤ 0.1, score 1)", "",
              f"| Subset | Points | Only {A} | Only {B} | Both | Neither |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for s, by_mo in by_sub.items():
            for mo, r in by_mo.items():
                if r is None:
                    continue
                am = r["all_model"]
                L.append(f"| `{s}` | {mo} | {f(am['only_' + a])} | {f(am['only_' + b])} | {f(am['both'])} | {f(am['neither'])} |")
        L.append("")
    # H1 verdict (OpenD4RT vs flow3r, dynamic, po and pstudio)
    key = "opend4rt|flow3r"
    if key in res:
        L += ["## H1 check (stated in advance)", "",
              "H1: on dynamic points, Spearman ρ < 0.5 and the cross-validated per-track oracle gains ≥ 0.03 APD "
              "on `po_mini` and `pstudio_mini`.", "",
              "| Subset | ρ per clip (median) | ρ pooled | CV track gain | 95% CI | ρ < 0.5 | gain ≥ 0.03 |",
              "| --- | ---: | ---: | ---: | --- | --- | --- |"]
        for s in ("po_mini", "pstudio_mini"):
            r = res[key].get(s, {}).get("dynamic")
            if r is None:
                continue
            g = r["oracle"]["track_cv"]
            L.append(f"| `{s}` | {f(r['rho_clip_median'], 2)} | {f(r['rho_pooled'], 2)} | {f(g['gain'])} | {ci(g['gain_ci'])} | "
                     f"{'yes' if r['rho_clip_median'] < 0.5 else 'no'} | {'yes' if g['gain_ci'][0] >= 0.03 else ('no' if g['gain_ci'][1] < 0.03 else 'uncertain')} |")
        L.append("")
    (out / "q1_complementarity.md").write_text("\n".join(L))
    # figure: CV-track and unit gains, dynamic and static
    for pair, by_sub in res.items():
        a, b = pair.split("|")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        subs = list(by_sub)
        x = np.arange(len(subs))
        for j, (mo, kind, col) in enumerate((("static", "track_cv", "#7aa6c2"), ("dynamic", "track_cv", "#d1495b"),
                                              ("static", "unit", "#c9dbe6"), ("dynamic", "unit", "#efb7bf"))):
            vals = [by_sub[s][mo]["oracle"][kind]["gain"] if by_sub[s][mo] else np.nan for s in subs]
            ax.bar(x + (j - 1.5) * 0.2, vals, 0.2, color=col, label=f"{mo}, {'track (CV)' if kind == 'track_cv' else 'unit (upper bound)'}")
        ax.axhline(0.03, color="k", lw=0.8, ls="--")
        ax.set_xticks(x, [s.replace("_mini", "") for s in subs])
        ax.set_ylabel("APD gain over better model")
        ax.set_title(f"Oracle headroom: {LABEL.get(a, a)} + {LABEL.get(b, b)}")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"q1_oracle_{a}_{b}.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------------------
# Q2
# --------------------------------------------------------------------------------------


def report_q2(res: dict, reg: dict, cam: dict, models: list[str], out: Path) -> None:
    L = ["# Q2: under which conditions does each model win?", "",
         "Stage 1 point table (`docs/l3d/STAGE1_PLAN.md`, section 4.2). Units are pooled inside a subset (like "
         "`results/l3d/splits`); every label comes from GT. Δ = APD(first) − APD(second). Win rate = share of units "
         "where the first model's score is higher, ties left out. 95% CIs: bootstrap over clips. **Real** = CI excludes "
         "0 and |Δ| ≥ 0.02. Bins with < 5 clips or < 1000 units show n/a. Results are exploratory except H2.", ""]
    for pair, by_sub in res.items():
        a, b = pair.split("|")
        A, B = LABEL.get(a, a), LABEL.get(b, b)
        L += [f"## {A} vs {B}", ""]
        labels = list(next(iter(by_sub.values())).keys())
        for lab in labels:
            L += [f"### {lab}", "",
                  f"| Subset | Bin | Units | Clips | APD {A} | APD {B} | Δ | 95% CI | Real? | Win rate {A} | 95% CI | Ties |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: |"]
            for s, by_lab in by_sub.items():
                for r in by_lab[lab]:
                    if r.get("na"):
                        L.append(f"| `{s}` | {r['bin']} | {r['units']} | {r['clips']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                        continue
                    L.append(f"| `{s}` | {r['bin']} | {r['units']} | {r['clips']} | {f(r['apd_' + a])} | {f(r['apd_' + b])} | "
                             f"{f(r['delta'])} | {ci(r['delta_ci'])} | {r['verdict']} | {f(r['win_rate'])} | {ci(r['win_ci'])} | {f(r['ties'], 2)} |")
            L.append("")
        L += ["### Regression: which labels matter when the others are held fixed", "",
              f"OLS of s({A}) − s({B}) per unit on bin indicators (the first bin of each label is the reference; "
              "\"speed: static\" is the motion label; parallax is left out because inside a clip it follows from camera "
              "motion and the point). One model per subset, so no subset effects are needed. CIs: 500 clip resamples.", ""]
        for s, rows in reg[pair].items():
            L += [f"**`{s}`**", "", "| Term | Coefficient | 95% CI | Real? |", "| --- | ---: | --- | --- |"]
            L += [f"| {r['term']} | {f(r['coef'])} | {ci(r['ci'])} | {r['verdict']} |" for r in rows]
            L.append("")
        rows = cam[pair]
        if rows:
            L += ["### Test of the fixed-camera explanation", "",
                  "Panoptic Studio has fixed cameras, but its content (close-range people) also differs. Inside the other "
                  "subsets, clips with a **still camera** (GT camera translation < 1e-6 and rotation < 1e-3 deg in every "
                  "frame) are compared with moving-camera clips, split into tertiles of mean camera translation from frame 0 "
                  "(/ median depth). If the deficit of the second model is larger on still-camera clips of the same dataset, "
                  "the data support a fixed-camera explanation.", "",
                  "| Subset | Clips group | Clips | Mean cam. translation | Mean cam. rotation (deg) | Mean Δ | 95% CI |",
                  "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
            for s in sorted({r["subset"] for r in rows}, key=SUBSETS.index):
                rs = [r for r in rows if r["subset"] == s]
                still = np.array([r["still_camera"] for r in rs])
                tr = np.array([r["cam_trans_mean"] for r in rs])
                rot = np.array([r["cam_rot_mean"] for r in rs])
                dl_all = np.array([r["delta"] for r in rs])
                groups = []
                if still.any():
                    groups.append(("still camera", still))
                mov = ~still
                q = np.quantile(tr[mov], [1 / 3, 2 / 3])
                k = np.digitize(tr, q, right=True)
                for i, nm in enumerate(("moving, low", "moving, mid", "moving, high")):
                    groups.append((nm, mov & (k == i)))
                for nm, sel in groups:
                    if not sel.any():
                        continue
                    L.append(f"| `{s}` | {nm} | {sel.sum()} | {f(tr[sel].mean())} | {f(rot[sel].mean(), 2)} | "
                             f"{f(dl_all[sel].mean())} | {ci(C.boot_ci(dl_all[sel]))} |")
            L.append("")
    (out / "q2_conditions.md").write_text("\n".join(L))
    # figure: win-rate heatmap per pair
    for pair, by_sub in res.items():
        a, b = pair.split("|")
        subs = list(by_sub)
        rows_lab, mat = [], []
        for lab in by_sub[subs[0]]:
            names = []
            for s in subs:
                for r in by_sub[s][lab]:
                    if r["bin"] not in names:
                        names.append(r["bin"])
            for nm in names:
                vals = []
                for s in subs:
                    r = next((r for r in by_sub[s][lab] if r["bin"] == nm), None)
                    vals.append(np.nan if r is None or r.get("na") else r["win_rate"])
                rows_lab.append(f"{lab}: {nm}")
                mat.append(vals)
        mat = np.array(mat)
        fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(rows_lab) + 1.2))
        im = ax.imshow(mat, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(mat[i, j] - 0.5) > 0.3 else "black")
        ax.set_xticks(range(len(subs)), [s.replace("_mini", "") for s in subs])
        ax.set_yticks(range(len(rows_lab)), rows_lab, fontsize=7)
        ax.set_title(f"Win rate of {LABEL.get(a, a)} over {LABEL.get(b, b)} (blue = {LABEL.get(a, a)})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(out / f"q2_winrate_{a}_{b}.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------------------
# Q3
# --------------------------------------------------------------------------------------


def report_q3(res: dict, models: list[str], out: Path) -> None:
    L = ["# Q3: which kind of error makes the difference?", "",
         "Stage 1 point table (`docs/l3d/STAGE1_PLAN.md`, section 4.3). Values are means of squared camera-0 error "
         "(m², after the upstream median scale) per unit, on units where every model has an output.", "",
         "- **Tier A** (any model): **gauge** = part removed by one Sim(3) per clip; then of the rest: **offset** = time "
         "mean of the GT motion group's error, **jitter** = the group's per-frame mean minus the offset, **residual** = per "
         "unit. Shares are of the mean squared error (MSE).",
         "- **Tier B** (decoded-query models): **point** = target-camera point error moved into camera 0 by the GT pose "
         "(split into **depth** along the GT ray and **image-plane**), **pose** = the rest (camera error), **cross** = 2 "
         "point·pose. **2D px** = median error of the predicted pixel at the target frame (256-px units, `t > 0`).",
         "- Squared errors are dominated by a few very large errors. **All units** shows the full MSE; **d ≤ 1** keeps "
         "units where every model's error is at most 1.0 (the largest APD threshold), so it describes errors that APD can "
         "still see.", ""]
    titles = {"all_units": "All units", "d_le_1": "Units with d ≤ 1 for every model",
              "px_le_2": "Units with d ≤ 1 and a correct pixel (2D error ≤ 2 px) for every model, t > 0"}
    for variant in ("all_units", "d_le_1", "px_le_2"):
        L += [f"## {titles[variant]}", "",
              "| Subset | Points | Units | Model | MSE | Median d | Gauge | Offset | Jitter | Residual | Point (depth) | Point (image-plane) | Pose | Cross | 2D px (median) |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for s, by_mo in res.items():
            for mo, cells in by_mo.items():
                r = cells[variant]
                if r is None:
                    continue
                for m in models:
                    x = r[m]
                    mse = x["mse"]
                    sh = {k: v / mse for k, v in x["tierA"].items()}
                    tb = x.get("tierB")
                    tbs = ["n/a"] * 4 if tb is None else [f(tb["radial_sq"] / mse, 2), f(tb["perp_sq"] / mse, 2),
                                                        f(tb["pose_sq"] / mse, 2), f(tb["cross"] / mse, 2)]
                    L.append(f"| `{s}` | {mo} | {r['units']} | {LABEL.get(m, m)} | {f(mse, 4)} | {f(x['median_d'])} | "
                             f"{f(sh['gauge'], 2)} | {f(sh['offset'], 2)} | {f(sh['jitter'], 2)} | {f(sh['resid'], 2)} | "
                             + " | ".join(tbs) + f" | {f(x.get('px_median', float('nan')), 2)} |")
        L.append("")
    # where the difference comes from (pairs, d<=1 and correct-pixel units, dynamic and static)
    for (a, b), variant in itertools.product(itertools.combinations(models, 2), ("d_le_1", "px_le_2")):
        A, B = LABEL.get(a, a), LABEL.get(b, b)
        L += [f"## Which parts carry the difference: {B} − {A} ({'units with d ≤ 1' if variant == 'd_le_1' else 'd ≤ 1 and correct pixel'})", "",
              f"Each cell is MSE part({B}) − part({A}); positive = {B} has more of that error. Parts add up to ΔMSE "
              "(Tier A and Tier B are two separate splits of the same ΔMSE).", "",
              "| Subset | Points | ΔMSE | Δ gauge | Δ offset | Δ jitter | Δ residual | Δ point (depth) | Δ point (image-plane) | Δ pose | Δ cross |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for s, by_mo in res.items():
            for mo, cells in by_mo.items():
                r = cells[variant]
                if r is None:
                    continue
                xa, xb = r[a], r[b]
                dA = {k: xb["tierA"][k] - xa["tierA"][k] for k in C.TIER_A_PARTS}
                cols = [f(xb["mse"] - xa["mse"], 4)] + [f(dA[k], 4) for k in C.TIER_A_PARTS]
                if "tierB" in xa and "tierB" in xb:
                    cols += [f(xb["tierB"][k] - xa["tierB"][k], 4) for k in ("radial_sq", "perp_sq", "pose_sq", "cross")]
                else:
                    cols += ["n/a"] * 4
                L.append(f"| `{s}` | {mo} | " + " | ".join(cols) + " |")
        L.append("")
    (out / "q3_error_type.md").write_text("\n".join(L))
    # figure: stacked shares (d <= 1), Tier A and Tier B, per subset x motion x model
    for tier, parts, colors in (("A", ("gauge", "offset", "jitter", "resid"), ("#9ab7d3", "#f2b134", "#ed553b", "#5c6b73")),
                                ("B", ("radial_sq", "perp_sq", "pose_sq"), ("#3c6e71", "#99c1b9", "#d9a5b3"))):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, mo in zip(axes, ("static", "dynamic")):
            labels, bottom = [], None
            stacks = {p: [] for p in parts}
            for s, by_mo in res.items():
                r = by_mo[mo]["d_le_1"]
                for m in models:
                    labels.append(f"{s.replace('_mini', '')}\n{LABEL.get(m, m)[:8]}")
                    for p in parts:
                        if r is None or (tier == "B" and "tierB" not in r[m]):
                            stacks[p].append(np.nan)
                        else:
                            src = r[m]["tierA"] if tier == "A" else r[m]["tierB"]
                            stacks[p].append(src[p])
            x = np.arange(len(labels))
            bottom = np.zeros(len(labels))
            for p, col in zip(parts, colors):
                v = np.nan_to_num(np.array(stacks[p]))
                ax.bar(x, v, 0.7, bottom=bottom, color=col, label=p.replace("_sq", "").replace("radial", "point depth").replace("perp", "point image-plane").replace("resid", "residual"))
                bottom += v
            ax.set_xticks(x, labels, fontsize=6)
            ax.set_title(f"{mo} points, d ≤ 1", fontsize=9)
        axes[0].set_ylabel("mean squared error (m²)")
        axes[1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Q3 Tier {tier} error parts", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"q3_tier{tier}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=Path, default=Path("output/l3d/compare"))
    ap.add_argument("--out", type=Path, default=Path("results/l3d_compare"))
    ap.add_argument("--questions", default="q1,q2,q3")
    args = ap.parse_args()
    pts, models, _ = load(args.points)
    args.out.mkdir(parents=True, exist_ok=True)
    qs = set(args.questions.split(","))
    pairs = list(itertools.combinations(models, 2))
    if "q1" in qs:
        t = time.time()
        res = C.q1(pts, models)
        (args.out / "q1_complementarity.json").write_text(json.dumps(jsonable(res), indent=1))
        report_q1(res, models, args.out)
        print(f"q1 done ({time.time() - t:.0f} s)")
    if "q2" in qs:
        t = time.time()
        res = {f"{a}|{b}": {s: C.q2_pair(P, a, b) for s, P in pts.items()} for a, b in pairs}
        reg = {f"{a}|{b}": {s: C.q2_regression(P, a, b) for s, P in pts.items()} for a, b in pairs}
        cam = {f"{a}|{b}": C.q2_fixed_camera_test(pts, a, b) for a, b in pairs}
        (args.out / "q2_conditions.json").write_text(json.dumps(jsonable({"bins": res, "regression": reg, "camera": cam}), indent=1))
        report_q2(res, reg, cam, models, args.out)
        print(f"q2 done ({time.time() - t:.0f} s)")
    if "q3" in qs:
        t = time.time()
        res = C.q3(pts, models)
        (args.out / "q3_error_type.json").write_text(json.dumps(jsonable(res), indent=1))
        report_q3(res, models, args.out)
        print(f"q3 done ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
