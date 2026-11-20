# Stage 1 results: OpenD4RT vs flow3r, point by point

Plan: `docs/l3d/STAGE1_PLAN.md`. Status 2026-11-20: Phase 0 and Q1–Q3 done for the two Tier B models; Q4 has
the clip selection and figures, but no video overlays yet. ARROW and TrackEverything (Pi3) join when their raw
predictions arrive (dependency D1).

| File | Content |
| --- | --- |
| `q1_complementarity.md`, `q1_oracle_*.png` | Q1: error correlation, failure overlap, oracle headroom, H1 check |
| `q2_conditions.md`, `q2_winrate_*.png` | Q2: accuracy and win rate per condition bin, regression, fixed-camera test |
| `q3_error_type.md`, `q3_tierA.png`, `q3_tierB.png` | Q3: Tier A (gauge/offset/jitter/residual) and Tier B (depth/image-plane/pose) error parts |
| `gallery/` | Q4: 10 selected clips, per-clip figures, empty failure-tag table |
| `*.json` | All numbers behind the reports |

Reproduce (CPU, about 3 minutes in total, needs the two local dumps):

```bash
python scripts/l3d/build_point_table.py --dump opend4rt=output/l3d/dumps/flow3r_T48_opend4rt_d0 \
    --dump flow3r=output/l3d/dumps/flow3r_T48 --out output/l3d/compare
python scripts/l3d/run_compare.py --points output/l3d/compare --out results/l3d_compare
python scripts/l3d/run_gallery.py --points output/l3d/compare --out results/l3d_compare/gallery
```

## Data and checks (Phase 0)

- 199 clips (50 per subset; one `adt_mini` clip has no frame-0 queries), 6.57 M units = (clip, frame-0 track,
  target frame), 48 frames, camera-0 coordinates, upstream global median scale per model and clip.
- The design-0 queries of the two dumps are identical in every clip.
- Each model's per-clip APD from the point table equals the manifest's `d0_apd_global` (largest difference 1e-16).
- Not done yet: the context-length control (needs D1).

## Main findings

**1. The two models fail on different points, but one model is better on most moving points.**
On dynamic points, the per-clip Spearman ρ between the two models' errors is only 0.15–0.47. However, on
`po_mini` and `pstudio_mini`, OpenD4RT is better on almost every track, so choosing per track gains only
0.011 and 0.001 APD (cross-validated). The real headroom is elsewhere: 0.10 on `adt_mini` dynamic points,
0.085 on `ds_mini` static points, and 0.06 on `po_mini` static points.

**2. Point motion is the main condition, then occlusion.** In the regression with all labels together,
moving points favor OpenD4RT by 0.12–0.54 APD on `po_mini`, `pstudio_mini` and `ds_mini`, and more for faster
points. Occluded points favor OpenD4RT on all four subsets (+0.06 to +0.25); on `adt_mini` this reverses
flow3r's lead. The frame gap matters only on `pstudio_mini` (up to +0.14 at gaps of 32–48 frames).
Off-image targets favor OpenD4RT on `po_mini` (+0.24).

**3. A still camera hurts flow3r.** 14 `po_mini` clips (all from the `dancingroom1_3rd` scene) have a fully still
camera. There, flow3r's deficit is 0.49 APD, against 0.10–0.34 on the moving-camera clips of the same subset.
This supports the fixed-camera explanation for `pstudio_mini`. The still clips share one scene, so content is
still partly confounded.

**4. flow3r's deficit is depth (geometry), not 2D correspondence.**
- On units with d ≤ 1, the extra error of flow3r on dynamic points is mostly depth along the ray: on `po_mini`,
  +0.230 m² depth against +0.007 m² image-plane.
- This holds on units where both models have the correct pixel (2D error ≤ 2 px): +0.20 m² depth on `po_mini`
  dynamic points. So the depth error is not caused by a wrong correspondence.
- At the query frame itself (`t = 0`, no tracking at all), flow3r is already behind on dynamic points by 0.30
  (`po_mini`), 0.23 (`pstudio_mini`) and 0.08 (`ds_mini`) APD. That is 72%, 56% and 38% of its deficit over all
  frames.
- The gallery shows the same: on `pstudio_mini/softball_21`, flow3r's points spread along the camera ray between
  1.5 m and 4 m while the GT stays near 2.5 m.

**5. flow3r's lead on `adt_mini` comes from two sources.** On dynamic points it is mostly camera pose error:
OpenD4RT's pose part is 0.84 of its MSE, against 0.17 for flow3r. On static points it is point error (both
depth and image-plane).

**6. A large part of both models' error is one Sim(3) per clip.** On units with d ≤ 1, the gauge share is
0.25–0.97. flow3r's static error is more global than OpenD4RT's: its gauge share is 0.81 on `po_mini` and 0.57 on
`ds_mini`, against 0.47 and 0.27 for OpenD4RT. So a few anchors would correct more of flow3r's static error.

## Hypotheses (stated in advance)

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| **H1**: on dynamic points, ρ < 0.5, and the CV per-track oracle gains ≥ 0.03 on `po_mini` and `pstudio_mini` | **Partly holds.** ρ < 0.5: yes. Gain ≥ 0.03: no | ρ (per clip, median) 0.15 and 0.23; CV gain 0.011 [0.006, 0.017] and 0.001 [0.001, 0.002] |
| **H2**: flow3r wins static and visible points in moving-camera clips; OpenD4RT wins dynamic, occluded and large-gap points, and its lead grows with the gap | **Partly holds.** Dynamic: yes on 3 of 4 subsets. Occluded: yes on 4 of 4. flow3r wins static/visible: only on `adt_mini`. Lead grows with the gap: only on `pstudio_mini` | `q2_conditions.md` |
| **H2b**: ARROW leads in most bins | **Not tested** (needs D1) | — |
| **H3**: flow3r's deficit on dynamic points is mostly image-plane error; its `adt_mini` lead is mostly pose error | **First half rejected:** the deficit is depth, even with the correct pixel. **Second half holds** for dynamic points (pose), not for static points (point error) | `q3_error_type.md`, finding 4 |

## Caveats

- **Two models only.** ARROW and TrackEverything (Pi3) are not in yet (D1). OpenD4RT here is the design-0 dump on
  the same clips as flow3r.
- **PointOdyssey is in flow3r's training data**, so `po_mini` is not a held-out test for flow3r. It is still
  flow3r's weakest subset.
- **Squared errors have heavy tails.** Some flow3r clips have errors of tens of metres. So Q3 reports all units
  and two filtered views (d ≤ 1, and d ≤ 1 with a correct pixel). The filtered views select easier units.
- **Bin-level APDs pool units inside a subset** (like `results/l3d/splits`). Clip-level APDs are means over clips
  (the upstream protocol). The two differ slightly.
- **Static/dynamic rows keep the clip-level scale.** The upstream `dyn` metric refits the scale on dynamic points.
- **Single-unit Tier B parts are exact** (point + pose + cross = d²). **Tier A parts are exact per selection**
  (they are fit on the selected units), so values from different rows do not add up.

## Deviations from the plan

- **Metric check (task 0.4).** `results/l3d/splits` uses the self-frame scale fit on all designs, so it cannot be
  matched exactly. The exact check is against `d0_apd_global`, which uses the upstream camera-0 scale.
- **Tier A split.** It is computed at analysis time on the selected units, not stored per unit. A per-clip fit
  over all units let a few huge errors control the Sim(3), which gave negative gauge shares on subsets.
- **Fixed-camera test.** It uses the 14 still-camera `po_mini` clips found in the data, which is a direct
  within-subset test, instead of only bins of camera motion.
- **Added cuts** (exploratory): units with a correct pixel (Q3), and APD at `t = 0` (finding 4).

## Next steps

1. **D1:** ARROW and TrackEverything (Pi3) raw predictions from Alicia (`output/l3d/baselines/`). Then the
   context-length control, and Q1–Q3 for all four models.
2. **Q4 videos:** download the RGB frames of the 10 gallery clips, render overlays, and tag failures (two people).
3. **Per-frame scale:** test whether flow3r's depth error is a per-frame scale change (fit a scale per frame and
   compare). The `softball_21` figure suggests so.
