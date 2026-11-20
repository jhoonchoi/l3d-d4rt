# Stage 1: how 4D reconstruction models differ, point by point

Status: plan, 2026-11-20. Owner: Jaehoon (with Alicia). Builds on Part A (`docs/l3d/EXPERIMENTS.md`),
`results/l3d/baselines/` and `results/l3d_flow3r/`.

## 0. Purpose

Part A measured each model's error with averages per subset. Averages hide most of what we want to know.
For example, two models with the same APD can fail on the same points or on different points, for the
same reason or for different reasons. Before we try to correct errors (Part B), we want to know where
each model is strong and weak, and why.

Stage 1 compares models **on the same query, point by point**. Every model answers the same frame-0
tracks on the same 200 WorldTrack clips. So each point gives a paired comparison, and we can split the
comparison by condition (motion, visibility, frame gap, and so on) without a dataset confound inside a clip.

Stage 1 uses only predictions that already exist. It needs no GPU in the base plan.

## 1. Questions and hypotheses

We ask four questions, plus one optional question. We state the hypotheses before we see the
per-point data. Results that test them are confirmatory. All other cuts are exploratory, and we label
them so.

| # | Question | Hypothesis (stated in advance) |
| --- | --- | --- |
| Q1 | **Complementarity.** Do the models fail on the same points? | **H1.** On dynamic points, the per-point errors of OpenD4RT and flow3r have Spearman ρ < 0.5, and a cross-validated per-track oracle beats the better single model by ≥ 0.03 APD on `po_mini` and `pstudio_mini`. |
| Q2 | **Conditions.** Under which conditions does each model win? | **H2.** flow3r wins on static and visible points in moving-camera clips. OpenD4RT wins on dynamic points, occluded points and large frame gaps, and its lead grows with the frame gap. **H2b.** ARROW leads in most bins; its lead is smallest on static points in `adt_mini`. |
| Q3 | **Error type.** Which kind of error makes the difference? | **H3.** flow3r's deficit on dynamic points is mostly 2D correspondence (image-plane) error, not depth. Its lead on `adt_mini` is mostly camera (pose) error. |
| Q4 | **Examples.** What do the failures look like? | No hypothesis. Q4 gives a taxonomy of failure modes and checks that Q1–Q3 match what we see. |
| Q5 (optional) | **Label-free selection.** Can a model's own signals tell which model is right on a track? | No hypothesis. Q5 links Stage 1 to RQ2 (detection). |

Basis for H1–H3: flow3r is better on `adt_mini` (APD 0.81 vs 0.73) and much worse on `po_mini` and
`pstudio_mini` (0.40 vs 0.69, 0.41 vs 0.81). Its Gate 3 self-correspondence error on `pstudio_mini`
grows from 4 px to 16 px with the frame gap. Its pose error is 2–4 times lower than OpenD4RT's.

## 2. Models and data

| Model | Type | Predictions we have | Location |
| --- | --- | --- | --- |
| OpenD4RT (`48CLIP_9Mix_NoCropAUG`) | Query-based 4D (D4RT re-implementation) | (a) query tables, design 0, 48 frames; (b) upstream Gate 0 predictions, 64 frames | (a) local `output/l3d/dumps/flow3r_T48_opend4rt_d0`; full designs 0–5 on Alicia's machine. (b) Alicia's machine, `output/l3d/baselines/` |
| flow3r (`6366c77`) | Per-frame point maps + poses + flow head | Query tables, designs 0–5, 48 frames | local `output/l3d/dumps/flow3r_T48` |
| ARROW (`e0e1a76`) | 4D tracker | Gate 0 predictions `pred_tracks_ref0 [64, N, 3]`, camera 0 | Alicia's machine, `output/l3d/baselines/` |
| TrackEverything + Pi3 geometry (`dfb29b5`) | Tracker on top of a geometry model | Same format as ARROW | Alicia's machine, `output/l3d/baselines/` |

- **Clips.** 50 clips each from `adt_mini`, `po_mini`, `pstudio_mini` and `ds_mini` (200 clips). One
  `adt_mini` sequence has no frame-0 queries, so 199 clips can be scored. `tum` is not used.
- **TrackEverything** runs with Pi3, because the VGGT-Omega checkpoint is not public. We always call it
  "TrackEverything (Pi3)". Its numbers probably understate the paper's setup.
- **Training data.** PointOdyssey is in flow3r's training data. We will record each model's training sets
  (task 0.6) and mark in-domain subsets in every table.

## 3. Phase 0: one common point table (the critical step)

All four questions use one table with one row per **unit**:

> unit = (clip, track `n`, target frame `t`), for every frame-0-visible track and every `t` in `[0, 48)`.

This is the upstream WorldTrack protocol (frame-0 queries, all frames, occluded frames included), cut to
the 48 frames that our query tables cover. Every model gives a predicted 3D point for each unit.

### 3.1 Two tiers

- **Tier A (all four models): camera-0 point.** For OpenD4RT and flow3r, this is the design-0 row with
  `q_ts = 0`, `q_tt = t`, `q_tc = 0`. For ARROW and TrackEverything, it is `pred_tracks_ref0[t, n]`.
- **Tier B (OpenD4RT and flow3r only): extra outputs.** The point in the target camera (design-0 row with
  `q_tc = t`), the predicted 2D position `p_uv`, and the raw confidence and visibility logits. Tier B
  separates point error from camera error (Q3) and gives the signals for Q5.

### 3.2 Build steps

| Task | What | Check that must pass |
| --- | --- | --- |
| 0.1 | Get ARROW, TrackEverything and OpenD4RT Gate 0 raw predictions from Alicia (dependency D1). | All 199 files are present for each model. |
| 0.2 | Match each Gate 0 query (`query_uv`, native pixels) to a design-0 track (`q_u`, `q_v` normalized with `u = x / (W - 1)`). | Every query matches one track within 0.01 px, and the two query sets are equal. If they are not equal, use the intersection and report its size. |
| 0.3 | Build the point table: keys, GT labels (section 4.2) and, per model, the aligned error `d` and the per-unit score `s`. | Unit tests pass (section 7). |
| 0.4 | **Metric check.** Recompute existing numbers from the point table. | The per-clip camera-0 APD of OpenD4RT and flow3r matches each clip's `d0_apd_global` in the dump manifests (and so `results/l3d_flow3r/track_apd/`) to 1e-6, because both come from the same local tables. OpenD4RT's row-pooled camera-0 APD is within 0.005 of `results/l3d/splits/`, which comes from Alicia's separate dump on another GPU. |
| 0.5 | **Context-length control** (section 3.4). | OpenD4RT's two versions agree (section 3.4). |
| 0.6 | Record each model's training sets. | One table in the results README. |

### 3.3 Alignment and per-unit score

- **Scale.** For each model and clip, we use the upstream global median scale
  (`_compute_scale_factor_global`), fit on that clip's 48-frame units. A secondary analysis uses the
  upstream closed-form Sim(3) fit (`sim3_closed`).
- **Error.** `d = |s_clip * p - g|` in camera-0 coordinates (metres, after scale).
- **Score.** `s = mean over thresholds {0.1, 0.3, 0.5, 1.0} of 1[d <= threshold]`. So `s` is in
  {0, 0.25, 0.5, 0.75, 1}, and the mean of `s` over units is exactly the upstream APD. This lets every
  analysis work on single units and still add up to APD.

### 3.4 Context-length control

The Gate 0 predictions of ARROW and TrackEverything come from 64-frame runs, but the point table uses
frames 0–47. A model that sees 64 frames can predict frames 0–47 differently from a model that sees 48.

OpenD4RT has predictions in both forms, so it measures this effect directly. We compare OpenD4RT's
Gate 0 predictions (frames 0–47) with its design-0 predictions on the same units.

- **Accept** the 64-frame predictions if, for every subset, |ΔAPD| ≤ 0.01 and the per-unit error
  Spearman ρ ≥ 0.9.
- **Otherwise,** rerun ARROW and TrackEverything on the 48-frame clips with Alicia's existing adapters.
  This takes about 0.2 h for ARROW and about 2 h for TrackEverything on one GPU (less than $3).

## 4. Analyses

### 4.1 Q1: complementarity

For each pair of models, per subset and per motion class (static, dynamic):

1. **Error correlation.** Spearman ρ of `d` across units. We report the median of per-clip ρ and the
   pooled ρ, because pooled ρ mixes differences between clips with differences inside a clip.
2. **Failure overlap.** A unit "fails" for a model if it is in that model's worst 20% of the subset (the
   same rule as A3's AUROC target). We report the share of units that fail for both models, and compare
   it with 0.04, the value we expect if failures are independent. We also report the Jaccard index.
3. **Oracle headroom.** The APD of an oracle that picks the better model:
   - **per clip**: one model for the whole clip;
   - **per track**: one model for all frames of a track;
   - **per unit**: one model for each unit (an upper bound only).

   The headroom is the oracle APD minus the APD of the best single model. A minimum of two noisy errors
   is always better than either one, even when the models are equal. So the per-track oracle is
   **cross-validated**: it chooses the model on even frames and is scored on odd frames, and the reverse.
   H1 uses the cross-validated per-track oracle.
4. **All-model view.** For all four models: the share of units that each model alone gets right
   (`s = 1`), that all models get right, and that no model gets right. The "no model" units are the hard
   cases for Stage 2.

### 4.2 Q2: conditions

**Labels.** Each unit gets these labels. All come from GT, never from a model.

| Label | Definition | Bins |
| --- | --- | --- |
| Subset | — | 4 |
| Motion | `is_static` (GT world displacement over the clip < 1% of median depth) | static, dynamic |
| Visibility | `gt_vis[t, n]` | visible, occluded |
| Frame gap | `t` (the source frame is 0) | [0, 4), [4, 8), [8, 16), [16, 32), [32, 48) |
| Parallax | `l3d.splits.parallax_deg` between cameras 0 and `t` | [0, 1), [1, 5), ≥ 5 degrees (as in `splits.py`) |
| Point speed | GT 3D displacement of the track from `t - 1` to `t`, divided by the clip's median depth | clip-pooled quartiles |
| Depth | GT depth at `t` divided by the clip's median depth | < 0.5, 0.5–1, 1–2, ≥ 2 |
| Camera motion | GT rotation angle and translation / median depth between cameras 0 and `t` | quartiles, per component |
| Image position | GT pixel at `t`: distance to the image border, and an off-image flag | inside > 32 px, border, off-image |

**Measures per bin.** The APD of each model, the paired difference ΔAPD with a 95% clip-bootstrap CI,
and the **win rate**: the share of units where model A's score is higher than model B's (ties left out).

**Confounds.** These labels are not independent. Frame gap, parallax and camera motion grow together.
Every `pstudio_mini` clip has a fixed camera and only dynamic humans, and no other subset has a fixed
camera. We use three tools:

1. **Report every split within each subset.** We never pool subsets for a condition claim.
2. **Use a regression to separate the labels.** Per unit, regress the score difference
   Δs = s_A − s_B on binned indicators of all labels, with subset fixed effects. We report the
   coefficients with clip-bootstrap CIs. A label "matters" only if its coefficient stays clear of zero
   when the other labels are in the model.
3. **Use a partial test for the fixed camera.** We cannot separate "fixed camera" from "Panoptic Studio"
   with these clips. As a partial test, we bin the moving-camera clips of the other subsets by their total
   camera motion. If flow3r's deficit grows as camera motion goes to zero, the data support the
   fixed-camera explanation. If it does not, the explanation is more likely the content (humans, close
   range).

### 4.3 Q3: error type

**Tier A (all models): structure of the camera-0 error.** After scale alignment, we split each
model's squared error into nested parts, with the same group definitions as A1:

1. **Gauge**: the part that one per-clip Sim(3) fit removes.
2. **Per-track offset**: the time mean of each track's remaining error vector.
3. **Per-group jitter**: the per-frame mean over each GT motion group, after the offset is removed.
4. **Residual**: what remains.

A model with a large gauge or offset share makes errors that a few anchors could correct (compare
GAUGE). A model with a large residual share makes errors that only better per-point prediction can fix.

**Tier B (OpenD4RT, flow3r): point error vs camera error.** For each unit, we write the camera-0 error
as the target-camera point error moved into camera 0 by the GT pose, plus the remaining part. The
remaining part is the error of the predicted camera pose (`R_gt` is the GT rotation from camera `t` to
camera 0):

    e_cam0 = (p_cam0 - g_cam0)
    e_point = R_gt · (p_cam_t - g_cam_t)
    e_pose = e_cam0 - e_point

We split `e_point` further into a **depth** part (along the GT ray) and an **image-plane** part. For the
image-plane part, we also report the 2D error of `p_uv` in pixels (256-px units, as in Gate 3). A large
2D error means a lost track. A large depth error at a small 2D error means a correct pixel with a wrong
depth.

**Main output.** For each (subset × motion) cell, a stacked bar of the parts for each model. For each
cell where one model wins (from Q2), the parts that carry the difference.

### 4.4 Q4: failure gallery

**Selection.** We fix the rule before we look at the data:

- for each model pair, the 3 clips with the largest ΔAPD in each direction;
- the 2 clips where the best of the four models is worst;
- 2 clips at random, as a reference.

After duplicates are removed, this gives about 16–20 clips.

**For each clip:**

- the video with GT and predicted 2D tracks, colored by error (Tier A models are projected with the GT
  pose and intrinsics);
- a top-down plot of the 3D trajectories;
- a per-frame error curve for each model.

**Use.** Two people (Jaehoon, Alicia) tag each failure on their own with a short taxonomy. The tags are
for example "lost after occlusion", "jumps to another object", "stuck to background", "depth drift" and
"scale jump"; we fix the list after the first 5 clips. We report agreement, then check whether the tags
match the Q2 and Q3 numbers.

**Data.** This needs the RGB frames of the selected clips only (dependency D3).

### 4.5 Q5 (optional): label-free selection

For OpenD4RT and flow3r, can the difference of their confidences per track predict which model is
better on that track? We report the AUROC of (confidence_A − confidence_B) for the label "A has the
lower error". This is the same question as RQ2 (detection), but between models. If the raw files of
ARROW or TrackEverything contain a confidence or visibility output, we add them.

## 5. Statistics

- **Unit of resampling: the clip.** Units inside one clip are not independent. We use a paired bootstrap
  over clips inside each subset (2000 resamples, 95% percentile CI), as in `run_bootstrap.py`.
- **Report per subset.** A "mean over subsets" is only a summary line.
- **A difference is real** only if its CI excludes 0 **and** |ΔAPD| ≥ 0.02 (the Gate 0 tolerance).
- **Small bins.** A bin with fewer than 5 clips or 1000 units shows "n/a".
- **No search over cuts.** H1–H3 are the confirmatory tests. All other results are exploratory, and the
  write-up says so.

## 6. Dependencies, risks and cost

| ID | Item | Effect if missing | Plan |
| --- | --- | --- | --- |
| D1 | ARROW, TrackEverything and OpenD4RT Gate 0 raw predictions (Alicia's `output/l3d/baselines/`) | Tier A has only 2 models | Ask Alicia to upload them (S3 or Drive). Without them, run Q1–Q3 on OpenD4RT and flow3r first. |
| D2 | OpenD4RT full dump (designs 1–5) | Only Q5 and any T/S/C cut need it | Not needed for Q1–Q4. |
| D3 | WorldTrack RGB for the gallery clips (about 20 files) | No Q4 videos | Download only those files. Google Drive blocked about 50 files per IP per day before, so 20 files are safe. |

**Risks**

- **Query sets differ (task 0.2).** The Gate 0 rule also requires a finite pixel and depth at frame 0.
  If the sets differ, we use the intersection and report its size.
- **Context length (section 3.4).** The control decides whether we rerun.
- **Off-image queries on `ds_mini`.** 13.9% of the queries project off-image, some with near-zero depth.
  We report `ds_mini` with all queries and with in-image queries (ARROW's rule).
- **In-domain subsets.** We mark them in every table (task 0.6).

**Cost.** No GPU in the base plan. The fallback reruns cost less than $3.

## 7. Deliverables

**Code**
- `l3d/pointtable.py`: build and load the point table.
- `l3d/compare.py`: Q1–Q3 measures.
- `scripts/l3d/build_point_table.py`, `run_compare.py`, `run_gallery.py`.

**Tests (CPU)**
- GT as the prediction gives APD = 1.
- The per-unit mean of `s` equals the upstream APD on a synthetic clip.
- The oracle is never worse than either model.
- The cross-validated oracle shows no gain for two models with identical-quality independent noise.
- Query matching works with shuffled queries.

**Results: `results/l3d_compare/`**
- `README.md` (findings, caveats, training-set table);
- `q1_complementarity.md`;
- `q2_conditions.md` with figures;
- `q3_error_type.md` with figures;
- `gallery/` (index and stills; large videos not committed).

**Poster figure candidates**
1. Win-rate map over conditions for each model pair.
2. Oracle headroom bars.
3. Error-type stacked bars.
4. Two or three gallery stills.

## 8. Timeline

| Dates | Work | Output |
| --- | --- | --- |
| Nov 20–23 | Phase 0: D1 handoff, point table, metric check, context control | point table, checks pass |
| Nov 23–25 | Q1 and Q2 | `q1_*.md`, `q2_*.md` |
| Nov 26–29 | Q3 (Thanksgiving week: lighter) | `q3_*.md` |
| Nov 30–Dec 2 | Q4 gallery, tagging, write-up | gallery, README with findings |

If D1 is late, Phase 0 builds Tier B (OpenD4RT and flow3r) first, and Q1–Q3 run on two models. ARROW
and TrackEverything join when their files arrive.

## 9. What each outcome means for the next step

| Outcome | Meaning | Next step |
| --- | --- | --- |
| H1 holds (low ρ, cross-validated oracle gain ≥ 0.03) | The models make different errors. | Stage 2 adds more model types. Model selection or combination becomes a candidate method. |
| ARROW wins > 80% of units in every bin | One model dominates on this data. | Study where ARROW still fails ("no model" units, Q1 step 4). These are the hard cases. |
| Condition effects vanish in the regression after subset fixed effects | The differences come from the dataset (domain), not from the conditions. | Stage 2 needs controlled data, for example Kubric clips with set camera motion and object speed. |
| H3 holds (flow3r loses on 2D correspondence) | flow3r's geometry is good, and its flow is the weak part. | Test "flow3r geometry + a strong 2D tracker" as a modular baseline in Stage 2. |
