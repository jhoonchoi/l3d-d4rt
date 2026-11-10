# Part A baseline measures for flow3r

The same Part A measures as `results/l3d` (OpenD4RT), run on the same WorldTrack eval clips
(50 per subset, clip `[0, 48)`, query designs 0–5, same seeds), with flow3r answering the queries
through `l3d/flow3r_engine.py`.

- Model: flow3r (Kidrauh/flow3r at `6366c77`, weights `Clara211111/flow3r` `flow3r.bin`), bf16, `pair_chunk=8`.
- Repo commit for the dump: `80ed562`. GPU: 1× A10G (AWS g5.xlarge). Dump time: 3.3 h for 200 clips
  (encode ≈ 9 s, decode ≈ 55 s per clip).
- `track_apd/` also needed an OpenD4RT design-0 dump on the same clips (same checkpoint as `results/l3d`).
- The dump ran in parts (Google Drive limited the data download), and `scripts/l3d/merge_dumps.py` merged them.
  The analyses then ran on the merged dump (same steps as `scripts/l3d/run_flow3r_baseline.sh`).
- `a2_synthetic/` is a copy of `results/l3d/a2_synthetic` (model-free).

## Read these numbers with care

1. **Report headers say "OpenD4RT 48CLIP" and "256x256 model input".** Alicia's report scripts hard-code
   these labels. All numbers in this folder are flow3r's. Pixel errors use the same normalized 256-px unit as
   `results/l3d`, so the two folders compare directly.
2. **Camera swap is degenerate.** flow3r moves a point between cameras with its predicted poses, which is a
   rigid transform. So the swap residual is zero (up to float noise), C and SC in A3 are zero, and the Gate 1
   camera-swap ρ (0.57–0.70) is a correlation of float noise. It is not a pass.
3. **Ray center reduces to reprojection** for the same reason. Its Gate 1 row tests flow3r's local point map
   against its own intrinsics.
4. **Visibility is a proxy.** flow3r has no visibility output. The engine gives +10 logits when the target
   pixel is inside the frame and −10 otherwise, so "model-visible" filters keep 100% of the rows.
5. **All S-type disagreement comes from the flow head.** Source swap with GT `u2` is zero by construction.
6. **PointOdyssey is in flow3r's training data**, so `po_mini` is not a held-out test for flow3r.
7. One `adt_mini` clip has no finite design-0 APD for either model, so `track_apd` uses 49 clips there.

## Headline comparison (flow3r vs OpenD4RT)

| Measure (flow3r / OpenD4RT) | `adt_mini` | `po_mini` | `pstudio_mini` | `ds_mini` |
| --- | --- | --- | --- | --- |
| Design-0 APD (↑) | **0.806** / 0.734 | 0.399 / **0.690** | 0.412 / **0.811** | 0.637 / **0.726** |
| Gate 3 median px (↓) | **1.54** / 1.98 | **1.75** / 1.90 | 9.07 / **2.44** | 1.05 / **0.97** |
| A4 Umeyama rot err, deg (↓) | **0.20** / 0.50 | **0.08** / 0.30 | **0.14** / 0.16 | **0.04** / 0.12 |
| A3 disagreement (1 − T) | 0.002 / 0.060 | 0.004 / 0.160 | 0.352 / 0.249 | 0.229 / 0.096 |
| Gate 1 losses | source swap on `pstudio_mini` only | | | |

OpenD4RT's Gate 1 losses were camera swap on `po_mini` and `ds_mini`.
