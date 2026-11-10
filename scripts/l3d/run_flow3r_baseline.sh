#!/usr/bin/env bash
# Part A baseline measures for flow3r, on the same WorldTrack clips and query designs as OpenD4RT.
# Prereqs on a CUDA box: SKIP_CKPT=1 bash scripts/l3d/setup_cuda_box.sh (env + data) and
# bash scripts/l3d/setup_flow3r.sh (code + 4 GB checkpoint). OpenD4RT design 0 (for track_apd) needs its
# checkpoint: bash scripts/l3d/setup_cuda_box.sh with SKIP_ENV=1 SKIP_DATA=1, or set SKIP_OPEND4RT=1.
#
#   SMOKE=1 bash scripts/l3d/run_flow3r_baseline.sh   # one sequence per subset -> output/l3d/flow3r_smoke
#   bash scripts/l3d/run_flow3r_baseline.sh           # all clips -> results/l3d_flow3r
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-data/worldtrack_release}"
DUMPS="${DUMPS:-output/l3d/dumps}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  LIMIT=(--limit-seqs 1); RUN="${RUN_NAME:-flow3r_smoke}"; RES="${RES:-output/l3d/flow3r_smoke}"; ALIMIT=(--limit 1)
else
  LIMIT=(); RUN="${RUN_NAME:-flow3r_T48}"; RES="${RES:-results/l3d_flow3r}"; ALIMIT=()
fi
D="$DUMPS/$RUN"

echo "[flow3r] dump -> $D"
"$PY" scripts/l3d/dump_queries.py --model flow3r --ckpt-path checkpoints/flow3r/flow3r.bin --amp bf16 \
  --data-root "$DATA_ROOT" --clip-len 48 --designs 0,1,2,3,4,5 --out-root "$DUMPS" --run-name "$RUN" \
  --pair-chunk "${PAIR_CHUNK:-8}" "${LIMIT[@]}"

if [[ "${SKIP_OPEND4RT:-0}" != "1" ]]; then
  echo "[opend4rt] design-0 dump on the same clips"
  "$PY" scripts/l3d/dump_queries.py --model opend4rt \
    --ckpt-path checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt --ckpt-mmap \
    --data-root "$DATA_ROOT" --clip-len 48 --designs 0 --out-root "$DUMPS" --run-name "${RUN}_opend4rt_d0" "${LIMIT[@]}"
fi

mkdir -p "$RES"
cp -r results/l3d/a2_synthetic "$RES/"   # A2 is model-free (synthetic scenes)
for pair in a1_decompose:run_a1_decompose a3_tsc:run_a3_tsc a3_detection:run_a3_detection a4_pose:run_a4_pose \
            gauge:run_gauge gate3_selfcorr:run_gate3_selfcorr; do
  name="${pair%%:*}"; script="${pair#*:}"
  echo "[analysis] $name"
  "$PY" "scripts/l3d/$script.py" --dump "$D" --out "$RES/$name" "${ALIMIT[@]}"
done
"$PY" scripts/l3d/run_gate1.py --results "$RES" --out "$RES/gate1"
"$PY" scripts/l3d/run_bootstrap.py --dump "$D" --results "$RES" "${ALIMIT[@]}"
if [[ "${SKIP_OPEND4RT:-0}" != "1" ]]; then
  "$PY" scripts/l3d/run_track_apd.py --dump "flow3r=$D" --dump "opend4rt=$DUMPS/${RUN}_opend4rt_d0" --out "$RES/track_apd"
fi
cp "$D/manifest.json" "$RES/flow3r_manifest.json"
echo "[done] results in $RES"
