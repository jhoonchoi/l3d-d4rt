#!/usr/bin/env bash
# Fetch flow3r (code pinned to a commit, weights from Hugging Face) for l3d/flow3r_engine.py.
#   bash scripts/l3d/setup_flow3r.sh            # code + 4.0 GB checkpoint
#   SKIP_CKPT=1 bash scripts/l3d/setup_flow3r.sh  # code only (CPU tests)
set -euo pipefail
cd "$(dirname "$0")/../.."

FLOW3R_DIR="${FLOW3R_DIR:-third_party/flow3r}"
FLOW3R_COMMIT="${FLOW3R_COMMIT:-6366c77}"
CKPT_DIR="${CKPT_DIR:-checkpoints/flow3r}"

if [ ! -d "$FLOW3R_DIR/.git" ]; then
  git clone https://github.com/Kidrauh/flow3r "$FLOW3R_DIR"
fi
git -C "$FLOW3R_DIR" fetch --quiet origin
git -C "$FLOW3R_DIR" checkout --quiet "$FLOW3R_COMMIT"
echo "flow3r code at $(git -C "$FLOW3R_DIR" rev-parse --short HEAD) in $FLOW3R_DIR"

if [ "${SKIP_CKPT:-0}" != "1" ]; then
  mkdir -p "$CKPT_DIR"
  if [ ! -s "$CKPT_DIR/flow3r.bin" ]; then
    python - "$CKPT_DIR" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id="Clara211111/flow3r", filename="flow3r.bin", local_dir=sys.argv[1]))
EOF
  fi
  ls -l "$CKPT_DIR/flow3r.bin"
fi
