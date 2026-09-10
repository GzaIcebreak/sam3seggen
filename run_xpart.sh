#!/usr/bin/env bash
# Complete our open, surface-split parts into closed solids with X-Part.
# Example: ./run_xpart.sh --glb model.glb --parts out/parts.glb --out_dir out/xpart
#
# Runs in its own venv (a clone of the SegviGen one plus spconv/torch_scatter/torch_cluster
# /pymeshlab/fpsample/diffusers, and with xformers removed -- it refuses flash_attn 2.8.3
# at import time, which is the version the rest of this box is built on).
set -euo pipefail
ROOT=/root/autodl-tmp/sam3seggen
export TMPDIR=/root/autodl-tmp/tmp
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
XPART_ROOT=${XPART_ROOT:-/root/autodl-tmp/Hunyuan3D-Part/XPart}
XPART_WEIGHTS=${XPART_WEIGHTS:-/root/autodl-tmp/Hunyuan3D-Part/weights}
PY=/root/autodl-tmp/envs/xpart/bin/python
exec "$PY" "$ROOT/xpart_complete.py" \
  --xpart_root "$XPART_ROOT" --model_path "$XPART_WEIGHTS" "$@"
