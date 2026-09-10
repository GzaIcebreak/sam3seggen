#!/usr/bin/env bash
# Resume-friendly weight downloader (hf-mirror + curl).
# SegviGen ckpts + TRELLIS.2-4B + RMBG do not need a token.
# SAM3 / DINOv3 are gated: accept the licenses on Hugging Face, then:
#   export HF_TOKEN=hf_xxx
#   ./download_weights.sh --gated
set -euo pipefail
ROOT=/root/autodl-tmp/sam3seggen
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export TMPDIR=/root/autodl-tmp/tmp

curl_get() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  echo ">> $dest"
  curl -L --retry 40 --retry-delay 5 --retry-all-errors -C - --output "$dest" "$url"
}

GATED=0
[[ "${1:-}" == "--gated" ]] && GATED=1

# 1) SegviGen checkpoints (~22 GB)
declare -A CKPT=(
  [full_seg.ckpt]=7860370137
  [full_seg_w_2d_map.ckpt]=7860370137
  [interactive_seg.ckpt]=7860389817
)
for name in "${!CKPT[@]}"; do
  dest="$ROOT/ckpt/$name"
  if [[ -f "$dest" && $(stat -c%s "$dest") -eq ${CKPT[$name]} ]]; then
    echo "skip $name"
    continue
  fi
  curl_get "$HF_ENDPOINT/fenghora/SegviGen/resolve/main/$name" "$dest"
done

# 2) TRELLIS.2-4B files used by inference_full.py
T="$ROOT/microsoft/TRELLIS.2-4B"
for f in \
  pipeline.json \
  ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json \
  ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors \
  ckpts/shape_enc_next_dc_f16c32_fp16.json \
  ckpts/shape_enc_next_dc_f16c32_fp16.safetensors \
  ckpts/tex_enc_next_dc_f16c32_fp16.json \
  ckpts/tex_enc_next_dc_f16c32_fp16.safetensors \
  ckpts/shape_dec_next_dc_f16c32_fp16.json \
  ckpts/shape_dec_next_dc_f16c32_fp16.safetensors \
  ckpts/tex_dec_next_dc_f16c32_fp16.json \
  ckpts/tex_dec_next_dc_f16c32_fp16.safetensors
do
  [[ -s "$T/$f" ]] && continue
  curl_get "$HF_ENDPOINT/microsoft/TRELLIS.2-4B/resolve/main/$f" "$T/$f"
done

# 3) RMBG
R="$ROOT/weights/briaai/RMBG-2.0"
for f in model.safetensors config.json preprocessor_config.json birefnet.py BiRefNet_config.py; do
  [[ -s "$R/$f" ]] && continue
  curl_get "$HF_ENDPOINT/briaai/RMBG-2.0/resolve/main/$f" "$R/$f"
done

if [[ "$GATED" -eq 1 ]]; then
  PY=/root/autodl-tmp/envs/sam3/bin/python
  "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.cache/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
root = "/root/autodl-tmp/sam3seggen/weights"
snapshot_download("facebook/sam3", local_dir=f"{root}/facebook/sam3")
snapshot_download("facebook/dinov3-vitl16-pretrain-lvd1689m",
                  local_dir=f"{root}/facebook/dinov3-vitl16-pretrain-lvd1689m")
print("gated models done")
PY
fi

echo "download_weights.sh finished"
