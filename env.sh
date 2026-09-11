# SAM3-SegviGen runtime for this AutoDL box (RTX 5090 / CUDA 12.8).
# Usage:  source /root/autodl-tmp/sam3seggen/env.sh

ROOT=/root/autodl-tmp/sam3seggen
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export OPENCV_IO_ENABLE_OPENEXR=1
export ATTN_BACKEND=flash_attn
export SPARSE_CONV_BACKEND=flex_gemm
export FLEX_GEMM_ALGO=explicit_gemm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEGVIGEN_PY_SAM3=/root/autodl-tmp/envs/sam3/bin/python
# X-Part (Hunyuan3D-Part) closes our open, cut parts into solids; its own venv.
export SEGVIGEN_PY_XPART=/root/autodl-tmp/envs/xpart/bin/python
export SEGVIGEN_XPART_ROOT=/root/autodl-tmp/Hunyuan3D-Part/XPart
export SEGVIGEN_XPART_WEIGHTS=/root/autodl-tmp/Hunyuan3D-Part/weights
export SEGVIGEN_SAM3="$ROOT/weights/facebook/sam3"
export SEGVIGEN_DINOV3="$ROOT/weights/facebook/dinov3-vitl16-pretrain-lvd1689m"
# briaai/RMBG-2.0 is gated; public BiRefNet is a drop-in via SEGVIGEN_RMBG
export SEGVIGEN_RMBG="$ROOT/weights/ZhengPeng7/BiRefNet"

# SegviGen / TRELLIS.2 env
export PATH="/root/autodl-tmp/envs/trellis2/bin:$PATH"
alias python-seg=/root/autodl-tmp/envs/trellis2/bin/python
alias python-sam3=/root/autodl-tmp/envs/sam3/bin/python

cd "$ROOT"
echo "sam3seggen ready"
echo "  SegviGen python: /root/autodl-tmp/envs/trellis2/bin/python"
echo "  SAM3 python:     $SEGVIGEN_PY_SAM3"
echo "  SAM3 weights:    $SEGVIGEN_SAM3"
echo "  DINOv3 weights:  $SEGVIGEN_DINOV3"
echo "  cwd:             $ROOT"
