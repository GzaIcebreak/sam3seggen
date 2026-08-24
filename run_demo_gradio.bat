@echo off
REM SegviGen local Gradio demo (http://127.0.0.1:7860)
set "HF_ENDPOINT=https://hf-mirror.com"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "ATTN_BACKEND=flash_attn"
set "SPARSE_CONV_BACKEND=flex_gemm"
set "FLEX_GEMM_ALGO=explicit_gemm"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "SEGVIGEN_DINOV3=E:\AI_New\ModelGen\weights\facebook\dinov3-vitl16-pretrain-lvd1689m"
set "SEGVIGEN_RMBG=E:\AI_New\ModelGen\weights\briaai\RMBG-2.0"
cd /d E:\AI_New\ModelGen\SegviGen
E:\AI_New\ModelGen\.venv\Scripts\python.exe app_local.py
