@echo off
REM Run any finetune/*.py script inside the SegviGen venv with the same environment
REM the inference .bat files use. Usage: finetune\run_ft.bat make_samples_b.py --dataset_root ...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39 >nul 2>&1
set "HF_ENDPOINT=https://hf-mirror.com"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "ATTN_BACKEND=flash_attn"
set "SPARSE_CONV_BACKEND=flex_gemm"
set "FLEX_GEMM_ALGO=explicit_gemm"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "SEGVIGEN_DINOV3=E:\AI_New\ModelGen\weights\facebook\dinov3-vitl16-pretrain-lvd1689m"
set "SEGVIGEN_RMBG=E:\AI_New\ModelGen\weights\briaai\RMBG-2.0"
REM Set SEGVIGEN_PROXY (e.g. http://127.0.0.1:7078) to reach services that need it, such as
REM wandb in online mode. Left unset, nothing is proxied.
if defined SEGVIGEN_PROXY set "HTTP_PROXY=%SEGVIGEN_PROXY%"
if defined SEGVIGEN_PROXY set "HTTPS_PROXY=%SEGVIGEN_PROXY%"
if defined SEGVIGEN_PROXY set "NO_PROXY=localhost,127.0.0.1,hf-mirror.com"
set "ROOT=E:\AI_New\ModelGen\SegviGen"
cd /d %ROOT%
E:\AI_New\ModelGen\.venv\Scripts\python.exe finetune\%*
