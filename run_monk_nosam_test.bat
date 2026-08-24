@echo off
REM monk.glb through segment_api.py WITHOUT SAM3, conditioned on a front view.
REM azimuth 135 is head-on front for monk.glb + data_toolkit/transforms.json
REM (the calibrated camera at azimuth 0 looks at the model's back).
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
set "ROOT=E:\AI_New\ModelGen\SegviGen"
set "OUT=%ROOT%\data_toolkit\assets\monk\nosam_test"
cd /d %ROOT%

E:\AI_New\ModelGen\.venv\Scripts\python.exe segment_api.py ^
  --glb %ROOT%\monk.glb ^
  --no_sam ^
  --azimuth 135 ^
  --out %OUT%\monk_parts.glb ^
  --work_dir %OUT%\work
if errorlevel 1 exit /b 1

echo [5/5] render 3 views of the result
E:\AI_New\ModelGen\.venv\Scripts\python.exe data_toolkit\render_cond_view.py ^
  --glb %OUT%\monk_parts.glb ^
  --transforms %ROOT%\data_toolkit\transforms.json ^
  --out %OUT%\view.png ^
  --azimuths 135,225,0
if errorlevel 1 exit /b 1

echo ALL DONE
