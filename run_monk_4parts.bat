@echo off
REM monk.glb -> one glb with exactly 4 sub-parts: armor / staff / base / body.
REM "body=..." merges several SAM3 concepts into a single part, and --unassigned_to body
REM folds unclaimed foreground into it so no fifth grey part appears.
REM azimuth 135 is head-on front for monk.glb + data_toolkit/transforms.json.
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
set "OUT=%ROOT%\data_toolkit\assets\monk\parts4"
cd /d %ROOT%

E:\AI_New\ModelGen\.venv\Scripts\python.exe segment_api.py ^
  --glb %ROOT%\monk.glb ^
  --azimuth 135 ^
  --unassigned_to body ^
  --out %OUT%\monk_parts.glb ^
  --work_dir %OUT%\work ^
  --prompts armor staff base body=head+face+hand+boot+leg
if errorlevel 1 exit /b 1

echo [5/5] render 3 views of the result
E:\AI_New\ModelGen\.venv\Scripts\python.exe data_toolkit\render_cond_view.py ^
  --glb %OUT%\monk_parts.glb ^
  --transforms %ROOT%\data_toolkit\transforms.json ^
  --out %OUT%\view.png ^
  --azimuths 135,225,0
if errorlevel 1 exit /b 1

echo ALL DONE
