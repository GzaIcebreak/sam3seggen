@echo off
REM Run finetune\p3sam_run.py (original P3-SAM auto-mask). P3-SAM's own conda env (`p3sam`) is gone, and the
REM SegviGen venv already carries torch 2.11+cu128 / spconv / torch_scatter / timm; the remaining pure-python
REM deps (addict, fpsample, numba, scikit-learn) were added to it on 2026-09-07.
REM Usage: finetune\run_p3sam.bat --out datasets\ext_bench\p3sam --glb a.glb b.glb
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "HF_ENDPOINT=https://hf-mirror.com"
E:\AI_New\ModelGen\.venv\Scripts\python.exe -u "%~dp0p3sam_run.py" %*
