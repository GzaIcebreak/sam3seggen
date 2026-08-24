@echo off
REM SAM3-only audit: render + 2D part map, no SegviGen / Blender.
REM Prompts stay as call-time arguments; nothing is hardcoded in the API.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39 >nul 2>&1
set "HF_ENDPOINT=https://hf-mirror.com"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "ROOT=E:\AI_New\ModelGen\SegviGen"
cd /d %ROOT%

set "AUD135=%ROOT%\data_toolkit\assets\monk\sam3_audit_135"
set "AUD90=%ROOT%\data_toolkit\assets\monk\sam3_audit_90"

echo [audit 135] front view
E:\AI_New\ModelGen\.venv\Scripts\python.exe segment_api.py ^
  --glb %ROOT%\monk.glb ^
  --azimuth 135 ^
  --unassigned_to body ^
  --sam3_only ^
  --work_dir %AUD135% ^
  --prompts armor staff base body=head+face+hand+boot+leg
if errorlevel 1 exit /b 1

echo [audit 90] 3/4 view
E:\AI_New\ModelGen\.venv\Scripts\python.exe segment_api.py ^
  --glb %ROOT%\monk.glb ^
  --azimuth 90 ^
  --unassigned_to body ^
  --sam3_only ^
  --work_dir %AUD90% ^
  --prompts armor staff base body=head+face+hand+boot+leg
if errorlevel 1 exit /b 1

echo ALL DONE
