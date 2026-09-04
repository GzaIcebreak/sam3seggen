@echo off
REM Restartable training driver. Runs train.py and, if it dies before reaching --max_steps,
REM resumes from <out_dir>\lora_last.pt; train.py picks the LR schedule back up at the
REM saved step, so an interrupted run continues instead of re-warming up.
REM
REM Completion is detected by the <out_dir>\done.json sentinel train.py writes, not by the
REM exit code: an exit code has to survive nested .bat calls to reach this loop, and when it
REM does not, the retry starts a fresh run from step 0 and silently overwrites a finished one.
REM Delete done.json to train the same out_dir further (raise --max_steps first).
REM
REM Usage: finetune\train_loop.bat <attempts> <out_dir> <train.py args...>
setlocal enabledelayedexpansion
REM shift moves %0 as well, so the script directory has to be captured before any shifting
set "HERE=%~dp0"
set "ATTEMPTS=%~1"
set "OUT_DIR=%~2"
shift
shift
set "ARGS="
:collect
if "%~1"=="" goto run
set "ARGS=!ARGS! %1"
shift
goto collect

:run
if exist "%OUT_DIR%\done.json" (
    echo [train_loop] %OUT_DIR%\done.json exists, this run already finished; nothing to do
    goto done
)
for /L %%i in (1,1,%ATTEMPTS%) do (
    set "RESUME="
    if exist "%OUT_DIR%\lora_last.pt" set "RESUME=--resume_lora "%OUT_DIR%\lora_last.pt""
    echo [train_loop] attempt %%i of %ATTEMPTS% !RESUME!
    call "!HERE!run_ft.bat" train.py --out_dir "%OUT_DIR%" !RESUME! !ARGS!
    set "RC=!errorlevel!"
    if exist "%OUT_DIR%\done.json" (
        echo [train_loop] train.py finished, done.json written
        goto done
    )
    echo [train_loop] attempt %%i ended with code !RC! and no done.json, retrying in 15s
    REM ping instead of timeout: timeout aborts when the process has no console (scheduled task)
    ping -n 16 127.0.0.1 >nul
)
echo [train_loop] giving up after %ATTEMPTS% attempts
:done
endlocal
