@echo off
REM Restartable training driver. Runs train.py and, if it dies before reaching --max_steps,
REM resumes from <out_dir>\lora_last.pt; train.py picks the LR schedule back up at the
REM saved step, so an interrupted run continues instead of re-warming up.
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
for /L %%i in (1,1,%ATTEMPTS%) do (
    set "RESUME="
    if exist "%OUT_DIR%\lora_last.pt" set "RESUME=--resume_lora "%OUT_DIR%\lora_last.pt""
    echo [train_loop] attempt %%i of %ATTEMPTS% !RESUME!
    call "!HERE!run_ft.bat" train.py --out_dir "%OUT_DIR%" !RESUME! !ARGS!
    if !errorlevel! EQU 0 (
        echo [train_loop] train.py finished cleanly
        goto done
    )
    echo [train_loop] attempt %%i exited with code !errorlevel!, retrying in 15s
    REM ping instead of timeout: timeout aborts when the process has no console (scheduled task)
    ping -n 16 127.0.0.1 >nul
)
echo [train_loop] giving up after %ATTEMPTS% attempts
:done
endlocal
