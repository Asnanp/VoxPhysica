@echo off
setlocal
set ROOT=%~dp0
cd /d "%ROOT%"

echo Launching detached real-world height training...
"%ROOT%\.venv-gpu\Scripts\python.exe" "%ROOT%scripts\start_two_cm_push_live.py" --seed 11 --device cuda

echo.
echo The training process is detached from this window.
echo A separate PowerShell watcher should open automatically.
echo If it does not, run:
echo   Get-Content "%ROOT%outputs\two_cm_push_realworld\seed_11\train.stdout.log" -Tail 40 -Wait
echo.
pause
