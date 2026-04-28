@echo off
setlocal
cd /d "%~dp0"
echo Running VoxPhysica Stage 4 flagship height-first GPU push...
"C:\Users\USER\anaconda3\python.exe" "%~dp0scripts\run_stage4_flagship_push.py"
echo.
echo Run finished. Press any key to close.
pause >nul
