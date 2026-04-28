@echo off
setlocal
cd /d "%~dp0"
echo Running VoxPhysica best-shot Stage 4 height-first GPU push...
"C:\Users\USER\anaconda3\python.exe" "%~dp0scripts\run_best_stage4_push.py"
echo.
echo Run finished. Press any key to close.
pause >nul
