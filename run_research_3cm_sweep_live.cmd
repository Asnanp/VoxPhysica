@echo off
setlocal
cd /d "%~dp0"
echo Running VoxPhysica research height sweep in this window...
"C:\Users\USER\anaconda3\python.exe" "%~dp0scripts\run_research_height_sweep.py"
echo.
echo Sweep finished. Press any key to close.
pause >nul
