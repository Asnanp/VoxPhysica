@echo off
setlocal
set ROOT=%~dp0
cd /d "%ROOT%"

echo Launching live 3cm research pipeline...
"C:\Users\USER\anaconda3\python.exe" "%ROOT%scripts\rebuild_and_run_research_live.py"

echo.
echo Window kept open so you can inspect the result.
pause

