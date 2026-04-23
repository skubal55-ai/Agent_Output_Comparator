@echo off
REM ─────────────────────────────────────────────────────────────
REM  Agent Output Comparator — Startup Script (Windows)
REM ─────────────────────────────────────────────────────────────

echo.
echo   ⚡  Agent Output Comparator
echo   ─────────────────────────────────────

REM Install Python dependencies
echo   Installing Python dependencies...
pip install -r requirements.txt --quiet

REM Start Flask backend in background
echo   Starting backend on http://localhost:5050 ...
start /B python server.py

REM Wait then open frontend
timeout /t 2 /nobreak >nul
echo   Opening frontend in browser...
start index.html

echo.
echo   Running! Close this window to stop the backend.
echo.
pause
