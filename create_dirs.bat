@echo off
mkdir source\architecture_agent 2>nul
mkdir source\tests 2>nul
python source\bootstrap.py
echo.
dir source
