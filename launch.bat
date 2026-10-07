@echo off
REM Open the routine launcher: profile -> routine -> plan -> practice.
cd /d "%~dp0"
python bootstrap.py --ui %*
