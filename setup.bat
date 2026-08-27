@echo off
REM One-time setup. Delegates to bootstrap.py, which repairs whatever is
REM missing or broken and then verifies the engine.
cd /d "%~dp0"
python bootstrap.py %*
