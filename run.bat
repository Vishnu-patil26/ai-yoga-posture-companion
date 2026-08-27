@echo off
REM Start the live trainer. Sets the environment up first if it needs it.
REM Extra arguments are passed through to app.py, e.g.  run.bat --no-voice
cd /d "%~dp0"
python bootstrap.py --run %*
