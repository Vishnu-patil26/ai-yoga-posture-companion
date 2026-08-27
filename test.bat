@echo off
REM Verify everything with no camera: engine self-test, then a scripted
REM practice session driven through the real pipeline, then the report.
cd /d "%~dp0"
python bootstrap.py --test
if errorlevel 1 exit /b 1
echo.
python bootstrap.py --demo -- --quiet
echo.
python bootstrap.py --report
