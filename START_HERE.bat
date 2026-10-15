@echo off
REM ============================================================================
REM  AI Yoga Posture & Wellness Companion - one-click setup and start (Windows)
REM
REM  Double-click this file.  The first time it
REM    1. finds Python 3.9-3.12 - or installs Python 3.11 for you,
REM    2. builds a private environment and installs the libraries,
REM    3. fetches the pose model if it is not already here,
REM    4. checks the engine, then opens the app.
REM  Every later time it only opens the app (a few seconds).
REM
REM  Needs: Windows 10/11 (64-bit), internet for the first run, a webcam.
REM  Options:  START_HERE.bat setup-only     set up and verify, do not open the app
REM            START_HERE.bat trainer        open the camera trainer directly
REM ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title AI Yoga Companion - setup and start
echo.
echo  AI Yoga Posture ^& Wellness Companion
echo  --------------------------------------------------------------
echo  First run: about 5 to 10 minutes and an internet connection.
echo  After that it opens straight away.
echo.

call :find_python
if defined PY goto have_python

echo  Python 3.9 to 3.12 was not found on this computer.
echo  Installing Python 3.11 for this user (no administrator needed) ...
call :install_python
call :find_python
if not defined PY goto no_python

:have_python
echo  Python: %PY%
echo.
if /i "%~1"=="setup-only" goto setup_only
if /i "%~1"=="trainer" goto trainer
"%PY%" bootstrap.py --ui
if errorlevel 1 goto fail
goto done

:setup_only
"%PY%" bootstrap.py
if errorlevel 1 goto fail
goto done

:trainer
"%PY%" bootstrap.py --run
if errorlevel 1 goto fail
goto done

:no_python
echo.
echo  Could not install Python automatically.
echo  Install Python 3.11 by hand from https://www.python.org/downloads/release/python-3119/
echo  (tick "Add python.exe to PATH"), then double-click this file again.
goto fail

:fail
echo.
echo  Something did not finish. Read the lines above (the last [FAIL] says why),
echo  fix it, and double-click START_HERE.bat again - it carries on where it stopped.
echo  Common causes: no internet on the first run; the camera is blocked in
echo  Settings ^> Privacy ^& security ^> Camera ^> "Let desktop apps access your camera".
echo.
pause
exit /b 1

:done
endlocal
exit /b 0

REM ---------------------------------------------------------------------------
REM  :find_python  sets PY to a Python 3.9-3.12 that has tkinter, or leaves it unset
REM ---------------------------------------------------------------------------
:find_python
set "PY="
if defined YC_FORCE_NO_PYTHON exit /b 0
for %%V in (3.12 3.11 3.10 3.9) do call :try py -%%V
call :try python
call :try "%LocalAppData%\Programs\Python\Python311\python.exe"
call :try "%LocalAppData%\Programs\Python\Python312\python.exe"
call :try "%LocalAppData%\Programs\Python\Python310\python.exe"
exit /b 0

:try
if defined PY exit /b 0
%* -c "import sys, tkinter; assert (3, 9) <= sys.version_info[:2] <= (3, 12); print(sys.executable)" > "%TEMP%\yc_python.txt" 2>nul
if errorlevel 1 exit /b 0
set /p PY=<"%TEMP%\yc_python.txt"
exit /b 0

REM ---------------------------------------------------------------------------
REM  :install_python  winget first, then the installer from python.org
REM ---------------------------------------------------------------------------
:install_python
if defined YC_DRYRUN goto install_dry
where winget >nul 2>nul
if errorlevel 1 goto install_download
echo  (using winget)
winget install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
call :find_python
if defined PY exit /b 0

:install_download
echo  (downloading the installer from python.org)
set "YC_INSTALLER=%TEMP%\python-3.11.9-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $env:YC_INSTALLER"
if not exist "%YC_INSTALLER%" exit /b 1
"%YC_INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_tcltk=1 Include_test=0
exit /b 0

:install_dry
echo  [dry run] winget install -e --id Python.Python.3.11 --scope user --silent
echo  [dry run] otherwise download python-3.11.9-amd64.exe from python.org and run it with /passive
exit /b 0
