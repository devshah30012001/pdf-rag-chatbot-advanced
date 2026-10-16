@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title PDF AI Assistant - One Click Launcher

echo =====================================================
echo             PDF AI ASSISTANT
echo =====================================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py"
) else (
  where python >nul 2>&1
  if %errorlevel%==0 (set "PY=python") else (
    echo ERROR: Python 3.11+ is required.
    pause
    exit /b 1
  )
)

if not exist "backend\.venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv "backend\.venv"
  if errorlevel 1 goto :error
)

set "VENV=backend\.venv\Scripts\python.exe"

echo Installing/checking required packages...
"%VENV%" -m pip install -r "backend\requirements.txt"
if errorlevel 1 goto :error

echo.
echo Checking Gemini API key...

REM Validate that .env exists and contains a non-placeholder key.
set "NEED_KEY=0"
if not exist "backend\.env" set "NEED_KEY=1"

if exist "backend\.env" (
  findstr /R /C:"^GOOGLE_API_KEY=" "backend\.env" >nul
  if errorlevel 1 set "NEED_KEY=1"
  findstr /R /C:"^GEMINI_API_KEY=" "backend\.env" >nul
  if not errorlevel 1 set "NEED_KEY=0"
)

if "%NEED_KEY%"=="1" (
  echo.
  echo A valid Gemini API key was not found.
  echo.
  set /p "GKEY=Paste your NEW Gemini API key: "
  if not defined GKEY (
    echo ERROR: No API key entered.
    pause
    exit /b 1
  )
  >"backend\.env" echo GOOGLE_API_KEY=%GKEY%
  >>"backend\.env" echo CHAT_MODEL=gemini-3.1-flash-lite
  >>"backend\.env" echo EMBEDDING_MODEL=gemini-embedding-001
  >>"backend\.env" echo EMBEDDING_PROVIDER=auto
  >>"backend\.env" echo TOP_K=5
  >>"backend\.env" echo CHUNK_SIZE=1000
  >>"backend\.env" echo CHUNK_OVERLAP=150
  >>"backend\.env" echo MAX_FILE_SIZE_MB=20
  echo API key saved.
) else (
  echo API key configuration file found.
)

echo.
echo Starting backend...
start "PDF RAG Backend" "%~dp0backend\START_BACKEND.bat"

echo Waiting for backend...
set /a COUNT=0
:wait
powershell -NoProfile -Command "$r=try { Invoke-WebRequest 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 1 } catch { $null }; if ($r -and $r.StatusCode -eq 200) { exit 0 } else { exit 1 }"
if not errorlevel 1 goto :ready
set /a COUNT+=1
if %COUNT% GEQ 30 goto :error_server
timeout /t 1 /nobreak >nul
goto :wait

:ready
echo Backend is ready.
start "" "http://127.0.0.1:8000"
echo.
echo Application: http://127.0.0.1:8000
echo.
exit /b 0

:error_server
echo ERROR: Backend did not become ready. Check the PDF RAG Backend window.
pause
exit /b 1

:error
echo ERROR: Setup failed. Read the message above.
pause
exit /b 1
