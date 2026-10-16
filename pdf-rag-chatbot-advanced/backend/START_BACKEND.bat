@echo off
cd /d "%~dp0"
title PDF RAG Backend
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run START_PROJECT.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
echo.
echo FastAPI stopped. Check the error above.
pause
