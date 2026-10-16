@echo off
taskkill /FI "WINDOWTITLE eq PDF AI Backend*" /T /F >nul 2>nul
echo PDF AI Assistant backend stopped.
pause
