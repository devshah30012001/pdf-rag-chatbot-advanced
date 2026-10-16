@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

title PDF AI Assistant - API Key Setup

echo.
echo ================================================
echo       GEMINI API KEY SETUP
echo ================================================
echo.
echo This will create/replace:
echo %CD%\.env
echo.

set /p "GKEY=Paste your NEW Gemini API key: "

if not defined GKEY (
    echo.
    echo ERROR: No API key entered.
    pause
    exit /b 1
)

> ".env" echo GOOGLE_API_KEY=%GKEY%
>> ".env" echo CHAT_MODEL=gemini-3.1-flash-lite
>> ".env" echo EMBEDDING_MODEL=gemini-embedding-001
>> ".env" echo EMBEDDING_PROVIDER=auto
>> ".env" echo LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
>> ".env" echo EMBEDDING_PROVIDER=auto
>> ".env" echo EMBEDDING_BATCH_SIZE=100
>> ".env" echo EMBEDDING_MAX_RETRIES=6
>> ".env" echo EMBEDDING_RETRY_BASE_SECONDS=5
>> ".env" echo EMBEDDING_MAX_RETRY_DELAY_SECONDS=70
>> ".env" echo TOP_K=5
>> ".env" echo CHUNK_SIZE=1000
>> ".env" echo CHUNK_OVERLAP=150
>> ".env" echo MAX_FILE_SIZE_MB=20
>> ".env" echo EMBEDDING_RPM_LIMIT=90
>> ".env" echo EMBEDDING_MAX_RETRIES=6
>> ".env" echo EMBEDDING_RETRY_BASE_SECONDS=5
>> ".env" echo EMBEDDING_MAX_RETRY_DELAY_SECONDS=70
>> ".env" echo EMBEDDING_BATCH_SIZE=100

echo.
echo API key saved successfully.
echo.
pause
