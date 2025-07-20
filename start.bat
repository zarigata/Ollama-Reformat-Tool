@echo off
echo Starting Ulama LLM Trainer...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Node.js is not installed or not in PATH. Please install Node.js 16 or higher.
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "backend\data" mkdir backend\data
if not exist "backend\uploads" mkdir backend\uploads
if not exist "backend\models" mkdir backend\models

REM Start the application
start "" cmd /k "cd /d "%~dp0" && python start.py"

echo.
echo Application is starting...
echo Frontend will be available at: http://localhost:3000
echo Backend API will be available at: http://localhost:8000
echo.
pause
