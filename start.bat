@echo off
setlocal enabledelayedexpansion

echo *******************************************************
echo *                                                     *
echo *           Starting Ulama LLM Trainer...            *
echo *                                                     *
echo *******************************************************
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
    echo [OK] Found Python version !PYTHON_VERSION!
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Please install Node.js 16 or higher from https://nodejs.org/
    pause
    exit /b 1
) else (
    for /f "tokens=1" %%a in ('node --version') do set NODE_VERSION=%%a
    echo [OK] Found Node.js version !NODE_VERSION!
)

echo.
echo Checking for required directories...

REM Create necessary directories
set "dirs_created=0"
if not exist "backend\data" (
    mkdir "backend\data"
    echo [CREATED] backend\data
    set "dirs_created=1"
)
if not exist "backend\uploads" (
    mkdir "backend\uploads"
    echo [CREATED] backend\uploads
    set "dirs_created=1"
)
if not exist "backend\models" (
    mkdir "backend\models"
    echo [CREATED] backend\models
    set "dirs_created=1"
)
if !dirs_created!==0 echo.

REM Check for virtual environment
if not exist "backend\venv" (
    echo Setting up Python virtual environment...
    python -m venv backend\venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    
    echo Installing Python dependencies...
    call backend\venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r backend\requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install Python dependencies
        pause
        exit /b 1
    )
    deactivate
    echo.
)

echo.
echo Starting application...
echo.

REM Start the application with a new console window
start "" cmd /k "cd /d "%~dp0" && python start.py"

echo *******************************************************
echo * Application is starting in a new window...          *
echo *                                                     *
echo * Frontend:    http://localhost:3000                  *
echo * Backend API: http://localhost:8000                  *
echo * API Docs:    http://localhost:8000/docs             *
echo *******************************************************
echo.
pause
