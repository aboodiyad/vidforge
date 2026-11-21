@echo off
rem Double-click to start VidForge. The first run sets everything up (needs internet).
cd /d "%~dp0"
if not exist ".venv\Scripts\vidforge-gui.exe" (
    echo First run: setting up VidForge. This takes a few minutes.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 (
        echo.
        echo Setup did not finish. Read the message above, then try again.
        pause
        exit /b 1
    )
)
start "" ".venv\Scripts\vidforge-gui.exe"
