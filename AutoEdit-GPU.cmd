@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv-gpu\Scripts\python.exe" (
    echo GPU environment missing. Run the GPU setup first.
    pause
    exit /b 1
)
"%~dp0.venv-gpu\Scripts\python.exe" -m autoedit gui -c "%~dp0config.yaml"
if errorlevel 1 pause
