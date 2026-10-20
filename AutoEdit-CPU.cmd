@echo off
setlocal
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
set NUMEXPR_NUM_THREADS=2
rem Reuse the environment with all installed dependencies; config selects CPU.
if exist "%~dp0.venv-gpu\Scripts\python.exe" (
    "%~dp0.venv-gpu\Scripts\python.exe" -m autoedit gui -c "%~dp0config.yaml"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m autoedit gui -c "%~dp0config.yaml"
) else (
    echo Python environment missing. Run setup first.
    pause
    exit /b 1
)
if errorlevel 1 pause
