@echo off
REM run_in_venv.bat — Activate uqala venv and run Python scripts
REM Usage: run_in_venv.bat <script.py> [args...]

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: run_in_venv.bat ^<script.py^> [args...]
    echo.
    echo Example:
    echo   run_in_venv.bat pipelines\level1_interpretable\p1_4_logistic_regression_v80.py --cv 5
    exit /b 1
)

REM Activate the conda environment
call C:\Users\augus\.conda\envs\uqala\Scripts\activate.bat

REM Run the Python script with all arguments
python %*

endlocal
